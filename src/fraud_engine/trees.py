"""Gradient-boosted trees served in pure Python, with exact TreeSHAP explanations.

The artifact is a plain JSON file holding every regression tree of a binary
gradient-boosting ensemble as flat node arrays (the layout scikit-learn uses internally):

* ``feature[j]`` / ``threshold[j]``: the split at node ``j`` (``x[feature] <= threshold``
  goes left, as in scikit-learn);
* ``left[j]`` / ``right[j]``: child node indices, ``-1`` for a leaf;
* ``value[j]``: the leaf's contribution to the log-odds (learning rate already applied);
* ``cover[j]``: the (weighted) number of training rows that reached node ``j``.

``predict_proba`` is ``sigmoid(base_score + sum(tree(x)))``. Explanations are path-dependent
TreeSHAP (Lundberg et al., 2020, Algorithm 2): exact Shapley values of the model's log-odds
where "feature unknown" means "follow both branches, weighted by training cover". They are
additive like the logistic baseline's contributions::

    sum(contributions) == log_odds(x) - expected_log_odds

so the explanation still accounts for the whole score; ``expected_log_odds`` is the
cover-weighted average log-odds of the training window. Cost is O(trees x leaves x depth^2)
per request with no numpy, which keeps the serving image slim.

Training (``fraud-engine-train --model gbt``) uses scikit-learn's
``GradientBoostingClassifier`` on exactly the same point-in-time features as the logistic
baseline, so the leakage guarantees and online/offline parity are unchanged.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .features import FEATURE_NAMES
from .model import ARTIFACT_SCHEMA_VERSION, sigmoid

MODEL_TYPE = "gradient_boosted_trees"
MAX_TREE_DEPTH = 32  # bounds recursion depth and per-request latency
_COVER_RTOL = 1e-6


@dataclass
class RegressionTree:
    """One tree as flat node arrays. Validated on construction."""

    feature: list[int]
    threshold: list[float]
    left: list[int]
    right: list[int]
    value: list[float]
    cover: list[float]
    depth: int = field(init=False, default=0)
    expected_value: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        n = len(self.feature)
        if n == 0:
            raise ValueError("Tree must have at least one node")
        if not (
            len(self.threshold)
            == len(self.left)
            == len(self.right)
            == len(self.value)
            == len(self.cover)
            == n
        ):
            raise ValueError("Tree node arrays must all have the same length")
        n_features = len(FEATURE_NAMES)
        for j in range(n):
            if not (math.isfinite(self.cover[j]) and self.cover[j] > 0):
                raise ValueError("Tree cover must contain finite positive numbers")
            lo, hi = self.left[j], self.right[j]
            if lo == -1 and hi == -1:
                if not math.isfinite(self.value[j]):
                    raise ValueError("Tree leaf values must be finite")
                continue
            # Children strictly after their parent => acyclic, every path terminates.
            if not (j < lo < n and j < hi < n and lo != hi):
                raise ValueError(f"Tree node {j} has invalid child indices ({lo}, {hi})")
            if not 0 <= self.feature[j] < n_features:
                raise ValueError(f"Tree node {j} splits on unknown feature {self.feature[j]}")
            if not math.isfinite(self.threshold[j]):
                raise ValueError("Tree thresholds must be finite")
            # TreeSHAP's additivity relies on child covers summing to the parent's.
            if not math.isclose(
                self.cover[lo] + self.cover[hi], self.cover[j], rel_tol=_COVER_RTOL
            ):
                raise ValueError(f"Tree node {j} cover does not equal the sum of its children")
        self.depth = self._depth()
        if self.depth > MAX_TREE_DEPTH:
            raise ValueError(f"Tree depth {self.depth} exceeds the maximum of {MAX_TREE_DEPTH}")
        self.expected_value = self._expected(0)

    def _depth(self) -> int:
        depth, stack = 0, [(0, 0)]
        while stack:
            j, d = stack.pop()
            if self.left[j] == -1:
                depth = max(depth, d)
            else:
                stack += [(self.left[j], d + 1), (self.right[j], d + 1)]
        return depth

    def _expected(self, j: int) -> float:
        if self.left[j] == -1:
            return self.value[j]
        lo, hi = self.left[j], self.right[j]
        return (
            self.cover[lo] * self._expected(lo) + self.cover[hi] * self._expected(hi)
        ) / self.cover[j]

    def predict(self, x: Sequence[float]) -> float:
        j = 0
        while self.left[j] != -1:
            j = self.left[j] if x[self.feature[j]] <= self.threshold[j] else self.right[j]
        return self.value[j]

    # ---- TreeSHAP (path-dependent) ------------------------------------------------
    def shap_into(self, x: Sequence[float], phi: list[float]) -> None:
        """Add this tree's exact SHAP values for ``x`` into ``phi`` (one per feature)."""
        self._recurse(x, phi, 0, [], [], [], [], 1.0, 1.0, -1)

    def _recurse(
        self,
        x: Sequence[float],
        phi: list[float],
        j: int,
        d: list[int],
        z: list[float],
        o: list[float],
        w: list[float],
        pz: float,
        po: float,
        pi: int,
    ) -> None:
        d, z, o, w = d[:], z[:], o[:], w[:]
        _extend(d, z, o, w, pz, po, pi)
        if self.left[j] == -1:
            leaf = self.value[j]
            for i in range(1, len(d)):
                total = _unwound_sum(z, o, w, i)
                phi[d[i]] += total * (o[i] - z[i]) * leaf
            return
        f = self.feature[j]
        if x[f] <= self.threshold[j]:
            hot, cold = self.left[j], self.right[j]
        else:
            hot, cold = self.right[j], self.left[j]
        iz = io = 1.0
        for k in range(1, len(d)):
            if d[k] == f:  # feature already on the path: undo its earlier split
                iz, io = z[k], o[k]
                _unwind(d, z, o, w, k)
                break
        cj = self.cover[j]
        self._recurse(x, phi, hot, d, z, o, w, iz * self.cover[hot] / cj, io, f)
        self._recurse(x, phi, cold, d, z, o, w, iz * self.cover[cold] / cj, 0.0, f)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "threshold": self.threshold,
            "left": self.left,
            "right": self.right,
            "value": self.value,
            "cover": self.cover,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RegressionTree:
        return cls(
            feature=[int(v) for v in d["feature"]],
            threshold=[float(v) for v in d["threshold"]],
            left=[int(v) for v in d["left"]],
            right=[int(v) for v in d["right"]],
            value=[float(v) for v in d["value"]],
            cover=[float(v) for v in d["cover"]],
        )


def _extend(
    d: list[int], z: list[float], o: list[float], w: list[float], pz: float, po: float, pi: int
) -> None:
    n = len(d)
    d.append(pi)
    z.append(pz)
    o.append(po)
    w.append(1.0 if n == 0 else 0.0)
    for i in range(n - 1, -1, -1):
        w[i + 1] += po * w[i] * (i + 1) / (n + 1)
        w[i] = pz * w[i] * (n - i) / (n + 1)


def _unwind(d: list[int], z: list[float], o: list[float], w: list[float], i: int) -> None:
    n = len(d) - 1
    carry = w[n]
    oi, zi = o[i], z[i]
    for j in range(n - 1, -1, -1):
        if oi != 0:
            tmp = w[j]
            w[j] = carry * (n + 1) / ((j + 1) * oi)
            carry = tmp - w[j] * zi * (n - j) / (n + 1)
        else:
            w[j] = w[j] * (n + 1) / (zi * (n - j))
    del d[i], z[i], o[i]
    w.pop()


def _unwound_sum(z: list[float], o: list[float], w: list[float], i: int) -> float:
    """Sum of path weights with element ``i`` unwound (without mutating the path)."""
    n = len(w) - 1
    carry = w[n]
    oi, zi = o[i], z[i]
    total = 0.0
    ww = w[:]
    for j in range(n - 1, -1, -1):
        if oi != 0:
            tmp = ww[j]
            ww[j] = carry * (n + 1) / ((j + 1) * oi)
            carry = tmp - ww[j] * zi * (n - j) / (n + 1)
        else:
            ww[j] = ww[j] * (n + 1) / (zi * (n - j))
        total += ww[j]
    return total


@dataclass
class GradientBoostedTrees:
    """Binary gradient-boosted tree ensemble with the same interface as the baseline."""

    feature_names: list[str]
    base_score: float
    trees: list[RegressionTree]
    version: str = "unversioned"
    metadata: dict[str, Any] = field(default_factory=dict)

    kind = "gbt"

    def __post_init__(self) -> None:
        if list(self.feature_names) != list(FEATURE_NAMES):
            raise ValueError(
                "Artifact features do not match the engine's FEATURE_NAMES; "
                "retrain the model with the current feature code."
            )
        if not math.isfinite(self.base_score):
            raise ValueError("Artifact base_score must be finite")
        if not self.trees:
            raise ValueError("Artifact must contain at least one tree")
        self.expected_log_odds = self.base_score + sum(t.expected_value for t in self.trees)

    # ---- inference -----------------------------------------------------------------
    def _vector(self, features: Mapping[str, float]) -> list[float]:
        x = [float(features[f]) for f in self.feature_names]
        if any(not math.isfinite(value) for value in x):
            raise ValueError("Model features must contain finite numbers")
        return x

    def log_odds(self, features: Mapping[str, float]) -> float:
        x = self._vector(features)
        return self.base_score + sum(t.predict(x) for t in self.trees)

    def predict_proba(self, features: Mapping[str, float]) -> float:
        return sigmoid(self.log_odds(features))

    def contributions(self, features: Mapping[str, float]) -> dict[str, float]:
        """Exact TreeSHAP values in log-odds; they sum to ``log_odds - expected_log_odds``."""
        x = self._vector(features)
        phi = [0.0] * len(self.feature_names)
        for tree in self.trees:
            tree.shap_into(x, phi)
        return dict(zip(self.feature_names, phi, strict=True))

    def explain(
        self, features: Mapping[str, float], top_k: int = 5
    ) -> tuple[float, list[tuple[str, float, float]]]:
        """Return probability and the top-k features pushing risk *up*."""
        p = self.predict_proba(features)
        contribs = self.contributions(features)
        ranked = sorted(contribs.items(), key=lambda kv: kv[1], reverse=True)
        top = [(f, float(features[f]), c) for f, c in ranked if c > 0][:top_k]
        return p, top

    # ---- persistence ---------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "model_type": MODEL_TYPE,
            "version": self.version,
            "feature_names": list(self.feature_names),
            "base_score": self.base_score,
            "trees": [t.to_dict() for t in self.trees],
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict()))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> GradientBoostedTrees:
        if d.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported artifact schema_version: {d.get('schema_version')}")
        if d.get("model_type") != MODEL_TYPE:
            raise ValueError(f"Not a {MODEL_TYPE} artifact: {d.get('model_type')!r}")
        return cls(
            feature_names=list(d["feature_names"]),
            base_score=float(d["base_score"]),
            trees=[RegressionTree.from_dict(t) for t in d["trees"]],
            version=str(d.get("version", "unversioned")),
            metadata=dict(d.get("metadata", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> GradientBoostedTrees:
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_sklearn(
        cls,
        clf: Any,
        feature_names: Sequence[str],
        x_sample: Any,
        version: str = "unversioned",
    ) -> GradientBoostedTrees:
        """Export a fitted binary ``sklearn.ensemble.GradientBoostingClassifier``.

        ``x_sample`` (at least one training row) recovers the ensemble's initial log-odds
        without touching private scikit-learn attributes. scikit-learn is only needed at
        training time; this method never imports it.
        """
        if getattr(clf, "n_classes_", None) != 2:
            raise ValueError("Only binary GradientBoostingClassifier models are supported")
        lr = float(clf.learning_rate)
        estimators = [row[0] for row in clf.estimators_]
        row = x_sample[:1]
        tree_sum = sum(float(e.predict(row)[0]) for e in estimators)
        base = float(clf.decision_function(row)[0]) - lr * tree_sum
        trees = []
        for est in estimators:
            t = est.tree_
            leaves = [int(v) == -1 for v in t.children_left]
            trees.append(
                RegressionTree(
                    feature=[
                        -1 if leaf else int(f) for f, leaf in zip(t.feature, leaves, strict=True)
                    ],
                    threshold=[
                        0.0 if leaf else float(v)
                        for v, leaf in zip(t.threshold, leaves, strict=True)
                    ],
                    left=[int(v) for v in t.children_left],
                    right=[int(v) for v in t.children_right],
                    value=[lr * float(v) for v in t.value[:, 0, 0]],
                    cover=[float(v) for v in t.weighted_n_node_samples],
                )
            )
        return cls(feature_names=list(feature_names), base_score=base, trees=trees, version=version)
