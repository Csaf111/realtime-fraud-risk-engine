"""Feature-level drift: per-feature PSI of live model inputs vs. the training window.

Score PSI (:mod:`fraud_engine.drift`) says *that* the score distribution moved; this module
says *which input* moved. Training writes a *feature reference*: for every feature in
``FEATURE_NAMES``, decile cut points taken from the most recent slice of the training
window and the share of rows in each resulting bin. At serving time a
:class:`FeatureDriftMonitor` keeps the bin indices of the features of the last ``window``
scored transactions and reports one PSI per feature.

Binning uses quantile edges deduplicated per feature, so binary features (``type_*``,
``orig_drained``) collapse to two or three bins instead of ten mostly-empty ones. A value
falls in bin ``i`` when ``edges[i-1] < x <= edges[i]`` (upper edge inclusive), and values
above the last edge land in the final bin, so live values outside the training range are
still counted (and show up as drift).

Leakage guarantee: the reference is built from the point-in-time features the engine
serves (the same ``compute_features`` code as the API) on the train-window tail. It never
reads labels, ``isFlaggedFraud``, raw IDs, raw ``step`` or the test window, and its feature
set must equal ``FEATURE_NAMES`` exactly, so online and offline binning cannot diverge.

Metric labels are the fixed feature names only (bounded cardinality, no customer data).
Pure Python and stdlib only, so it adds nothing to the serving image.
"""

from __future__ import annotations

import bisect
import json
import math
import threading
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .drift import psi
from .features import FEATURE_NAMES

FEATURE_REFERENCE_FORMAT = "fre-feature-reference/v1"
DEFAULT_QUANTILE_BINS = 10


def quantile_edges(values: Sequence[float], n_bins: int = DEFAULT_QUANTILE_BINS) -> list[float]:
    """Unique interior cut points (nearest-rank quantiles) splitting ``values`` into bins."""
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    xs = sorted(float(v) for v in values)
    if not xs:
        raise ValueError("values must contain at least one observation")
    if any(not math.isfinite(x) for x in xs):
        raise ValueError("values must be finite")
    n = len(xs)
    cuts = {xs[max(0, math.ceil(k * n / n_bins) - 1)] for k in range(1, n_bins)}
    return sorted(cuts)


def value_bin(x: float, edges: Sequence[float]) -> int:
    """Bin index of ``x``: upper edge inclusive, overflow goes to the last bin."""
    if not math.isfinite(x):
        raise ValueError("feature value must be finite")
    return bisect.bisect_left(edges, x)


@dataclass(frozen=True)
class FeatureBins:
    """Quantile edges and reference proportions for one feature."""

    edges: list[float]
    proportions: list[float]

    def __post_init__(self) -> None:
        if not self.edges:
            raise ValueError("a feature needs at least one bin edge")
        if any(not math.isfinite(e) for e in self.edges):
            raise ValueError("bin edges must be finite")
        if any(b <= a for a, b in zip(self.edges, self.edges[1:], strict=False)):
            raise ValueError("bin edges must be strictly increasing")
        if len(self.proportions) != len(self.edges) + 1:
            raise ValueError("need exactly one more proportion than edges")
        if any(not math.isfinite(p) or p < 0 for p in self.proportions):
            raise ValueError("proportions must be finite and non-negative")
        if not math.isclose(sum(self.proportions), 1.0, abs_tol=1e-6):
            raise ValueError("proportions must sum to 1")

    @property
    def n_bins(self) -> int:
        return len(self.proportions)

    @classmethod
    def from_values(
        cls, values: Sequence[float], n_bins: int = DEFAULT_QUANTILE_BINS
    ) -> FeatureBins:
        edges = quantile_edges(values, n_bins)
        counts = [0] * (len(edges) + 1)
        for v in values:
            counts[value_bin(float(v), edges)] += 1
        total = float(len(values))
        return cls(edges=edges, proportions=[c / total for c in counts])


@dataclass(frozen=True)
class FeatureReference:
    """Per-feature reference distributions saved by training and loaded by the API."""

    features: dict[str, FeatureBins]
    n: int
    steps: list[int] = field(default_factory=list)
    model_version: str | None = None

    def __post_init__(self) -> None:
        if tuple(self.features) != FEATURE_NAMES:
            raise ValueError(
                "feature reference does not match the engine's FEATURE_NAMES; "
                "retrain it with the current feature code"
            )
        if self.n < 1:
            raise ValueError("reference must be built from at least one row")

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Mapping[str, float]],
        n_bins: int = DEFAULT_QUANTILE_BINS,
        **meta: Any,
    ) -> FeatureReference:
        """Build from feature dicts (as produced by ``compute_features``)."""
        if not rows:
            raise ValueError("rows must contain at least one observation")
        features = {
            name: FeatureBins.from_values([float(r[name]) for r in rows], n_bins)
            for name in FEATURE_NAMES
        }
        return cls(features=features, n=len(rows), **meta)

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": FEATURE_REFERENCE_FORMAT,
            "n": self.n,
            "steps": list(self.steps),
            "model_version": self.model_version,
            "features": {
                name: {"edges": b.edges, "proportions": b.proportions}
                for name, b in self.features.items()
            },
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> FeatureReference:
        data = json.loads(Path(path).read_text())
        if data.get("format") != FEATURE_REFERENCE_FORMAT:
            raise ValueError(f"unsupported feature reference format: {data.get('format')!r}")
        raw = data.get("features")
        if not isinstance(raw, dict):
            raise ValueError("feature reference has no features")
        features = {
            str(name): FeatureBins(
                edges=[float(e) for e in spec["edges"]],
                proportions=[float(p) for p in spec["proportions"]],
            )
            for name, spec in raw.items()
        }
        return cls(
            features=features,
            n=int(data["n"]),
            steps=[int(s) for s in data.get("steps", [])],
            model_version=data.get("model_version"),
        )


class FeatureDriftMonitor:
    """Rolling-window PSI of every model input against a :class:`FeatureReference`.

    Stores one small tuple of bin indices per observation (bounded memory, O(features)
    per update). :meth:`psi` returns ``None`` until ``min_samples`` observations have been
    seen. Observations containing a non-finite value are skipped and counted rather than
    raised, so monitoring can never fail a live request.
    """

    def __init__(self, reference: FeatureReference, window: int = 5000, min_samples: int = 500):
        if window < 1:
            raise ValueError("window must be >= 1")
        if not 1 <= min_samples <= window:
            raise ValueError("min_samples must be between 1 and window")
        self.reference = reference
        self.window = window
        self.min_samples = min_samples
        self._names = tuple(reference.features)
        self._edges = [reference.features[n].edges for n in self._names]
        self._rows: deque[tuple[int, ...]] = deque()
        self._counts = [[0] * reference.features[n].n_bins for n in self._names]
        self.skipped = 0
        self._lock = threading.Lock()

    def observe(self, features: Mapping[str, float]) -> None:
        values = [float(features[n]) for n in self._names]
        if any(not math.isfinite(v) for v in values):
            with self._lock:
                self.skipped += 1
            return
        row = tuple(value_bin(v, e) for v, e in zip(values, self._edges, strict=True))
        with self._lock:
            if len(self._rows) == self.window:
                old = self._rows.popleft()
                for counts, b in zip(self._counts, old, strict=True):
                    counts[b] -= 1
            self._rows.append(row)
            for counts, b in zip(self._counts, row, strict=True):
                counts[b] += 1

    @property
    def count(self) -> int:
        return len(self._rows)

    def live_counts(self) -> dict[str, list[int]]:
        with self._lock:
            return {n: list(c) for n, c in zip(self._names, self._counts, strict=True)}

    def psi(self) -> dict[str, float] | None:
        with self._lock:
            if len(self._rows) < self.min_samples:
                return None
            snapshot = [list(c) for c in self._counts]
        return {
            name: psi(self.reference.features[name].proportions, counts)
            for name, counts in zip(self._names, snapshot, strict=True)
        }
