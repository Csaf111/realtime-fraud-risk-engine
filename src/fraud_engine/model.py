"""Explainable logistic-regression baseline, served in pure Python.

The artifact is a plain JSON file holding the standardisation parameters and the
coefficients. Because the model is linear in standardised features, each feature's
contribution to the log-odds is exactly ``coef_i * (x_i - mean_i) / scale_i`` — the
explanation is the model, not an approximation of it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .features import FEATURE_NAMES

ARTIFACT_SCHEMA_VERSION = 1


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


@dataclass
class LogisticBaseline:
    feature_names: list[str]
    mean: list[float]
    scale: list[float]
    coef: list[float]
    intercept: float
    version: str = "unversioned"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.feature_names)
        if not (len(self.mean) == len(self.scale) == len(self.coef) == n):
            raise ValueError("Artifact arrays must all match feature_names length")
        if list(self.feature_names) != list(FEATURE_NAMES):
            raise ValueError(
                "Artifact features do not match the engine's FEATURE_NAMES; "
                "retrain the model with the current feature code."
            )
        # Reject corrupt model parameters rather than silently changing inference.
        for label, values in (("mean", self.mean), ("scale", self.scale), ("coef", self.coef)):
            if any(not math.isfinite(value) for value in values):
                raise ValueError(f"Artifact {label} must contain finite numbers")
        if not math.isfinite(self.intercept):
            raise ValueError("Artifact intercept must be finite")
        # Preserve the existing zero-variance feature convention after validation.
        self.scale = [value if value > 0 else 1.0 for value in self.scale]

    # ---- inference -----------------------------------------------------------------
    def _standardise(self, x: list[float]) -> list[float]:
        return [(xi - m) / s for xi, m, s in zip(x, self.mean, self.scale, strict=True)]

    def contributions(self, features: Mapping[str, float]) -> dict[str, float]:
        x = [float(features[f]) for f in self.feature_names]
        if any(not math.isfinite(value) for value in x):
            raise ValueError("Model features must contain finite numbers")
        z = self._standardise(x)
        return {f: c * zi for f, c, zi in zip(self.feature_names, self.coef, z, strict=True)}

    def predict_proba(self, features: Mapping[str, float]) -> float:
        return sigmoid(self.intercept + sum(self.contributions(features).values()))

    def explain(
        self, features: Mapping[str, float], top_k: int = 5
    ) -> tuple[float, list[tuple[str, float, float]]]:
        """Return probability and the top-k features pushing risk *up*."""
        contribs = self.contributions(features)
        p = sigmoid(self.intercept + sum(contribs.values()))
        ranked = sorted(contribs.items(), key=lambda kv: kv[1], reverse=True)
        top = [(f, float(features[f]), c) for f, c in ranked if c > 0][:top_k]
        return p, top

    # ---- persistence ---------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "model_type": "logistic_regression",
            "version": self.version,
            "feature_names": list(self.feature_names),
            "mean": self.mean,
            "scale": self.scale,
            "coef": self.coef,
            "intercept": self.intercept,
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> LogisticBaseline:
        if d.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported artifact schema_version: {d.get('schema_version')}")
        return cls(
            feature_names=list(d["feature_names"]),
            mean=[float(v) for v in d["mean"]],
            scale=[float(v) for v in d["scale"]],
            coef=[float(v) for v in d["coef"]],
            intercept=float(d["intercept"]),
            version=str(d.get("version", "unversioned")),
            metadata=dict(d.get("metadata", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> LogisticBaseline:
        return cls.from_dict(json.loads(Path(path).read_text()))
