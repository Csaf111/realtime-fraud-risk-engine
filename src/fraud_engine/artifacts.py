"""Model artifact loading: one entry point for every servable model type.

Artifacts are JSON files tagged with ``model_type``. :func:`load_model` picks the right
pure-Python class, so ``FRE_MODEL_PATH`` and ``FRE_SHADOW_MODEL_PATH`` accept either a
logistic baseline or a gradient-boosted tree ensemble without extra configuration (a GBT
candidate can be shadowed against a live logistic model, and vice versa).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from .model import LogisticBaseline
from .trees import MODEL_TYPE as GBT_MODEL_TYPE
from .trees import GradientBoostedTrees

LOGISTIC_MODEL_TYPE = "logistic_regression"


class ScoringModel(Protocol):
    """What the engine needs from a model: a probability and additive explanations."""

    kind: str
    version: str
    metadata: dict[str, Any]

    def predict_proba(self, features: Mapping[str, float]) -> float: ...

    def explain(
        self, features: Mapping[str, float], top_k: int = 5
    ) -> tuple[float, list[tuple[str, float, float]]]: ...

    def to_dict(self) -> dict[str, Any]: ...

    def save(self, path: str | Path) -> None: ...


def model_from_dict(d: Mapping[str, Any]) -> ScoringModel:
    # Artifacts written before model_type existed are logistic baselines.
    model_type = d.get("model_type", LOGISTIC_MODEL_TYPE)
    if model_type == LOGISTIC_MODEL_TYPE:
        return LogisticBaseline.from_dict(d)
    if model_type == GBT_MODEL_TYPE:
        return GradientBoostedTrees.from_dict(d)
    raise ValueError(f"Unsupported model_type: {model_type!r}")


def load_model(path: str | Path) -> ScoringModel:
    """Load and validate any supported model artifact."""
    return model_from_dict(json.loads(Path(path).read_text()))
