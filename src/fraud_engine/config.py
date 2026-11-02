"""Runtime configuration, read from environment variables (prefix ``FRE_``)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    """Engine settings.

    Attributes:
        review_threshold: score at/above which a transaction is sent to manual review.
        block_threshold: score at/above which a transaction is blocked.
        model_path: optional path to a trained baseline model artifact (JSON).
            When unset, the engine runs in rules-only mode.
        model_weight: blend weight of the model probability vs. the rules score (0-1).
        velocity_window_steps: look-back window for velocity features, in PaySim
            steps (1 step = 1 simulated hour).
        high_amount_threshold: amount above which rule R003 fires.
        max_batch_size: maximum transactions accepted by the batch endpoint.
    """

    review_threshold: float = 0.5
    block_threshold: float = 0.8
    model_path: str | None = None
    model_weight: float = 0.6
    velocity_window_steps: int = 24
    high_amount_threshold: float = 200_000.0
    max_batch_size: int = 1000
    environment: str = field(default="local")

    def __post_init__(self) -> None:
        if not 0.0 <= self.review_threshold <= self.block_threshold <= 1.0:
            raise ValueError("Require 0 <= review_threshold <= block_threshold <= 1")
        if not 0.0 <= self.model_weight <= 1.0:
            raise ValueError("model_weight must be between 0 and 1")
        if self.velocity_window_steps < 1:
            raise ValueError("velocity_window_steps must be >= 1")

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            review_threshold=_float("FRE_REVIEW_THRESHOLD", 0.5),
            block_threshold=_float("FRE_BLOCK_THRESHOLD", 0.8),
            model_path=os.getenv("FRE_MODEL_PATH") or None,
            model_weight=_float("FRE_MODEL_WEIGHT", 0.6),
            velocity_window_steps=_int("FRE_VELOCITY_WINDOW_STEPS", 24),
            high_amount_threshold=_float("FRE_HIGH_AMOUNT_THRESHOLD", 200_000.0),
            max_batch_size=_int("FRE_MAX_BATCH_SIZE", 1000),
            environment=os.getenv("FRE_ENVIRONMENT", "local"),
        )
