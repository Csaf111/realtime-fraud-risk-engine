"""Runtime configuration, read from environment variables (prefix ``FRE_``)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .limits import DEFAULT_MAX_BODY_BYTES, MIN_MAX_BODY_BYTES


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


VELOCITY_BACKENDS = frozenset({"memory", "redis"})


@dataclass(frozen=True)
class Settings:
    """Engine settings.

    Attributes:
        review_threshold: score at/above which a transaction is sent to manual review.
        block_threshold: score at/above which a transaction is blocked.
        model_path: optional path to a trained baseline model artifact (JSON).
            When unset, the engine runs in rules-only mode.
        model_weight: blend weight of the model probability vs. the rules score (0-1).
        rules_floor: makes the rules a hard floor under the blend (0-1): the final score
            is never below ``rules_floor * rules_score``. ``0`` (default) disables it;
            ``1`` means the model can never pull a score below the rules-only score.
        velocity_window_steps: look-back window for velocity features, in PaySim
            steps (1 step = 1 simulated hour).
        high_amount_threshold: amount above which rule R003 fires.
        max_batch_size: maximum transactions accepted by the batch endpoint.
        max_body_bytes: maximum HTTP request body size in bytes, enforced before any JSON
            is parsed (``413`` above it). Must fit a full batch of ``max_batch_size``
            transactions (about 300 bytes each as JSON).
        velocity_backend: ``memory`` (per-process, default) or ``redis`` (shared across
            replicas; requires the ``redis`` extra).
        redis_url: connection URL used when ``velocity_backend == "redis"``.
        redis_key_prefix: namespace for velocity keys in Redis.
        score_reference_path: optional training-time score reference (JSON) for drift
            monitoring. When set, ``/metrics`` exports ``fre_score_psi``.
        feature_reference_path: optional training-time per-feature reference (JSON).
            When set, ``/metrics`` exports ``fre_feature_psi{feature=...}`` for every
            model input.
        drift_window: number of most recent live scores (and feature vectors) compared
            with the references.
        drift_min_samples: live observations needed before PSI is reported.
        shadow_model_path: optional path to a candidate model artifact scored in *shadow*
            mode: it sees exactly the same point-in-time features as the primary path,
            but its score never changes the API response, the decision or the velocity
            state. Only aggregate agreement metrics are exported.
        label_pending_capacity: served decisions (with a ``transaction_id``) kept in memory
            awaiting a delayed label via ``POST /v1/labels``. ``0`` disables label tracking.
        label_window: most recent labelled outcomes used for rolling precision/recall.
        feedback_log_path: optional JSONL file. When set, every label that joins a served
            decision appends ``(served features, label)`` there for offline retraining
            (see :mod:`fraud_engine.feedback`). Requires label tracking to be enabled.
    """

    review_threshold: float = 0.5
    block_threshold: float = 0.8
    model_path: str | None = None
    model_weight: float = 0.6
    rules_floor: float = 0.0
    velocity_window_steps: int = 24
    high_amount_threshold: float = 200_000.0
    max_batch_size: int = 1000
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    environment: str = field(default="local")
    velocity_backend: str = "memory"
    redis_url: str = "redis://localhost:6379/0"
    redis_key_prefix: str = "fre:velocity"
    score_reference_path: str | None = None
    feature_reference_path: str | None = None
    drift_window: int = 5000
    drift_min_samples: int = 500
    shadow_model_path: str | None = None
    label_pending_capacity: int = 100_000
    label_window: int = 5000
    feedback_log_path: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.review_threshold <= self.block_threshold <= 1.0:
            raise ValueError("Require 0 <= review_threshold <= block_threshold <= 1")
        if not 0.0 <= self.model_weight <= 1.0:
            raise ValueError("model_weight must be between 0 and 1")
        if not 0.0 <= self.rules_floor <= 1.0:  # also rejects NaN
            raise ValueError("rules_floor must be between 0 and 1")
        if self.velocity_window_steps < 1:
            raise ValueError("velocity_window_steps must be >= 1")
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if self.max_body_bytes < MIN_MAX_BODY_BYTES:
            raise ValueError(f"max_body_bytes must be >= {MIN_MAX_BODY_BYTES}")
        if self.velocity_backend not in VELOCITY_BACKENDS:
            raise ValueError(f"velocity_backend must be one of {sorted(VELOCITY_BACKENDS)}")
        if not self.redis_key_prefix:
            raise ValueError("redis_key_prefix must be non-empty")
        if self.drift_window < 1:
            raise ValueError("drift_window must be >= 1")
        if not 1 <= self.drift_min_samples <= self.drift_window:
            raise ValueError("drift_min_samples must be between 1 and drift_window")
        if self.label_pending_capacity < 0:
            raise ValueError("label_pending_capacity must be >= 0")
        if self.label_window < 1:
            raise ValueError("label_window must be >= 1")
        if self.feedback_log_path and self.label_pending_capacity == 0:
            raise ValueError("feedback_log_path requires label_pending_capacity > 0")

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            review_threshold=_float("FRE_REVIEW_THRESHOLD", 0.5),
            block_threshold=_float("FRE_BLOCK_THRESHOLD", 0.8),
            model_path=os.getenv("FRE_MODEL_PATH") or None,
            model_weight=_float("FRE_MODEL_WEIGHT", 0.6),
            rules_floor=_float("FRE_RULES_FLOOR", 0.0),
            velocity_window_steps=_int("FRE_VELOCITY_WINDOW_STEPS", 24),
            high_amount_threshold=_float("FRE_HIGH_AMOUNT_THRESHOLD", 200_000.0),
            max_batch_size=_int("FRE_MAX_BATCH_SIZE", 1000),
            max_body_bytes=_int("FRE_MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES),
            environment=os.getenv("FRE_ENVIRONMENT", "local"),
            velocity_backend=(os.getenv("FRE_VELOCITY_BACKEND") or "memory").strip().lower(),
            redis_url=os.getenv("FRE_REDIS_URL") or "redis://localhost:6379/0",
            redis_key_prefix=os.getenv("FRE_REDIS_KEY_PREFIX") or "fre:velocity",
            score_reference_path=os.getenv("FRE_SCORE_REFERENCE_PATH") or None,
            feature_reference_path=os.getenv("FRE_FEATURE_REFERENCE_PATH") or None,
            drift_window=_int("FRE_DRIFT_WINDOW", 5000),
            drift_min_samples=_int("FRE_DRIFT_MIN_SAMPLES", 500),
            shadow_model_path=os.getenv("FRE_SHADOW_MODEL_PATH") or None,
            label_pending_capacity=_int("FRE_LABEL_PENDING_CAPACITY", 100_000),
            label_window=_int("FRE_LABEL_WINDOW", 5000),
            feedback_log_path=os.getenv("FRE_FEEDBACK_LOG_PATH") or None,
        )
