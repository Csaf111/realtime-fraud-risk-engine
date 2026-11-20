"""Delayed-label feedback log: served features + late labels -> a retraining dataset.

The training pipeline rebuilds features offline from PaySim. Once the engine is live, the
richer source is its own traffic: the exact feature vector it served, joined to the fraud
label that arrived days later. :class:`FeedbackLog` writes that join as JSON Lines.

How a record is made:

1. At scoring time the engine computes point-in-time features (velocity snapshot taken
   *before* the event is recorded). When a feedback log is configured,
   :class:`~fraud_engine.outcomes.OutcomeTracker` keeps that dict next to the pending
   decision.
2. When ``POST /v1/labels`` matches the decision, one line is appended::

     {"schema": 1, "transaction_id": "t-1", "step": 7, "model_version": "...",
      "risk_score": 0.91, "decision": "BLOCK", "is_fraud": true,
      "features": {"log_amount": 12.1, ...}}

Why this is leakage-safe and parity-preserving:

* ``features`` are the served values, frozen before the label existed, so a retrain on the
  log sees exactly what the online path saw (online/offline parity by construction, no
  re-derivation that could drift from :func:`~fraud_engine.features.compute_features`).
* The label flows one way: into this file. It never reaches features, rules, the model or
  the ``VelocityStore`` of the running engine.
* ``transaction_id`` and ``step`` are *metadata*: the id dedupes, ``step`` orders records
  for a temporal train/test split. :func:`feedback_dataset` never puts either (nor
  ``risk_score`` / ``decision``, which would teach a model to copy the old one) in ``X``.

Writes are best-effort: an I/O error is logged and counted in
``fre_feedback_write_errors_total`` but never fails the label request. Stdlib only.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from .features import FEATURE_NAMES

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FeedbackRecord:
    """One served decision joined to its delayed label."""

    transaction_id: str
    step: int
    model_version: str
    risk_score: float
    decision: str
    is_fraud: bool
    features: Mapping[str, float]

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema": SCHEMA_VERSION,
                "transaction_id": self.transaction_id,
                "step": self.step,
                "model_version": self.model_version,
                "risk_score": self.risk_score,
                "decision": self.decision,
                "is_fraud": self.is_fraud,
                "features": {name: float(self.features[name]) for name in FEATURE_NAMES},
            },
            separators=(",", ":"),
            allow_nan=False,
        )

    @classmethod
    def from_json(cls, line: str) -> FeedbackRecord:
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError("feedback record must be a JSON object")
        if raw.get("schema") != SCHEMA_VERSION:
            raise ValueError(f"unsupported feedback schema {raw.get('schema')!r}")
        feats = raw.get("features")
        if not isinstance(feats, dict) or tuple(feats) != FEATURE_NAMES:
            # Guards parity: a log written by a different feature set must not be silently
            # re-ordered or padded into the current model's inputs.
            raise ValueError("feedback features do not match FEATURE_NAMES")
        values = {k: float(v) for k, v in feats.items()}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("feedback features must be finite")
        is_fraud = raw.get("is_fraud")
        if not isinstance(is_fraud, bool):
            raise ValueError("is_fraud must be a boolean")
        return cls(
            transaction_id=str(raw["transaction_id"]),
            step=int(raw["step"]),
            model_version=str(raw["model_version"]),
            risk_score=float(raw["risk_score"]),
            decision=str(raw["decision"]),
            is_fraud=is_fraud,
            features=values,
        )


class FeedbackLog:
    """Thread-safe, append-only JSONL sink for :class:`FeedbackRecord`."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self.written = 0
        self.errors = 0

    def append(self, record: FeedbackRecord) -> bool:
        """Append one record. Returns ``False`` (and counts it) on any failure."""
        try:
            line = record.to_json()
            with self._lock, self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                self.written += 1
            return True
        except Exception:  # feedback capture must never fail a label request
            with self._lock:
                self.errors += 1
            log.exception("Failed to append feedback record to %s", self.path)
            return False

    def render(self) -> list[str]:
        """Prometheus exposition lines."""
        with self._lock:
            written, errors = self.written, self.errors
        return [
            "# HELP fre_feedback_records_total Labelled feedback records written for retraining.",
            "# TYPE fre_feedback_records_total counter",
            f"fre_feedback_records_total {written}",
            "# HELP fre_feedback_write_errors_total Feedback records that failed to write.",
            "# TYPE fre_feedback_write_errors_total counter",
            f"fre_feedback_write_errors_total {errors}",
        ]


def read_feedback(path: str | Path) -> Iterator[FeedbackRecord]:
    """Yield records from a feedback log, raising ``ValueError`` with the line number on a
    malformed line (a corrupt training set should fail loudly, not shrink silently)."""
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                yield FeedbackRecord.from_json(line)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc


def dedupe_feedback(
    records: list[FeedbackRecord] | Iterator[FeedbackRecord],
) -> list[FeedbackRecord]:
    """Records in time order (by ``step``), one per transaction id; the last one wins."""
    latest: dict[str, FeedbackRecord] = {}
    for rec in records:
        latest.pop(rec.transaction_id, None)
        latest[rec.transaction_id] = rec
    return sorted(latest.values(), key=lambda r: r.step)  # stable: ties keep log order


def feedback_dataset(
    records: list[FeedbackRecord] | Iterator[FeedbackRecord],
) -> tuple[list[list[float]], list[int], list[int]]:
    """Turn records into ``(X, y, steps)`` in time order, deduplicated by transaction id.

    ``X`` rows follow ``FEATURE_NAMES``; ``y`` is ``is_fraud`` as 0/1; ``steps`` is returned
    only so callers can make a temporal split. If an id appears more than once (a log
    concatenated from several replicas or restarts), the last record wins.
    """
    ordered = dedupe_feedback(records)
    X = [[float(r.features[name]) for name in FEATURE_NAMES] for r in ordered]
    y = [int(r.is_fraud) for r in ordered]
    steps = [r.step for r in ordered]
    return X, y, steps
