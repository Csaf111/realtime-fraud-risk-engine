"""Score drift detection: Population Stability Index (PSI) of live vs. reference scores.

Training writes a *score reference*: the share of served risk scores falling in each of
the ten ``fre_risk_score`` histogram bins on the most recent slice of the training window.
At serving time a :class:`DriftMonitor` keeps the bins of the last ``window`` live scores
and compares them with that reference:

    PSI = sum_i (live_i - ref_i) * ln(live_i / ref_i)

Common reading: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant shift.

Leakage guarantee: the reference is built from served scores only (features, rules and
the model). It never reads labels, ``isFlaggedFraud`` or the test window.

Pure Python and stdlib only, so it adds nothing to the serving image.
"""

from __future__ import annotations

import json
import math
import threading
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Upper (inclusive) edges of the score bins; identical to the fre_risk_score histogram.
SCORE_BUCKETS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
N_BINS = len(SCORE_BUCKETS)
PSI_EPSILON = 1e-4
REFERENCE_FORMAT = "fre-score-reference/v1"


def score_bin(score: float) -> int:
    """Index of the bin holding ``score`` (Prometheus ``le`` semantics: upper edge inclusive).

    Scores are clipped to [0, 1] first; non-finite scores are rejected.
    """
    if not math.isfinite(score):
        raise ValueError("score must be finite")
    s = min(max(score, 0.0), 1.0)
    for i, edge in enumerate(SCORE_BUCKETS):
        if s <= edge:
            return i
    return N_BINS - 1  # pragma: no cover - unreachable after clipping


def bin_counts(scores: Iterable[float]) -> list[int]:
    counts = [0] * N_BINS
    for s in scores:
        counts[score_bin(s)] += 1
    return counts


def _proportions(counts: Sequence[float]) -> list[float]:
    total = float(sum(counts))
    if total <= 0:
        raise ValueError("counts must contain at least one observation")
    return [c / total for c in counts]


def psi(reference: Sequence[float], live: Sequence[float], epsilon: float = PSI_EPSILON) -> float:
    """Population Stability Index between two binned distributions.

    Both arguments may be counts or proportions; each is normalised. Empty bins are
    floored at ``epsilon`` so the index stays finite.
    """
    if len(reference) != len(live):
        raise ValueError("reference and live must have the same number of bins")
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if any(not math.isfinite(x) or x < 0 for x in (*reference, *live)):
        raise ValueError("bin values must be finite and non-negative")
    ref = [max(p, epsilon) for p in _proportions(reference)]
    cur = [max(p, epsilon) for p in _proportions(live)]
    return float(sum((c - r) * math.log(c / r) for r, c in zip(ref, cur, strict=True)))


@dataclass(frozen=True)
class ScoreReference:
    """Reference score distribution saved by training and loaded by the API."""

    proportions: list[float]
    n: int
    model_weight: float | None = None
    steps: list[int] = field(default_factory=list)
    model_version: str | None = None

    def __post_init__(self) -> None:
        if len(self.proportions) != N_BINS:
            raise ValueError(f"reference needs {N_BINS} bins, got {len(self.proportions)}")
        if any(not math.isfinite(p) or p < 0 for p in self.proportions):
            raise ValueError("reference proportions must be finite and non-negative")
        if not math.isclose(sum(self.proportions), 1.0, abs_tol=1e-6):
            raise ValueError("reference proportions must sum to 1")
        if self.n < 1:
            raise ValueError("reference must be built from at least one score")

    @classmethod
    def from_scores(cls, scores: Sequence[float], **meta: Any) -> ScoreReference:
        counts = bin_counts(scores)
        return cls(proportions=_proportions(counts), n=len(scores), **meta)

    def as_dict(self) -> dict[str, Any]:
        return {"format": REFERENCE_FORMAT, "bins": list(SCORE_BUCKETS), **asdict(self)}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> ScoreReference:
        data = json.loads(Path(path).read_text())
        if data.get("format") != REFERENCE_FORMAT:
            raise ValueError(f"unsupported score reference format: {data.get('format')!r}")
        if [float(b) for b in data.get("bins", [])] != list(SCORE_BUCKETS):
            raise ValueError("score reference bins do not match the serving histogram")
        return cls(
            proportions=[float(p) for p in data["proportions"]],
            n=int(data["n"]),
            model_weight=data.get("model_weight"),
            steps=[int(s) for s in data.get("steps", [])],
            model_version=data.get("model_version"),
        )


class DriftMonitor:
    """Rolling-window PSI of live risk scores against a :class:`ScoreReference`.

    Keeps only bin indices of the last ``window`` scores (O(1) per observation, bounded
    memory). :meth:`psi` returns ``None`` until ``min_samples`` scores have been seen, so a
    freshly started replica does not report noise as drift.
    """

    def __init__(self, reference: ScoreReference, window: int = 5000, min_samples: int = 500):
        if window < 1:
            raise ValueError("window must be >= 1")
        if not 1 <= min_samples <= window:
            raise ValueError("min_samples must be between 1 and window")
        self.reference = reference
        self.window = window
        self.min_samples = min_samples
        self._bins: deque[int] = deque()
        self._counts = [0] * N_BINS
        self._lock = threading.Lock()

    def observe(self, score: float) -> None:
        b = score_bin(score)
        with self._lock:
            if len(self._bins) == self.window:
                self._counts[self._bins.popleft()] -= 1
            self._bins.append(b)
            self._counts[b] += 1

    @property
    def count(self) -> int:
        return len(self._bins)

    def live_counts(self) -> list[int]:
        with self._lock:
            return list(self._counts)

    def psi(self) -> float | None:
        with self._lock:
            if len(self._bins) < self.min_samples:
                return None
            counts = list(self._counts)
        return psi(self.reference.proportions, counts)
