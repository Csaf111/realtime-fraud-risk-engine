"""Decision-threshold selection from an alert budget.

Fraud ops teams usually think in capacity, not probabilities: "we can review 1% of
traffic and auto-block at most 0.2%". This module turns those budgets into
``review_threshold`` / ``block_threshold`` values.

Leakage guarantee: thresholds are chosen from the *score distribution* of a validation
window only. No labels are read during selection, and the validation window is the most
recent slice of the training period, so the held-out test window is never touched until
the chosen thresholds are evaluated on it.

Pure Python on purpose so it can be reused from the serving path without numpy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any


def blend_risk(model_score: float, rules_score: float, model_weight: float) -> float:
    """Final risk score exactly as served: blend, clip to [0, 1], round to 6 dp."""
    risk = model_weight * model_score + (1 - model_weight) * rules_score
    return round(min(max(risk, 0.0), 1.0), 6)


def _check_rate(name: str, rate: float) -> None:
    if not (math.isfinite(rate) and 0.0 <= rate <= 1.0):
        raise ValueError(f"{name} must be a finite rate between 0 and 1, got {rate!r}")


def alert_rate(scores: Sequence[float], threshold: float) -> float:
    """Fraction of scores at or above ``threshold`` (the engine uses ``>=``)."""
    if not scores:
        raise ValueError("scores must be non-empty")
    return sum(1 for s in scores if s >= threshold) / len(scores)


def threshold_for_alert_rate(scores: Sequence[float], max_alert_rate: float) -> float:
    """Lowest threshold whose alert rate on ``scores`` does not exceed ``max_alert_rate``.

    Ties are respected: every score equal to the threshold alerts, so a block of tied
    scores is either fully in or fully out. If even the top score's tie group exceeds the
    budget, the threshold is placed just above the maximum score (zero alerts). Thresholds
    are capped at 1.0 because the engine's settings require ``<= 1``; if many scores sit
    exactly at 1.0 the budget may then be unattainable, which the caller can detect with
    :func:`alert_rate`.
    """
    _check_rate("max_alert_rate", max_alert_rate)
    if not scores:
        raise ValueError("scores must be non-empty")
    for s in scores:
        if not math.isfinite(s):
            raise ValueError("scores must be finite")

    n = len(scores)
    allowed = max_alert_rate * n
    ordered = sorted(scores, reverse=True)
    best: float | None = None
    i = 0
    while i < n:
        value = ordered[i]
        j = i
        while j < n and ordered[j] == value:
            j += 1
        # j scores are >= value
        if j > allowed + 1e-9:
            break
        best = value
        i = j
    if best is None:
        best = math.nextafter(ordered[0], math.inf)
    return min(best, 1.0)


@dataclass(frozen=True)
class ThresholdSelection:
    review_budget: float
    block_budget: float
    review_threshold: float
    block_threshold: float
    validation_review_rate: float
    validation_block_rate: float
    budget_met: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_thresholds(
    validation_scores: Sequence[float], review_budget: float, block_budget: float
) -> ThresholdSelection:
    """Choose review/block thresholds from alert budgets (fractions of traffic).

    ``review_budget`` is the share of traffic allowed to be REVIEW *or* BLOCK;
    ``block_budget`` is the share allowed to be BLOCK, so it must not exceed it.
    """
    _check_rate("review_budget", review_budget)
    _check_rate("block_budget", block_budget)
    if block_budget > review_budget:
        raise ValueError("block_budget must be <= review_budget")
    review_t = threshold_for_alert_rate(validation_scores, review_budget)
    block_t = max(threshold_for_alert_rate(validation_scores, block_budget), review_t)
    review_rate = alert_rate(validation_scores, review_t)
    block_rate = alert_rate(validation_scores, block_t)
    return ThresholdSelection(
        review_budget=review_budget,
        block_budget=block_budget,
        review_threshold=review_t,
        block_threshold=block_t,
        validation_review_rate=review_rate,
        validation_block_rate=block_rate,
        budget_met=review_rate <= review_budget + 1e-12 and block_rate <= block_budget + 1e-12,
    )


def evaluate_thresholds(
    scores: Sequence[float], labels: Sequence[int], review_threshold: float, block_threshold: float
) -> dict[str, float]:
    """Alert rate, precision and recall at each decision level on a labelled window."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    if not scores:
        raise ValueError("scores must be non-empty")
    positives = sum(1 for y in labels if y)
    out: dict[str, float] = {}
    for name, t in (("review", review_threshold), ("block", block_threshold)):
        flagged = [y for s, y in zip(scores, labels, strict=True) if s >= t]
        tp = sum(1 for y in flagged if y)
        out[f"{name}_alert_rate"] = len(flagged) / len(scores)
        out[f"{name}_precision"] = tp / len(flagged) if flagged else 0.0
        out[f"{name}_recall"] = tp / positives if positives else 0.0
    return out
