"""Operational alert-budget evaluation for chronological fraud holdouts.

This module is intentionally independent of training and threshold selection.
Scores must be generated on an untouched future holdout before calling evaluate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class BudgetMetrics:
    budget_fraction: float
    alerts: int
    actual_fraction: float
    precision: float
    recall: float
    captured_fraud: int
    total_fraud: int
    threshold: float | None

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "budget_fraction": self.budget_fraction,
            "alerts": self.alerts,
            "actual_fraction": self.actual_fraction,
            "precision": self.precision,
            "recall": self.recall,
            "captured_fraud": self.captured_fraud,
            "total_fraud": self.total_fraud,
            "threshold": self.threshold,
        }


def evaluate_alert_budgets(
    scores: Sequence[float],
    labels: Sequence[int],
    budgets: Sequence[float] = (0.001, 0.005, 0.01),
) -> list[BudgetMetrics]:
    """Report precision/recall at fixed review capacities.

    Select exactly floor(N * budget) highest scores; stable index tie-breaking
    makes results reproducible. This is a *ranking evaluation*, not a
    deployable score threshold: ties at the boundary require a separate
    deterministic production tie-breaker or capacity limiter.
    """
    if not scores or len(scores) != len(labels):
        raise ValueError("scores and labels must be nonempty and equal length")
    if any(not math.isfinite(float(s)) for s in scores):
        raise ValueError("scores must be finite")
    if any(y not in (0, 1) for y in labels):
        raise ValueError("labels must be binary")
    if any(not math.isfinite(float(b)) or not 0 < b <= 1 for b in budgets):
        raise ValueError("budgets must be finite fractions in (0, 1]")

    n = len(scores)
    fraud_total = sum(labels)
    ranking = sorted(range(n), key=lambda i: (-scores[i], i))
    prefix = [0]
    for index in ranking:
        prefix.append(prefix[-1] + labels[index])

    results = []
    for budget in budgets:
        k = math.floor(n * budget)
        captured = prefix[k]
        results.append(
            BudgetMetrics(
                budget_fraction=float(budget),
                alerts=k,
                actual_fraction=k / n,
                precision=captured / k if k else 0.0,
                recall=captured / fraud_total if fraud_total else 0.0,
                captured_fraud=captured,
                total_fraud=fraud_total,
                threshold=float(scores[ranking[k - 1]]) if k else None,
            )
        )
    return results
