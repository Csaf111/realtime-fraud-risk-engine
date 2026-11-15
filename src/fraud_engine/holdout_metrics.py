"""Evaluate chronological fraud scores without selecting thresholds on test labels.

The caller must supply point-in-time scores from a causal feature pipeline.
This module does not train a model or claim that scores are leakage-free.
"""
from __future__ import annotations

from math import ceil, isfinite
from typing import Sequence

from .temporal_holdout import HoldoutSplit


def evaluate_partition(
    scores: Sequence[float], labels: Sequence[int], indices: Sequence[int],
    *, budgets: tuple[float, ...] = (0.001, 0.005, 0.01),
) -> dict:
    """Compute exact-capacity top-k metrics for one fixed partition.

    Stable tie breaking uses original transaction index. A budget is a
    fraction of the evaluated partition, never of the full dataset.
    """
    if not indices:
        raise ValueError("partition must not be empty")
    if not budgets or any(not isfinite(b) or not 0 < b <= 1 for b in budgets):
        raise ValueError("budgets must be finite fractions in (0, 1]")
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have the same length")
    if len(set(indices)) != len(indices):
        raise ValueError("duplicate indices")
    if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= len(scores) for i in indices):
        raise ValueError("invalid partition index")
    if any(not isinstance(labels[i], int) or isinstance(labels[i], bool) or labels[i] not in (0, 1) for i in indices):
        raise ValueError("labels must be binary integers")
    if any(not isinstance(scores[i], (float, int)) or not isfinite(scores[i]) for i in indices):
        raise ValueError("scores must be finite numbers")

    ranked = sorted(indices, key=lambda i: (-scores[i], i))
    fraud = sum(labels[i] for i in indices)
    results = {}
    for budget in budgets:
        capacity = ceil(len(indices) * budget)
        caught = sum(labels[i] for i in ranked[:capacity])
        results[str(budget)] = {
            "alerts": capacity,
            "true_positives": caught,
            "precision": caught / capacity,
            "recall": caught / fraud if fraud else 0.0,
        }
    return {"records": len(indices), "fraud_records": fraud, "prevalence": fraud / len(indices),
            "alert_budgets": results}


def evaluate_holdout(
    scores: Sequence[float], labels: Sequence[int], split: HoldoutSplit,
    *, budgets: tuple[float, ...] = (0.001, 0.005, 0.01),
) -> dict:
    """Report partitions separately; never aggregate embargoed records."""
    split.assert_disjoint()
    all_indices = (*split.train, *split.validation, *split.test, *split.embargoed)
    if len(all_indices) != len(scores) or set(all_indices) != set(range(len(scores))):
        raise ValueError("split must cover every input record exactly once")
    return {
        "train": evaluate_partition(scores, labels, split.train, budgets=budgets),
        "validation": evaluate_partition(scores, labels, split.validation, budgets=budgets),
        "test": evaluate_partition(scores, labels, split.test, budgets=budgets),
        "embargoed_records": len(split.embargoed),
        "warning": "Scores must be generated point-in-time; do not tune models or thresholds on test.",
    }
