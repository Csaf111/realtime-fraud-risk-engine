"""Tie-aware average precision for chronological fraud holdout partitions.

Average precision is the step-wise area under the precision-recall curve,
not trapezoidal PR-AUC. Equal scores are evaluated as one threshold group:
their arbitrary input ordering cannot inflate performance.
"""
from __future__ import annotations

from math import isfinite
from typing import Sequence


def average_precision(
    scores: Sequence[float], labels: Sequence[int], indices: Sequence[int],
) -> float | None:
    """Return AP, or None when the partition has no positive examples.

    No threshold selection or fitting occurs. A partition without positives
    has undefined AP and must not be silently reported as a zero score.
    """
    if not indices:
        raise ValueError("partition must not be empty")
    if len(scores) != len(labels):
        raise ValueError("scores and labels length mismatch")
    if len(set(indices)) != len(indices):
        raise ValueError("duplicate partition indices")
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(scores):
            raise ValueError("invalid partition index")
        score, label = scores[index], labels[index]
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not isfinite(score):
            raise ValueError("scores must be finite numbers")
        if isinstance(label, bool) or not isinstance(label, int) or label not in (0, 1):
            raise ValueError("labels must be binary integers")

    positives = sum(labels[i] for i in indices)
    if not positives:
        return None
    ordered = sorted(indices, key=lambda i: -scores[i])
    tp = 0
    seen = 0
    ap = 0.0
    pos = 0
    while pos < len(ordered):
        end = pos + 1
        while end < len(ordered) and scores[ordered[end]] == scores[ordered[pos]]:
            end += 1
        group_tp = sum(labels[i] for i in ordered[pos:end])
        seen += end - pos
        tp += group_tp
        ap += (group_tp / positives) * (tp / seen)
        pos = end
    return ap
