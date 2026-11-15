"""Regression tests for fixed-capacity, partition-aware evaluation."""
import math

import pytest

from fraud_engine.holdout_metrics import evaluate_holdout, evaluate_partition
from fraud_engine.temporal_holdout import HoldoutSplit, chronological_holdout


def test_partition_uses_local_budget_and_stable_ties():
    result = evaluate_partition([0.8, 0.8, 0.2, 0.1], [0, 1, 1, 0], (0, 1, 2, 3), budgets=(0.25, 0.5))
    assert result["alert_budgets"]["0.25"]["alerts"] == 1
    assert result["alert_budgets"]["0.25"]["true_positives"] == 0
    assert result["alert_budgets"]["0.5"]["true_positives"] == 1
    assert result["prevalence"] == 0.5


def test_holdout_keeps_test_separate_and_excludes_embargo():
    steps = list(range(1, 31))
    split = chronological_holdout(steps, embargo_steps=1)
    scores = [i / 30 for i in range(30)]
    labels = [int(i % 3 == 0) for i in range(30)]
    result = evaluate_holdout(scores, labels, split, budgets=(0.5,))
    assert result["test"]["records"] == len(split.test)
    assert result["embargoed_records"] == len(split.embargoed)
    assert sum(result[p]["records"] for p in ("train", "validation", "test")) + result["embargoed_records"] == 30


def test_zero_fraud_population():
    result = evaluate_partition([0.1, 0.2], [0, 0], [0, 1], budgets=(0.5,))
    assert result["alert_budgets"]["0.5"]["recall"] == 0


@pytest.mark.parametrize("scores,labels,indices,budgets", [
    ([math.nan], [1], [0], (1.0,)),
    ([math.inf], [1], [0], (1.0,)),
    ([0.2], [2], [0], (1.0,)),
    ([0.2], [True], [0], (1.0,)),
    ([0.2], [1], [0, 0], (1.0,)),
    ([0.2], [1], [1], (1.0,)),
    ([0.2], [1], [], (1.0,)),
    ([0.2], [1], [0], (math.nan,)),
    ([0.2], [1], [0], (0.0,)),
])
def test_invalid_inputs(scores, labels, indices, budgets):
    with pytest.raises(ValueError):
        evaluate_partition(scores, labels, indices, budgets=budgets)


def test_incomplete_split_is_rejected():
    split = HoldoutSplit((0,), (1,), (2,), ())
    with pytest.raises(ValueError, match="cover"):
        evaluate_holdout([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 0], split)
