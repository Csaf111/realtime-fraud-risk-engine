"""Tests for reproducible operational ranking metrics."""
import pytest

from fraud_engine.alert_budget import evaluate_alert_budgets


def test_top_k_precision_recall_and_capacity():
    result = evaluate_alert_budgets([0.2, 0.9, 0.8, 0.1], [0, 1, 0, 1], [0.25, 0.5])
    assert result[0].alerts == 1
    assert result[0].precision == 1
    assert result[0].recall == 0.5
    assert result[1].alerts == 2
    assert result[1].precision == 0.5
    assert result[1].recall == 0.5


def test_ties_are_stable_in_original_order():
    result = evaluate_alert_budgets([0.5, 0.5, 0.5, 0.5], [1, 0, 0, 0], [0.25])
    assert result[0].captured_fraud == 1
    assert result[0].threshold == 0.5


def test_tiny_budget_does_not_exceed_capacity():
    result = evaluate_alert_budgets([0.8, 0.1], [1, 0], [0.01])[0]
    assert result.alerts == 0
    assert result.threshold is None
    assert result.precision == result.recall == 0


def test_zero_positive_holdout():
    result = evaluate_alert_budgets([0.8, 0.1], [0, 0], [0.5])[0]
    assert result.recall == 0
    assert result.precision == 0


@pytest.mark.parametrize("scores,labels,budgets", [
    ([], [], [0.5]),
    ([0.1], [], [0.5]),
    ([float("nan")], [1], [0.5]),
    ([float("inf")], [1], [0.5]),
    ([0.1], [2], [0.5]),
    ([0.1], [1], [0]),
    ([0.1], [1], [1.1]),
    ([0.1], [1], [float("nan")]),
])
def test_rejects_invalid_inputs(scores, labels, budgets):
    with pytest.raises(ValueError):
        evaluate_alert_budgets(scores, labels, budgets)


def test_serialisable_result():
    result = evaluate_alert_budgets([0.1, 0.9], [0, 1], [0.5])[0]
    assert result.as_dict()["captured_fraud"] == 1
