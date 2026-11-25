"""Regression tests for threshold-grouped average precision."""
import pytest

from fraud_engine.average_precision import average_precision


def test_perfect_ranking():
    assert average_precision([0.9, 0.8, 0.1], [1, 1, 0], [0, 1, 2]) == 1.0


def test_tied_scores_are_order_invariant():
    assert average_precision([0.5, 0.5], [1, 0], [0, 1]) == 0.5
    assert average_precision([0.5, 0.5], [1, 0], [1, 0]) == 0.5


def test_no_positive_is_undefined():
    assert average_precision([0.1, 0.2], [0, 0], [0, 1]) is None


def test_partition_isolation():
    assert average_precision([0.99, 0.8, 0.1], [0, 1, 0], [1, 2]) == 1.0


@pytest.mark.parametrize("scores,labels,indices", [
    ([float("nan")], [1], [0]),
    ([True], [1], [0]),
    ([0.2], [2], [0]),
    ([0.2], [1], [0, 0]),
    ([0.2], [1], [1]),
    ([0.2], [1], []),
])
def test_reject_invalid_inputs(scores, labels, indices):
    with pytest.raises(ValueError):
        average_precision(scores, labels, indices)
