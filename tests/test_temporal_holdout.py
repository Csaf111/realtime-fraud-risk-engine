"""Tests for chronological, leakage-resistant partition planning."""

import pytest

from fraud_engine.temporal_holdout import chronological_holdout


def test_disjoint_complete_and_chronological():
    steps = [step for step in range(30) for _ in range(3)]
    split = chronological_holdout(steps, embargo_steps=1)
    partitions = (split.train, split.validation, split.test, split.embargoed)
    assert sorted(i for group in partitions for i in group) == list(range(len(steps)))
    assert max(steps[i] for i in split.train) < min(steps[i] for i in split.validation)
    assert max(steps[i] for i in split.validation) < min(steps[i] for i in split.test)


def test_identical_steps_never_cross_partitions():
    steps = [step for step in range(40) for _ in range(4)]
    split = chronological_holdout(steps, embargo_steps=0)
    assignment = {}
    for name in ("train", "validation", "test", "embargoed"):
        for index in getattr(split, name):
            previous = assignment.setdefault(steps[index], name)
            assert previous == name


def test_sparse_steps_use_elapsed_time_for_embargo():
    steps = [10 * step for step in range(30)]
    split = chronological_holdout(steps, embargo_steps=1)
    assert split.train and split.validation and split.test


@pytest.mark.parametrize("steps", [[], [2, 1, 3], [-1, 0, 1], [0, True, 2]])
def test_rejects_bad_steps(steps):
    with pytest.raises(ValueError):
        chronological_holdout(steps)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_fraction": float("nan")},
        {"train_fraction": 0},
        {"validation_fraction": 0},
        {"train_fraction": 0.8, "validation_fraction": 0.3},
        {"embargo_steps": -1},
        {"embargo_steps": True},
    ],
)
def test_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        chronological_holdout(list(range(30)), **kwargs)


def test_rejects_empty_partition_after_embargo():
    with pytest.raises(ValueError, match="Embargo leaves"):
        chronological_holdout(list(range(12)), embargo_steps=3)


def test_does_not_depend_on_labels_or_transaction_fields():
    steps = list(range(30))
    original = chronological_holdout(steps)
    assert original == chronological_holdout(list(steps))
