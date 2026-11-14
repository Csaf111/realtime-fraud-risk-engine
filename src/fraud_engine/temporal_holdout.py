"""Point-in-time split planner for chronological fraud evaluation.

Split by *distinct time steps*, never by rows: simultaneous events must not
straddle train/validation/test. The embargo excludes steps on both sides of
each boundary and is deliberately not used for model fitting or tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence


@dataclass(frozen=True)
class HoldoutSplit:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    embargoed: tuple[int, ...]

    def assert_disjoint(self) -> None:
        groups = (self.train, self.validation, self.test, self.embargoed)
        flattened = [index for group in groups for index in group]
        if len(flattened) != len(set(flattened)):
            raise ValueError("Split partitions overlap")


def chronological_holdout(
    steps: Sequence[int],
    *,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    embargo_steps: int = 1,
) -> HoldoutSplit:
    """Partition row indices by event time, with a gap around each boundary.

    Fractions determine the approximate number of *distinct* steps assigned
    to training and validation before embargo. Test receives the remainder.
    No transaction label, amount or post-event balance is consulted.

    The returned indices preserve input order. Empty post-embargo partitions
    are rejected instead of silently reporting misleading test metrics.
    """
    if not isfinite(train_fraction) or not isfinite(validation_fraction):
        raise ValueError("Fractions must be finite")
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1):
        raise ValueError("Fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Fractions must leave a test partition")
    if isinstance(embargo_steps, bool) or not isinstance(embargo_steps, int):
        raise ValueError("embargo_steps must be a nonnegative integer")
    if embargo_steps < 0:
        raise ValueError("embargo_steps must be nonnegative")
    if not steps:
        raise ValueError("Cannot split an empty dataset")
    if any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in steps):
        raise ValueError("Steps must be nonnegative integers")
    if any(current < previous for previous, current in zip(steps, steps[1:])):
        raise ValueError("Input must be ordered by nondecreasing step")

    distinct = sorted(set(steps))
    n = len(distinct)
    train_end = int(n * train_fraction)
    validation_end = train_end + int(n * validation_fraction)
    if train_end == 0 or validation_end == train_end or validation_end >= n:
        raise ValueError("Not enough distinct steps for all partitions")

    # Use elapsed step values for the embargo, not the number of records.
    # Exclude each boundary's adjacent time interval from BOTH partitions.
    boundaries = (distinct[train_end], distinct[validation_end])
    def embargoed(step: int) -> bool:
        return any(abs(step - boundary) <= embargo_steps for boundary in boundaries)

    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    excluded: list[int] = []
    first_validation = distinct[train_end]
    first_test = distinct[validation_end]
    for index, step in enumerate(steps):
        if embargoed(step):
            excluded.append(index)
        elif step < first_validation:
            train.append(index)
        elif step < first_test:
            validation.append(index)
        else:
            test.append(index)

    if not train or not validation or not test:
        raise ValueError("Embargo leaves an empty train, validation or test partition")
    result = HoldoutSplit(tuple(train), tuple(validation), tuple(test), tuple(excluded))
    result.assert_disjoint()
    return result
