"""Deterministic contiguous cross-validation folds with temporal guards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True, order=True)
class SampleSpan:
    """Half-open sample interval ``[start, stop)``."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("SampleSpan requires 0 <= start < stop")

    @property
    def size(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class BlockedFold:
    """One contiguous test block and its guarded training spans."""

    fold_index: int
    test: SampleSpan
    train: Tuple[SampleSpan, ...]
    guard: Tuple[SampleSpan, ...]
    guard_samples: int


def _partition_bounds(n_samples: int, n_splits: int) -> Tuple[int, ...]:
    quotient, remainder = divmod(n_samples, n_splits)
    bounds = [0]
    for fold_index in range(n_splits):
        width = quotient + (1 if fold_index < remainder else 0)
        bounds.append(bounds[-1] + width)
    return tuple(bounds)


def make_guarded_blocked_folds(
    n_samples: int,
    sampling_rate_hz: float,
    n_splits: int = 5,
    guard_seconds: float = 5.0,
) -> Tuple[BlockedFold, ...]:
    """Construct contiguous test folds with excluded guards on both sides.

    Test blocks partition the full recording.  For each fold, training spans
    exclude the test block and up to ``guard_seconds`` immediately before and
    after it.
    """
    if type(n_samples) is not int or n_samples < 2:
        raise ValueError("n_samples must be an integer of at least 2")
    if type(n_splits) is not int or not 2 <= n_splits <= n_samples:
        raise ValueError("n_splits must lie between 2 and n_samples")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    if guard_seconds < 0:
        raise ValueError("guard_seconds cannot be negative")
    guard_samples = int(round(guard_seconds * sampling_rate_hz))
    bounds = _partition_bounds(n_samples, n_splits)
    folds: List[BlockedFold] = []
    for fold_index, (test_start, test_stop) in enumerate(
        zip(bounds[:-1], bounds[1:])
    ):
        left_stop = max(0, test_start - guard_samples)
        right_start = min(n_samples, test_stop + guard_samples)
        train = []
        guard = []
        if left_stop > 0:
            train.append(SampleSpan(0, left_stop))
        if left_stop < test_start:
            guard.append(SampleSpan(left_stop, test_start))
        if test_stop < right_start:
            guard.append(SampleSpan(test_stop, right_start))
        if right_start < n_samples:
            train.append(SampleSpan(right_start, n_samples))
        if not train:
            raise ValueError(
                "guard leaves no training samples; reduce guard_seconds or n_splits"
            )
        folds.append(
            BlockedFold(
                fold_index=fold_index,
                test=SampleSpan(test_start, test_stop),
                train=tuple(train),
                guard=tuple(guard),
                guard_samples=guard_samples,
            )
        )
    validate_blocked_folds(
        tuple(folds), n_samples=n_samples, expected_n_splits=n_splits
    )
    return tuple(folds)


def validate_blocked_folds(
    folds: Tuple[BlockedFold, ...], n_samples: int, expected_n_splits: int
) -> None:
    """Validate partitioning, train/test disjointness, and exact guards."""
    if len(folds) != expected_n_splits:
        raise ValueError("unexpected number of folds")
    expected_start = 0
    for expected_index, fold in enumerate(folds):
        if fold.fold_index != expected_index:
            raise ValueError("fold indices must be consecutive")
        if fold.test.start != expected_start:
            raise ValueError("test spans must form one contiguous partition")
        expected_start = fold.test.stop

        excluded_start = max(0, fold.test.start - fold.guard_samples)
        excluded_stop = min(n_samples, fold.test.stop + fold.guard_samples)
        expected_train = []
        if excluded_start > 0:
            expected_train.append(SampleSpan(0, excluded_start))
        if excluded_stop < n_samples:
            expected_train.append(SampleSpan(excluded_stop, n_samples))
        if fold.train != tuple(expected_train):
            raise ValueError("training spans do not match the guarded complement")

        expected_guard = []
        if excluded_start < fold.test.start:
            expected_guard.append(SampleSpan(excluded_start, fold.test.start))
        if fold.test.stop < excluded_stop:
            expected_guard.append(SampleSpan(fold.test.stop, excluded_stop))
        if fold.guard != tuple(expected_guard):
            raise ValueError("guard spans do not match the requested exclusion")
    if expected_start != n_samples:
        raise ValueError("test spans do not cover all samples")
