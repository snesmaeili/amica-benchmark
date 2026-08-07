from __future__ import annotations

import sys
from pathlib import Path

import pytest


MODULE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "multimodel"
sys.path.insert(0, str(MODULE_DIR))

from stage1.folds import make_guarded_blocked_folds  # noqa: E402


def _indices(spans):
    return {
        index
        for span in spans
        for index in range(span.start, span.stop)
    }


def test_five_contiguous_folds_use_exact_five_second_guards():
    folds = make_guarded_blocked_folds(
        n_samples=25_003,
        sampling_rate_hz=250.0,
        n_splits=5,
        guard_seconds=5.0,
    )
    assert len(folds) == 5
    assert all(fold.guard_samples == 1_250 for fold in folds)
    test_indices = [_indices((fold.test,)) for fold in folds]
    assert set.union(*test_indices) == set(range(25_003))
    assert sum(len(indices) for indices in test_indices) == 25_003

    for fold in folds:
        train = _indices(fold.train)
        test = _indices((fold.test,))
        guard = _indices(fold.guard)
        assert train.isdisjoint(test)
        assert train.isdisjoint(guard)
        assert test.isdisjoint(guard)
        assert train | test | guard == set(range(25_003))


def test_folds_are_deterministic():
    kwargs = dict(
        n_samples=12_000,
        sampling_rate_hz=200.0,
        n_splits=5,
        guard_seconds=5.0,
    )
    assert make_guarded_blocked_folds(**kwargs) == make_guarded_blocked_folds(
        **kwargs
    )


def test_guard_that_removes_all_training_data_is_rejected():
    with pytest.raises(ValueError, match="leaves no training samples"):
        make_guarded_blocked_folds(
            n_samples=100,
            sampling_rate_hz=10.0,
            n_splits=2,
            guard_seconds=10.0,
        )
