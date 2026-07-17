from types import SimpleNamespace

import numpy as np

from scripts.heldout.core import (
    complete_mir_from_fitted_ica,
    contiguous_folds,
    evaluation_indices,
    project_with_fitted_ica,
)


class DummyRaw:
    def __init__(self, data, sfreq=250.0):
        self._data = np.asarray(data)
        self.info = {"sfreq": sfreq}

    def get_data(self):
        return self._data


def test_guard_banded_folds_are_disjoint():
    folds = contiguous_folds(1_000, 5, 20)
    assert len(folds) == 5
    for fold in folds:
        assert not np.intersect1d(fold.train_indices, fold.test_indices).size
        assert not np.any(
            (fold.train_indices >= fold.excluded_start)
            & (fold.train_indices < fold.excluded_stop)
        )
    np.testing.assert_array_equal(
        np.concatenate([fold.test_indices for fold in folds]), np.arange(1_000)
    )


def test_evaluation_indices_are_deterministic_and_shared():
    first = evaluation_indices(50_000, 20_000, 42)
    second = evaluation_indices(50_000, 20_000, 42)
    np.testing.assert_array_equal(first, second)
    assert len(first) == 20_000
    assert len(np.unique(first)) == 20_000


def test_projection_uses_training_fitted_mean():
    raw = DummyRaw([[10.0, 11.0], [20.0, 22.0]])
    ica = SimpleNamespace(
        pca_components_=np.eye(2),
        unmixing_matrix_=np.eye(2),
        pre_whitener_=np.ones((2, 1)),
        pca_mean_=np.array([1.0, 2.0]),
    )
    projected, _ = project_with_fitted_ica(raw, ica)
    np.testing.assert_allclose(projected, [[9.0, 10.0], [18.0, 20.0]])


def test_identity_complete_mir_is_zero():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(3, 5_000))
    raw = DummyRaw(data)
    ica = SimpleNamespace(
        pca_components_=np.eye(3),
        unmixing_matrix_=np.eye(3),
        pre_whitener_=np.ones((3, 1)),
        pca_mean_=np.zeros(3),
    )
    result = complete_mir_from_fitted_ica(
        raw, ica, indices=np.arange(data.shape[1]), n_bins=100
    )
    assert abs(result["bits_per_sample"]) < 1e-12
