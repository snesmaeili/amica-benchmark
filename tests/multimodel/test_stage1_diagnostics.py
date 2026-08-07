from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "multimodel"
sys.path.insert(0, str(MODULE_DIR))

from stage1.diagnostics import occupancy_kish_diagnostics  # noqa: E402


def test_one_hot_posteriors_have_expected_occupancy_and_kish_n():
    posteriors = np.asarray(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    result = occupancy_kish_diagnostics(
        posteriors,
        fitted_priors=np.asarray([0.5, 0.5]),
        n_components=1,
        min_fitted_prior=0.1,
        primary_kish_per_c_squared=2.0,
    )
    np.testing.assert_allclose(result.fitted_priors, [0.5, 0.5])
    np.testing.assert_allclose(result.posterior_occupancy, [0.5, 0.5])
    np.testing.assert_allclose(result.kish_effective_n, [2.0, 2.0])
    np.testing.assert_allclose(result.hard_occupancy, [0.5, 0.5])
    assert not result.any_degenerate


def test_uniform_soft_assignments_use_all_samples_for_each_model():
    posteriors = np.full((2, 6), 0.5)
    result = occupancy_kish_diagnostics(
        posteriors,
        fitted_priors=np.asarray([0.5, 0.5]),
        n_components=1,
        min_fitted_prior=0.1,
        primary_kish_per_c_squared=5.0,
    )
    np.testing.assert_allclose(result.posterior_occupancy, [0.5, 0.5])
    np.testing.assert_allclose(result.kish_effective_n, [6.0, 6.0])
    assert result.as_records()[0]["model_index"] == 0


def test_sample_weights_and_degeneracy_flags_are_respected():
    posteriors = np.asarray(
        [
            [0.99, 0.99, 0.99, 0.99],
            [0.01, 0.01, 0.01, 0.01],
        ]
    )
    result = occupancy_kish_diagnostics(
        posteriors,
        fitted_priors=np.asarray([0.99, 0.01]),
        n_components=1,
        sample_weights=np.asarray([1.0, 1.0, 0.0, 0.0]),
        min_fitted_prior=0.05,
        min_posterior_occupancy=0.05,
        primary_kish_per_c_squared=3.0,
    )
    np.testing.assert_allclose(result.fitted_priors, [0.99, 0.01])
    np.testing.assert_allclose(result.posterior_occupancy, [0.99, 0.01])
    np.testing.assert_allclose(result.kish_effective_n, [2.0, 2.0])
    assert result.low_fitted_prior.tolist() == [False, True]
    assert result.low_posterior_occupancy.tolist() == [False, True]
    assert result.low_kish_effective_n.tolist() == [True, True]
    assert result.any_degenerate


@pytest.mark.parametrize(
    "posteriors,match",
    [
        (np.asarray([[0.6, 0.6], [0.5, 0.5]]), "sum to one"),
        (np.asarray([[1.0, np.nan], [0.0, np.nan]]), "non-finite"),
        (np.asarray([[1.1, 0.5], [-0.1, 0.5]]), "negative"),
    ],
)
def test_invalid_posteriors_are_rejected(posteriors, match):
    with pytest.raises(ValueError, match=match):
        occupancy_kish_diagnostics(
            posteriors,
            fitted_priors=np.asarray([0.5, 0.5]),
            n_components=1,
        )


def test_fitted_prior_and_evaluation_occupancy_are_not_conflated():
    posteriors = np.full((2, 8), 0.5)
    result = occupancy_kish_diagnostics(
        posteriors,
        fitted_priors=np.asarray([0.99, 0.01]),
        n_components=1,
        min_fitted_prior=0.02,
        min_posterior_occupancy=0.02,
        primary_kish_per_c_squared=1.0,
    )
    np.testing.assert_allclose(result.posterior_occupancy, [0.5, 0.5])
    assert result.low_fitted_prior.tolist() == [False, True]
    assert result.low_posterior_occupancy.tolist() == [False, False]
    assert result.any_degenerate
    record = result.as_records()[1]
    assert record["fitted_prior"] == pytest.approx(0.01)
    assert record["posterior_occupancy"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("priors", "match"),
    [
        (np.asarray([1.0]), "shape"),
        (np.asarray([0.6, 0.5]), "sum to one"),
        (np.asarray([1.1, -0.1]), "non-negative"),
    ],
)
def test_invalid_fitted_priors_are_rejected(priors, match):
    with pytest.raises(ValueError, match=match):
        occupancy_kish_diagnostics(
            np.full((2, 4), 0.5),
            fitted_priors=priors,
            n_components=1,
        )
