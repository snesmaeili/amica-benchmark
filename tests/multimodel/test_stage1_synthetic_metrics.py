from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "multimodel"
sys.path.insert(0, str(MODULE_DIR))

from stage1.synthetic_metrics import (  # noqa: E402
    align_candidate_model_axis,
    classwise_expected_calibration_error,
    cross_seed_posterior_stability,
    evaluate_regime_recovery,
    evaluate_source_recovery,
    hungarian_align_labels,
    hungarian_align_posteriors,
    hungarian_model_alignment,
    multiclass_brier_score,
    occupancy_recovery,
    square_amari_distance,
    smallest_model_within_one_standard_error,
    transition_boundaries,
    transition_boundary_f1,
    variation_of_information,
    weighted_source_correlation_matrix,
)


def _one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    result = np.zeros((n_classes, labels.size), dtype=float)
    result[labels, np.arange(labels.size)] = 1.0
    return result


def test_hungarian_alignment_recovers_a_label_permutation():
    truth = np.asarray([0, 0, 1, 1, 2, 2])
    predicted = np.asarray([2, 2, 0, 0, 1, 1])
    label_result = hungarian_align_labels(truth, predicted)
    np.testing.assert_array_equal(label_result.aligned_predictions, truth)
    assert label_result.accuracy == 1.0

    posterior_result = hungarian_align_posteriors(
        truth, _one_hot(predicted, 3)
    )
    assert posterior_result.is_bijective
    np.testing.assert_allclose(
        posterior_result.aligned_posteriors, _one_hot(truth, 3)
    )
    np.testing.assert_allclose(
        posterior_result.unmatched_probability_mass, 0.0
    )


def test_unequal_model_count_is_explicit_and_proper_scores_reject_it():
    truth = np.asarray([0, 0, 1, 1])
    predicted = np.asarray([0, 2, 1, 2])
    posteriors = _one_hot(predicted, 3)
    aligned = hungarian_align_posteriors(truth, posteriors)
    assert not aligned.is_bijective
    assert np.any(aligned.predicted_to_true_position == -1)
    assert np.any(aligned.unmatched_probability_mass > 0)
    with pytest.raises(ValueError, match="equal fitted and true"):
        evaluate_regime_recovery(truth, posteriors)


def test_regime_metrics_are_permutation_invariant_and_proper_scores_are_zero():
    truth = np.asarray([0, 0, 1, 1, 2, 2])
    predicted = np.asarray([1, 1, 2, 2, 0, 0])
    result = evaluate_regime_recovery(
        truth, _one_hot(predicted, n_classes=3), n_calibration_bins=5
    )
    assert result.accuracy == pytest.approx(1.0)
    assert result.adjusted_rand_index == pytest.approx(1.0)
    assert result.variation_of_information_nats == pytest.approx(0.0)
    assert result.multiclass_brier_score == pytest.approx(0.0)
    assert result.classwise_calibration_error == pytest.approx(0.0)
    assert variation_of_information(truth, predicted) == pytest.approx(0.0)


def test_brier_and_classwise_ece_have_declared_multiclass_definitions():
    truth = np.asarray([0, 1])
    probabilities = np.asarray([[0.75, 0.25], [0.25, 0.75]])
    # Per sample: (0.25**2 + 0.25**2) = 0.125.
    assert multiclass_brier_score(truth, probabilities) == pytest.approx(
        0.125
    )
    # Across four class-sample pairs each bin has a 0.25 calibration gap.
    assert classwise_expected_calibration_error(
        truth, probabilities, n_bins=4
    ) == pytest.approx(0.25)


def test_gradual_regimes_use_soft_targets_for_alignment_and_proper_scores():
    planted = np.asarray(
        [
            [1.0, 0.8, 0.55, 0.2, 0.0],
            [0.0, 0.2, 0.45, 0.8, 1.0],
        ]
    )
    predicted_with_swapped_rows = planted[::-1].copy()
    aligned = hungarian_align_posteriors(
        planted, predicted_with_swapped_rows
    )
    assert aligned.is_bijective
    np.testing.assert_allclose(aligned.aligned_posteriors, planted)

    result = evaluate_regime_recovery(
        planted,
        predicted_with_swapped_rows,
        n_calibration_bins=4,
    )
    assert result.accuracy == pytest.approx(1.0)
    assert result.multiclass_brier_score == pytest.approx(0.0)
    assert result.classwise_calibration_error == pytest.approx(0.0)
    assert multiclass_brier_score(planted, planted) == pytest.approx(0.0)
    assert classwise_expected_calibration_error(
        planted, planted, n_bins=4
    ) == pytest.approx(0.0)


def test_soft_target_score_does_not_collapse_transition_samples_to_one_hot():
    planted = np.asarray([[0.75, 0.5, 0.25], [0.25, 0.5, 0.75]])
    hard_collapse = _one_hot(np.argmax(planted, axis=0), 2)
    assert multiclass_brier_score(planted, planted) == pytest.approx(0.0)
    assert multiclass_brier_score(planted, hard_collapse) > 0.0


def test_boundary_f1_uses_one_to_one_matching_with_tolerance():
    truth = np.repeat([0, 1, 0], [5, 5, 5])
    predicted = np.repeat([0, 1, 0], [6, 3, 6])
    np.testing.assert_array_equal(transition_boundaries(truth), [5, 10])
    np.testing.assert_array_equal(transition_boundaries(predicted), [6, 9])
    within = transition_boundary_f1(
        truth, predicted, tolerance_samples=1
    )
    assert within.matched_pairs == ((5, 6), (10, 9))
    assert within.precision == pytest.approx(1.0)
    assert within.recall == pytest.approx(1.0)
    assert within.f1 == pytest.approx(1.0)

    exact = transition_boundary_f1(
        truth, predicted, tolerance_samples=0
    )
    assert exact.f1 == pytest.approx(0.0)


def test_boundary_f1_edge_cases_are_defined():
    no_transitions = np.zeros(8, dtype=int)
    perfect_empty = transition_boundary_f1(
        no_transitions, no_transitions, tolerance_samples=2
    )
    assert perfect_empty.precision == perfect_empty.recall == 1.0
    assert perfect_empty.f1 == 1.0

    one_transition = np.asarray([0, 0, 0, 1, 1, 1, 1, 1])
    false_positive = transition_boundary_f1(
        no_transitions, one_transition, tolerance_samples=2
    )
    assert false_positive.precision == false_positive.recall == 0.0
    assert false_positive.f1 == 0.0


def test_occupancy_recovery_uses_soft_posteriors_and_optional_weights():
    truth = np.asarray([0, 0, 0, 1])
    posteriors = np.asarray(
        [[0.9, 0.8, 0.7, 0.2], [0.1, 0.2, 0.3, 0.8]]
    )
    result = occupancy_recovery(truth, posteriors)
    np.testing.assert_allclose(result.true_occupancy, [0.75, 0.25])
    np.testing.assert_allclose(result.posterior_occupancy, [0.65, 0.35])
    np.testing.assert_allclose(result.hard_occupancy, [0.75, 0.25])
    np.testing.assert_allclose(result.posterior_error, [-0.1, 0.1])
    assert result.mean_absolute_error == pytest.approx(0.1)
    assert result.total_variation_distance == pytest.approx(0.1)

    weighted = occupancy_recovery(
        truth,
        posteriors,
        sample_weights=np.asarray([1.0, 1.0, 0.0, 2.0]),
    )
    np.testing.assert_allclose(weighted.true_occupancy, [0.5, 0.5])


def test_generic_model_alignment_and_axis_reordering_support_map_scores():
    scores = np.asarray([[0.1, 0.9], [0.8, 0.2]])
    alignment = hungarian_model_alignment(scores)
    np.testing.assert_array_equal(alignment.reference_to_candidate, [1, 0])
    candidate_maps = np.asarray([[20.0, 21.0], [10.0, 11.0]])
    np.testing.assert_allclose(
        align_candidate_model_axis(candidate_maps, alignment),
        [[10.0, 11.0], [20.0, 21.0]],
    )


def test_source_recovery_handles_permutation_sign_scale_and_weights():
    rng = np.random.default_rng(31)
    true_sources = rng.standard_normal((3, 500))
    true_mixing = np.asarray(
        [[1.0, 0.2, -0.1], [0.1, 1.1, 0.3], [0.25, -0.2, 0.9]]
    )
    permutation = np.asarray([2, 0, 1])
    multipliers = np.asarray([-2.0, 0.5, -1.5])
    estimated_sources = (
        multipliers[:, np.newaxis] * true_sources[permutation]
    )
    estimated_mixing = (
        true_mixing[:, permutation] / multipliers[np.newaxis, :]
    )
    sample_weights = np.linspace(0.2, 1.0, true_sources.shape[1])
    source_weights = np.asarray([0.6, 0.3, 0.1])

    result = evaluate_source_recovery(
        true_sources,
        estimated_sources,
        true_mixing,
        estimated_mixing,
        sample_weights=sample_weights,
        source_weights=source_weights,
    )
    inverse_permutation = np.argsort(permutation)
    np.testing.assert_array_equal(
        result.alignment.reference_to_candidate, inverse_permutation
    )
    expected_multipliers = multipliers[inverse_permutation]
    np.testing.assert_array_equal(
        result.source_signs, np.sign(expected_multipliers)
    )
    np.testing.assert_allclose(
        result.source_scales, 1.0 / np.abs(expected_multipliers)
    )
    np.testing.assert_allclose(result.matched_source_correlations, 1.0)
    np.testing.assert_allclose(result.matched_map_errors, 0.0, atol=1e-15)
    assert np.isinf(result.source_sir_db).all()
    assert np.isinf(result.aggregate_sir_db)
    assert result.square_amari_distance == pytest.approx(0.0, abs=1e-15)
    np.testing.assert_allclose(result.sample_weights.sum(), 1.0)
    np.testing.assert_allclose(result.source_weights, source_weights)


def test_source_metrics_degrade_with_interference_and_amari_is_bounded():
    rng = np.random.default_rng(32)
    true_sources = rng.standard_normal((3, 600))
    true_mixing = np.asarray(
        [[1.0, 0.1, 0.2], [0.2, 1.0, -0.1], [-0.1, 0.3, 1.1]]
    )
    estimated_sources = true_sources.copy()
    estimated_sources[0] += 0.5 * true_sources[1]
    estimated_mixing = true_mixing @ np.asarray(
        [[1.0, 0.2, 0.0], [0.1, 1.0, 0.1], [0.0, 0.15, 1.0]]
    )
    result = evaluate_source_recovery(
        true_sources, estimated_sources, true_mixing, estimated_mixing
    )
    assert result.weighted_mean_source_correlation < 1.0
    assert np.isfinite(result.aggregate_sir_db)
    assert result.aggregate_sir_db > 0.0
    assert 0.0 < result.square_amari_distance <= 1.0


def test_source_sir_uses_gain_matrix_interference_definition():
    rng = np.random.default_rng(33)
    true_sources = rng.standard_normal((2, 2_000))
    true_mixing = np.eye(2)
    estimated_unmixing = np.asarray([[1.0, 0.1], [0.0, 1.0]])
    estimated_mixing = np.linalg.inv(estimated_unmixing)
    estimated_sources = estimated_unmixing @ true_sources

    result = evaluate_source_recovery(
        true_sources,
        estimated_sources,
        true_mixing,
        estimated_mixing,
    )
    assert result.source_sir_db[0] == pytest.approx(20.0)
    assert np.isinf(result.source_sir_db[1])
    assert result.aggregate_sir_db == pytest.approx(
        10.0 * np.log10(1.0 / 0.005)
    )


def test_source_recovery_validation_rejects_undefined_endpoints():
    varying = np.vstack([np.arange(8.0), np.arange(8.0)[::-1]])
    constant = varying.copy()
    constant[1] = 1.0
    with pytest.raises(ValueError, match="constant source"):
        weighted_source_correlation_matrix(varying, constant)
    with pytest.raises(ValueError, match="nonsingular"):
        square_amari_distance(np.eye(2), np.ones((2, 2)))
    with pytest.raises(ValueError, match="equal true/estimated"):
        evaluate_source_recovery(
            varying,
            varying[:1],
            np.eye(2),
            np.eye(2),
        )


def test_cross_seed_stability_removes_model_label_switching():
    reference = np.asarray(
        [
            [0.95, 0.9, 0.1, 0.05],
            [0.05, 0.1, 0.9, 0.95],
        ]
    )
    candidate = reference[::-1].copy()
    result = cross_seed_posterior_stability(reference, candidate)
    np.testing.assert_allclose(result.matched_correlations, 1.0)
    assert result.mean_matched_correlation == pytest.approx(1.0)
    assert result.mean_absolute_posterior_difference == pytest.approx(0.0)
    assert result.hard_assignment_agreement == pytest.approx(1.0)
    assert result.hard_assignment_adjusted_rand_index == pytest.approx(1.0)


def test_one_standard_error_rule_selects_smallest_eligible_model():
    # M=3 is empirically best but uncertainty at M=3 admits M=2, not M=1.
    scores = np.asarray(
        [
            [0.50, 0.91, 0.90, 0.88],
            [0.52, 0.93, 0.98, 0.89],
            [0.48, 0.92, 0.94, 0.90],
            [0.50, 0.92, 0.90, 0.89],
        ]
    )
    result = smallest_model_within_one_standard_error(
        [1, 2, 3, 5], scores
    )
    assert result.empirical_best_order == 3
    assert result.selected_order == 2
    np.testing.assert_array_equal(result.eligible, [False, True, True, False])


def test_one_standard_error_rule_supports_minimised_metrics():
    scores = np.asarray(
        [
            [1.0, 0.75, 0.75],
            [1.1, 0.77, 0.77],
            [0.9, 0.75, 0.73],
        ]
    )
    result = smallest_model_within_one_standard_error(
        [1, 2, 3], scores, higher_is_better=False
    )
    assert result.empirical_best_order == 3
    assert result.selected_order == 2


def test_one_standard_error_groups_initialisations_before_seed_level_se():
    # Three optimiser initialisations per generating seed are not six
    # independent simulations.  The endpoint is averaged within seed first.
    scores = np.asarray(
        [
            [0.0, 0.9],
            [0.0, 0.8],
            [0.0, 0.7],
            [1.0, 0.1],
            [1.0, 0.2],
            [1.0, 0.3],
        ]
    )
    result = smallest_model_within_one_standard_error(
        [1, 2],
        scores,
        generating_seeds=[10, 10, 10, 20, 20, 20],
    )
    np.testing.assert_array_equal(result.generating_seeds, [10, 20])
    np.testing.assert_array_equal(result.replicates_per_seed, [3, 3])
    np.testing.assert_allclose(
        result.within_seed_scores, [[0.0, 0.8], [1.0, 0.2]]
    )
    np.testing.assert_allclose(result.mean_scores, [0.5, 0.5])
    np.testing.assert_allclose(result.standard_errors, [0.5, 0.3])
    assert result.within_seed_method == "mean"


def test_one_standard_error_can_select_restart_by_independent_score():
    endpoint = np.asarray(
        [
            [[0.5, 0.9], [0.6, 0.7]],
            [[0.4, 0.8], [0.7, 0.6]],
        ]
    )
    training_objective = np.asarray(
        [
            [[10.0, 1.0], [9.0, 2.0]],
            [[1.0, 9.0], [2.0, 8.0]],
        ]
    )
    result = smallest_model_within_one_standard_error(
        [1, 2],
        endpoint,
        generating_seeds=[101, 202],
        initialization_selection_scores=training_objective,
    )
    # For each generating seed and model order, endpoint values come from the
    # restart selected by the separate training objective—not endpoint maxima.
    np.testing.assert_allclose(
        result.within_seed_scores, [[0.5, 0.7], [0.7, 0.8]]
    )
    np.testing.assert_allclose(result.mean_scores, [0.6, 0.75])
    np.testing.assert_array_equal(result.replicates_per_seed, [2, 2])
    assert result.within_seed_method == "selected_by_initialization_score"


def test_one_standard_error_requires_independent_generating_seeds():
    with pytest.raises(ValueError, match="independent generating seeds"):
        smallest_model_within_one_standard_error(
            [1, 2],
            np.asarray([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]),
            generating_seeds=[7, 7, 7],
        )


@pytest.mark.parametrize(
    "function,args,match",
    [
        (
            multiclass_brier_score,
            (np.asarray([0, 1]), np.asarray([[0.8, 0.8], [0.3, 0.3]])),
            "sum to one",
        ),
        (
            transition_boundary_f1,
            (np.asarray([0, 1]), np.asarray([0, 1])),
            "non-negative",
        ),
        (
            smallest_model_within_one_standard_error,
            ([1, 2], np.asarray([[0.1, 0.2]])),
            "at least two",
        ),
    ],
)
def test_invalid_metric_inputs_fail_loudly(function, args, match):
    kwargs = {}
    if function is transition_boundary_f1:
        kwargs["tolerance_samples"] = -1
    with pytest.raises(ValueError, match=match):
        function(*args, **kwargs)
