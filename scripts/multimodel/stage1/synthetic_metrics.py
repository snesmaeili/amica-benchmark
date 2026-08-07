"""Predeclared evaluation metrics for Stage I synthetic regime recovery.

Label conventions
-----------------
Ground-truth and hard predicted labels are one-dimensional arrays of
non-negative integers.  They need not be contiguous: functions sort the
observed label values before constructing contingency tables.  Posterior
arrays always have shape ``(n_models, n_samples)`` and row ``m`` corresponds
to fitted model index ``m``.  Gradual generating regimes are represented by a
soft planted target with shape ``(n_true_regimes, n_samples)``; proper scores
compare against that target directly rather than collapsing it to hard labels.

Hungarian alignment is one-to-one.  When fitted and true model counts differ,
unmatched fitted models map to ``-1`` and unmatched true regimes receive a
zero posterior row.  Proper multiclass scores require a bijection and
therefore reject unequal model counts rather than silently merging or dropping
posterior mass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score


@dataclass(frozen=True)
class LabelAlignment:
    """One-to-one alignment of fitted labels to planted regime labels."""

    true_classes: np.ndarray
    predicted_classes: np.ndarray
    contingency: np.ndarray
    predicted_to_true: np.ndarray
    aligned_predictions: np.ndarray
    matched_count: int
    accuracy: float


@dataclass(frozen=True)
class PosteriorAlignment:
    """Posterior rows reordered into sorted ground-truth regime order."""

    true_classes: np.ndarray
    predicted_to_true_position: np.ndarray
    true_to_predicted_model: np.ndarray
    aligned_posteriors: np.ndarray
    unmatched_probability_mass: np.ndarray
    alignment_score: np.ndarray

    @property
    def is_bijective(self) -> bool:
        """Whether every fitted model and planted regime has one match."""
        return bool(
            np.all(self.predicted_to_true_position >= 0)
            and np.all(self.true_to_predicted_model >= 0)
            and self.predicted_to_true_position.size
            == self.true_to_predicted_model.size
        )


@dataclass(frozen=True)
class RegimeRecoveryMetrics:
    """Ground-truth regime recovery and posterior calibration metrics."""

    accuracy: float
    adjusted_rand_index: float
    variation_of_information_nats: float
    multiclass_brier_score: float
    classwise_calibration_error: float
    alignment: PosteriorAlignment


@dataclass(frozen=True)
class SourceRecoveryMetrics:
    """Aligned source, scalp-map, SIR and square-Amari recovery metrics."""

    alignment: ModelAlignment
    source_signs: np.ndarray
    source_scales: np.ndarray
    matched_source_correlations: np.ndarray
    weighted_mean_source_correlation: float
    matched_map_errors: np.ndarray
    weighted_mean_map_error: float
    source_sir_db: np.ndarray
    aggregate_sir_db: float
    square_amari_distance: float
    sample_weights: np.ndarray
    source_weights: np.ndarray


@dataclass(frozen=True)
class BoundaryMetrics:
    """One-to-one transition-boundary matching results."""

    true_boundaries: np.ndarray
    predicted_boundaries: np.ndarray
    matched_pairs: Tuple[Tuple[int, int], ...]
    tolerance_samples: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class OccupancyRecovery:
    """Posterior and hard-assignment occupancy relative to planted occupancy."""

    classes: np.ndarray
    true_occupancy: np.ndarray
    posterior_occupancy: np.ndarray
    hard_occupancy: np.ndarray
    posterior_error: np.ndarray
    mean_absolute_error: float
    maximum_absolute_error: float
    total_variation_distance: float


@dataclass(frozen=True)
class ModelAlignment:
    """Maximum-score one-to-one alignment of candidate to reference models."""

    score_matrix: np.ndarray
    reference_to_candidate: np.ndarray
    candidate_to_reference: np.ndarray
    matched_scores: np.ndarray

    @property
    def is_bijective(self) -> bool:
        """Whether all reference and candidate models were matched."""
        return bool(
            self.reference_to_candidate.size == self.candidate_to_reference.size
            and np.all(self.reference_to_candidate >= 0)
            and np.all(self.candidate_to_reference >= 0)
        )


@dataclass(frozen=True)
class PosteriorStability:
    """Cross-seed posterior stability after model-label alignment."""

    alignment: ModelAlignment
    matched_correlations: np.ndarray
    mean_matched_correlation: float
    mean_absolute_posterior_difference: float
    hard_assignment_agreement: float
    hard_assignment_adjusted_rand_index: float


@dataclass(frozen=True)
class OneStandardErrorSelection:
    """Result of the smallest-model-within-one-standard-error rule."""

    model_orders: np.ndarray
    mean_scores: np.ndarray
    standard_errors: np.ndarray
    empirical_best_order: int
    threshold: float
    eligible: np.ndarray
    selected_order: int
    higher_is_better: bool
    generating_seeds: np.ndarray
    within_seed_scores: np.ndarray
    replicates_per_seed: np.ndarray
    within_seed_method: str


def _validate_integer_labels(labels: np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 1 or values.size < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.issubdtype(values.dtype, np.integer):
        if not (
            np.issubdtype(values.dtype, np.floating)
            and np.all(np.isfinite(values))
            and np.all(values == np.floor(values))
        ):
            raise ValueError(f"{name} must contain integer-valued labels")
    values = values.astype(np.int64, copy=False)
    if np.any(values < 0):
        raise ValueError(f"{name} cannot contain negative labels")
    return values


def _validate_posteriors(
    posteriors: np.ndarray,
    *,
    n_samples: Optional[int] = None,
    atol: float = 1e-8,
) -> np.ndarray:
    values = np.asarray(posteriors, dtype=float)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError(
            "posteriors must have shape (n_models, n_samples)"
        )
    if n_samples is not None and values.shape[1] != n_samples:
        raise ValueError("posterior sample count does not match labels")
    if not np.all(np.isfinite(values)):
        raise ValueError("posteriors contain non-finite values")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("posterior probabilities must lie in [0, 1]")
    if not np.allclose(values.sum(axis=0), 1.0, rtol=0.0, atol=atol):
        raise ValueError("posterior probabilities must sum to one per sample")
    return values


def _validate_soft_targets(
    targets: np.ndarray,
    *,
    n_samples: Optional[int] = None,
    atol: float = 1e-8,
) -> np.ndarray:
    values = np.asarray(targets, dtype=float)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError(
            "soft targets must have shape (n_true_regimes, n_samples)"
        )
    if n_samples is not None and values.shape[1] != n_samples:
        raise ValueError("soft-target sample count does not match posteriors")
    if not np.all(np.isfinite(values)):
        raise ValueError("soft targets contain non-finite values")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("soft-target probabilities must lie in [0, 1]")
    if not np.allclose(values.sum(axis=0), 1.0, rtol=0.0, atol=atol):
        raise ValueError("soft-target probabilities must sum to one per sample")
    return values


def _target_matrix(
    targets: np.ndarray,
    *,
    classes: Optional[Sequence[int]] = None,
    n_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Return target probabilities, row labels, hard positions and softness."""
    values = np.asarray(targets)
    if values.ndim == 1:
        labels = _validate_integer_labels(values, name="true_labels")
        if n_samples is not None and labels.size != n_samples:
            raise ValueError("target sample count does not match posteriors")
        if classes is None:
            class_values = np.unique(labels)
        else:
            class_values = _validate_integer_labels(
                np.asarray(classes), name="classes"
            )
            if np.unique(class_values).size != class_values.size:
                raise ValueError("classes must contain unique labels")
            if not np.all(np.isin(labels, class_values)):
                raise ValueError(
                    "true_labels include a class absent from classes"
                )
        lookup = {
            int(value): index for index, value in enumerate(class_values)
        }
        hard_positions = np.asarray(
            [lookup[int(value)] for value in labels], dtype=np.int64
        )
        probabilities = np.zeros(
            (class_values.size, labels.size), dtype=float
        )
        probabilities[hard_positions, np.arange(labels.size)] = 1.0
        return probabilities, class_values, hard_positions, False

    probabilities = _validate_soft_targets(values, n_samples=n_samples)
    if classes is None:
        class_values = np.arange(
            probabilities.shape[0], dtype=np.int64
        )
    else:
        class_values = _validate_integer_labels(
            np.asarray(classes), name="classes"
        )
        if (
            class_values.size != probabilities.shape[0]
            or np.unique(class_values).size != class_values.size
        ):
            raise ValueError(
                "classes must uniquely identify every soft-target row"
            )
    return (
        probabilities,
        class_values,
        np.argmax(probabilities, axis=0).astype(np.int64),
        True,
    )


def _finite_real_matrix(
    values: np.ndarray,
    *,
    name: str,
    ndim: int = 2,
) -> np.ndarray:
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    matrix = np.asarray(raw, dtype=float)
    if matrix.ndim != ndim or min(matrix.shape) < 1:
        raise ValueError(f"{name} must be a non-empty {ndim}-dimensional array")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return matrix


def _normalised_weights(
    values: Optional[np.ndarray],
    *,
    size: int,
    name: str,
) -> np.ndarray:
    if values is None:
        weights = np.ones(size, dtype=float)
    else:
        weights = np.asarray(values, dtype=float)
        if weights.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},)")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError(f"{name} must be finite and non-negative")
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError(f"{name} must have positive total weight")
    return weights / total


def hungarian_align_labels(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> LabelAlignment:
    """Align arbitrary fitted cluster labels to ground truth.

    Unmatched predicted labels are represented by ``-1`` in both the mapping
    and aligned predictions and therefore count as incorrect.
    """
    truth = _validate_integer_labels(true_labels, name="true_labels")
    predicted = _validate_integer_labels(
        predicted_labels, name="predicted_labels"
    )
    if truth.shape != predicted.shape:
        raise ValueError("true_labels and predicted_labels must have one shape")

    true_classes, true_inverse = np.unique(truth, return_inverse=True)
    predicted_classes, predicted_inverse = np.unique(
        predicted, return_inverse=True
    )
    contingency = np.zeros(
        (true_classes.size, predicted_classes.size), dtype=np.int64
    )
    np.add.at(contingency, (true_inverse, predicted_inverse), 1)
    true_positions, predicted_positions = linear_sum_assignment(-contingency)

    predicted_to_true = np.full(predicted_classes.size, -1, dtype=np.int64)
    predicted_to_true[predicted_positions] = true_classes[true_positions]
    aligned = predicted_to_true[predicted_inverse]
    matched_count = int(np.sum(aligned == truth))
    return LabelAlignment(
        true_classes=true_classes,
        predicted_classes=predicted_classes,
        contingency=contingency,
        predicted_to_true=predicted_to_true,
        aligned_predictions=aligned,
        matched_count=matched_count,
        accuracy=matched_count / truth.size,
    )


def hungarian_align_posteriors(
    planted_targets: np.ndarray,
    posteriors: np.ndarray,
    *,
    classes: Optional[Sequence[int]] = None,
) -> PosteriorAlignment:
    """Align fitted posterior rows to planted regimes.

    ``planted_targets`` may be hard labels or a soft
    ``(n_true_regimes, n_samples)`` target.  The assignment score for true
    regime ``r`` and fitted model ``m`` is ``sum_t q(r,t) p(m|t)``.  Thus
    gradual transitions are aligned against their planted interpolation
    weights rather than an arbitrary hard collapse.
    """
    probabilities = _validate_posteriors(posteriors)
    target_probabilities, true_classes, _hard_positions, _is_soft = (
        _target_matrix(
            planted_targets,
            classes=classes,
            n_samples=probabilities.shape[1],
        )
    )
    n_true = true_classes.size
    n_predicted = probabilities.shape[0]
    scores = target_probabilities @ probabilities.T

    true_positions, predicted_models = linear_sum_assignment(-scores)
    predicted_to_true_position = np.full(
        n_predicted, -1, dtype=np.int64
    )
    true_to_predicted_model = np.full(n_true, -1, dtype=np.int64)
    predicted_to_true_position[predicted_models] = true_positions
    true_to_predicted_model[true_positions] = predicted_models

    aligned = np.zeros((n_true, probabilities.shape[1]), dtype=float)
    for true_position, predicted_model in zip(
        true_positions, predicted_models
    ):
        aligned[true_position] = probabilities[predicted_model]
    unmatched = probabilities[
        predicted_to_true_position < 0
    ].sum(axis=0)
    return PosteriorAlignment(
        true_classes=true_classes,
        predicted_to_true_position=predicted_to_true_position,
        true_to_predicted_model=true_to_predicted_model,
        aligned_posteriors=aligned,
        unmatched_probability_mass=unmatched,
        alignment_score=scores,
    )


def variation_of_information(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
) -> float:
    """Return variation of information in nats.

    ``VI(A, B) = H(A) + H(B) - 2 I(A; B)``.  It is label-permutation
    invariant, equals zero for identical partitions, and uses natural logs.
    """
    first = _validate_integer_labels(labels_a, name="labels_a")
    second = _validate_integer_labels(labels_b, name="labels_b")
    if first.shape != second.shape:
        raise ValueError("labels_a and labels_b must have one shape")
    _, first_inverse = np.unique(first, return_inverse=True)
    _, second_inverse = np.unique(second, return_inverse=True)
    contingency = np.zeros(
        (
            int(first_inverse.max()) + 1,
            int(second_inverse.max()) + 1,
        ),
        dtype=float,
    )
    np.add.at(contingency, (first_inverse, second_inverse), 1.0)
    joint = contingency / first.size
    marginal_a = joint.sum(axis=1)
    marginal_b = joint.sum(axis=0)
    nonzero_joint = joint > 0
    expected = marginal_a[:, None] * marginal_b[None, :]
    mutual_information = float(
        np.sum(
            joint[nonzero_joint]
            * np.log(joint[nonzero_joint] / expected[nonzero_joint])
        )
    )
    entropy_a = float(
        -np.sum(marginal_a[marginal_a > 0] * np.log(marginal_a[marginal_a > 0]))
    )
    entropy_b = float(
        -np.sum(marginal_b[marginal_b > 0] * np.log(marginal_b[marginal_b > 0]))
    )
    return max(0.0, entropy_a + entropy_b - 2.0 * mutual_information)


def multiclass_brier_score(
    planted_targets: np.ndarray,
    posteriors: np.ndarray,
    *,
    classes: Optional[Sequence[int]] = None,
) -> float:
    """Return the unnormalised multiclass Brier score.

    The score is ``mean_t sum_c (p_ct - q_ct)**2`` and ranges from zero to
    two.  For abrupt regimes ``q`` is the one-hot encoding of hard labels; for
    gradual regimes it is the planted interpolation weight.  Posterior and
    soft-target rows follow ``classes`` when supplied.
    """
    probabilities = _validate_posteriors(posteriors)
    target_classes = classes
    if np.asarray(planted_targets).ndim == 1 and classes is None:
        target_classes = np.arange(probabilities.shape[0], dtype=np.int64)
    targets, class_values, _hard_positions, _is_soft = _target_matrix(
        planted_targets,
        classes=target_classes,
        n_samples=probabilities.shape[1],
    )
    if targets.shape != probabilities.shape:
        raise ValueError(
            "planted targets and posteriors must have the same class rows"
        )
    if class_values.size != probabilities.shape[0]:
        raise ValueError(
            "classes must uniquely identify every posterior row"
        )
    return float(np.mean(np.square(probabilities - targets).sum(axis=0)))


def classwise_expected_calibration_error(
    planted_targets: np.ndarray,
    posteriors: np.ndarray,
    *,
    classes: Optional[Sequence[int]] = None,
    n_bins: int = 10,
) -> float:
    """Return equal-width classwise expected calibration error.

    For every class, probabilities are divided into ``n_bins`` equal-width
    bins over ``[0, 1]``.  Within each class-bin pair, absolute difference
    between mean predicted probability and mean planted class weight is
    weighted by that pair's count divided by ``n_samples * n_classes``.
    Unlike top-label ECE, this definition audits every posterior entry.
    """
    if type(n_bins) is not int or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    probabilities = _validate_posteriors(posteriors)
    target_classes = classes
    if np.asarray(planted_targets).ndim == 1 and classes is None:
        target_classes = np.arange(probabilities.shape[0], dtype=np.int64)
    targets, class_values, _hard_positions, _is_soft = _target_matrix(
        planted_targets,
        classes=target_classes,
        n_samples=probabilities.shape[1],
    )
    if targets.shape != probabilities.shape:
        raise ValueError(
            "planted targets and posteriors must have the same class rows"
        )
    if class_values.size != probabilities.shape[0]:
        raise ValueError(
            "classes must uniquely identify every posterior row"
        )

    total_pairs = probabilities.shape[1] * class_values.size
    error = 0.0
    for class_position in range(class_values.size):
        probabilities_for_class = probabilities[class_position]
        outcomes = targets[class_position]
        bin_indices = np.minimum(
            (probabilities_for_class * n_bins).astype(int), n_bins - 1
        )
        for bin_index in range(n_bins):
            selected = bin_indices == bin_index
            count = int(selected.sum())
            if count == 0:
                continue
            calibration_gap = abs(
                float(probabilities_for_class[selected].mean())
                - float(outcomes[selected].mean())
            )
            error += count / total_pairs * calibration_gap
    return float(error)


def evaluate_regime_recovery(
    planted_targets: np.ndarray,
    posteriors: np.ndarray,
    *,
    classes: Optional[Sequence[int]] = None,
    n_calibration_bins: int = 10,
) -> RegimeRecoveryMetrics:
    """Evaluate planted-regime recovery after one-to-one alignment.

    ARI and variation of information use the unaligned hard partition because
    both are label-permutation invariant.  Accuracy and proper posterior scores
    use the Hungarian-aligned model order.  For gradual regimes, hard partition
    metrics use ``argmax`` of the planted weights, while Brier score and
    calibration retain the full soft planted target.
    """
    probabilities = _validate_posteriors(posteriors)
    target_probabilities, class_values, true_positions, _is_soft = (
        _target_matrix(
            planted_targets,
            classes=classes,
            n_samples=probabilities.shape[1],
        )
    )
    alignment = hungarian_align_posteriors(
        target_probabilities,
        probabilities,
        classes=class_values,
    )
    if not alignment.is_bijective:
        raise ValueError(
            "proper posterior scores require equal fitted and true model counts"
        )
    hard_unaligned = np.argmax(probabilities, axis=0)
    hard_aligned = alignment.predicted_to_true_position[hard_unaligned]
    accuracy = float(np.mean(hard_aligned == true_positions))
    return RegimeRecoveryMetrics(
        accuracy=accuracy,
        adjusted_rand_index=float(
            adjusted_rand_score(true_positions, hard_unaligned)
        ),
        variation_of_information_nats=variation_of_information(
            true_positions, hard_unaligned
        ),
        multiclass_brier_score=multiclass_brier_score(
            target_probabilities, alignment.aligned_posteriors
        ),
        classwise_calibration_error=classwise_expected_calibration_error(
            target_probabilities,
            alignment.aligned_posteriors,
            n_bins=n_calibration_bins,
        ),
        alignment=alignment,
    )


def transition_boundaries(labels: np.ndarray) -> np.ndarray:
    """Return transition coordinates.

    A boundary at index ``i`` means sample ``i`` is the first sample carrying
    a label different from sample ``i - 1``.  Index zero is never a boundary.
    """
    values = _validate_integer_labels(labels, name="labels")
    return np.flatnonzero(values[1:] != values[:-1]).astype(np.int64) + 1


def transition_boundary_f1(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    *,
    tolerance_samples: int,
) -> BoundaryMetrics:
    """Match planted and predicted transitions within a sample tolerance.

    Matching is one-to-one and maximises the number of valid matches before
    minimising temporal distance.  If neither sequence contains a transition,
    precision, recall and F1 are defined as one.
    """
    if type(tolerance_samples) is not int or tolerance_samples < 0:
        raise ValueError("tolerance_samples must be a non-negative integer")
    truth = _validate_integer_labels(true_labels, name="true_labels")
    predicted = _validate_integer_labels(
        predicted_labels, name="predicted_labels"
    )
    if truth.shape != predicted.shape:
        raise ValueError("true_labels and predicted_labels must have one shape")
    true_boundaries = transition_boundaries(truth)
    predicted_boundaries = transition_boundaries(predicted)

    matched_pairs: Tuple[Tuple[int, int], ...] = ()
    if true_boundaries.size and predicted_boundaries.size:
        distances = np.abs(
            true_boundaries[:, None] - predicted_boundaries[None, :]
        )
        n_assignments = min(true_boundaries.size, predicted_boundaries.size)
        invalid_cost = n_assignments * (tolerance_samples + 1) + 1
        costs = np.where(
            distances <= tolerance_samples, distances, invalid_cost
        )
        true_indices, predicted_indices = linear_sum_assignment(costs)
        matched_pairs = tuple(
            (int(true_boundaries[true_index]), int(predicted_boundaries[pred_index]))
            for true_index, pred_index in zip(true_indices, predicted_indices)
            if distances[true_index, pred_index] <= tolerance_samples
        )
    n_matched = len(matched_pairs)
    if true_boundaries.size == 0 and predicted_boundaries.size == 0:
        precision = recall = f1 = 1.0
    else:
        precision = (
            n_matched / predicted_boundaries.size
            if predicted_boundaries.size
            else 0.0
        )
        recall = (
            n_matched / true_boundaries.size
            if true_boundaries.size
            else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return BoundaryMetrics(
        true_boundaries=true_boundaries,
        predicted_boundaries=predicted_boundaries,
        matched_pairs=matched_pairs,
        tolerance_samples=tolerance_samples,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )


def occupancy_recovery(
    true_labels: np.ndarray,
    aligned_posteriors: np.ndarray,
    *,
    classes: Optional[Sequence[int]] = None,
    sample_weights: Optional[np.ndarray] = None,
) -> OccupancyRecovery:
    """Compare planted occupancy with posterior and hard occupancy."""
    truth = _validate_integer_labels(true_labels, name="true_labels")
    probabilities = _validate_posteriors(
        aligned_posteriors, n_samples=truth.size
    )
    if classes is None:
        class_values = np.arange(probabilities.shape[0], dtype=np.int64)
    else:
        class_values = _validate_integer_labels(
            np.asarray(classes), name="classes"
        )
        if (
            class_values.size != probabilities.shape[0]
            or np.unique(class_values).size != class_values.size
        ):
            raise ValueError(
                "classes must uniquely identify every posterior row"
            )
    if not np.all(np.isin(truth, class_values)):
        raise ValueError("true_labels include a class absent from classes")

    if sample_weights is None:
        weights = np.ones(truth.size, dtype=float)
    else:
        weights = np.asarray(sample_weights, dtype=float)
        if weights.shape != truth.shape:
            raise ValueError("sample_weights must have shape (n_samples,)")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("sample_weights must be finite and non-negative")
    total_weight = float(weights.sum())
    if total_weight <= 0:
        raise ValueError("sample_weights must contain positive total weight")

    true_occupancy = np.asarray(
        [weights[truth == value].sum() for value in class_values],
        dtype=float,
    ) / total_weight
    posterior_occupancy = (probabilities * weights[None, :]).sum(
        axis=1
    ) / total_weight
    hard_positions = np.argmax(probabilities, axis=0)
    hard_occupancy = np.bincount(
        hard_positions, weights=weights, minlength=class_values.size
    ) / total_weight
    error = posterior_occupancy - true_occupancy
    absolute_error = np.abs(error)
    return OccupancyRecovery(
        classes=class_values,
        true_occupancy=true_occupancy,
        posterior_occupancy=posterior_occupancy,
        hard_occupancy=hard_occupancy,
        posterior_error=error,
        mean_absolute_error=float(absolute_error.mean()),
        maximum_absolute_error=float(absolute_error.max()),
        total_variation_distance=float(0.5 * absolute_error.sum()),
    )


def hungarian_model_alignment(score_matrix: np.ndarray) -> ModelAlignment:
    """Maximise an arbitrary reference-by-candidate model score matrix."""
    scores = np.asarray(score_matrix, dtype=float)
    if scores.ndim != 2 or min(scores.shape) < 1:
        raise ValueError(
            "score_matrix must have shape (n_reference, n_candidate)"
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("score_matrix contains non-finite values")
    reference_indices, candidate_indices = linear_sum_assignment(-scores)
    reference_to_candidate = np.full(scores.shape[0], -1, dtype=np.int64)
    candidate_to_reference = np.full(scores.shape[1], -1, dtype=np.int64)
    reference_to_candidate[reference_indices] = candidate_indices
    candidate_to_reference[candidate_indices] = reference_indices
    return ModelAlignment(
        score_matrix=scores,
        reference_to_candidate=reference_to_candidate,
        candidate_to_reference=candidate_to_reference,
        matched_scores=scores[reference_indices, candidate_indices],
    )


def weighted_source_correlation_matrix(
    true_sources: np.ndarray,
    estimated_sources: np.ndarray,
    *,
    sample_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return weighted Pearson correlations for every true/estimated pair.

    Rows index planted sources and columns index estimated sources.  The
    sample weights should describe the planted regime being evaluated (for
    example its hard membership or gradual interpolation weights), not fitted
    posteriors selected after inspecting recovery.
    """
    truth = _finite_real_matrix(true_sources, name="true_sources")
    estimate = _finite_real_matrix(
        estimated_sources, name="estimated_sources"
    )
    if truth.shape[1] != estimate.shape[1]:
        raise ValueError(
            "true_sources and estimated_sources must share the sample axis"
        )
    weights = _normalised_weights(
        sample_weights,
        size=truth.shape[1],
        name="sample_weights",
    )
    truth_centered = truth - (truth * weights).sum(axis=1, keepdims=True)
    estimate_centered = estimate - (
        estimate * weights
    ).sum(axis=1, keepdims=True)
    truth_norm = np.sqrt(
        np.sum(np.square(truth_centered) * weights, axis=1)
    )
    estimate_norm = np.sqrt(
        np.sum(np.square(estimate_centered) * weights, axis=1)
    )
    if np.any(truth_norm <= 0.0):
        raise ValueError(
            "true_sources contain a constant source under sample_weights"
        )
    if np.any(estimate_norm <= 0.0):
        raise ValueError(
            "estimated_sources contain a constant source under sample_weights"
        )
    covariance = (truth_centered * weights) @ estimate_centered.T
    correlations = covariance / (
        truth_norm[:, np.newaxis] * estimate_norm[np.newaxis, :]
    )
    return np.clip(correlations, -1.0, 1.0)


def square_amari_distance(
    true_mixing: np.ndarray,
    estimated_mixing: np.ndarray,
) -> float:
    """Return the scale/sign/permutation-invariant square Amari distance.

    Both matrices must be finite, nonsingular ``n x n`` mixing matrices with
    ``n >= 2``.  The gain matrix is
    ``P = inv(estimated_mixing) @ true_mixing`` and the conventional
    row-and-column normalised error is divided by ``2 n (n - 1)``, yielding
    zero for exact recovery up to source permutation, sign and scale.
    """
    truth = _finite_real_matrix(true_mixing, name="true_mixing")
    estimate = _finite_real_matrix(
        estimated_mixing, name="estimated_mixing"
    )
    if truth.shape != estimate.shape or truth.shape[0] != truth.shape[1]:
        raise ValueError(
            "square Amari distance requires equal square mixing matrices"
        )
    n_sources = truth.shape[0]
    if n_sources < 2:
        raise ValueError("square Amari distance requires at least two sources")
    if (
        np.linalg.matrix_rank(truth) < n_sources
        or np.linalg.matrix_rank(estimate) < n_sources
    ):
        raise ValueError(
            "square Amari distance requires nonsingular mixing matrices"
        )
    try:
        gain = np.linalg.solve(estimate, truth)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "estimated_mixing could not be inverted for Amari distance"
        ) from error
    absolute = np.abs(gain)
    row_maximum = absolute.max(axis=1)
    column_maximum = absolute.max(axis=0)
    if np.any(row_maximum <= 0.0) or np.any(column_maximum <= 0.0):
        raise ValueError("Amari gain matrix contains an empty row or column")
    row_error = np.sum(absolute / row_maximum[:, np.newaxis]) - n_sources
    column_error = (
        np.sum(absolute / column_maximum[np.newaxis, :]) - n_sources
    )
    distance = (row_error + column_error) / (
        2.0 * n_sources * (n_sources - 1)
    )
    return float(max(0.0, distance))


def evaluate_source_recovery(
    true_sources: np.ndarray,
    estimated_sources: np.ndarray,
    true_mixing: np.ndarray,
    estimated_mixing: np.ndarray,
    *,
    sample_weights: Optional[np.ndarray] = None,
    source_weights: Optional[np.ndarray] = None,
) -> SourceRecoveryMetrics:
    """Evaluate one aligned planted-regime/estimated-model source solution.

    Alignment maximises absolute weighted source-time-course correlation.
    Each matched estimated source is sign-corrected and least-squares scaled
    to the centred planted source for reporting. Map error is ``1 - |r|`` for
    the correspondingly aligned mixing columns. SIR follows the conventional
    gain-matrix definition using ``G = inv(A_est) @ A_true``: the matched
    diagonal contribution is signal and all off-target planted sources are
    interference. Sample weights define the regime-specific time support for
    correlation; source weights define only across-source summaries and never
    alter the Hungarian assignment.
    """
    truth = _finite_real_matrix(true_sources, name="true_sources")
    estimate = _finite_real_matrix(
        estimated_sources, name="estimated_sources"
    )
    if truth.shape != estimate.shape:
        raise ValueError(
            "source recovery requires equal true/estimated source shapes"
        )
    n_sources, n_samples = truth.shape
    true_maps = _finite_real_matrix(true_mixing, name="true_mixing")
    estimated_maps = _finite_real_matrix(
        estimated_mixing, name="estimated_mixing"
    )
    if (
        true_maps.shape != estimated_maps.shape
        or true_maps.shape[1] != n_sources
    ):
        raise ValueError(
            "mixing matrices must share shape (n_sensors, n_sources)"
        )
    if true_maps.shape[0] != n_sources:
        raise ValueError(
            "square Amari distance requires n_sensors == n_sources"
        )

    time_weights = _normalised_weights(
        sample_weights, size=n_samples, name="sample_weights"
    )
    component_weights = _normalised_weights(
        source_weights, size=n_sources, name="source_weights"
    )
    correlations = weighted_source_correlation_matrix(
        truth,
        estimate,
        sample_weights=time_weights,
    )
    alignment = hungarian_model_alignment(np.abs(correlations))
    if not alignment.is_bijective:
        raise ValueError(
            "source recovery requires equal planted and estimated source counts"
        )

    signs = np.empty(n_sources, dtype=float)
    scales = np.empty(n_sources, dtype=float)
    matched_correlations = np.empty(n_sources, dtype=float)
    map_errors = np.empty(n_sources, dtype=float)
    source_sir = np.empty(n_sources, dtype=float)
    gain = np.linalg.solve(estimated_maps, true_maps)
    signal_powers = np.empty(n_sources, dtype=float)
    interference_powers = np.empty(n_sources, dtype=float)

    for true_index, estimated_index in enumerate(
        alignment.reference_to_candidate
    ):
        raw_correlation = correlations[true_index, estimated_index]
        sign = 1.0 if raw_correlation >= 0.0 else -1.0
        true_source = truth[true_index]
        estimated_source = estimate[estimated_index]
        true_centered = true_source - np.sum(time_weights * true_source)
        estimated_centered = sign * (
            estimated_source - np.sum(time_weights * estimated_source)
        )
        estimated_power = float(
            np.sum(time_weights * np.square(estimated_centered))
        )
        if estimated_power <= 0.0:
            raise ValueError(
                "matched estimated source has zero weighted variance"
            )
        scale = float(
            np.sum(time_weights * true_centered * estimated_centered)
            / estimated_power
        )
        signal_power = float(np.abs(gain[estimated_index, true_index]) ** 2)
        interference_power = float(
            np.sum(np.abs(gain[estimated_index]) ** 2) - signal_power
        )
        tolerance = np.finfo(float).eps * max(1.0, signal_power)
        sir = (
            np.inf
            if interference_power <= tolerance
            else 10.0 * np.log10(signal_power / interference_power)
        )

        true_map = true_maps[:, true_index]
        estimated_map = estimated_maps[:, estimated_index]
        true_map_centered = true_map - true_map.mean()
        estimated_map_centered = estimated_map - estimated_map.mean()
        map_denominator = float(
            np.linalg.norm(true_map_centered)
            * np.linalg.norm(estimated_map_centered)
        )
        if map_denominator <= 0.0:
            raise ValueError(
                "mixing matrices contain a constant scalp-map column"
            )
        map_correlation = float(
            np.dot(true_map_centered, estimated_map_centered)
            / map_denominator
        )

        signs[true_index] = sign
        scales[true_index] = scale
        matched_correlations[true_index] = abs(raw_correlation)
        map_errors[true_index] = 1.0 - min(1.0, abs(map_correlation))
        signal_powers[true_index] = signal_power
        interference_powers[true_index] = interference_power
        source_sir[true_index] = sir

    aggregate_signal = float(np.sum(component_weights * signal_powers))
    aggregate_interference = float(
        np.sum(component_weights * interference_powers)
    )
    aggregate_tolerance = np.finfo(float).eps * max(1.0, aggregate_signal)
    aggregate_sir = (
        np.inf
        if aggregate_interference <= aggregate_tolerance
        else 10.0 * np.log10(
            aggregate_signal / aggregate_interference
        )
    )
    return SourceRecoveryMetrics(
        alignment=alignment,
        source_signs=signs,
        source_scales=scales,
        matched_source_correlations=matched_correlations,
        weighted_mean_source_correlation=float(
            np.sum(component_weights * matched_correlations)
        ),
        matched_map_errors=map_errors,
        weighted_mean_map_error=float(
            np.sum(component_weights * map_errors)
        ),
        source_sir_db=source_sir,
        aggregate_sir_db=float(aggregate_sir),
        square_amari_distance=square_amari_distance(
            true_maps, estimated_maps
        ),
        sample_weights=time_weights,
        source_weights=component_weights,
    )


def posterior_similarity_matrix(
    reference_posteriors: np.ndarray,
    candidate_posteriors: np.ndarray,
) -> np.ndarray:
    """Return pairwise Pearson correlations between model posteriors.

    A pair of exactly equal constant rows is assigned correlation one; another
    constant-row comparison is assigned zero because Pearson correlation is
    otherwise undefined.
    """
    reference = _validate_posteriors(reference_posteriors)
    candidate = _validate_posteriors(
        candidate_posteriors, n_samples=reference.shape[1]
    )
    scores = np.empty((reference.shape[0], candidate.shape[0]), dtype=float)
    for reference_index, reference_row in enumerate(reference):
        reference_centered = reference_row - reference_row.mean()
        reference_norm = float(np.linalg.norm(reference_centered))
        for candidate_index, candidate_row in enumerate(candidate):
            candidate_centered = candidate_row - candidate_row.mean()
            candidate_norm = float(np.linalg.norm(candidate_centered))
            if reference_norm == 0.0 or candidate_norm == 0.0:
                scores[reference_index, candidate_index] = (
                    1.0
                    if np.array_equal(reference_row, candidate_row)
                    else 0.0
                )
            else:
                scores[reference_index, candidate_index] = float(
                    np.dot(reference_centered, candidate_centered)
                    / (reference_norm * candidate_norm)
                )
    return np.clip(scores, -1.0, 1.0)


def align_candidate_model_axis(
    candidate_values: np.ndarray,
    alignment: ModelAlignment,
    *,
    axis: int = 0,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Reorder a candidate model axis into reference-model order.

    Unmatched reference models are filled with ``fill_value``.  Extra candidate
    models are omitted rather than silently combined.
    """
    values = np.asarray(candidate_values)
    axis = int(axis)
    if axis < 0:
        axis += values.ndim
    if axis < 0 or axis >= values.ndim:
        raise ValueError("axis is outside candidate_values")
    if values.shape[axis] != alignment.candidate_to_reference.size:
        raise ValueError("candidate model axis does not match alignment")
    output_shape = list(values.shape)
    output_shape[axis] = alignment.reference_to_candidate.size
    dtype = np.result_type(values.dtype, type(fill_value))
    output = np.full(output_shape, fill_value, dtype=dtype)
    for reference_index, candidate_index in enumerate(
        alignment.reference_to_candidate
    ):
        if candidate_index < 0:
            continue
        source = [slice(None)] * values.ndim
        target = [slice(None)] * values.ndim
        source[axis] = int(candidate_index)
        target[axis] = reference_index
        output[tuple(target)] = values[tuple(source)]
    return output


def cross_seed_posterior_stability(
    reference_posteriors: np.ndarray,
    candidate_posteriors: np.ndarray,
) -> PosteriorStability:
    """Evaluate seed-to-seed stability after posterior-correlation alignment."""
    reference = _validate_posteriors(reference_posteriors)
    candidate = _validate_posteriors(
        candidate_posteriors, n_samples=reference.shape[1]
    )
    alignment = hungarian_model_alignment(
        posterior_similarity_matrix(reference, candidate)
    )
    if not alignment.is_bijective:
        raise ValueError(
            "posterior stability requires equal reference and candidate model counts"
        )
    aligned_candidate = align_candidate_model_axis(candidate, alignment)
    matched_correlations = np.asarray(
        [
            alignment.score_matrix[
                reference_index, candidate_index
            ]
            for reference_index, candidate_index in enumerate(
                alignment.reference_to_candidate
            )
        ],
        dtype=float,
    )
    reference_hard = np.argmax(reference, axis=0)
    candidate_hard = np.argmax(aligned_candidate, axis=0)
    return PosteriorStability(
        alignment=alignment,
        matched_correlations=matched_correlations,
        mean_matched_correlation=float(matched_correlations.mean()),
        mean_absolute_posterior_difference=float(
            np.mean(np.abs(reference - aligned_candidate))
        ),
        hard_assignment_agreement=float(
            np.mean(reference_hard == candidate_hard)
        ),
        hard_assignment_adjusted_rand_index=float(
            adjusted_rand_score(reference_hard, candidate_hard)
        ),
    )


def smallest_model_within_one_standard_error(
    model_orders: Sequence[int],
    replicate_scores: np.ndarray,
    *,
    higher_is_better: bool = True,
    generating_seeds: Optional[Sequence[int]] = None,
    within_seed_aggregation: str = "mean",
    initialization_selection_scores: Optional[np.ndarray] = None,
    selection_higher_is_better: bool = True,
) -> OneStandardErrorSelection:
    """Select the smallest model within one generating-seed SE of the best.

    ``replicate_scores`` may have shape ``(n_runs, n_model_orders)`` or
    ``(n_generating_seeds, n_initialisations, n_model_orders)``.  Repeated fit
    initialisations are first reduced *within* each generating seed; only then
    are means and standard errors computed across independent generating
    seeds.

    For a two-dimensional input, ``generating_seeds`` identifies rows sharing
    a data-generating seed.  If omitted, every row is explicitly treated as an
    independent generating seed.  For a three-dimensional input, axis zero is
    already the generating-seed axis.

    Without ``initialization_selection_scores``, repeated fits are combined
    by ``within_seed_aggregation`` (``"mean"``, ``"median"`` or an explicitly
    requested ``"best"`` endpoint).  Prefer supplying an independent
    training-objective array to select an initialisation without optimistically
    selecting on the validation endpoint itself.
    """
    orders = np.asarray(model_orders)
    if (
        orders.ndim != 1
        or orders.size < 1
        or not np.issubdtype(orders.dtype, np.integer)
        or np.any(orders < 1)
        or np.unique(orders).size != orders.size
    ):
        raise ValueError("model_orders must be unique positive integers")
    scores = np.asarray(replicate_scores, dtype=float)
    if scores.ndim not in {2, 3} or scores.shape[-1] != orders.size:
        raise ValueError(
            "replicate_scores must have shape (n_runs, n_model_orders) or "
            "(n_generating_seeds, n_initialisations, n_model_orders)"
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("replicate_scores contain non-finite values")
    if within_seed_aggregation not in {"mean", "median", "best"}:
        raise ValueError(
            "within_seed_aggregation must be 'mean', 'median', or 'best'"
        )

    selection = None
    if initialization_selection_scores is not None:
        selection = np.asarray(initialization_selection_scores, dtype=float)
        if selection.shape != scores.shape:
            raise ValueError(
                "initialization_selection_scores must match replicate_scores"
            )
        if not np.all(np.isfinite(selection)):
            raise ValueError(
                "initialization_selection_scores contain non-finite values"
            )

    if scores.ndim == 3:
        n_generating_seeds, n_initializations, _n_orders = scores.shape
        if n_initializations < 1:
            raise ValueError("each generating seed needs an initialisation")
        if generating_seeds is None:
            seed_values = np.arange(n_generating_seeds, dtype=np.int64)
        else:
            seed_values = _validate_integer_labels(
                np.asarray(generating_seeds), name="generating_seeds"
            )
            if seed_values.size != n_generating_seeds:
                raise ValueError(
                    "generating_seeds must identify axis zero of replicate_scores"
                )
            if np.unique(seed_values).size != seed_values.size:
                raise ValueError(
                    "three-dimensional generating_seeds must be unique"
                )
        grouped_scores = [
            scores[seed_index]
            for seed_index in range(n_generating_seeds)
        ]
        grouped_selection = (
            None
            if selection is None
            else [
                selection[seed_index]
                for seed_index in range(n_generating_seeds)
            ]
        )
    else:
        n_runs = scores.shape[0]
        if generating_seeds is None:
            run_seeds = np.arange(n_runs, dtype=np.int64)
        else:
            run_seeds = _validate_integer_labels(
                np.asarray(generating_seeds), name="generating_seeds"
            )
            if run_seeds.size != n_runs:
                raise ValueError(
                    "generating_seeds must identify every score row"
                )
        seed_values = np.unique(run_seeds)
        grouped_scores = [scores[run_seeds == seed] for seed in seed_values]
        grouped_selection = (
            None
            if selection is None
            else [
                selection[run_seeds == seed]
                for seed in seed_values
            ]
        )

    if seed_values.size < 2:
        raise ValueError(
            "at least two independent generating seeds are required"
        )
    replicates_per_seed = np.asarray(
        [group.shape[0] for group in grouped_scores], dtype=np.int64
    )
    within_scores = np.empty((seed_values.size, orders.size), dtype=float)
    if grouped_selection is not None:
        for seed_index, (endpoint_group, selection_group) in enumerate(
            zip(grouped_scores, grouped_selection)
        ):
            selected_indices = (
                np.argmax(selection_group, axis=0)
                if selection_higher_is_better
                else np.argmin(selection_group, axis=0)
            )
            within_scores[seed_index] = endpoint_group[
                selected_indices, np.arange(orders.size)
            ]
        within_seed_method = "selected_by_initialization_score"
    else:
        for seed_index, endpoint_group in enumerate(grouped_scores):
            if within_seed_aggregation == "mean":
                within_scores[seed_index] = endpoint_group.mean(axis=0)
            elif within_seed_aggregation == "median":
                within_scores[seed_index] = np.median(
                    endpoint_group, axis=0
                )
            elif higher_is_better:
                within_scores[seed_index] = endpoint_group.max(axis=0)
            else:
                within_scores[seed_index] = endpoint_group.min(axis=0)
        within_seed_method = within_seed_aggregation

    means = within_scores.mean(axis=0)
    standard_errors = within_scores.std(axis=0, ddof=1) / np.sqrt(
        within_scores.shape[0]
    )
    best_value = means.max() if higher_is_better else means.min()
    tied_best = np.flatnonzero(
        np.isclose(means, best_value, rtol=1e-12, atol=1e-15)
    )
    best_index = int(
        tied_best[np.argmin(orders[tied_best].astype(np.int64))]
    )
    if higher_is_better:
        threshold = float(means[best_index] - standard_errors[best_index])
        eligible = means >= threshold
    else:
        threshold = float(means[best_index] + standard_errors[best_index])
        eligible = means <= threshold
    selected_order = int(np.min(orders[eligible]))
    return OneStandardErrorSelection(
        model_orders=orders.astype(np.int64),
        mean_scores=means,
        standard_errors=standard_errors,
        empirical_best_order=int(orders[best_index]),
        threshold=threshold,
        eligible=eligible,
        selected_order=selected_order,
        higher_is_better=higher_is_better,
        generating_seeds=seed_values,
        within_seed_scores=within_scores,
        replicates_per_seed=replicates_per_seed,
        within_seed_method=within_seed_method,
    )
