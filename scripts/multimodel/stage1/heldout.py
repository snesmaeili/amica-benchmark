"""Leakage-resistant held-out evaluation primitives for real EEG.

This module deliberately contains no dataset loading, AMICA fitting, or Slurm
submission logic.  It defines the contracts that a Stage I runner must obey:
guarded blocked folds, training-only dimensionality reduction, common
evaluation samples across fitted model orders, and subject-level predictive
density aggregation.

``PreparedHeldoutFold.train_data`` and ``test_data`` are already centred,
projected, and scaled with the training-only transform.  A fitting runner must
therefore disable AMICA's internal centring and sphering/PCA steps (for
example, ``do_mean=False`` and ``do_sphere=False``) and score the returned
model on these exact arrays.  Re-estimating either transform inside AMICA
would change the evaluation contract and can leak held-out information.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence, Tuple

import numpy as np

from .folds import BlockedFold, SampleSpan
from .provenance import PROVENANCE_SCHEMA_VERSION, validate_provenance


HELDOUT_RESULT_SCHEMA_VERSION = "amica-multimodel-heldout-fold-v1"
HELDOUT_CAMPAIGN = "real_eeg_heldout"
DEFAULT_MODEL_ORDERS = (1, 2, 3, 5, 7, 10)
EXPECTED_N_SPLITS = 5
EXPECTED_GUARD_SECONDS = 5.0
EXPECTED_FIT_SEEDS = (0, 1, 2)
HELDOUT_FIT_CANDIDATE_SCHEMA_VERSION = (
    "amica-multimodel-heldout-fit-candidate-v1"
)
HELDOUT_SELECTION_AUDIT_SCHEMA_VERSION = (
    "amica-multimodel-heldout-selection-audit-v1"
)


class HeldoutCandidateSelectionRule(str, Enum):
    """Predeclared rule for selecting one fit before held-out scoring."""

    MAX_FINITE_TRAINING_LL = (
        "max_finite_recomputed_training_ll_then_smallest_seed_v1"
    )

_HEX40 = frozenset("0123456789abcdef")
_RESULT_FIELDS = {
    "schema_version",
    "run_id",
    "campaign",
    "dataset",
    "subject",
    "fold_index",
    "n_splits",
    "guard_seconds",
    "fit_model_order",
    "fit_seed",
    "selection_rule",
    "n_components",
    "sampling_rate_hz",
    "n_recording_samples",
    "guard_samples",
    "test_start",
    "test_stop",
    "train_spans",
    "guard_spans",
    "n_train_samples",
    "n_guard_samples",
    "n_test_samples",
    "train_index_sha256",
    "test_index_sha256",
    "evaluation_sample_sha256",
    "preprocessing_state_sha256",
    "transform_fit_scope",
    "lpd_test_sum",
    "lpd_test_mean",
    "lpd_unit",
    "result_path",
    "manifest_sha256",
    "manifest_row_index",
    "manifest_row_sha256",
    "benchmark_git_sha",
    "package_git_sha",
    "provenance_schema_version",
    "provenance_run_id",
}
_FIT_CANDIDATE_FIELDS = {
    "schema_version",
    "candidate_id",
    "dataset",
    "subject",
    "fold_index",
    "fit_model_order",
    "fit_seed",
    "training_ll_recomputed",
    "training_ll_unit",
    "fit_healthy",
    "stopping_reason",
    "converged",
    "reached_iteration_cap",
    "occupancy_ok",
    "kish_effective_sample_size_ok",
    "fitted_state_sha256",
    "output_sha256",
    "result_path",
}
_SELECTION_AUDIT_FIELDS = {
    "schema_version",
    "selection_rule",
    "dataset",
    "subject",
    "fold_index",
    "fit_model_order",
    "expected_fit_seeds",
    "selected_candidate_id",
    "selected_fit_seed",
    "selected_record",
    "candidates",
}
_SELECTION_AUDIT_CANDIDATE_FIELDS = {
    "candidate_id",
    "fit_seed",
    "training_ll_recomputed",
    "finite_training_ll",
    "selection_rank",
    "selected",
    "fit_healthy",
    "stopping_reason",
    "converged",
    "reached_iteration_cap",
    "occupancy_ok",
    "kish_effective_sample_size_ok",
    "fitted_state_sha256",
    "output_sha256",
    "result_path",
}
_SELECTED_RECORD_LINK_FIELDS = {
    "candidate_id",
    "fit_seed",
    "fitted_state_sha256",
    "output_sha256",
    "result_path",
}


def _validate_matrix(x: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(x, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (channels, samples)")
    if min(array.shape) < 1:
        raise ValueError(f"{name} cannot have an empty dimension")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _sha256_array(array: np.ndarray, dtype: str) -> str:
    normalized = np.ascontiguousarray(np.asarray(array, dtype=dtype))
    header = f"{normalized.dtype.str}|{normalized.shape}".encode("ascii")
    return hashlib.sha256(header + normalized.tobytes(order="C")).hexdigest()


def _sha256_state(
    mean: np.ndarray,
    components: np.ndarray,
    scales: np.ndarray,
    training_index_sha256: str,
) -> str:
    digest = hashlib.sha256(training_index_sha256.encode("ascii"))
    for array in (mean, components, scales):
        normalized = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
        digest.update(f"{normalized.dtype.str}|{normalized.shape}".encode("ascii"))
        digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def spans_to_indices(spans: Sequence[SampleSpan]) -> np.ndarray:
    """Expand ordered, non-overlapping half-open spans to int64 indices."""
    indices = []
    previous_stop = -1
    for span in spans:
        if span.start < previous_stop:
            raise ValueError("spans must be ordered and non-overlapping")
        indices.append(np.arange(span.start, span.stop, dtype=np.int64))
        previous_stop = span.stop
    if not indices:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(indices)


@dataclass(frozen=True)
class TrainingOnlyPCATransform:
    """PCA projection and whitening fitted exclusively on training samples."""

    channel_mean: np.ndarray
    components: np.ndarray
    pc_scales: np.ndarray
    n_training_samples: int
    training_index_sha256: str
    state_sha256: str

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        training_indices: np.ndarray,
        *,
        n_components: int,
    ) -> "TrainingOnlyPCATransform":
        """Fit centring, PCA directions, and PC scales on training samples."""
        data = _validate_matrix(x, "x")
        indices = np.asarray(training_indices)
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise ValueError("training_indices must be a one-dimensional integer array")
        indices = indices.astype(np.int64, copy=False)
        if indices.size < 2:
            raise ValueError("at least two training samples are required")
        if np.any(np.diff(indices) <= 0):
            raise ValueError("training_indices must be strictly increasing")
        if indices[0] < 0 or indices[-1] >= data.shape[1]:
            raise ValueError("training_indices are outside the sample range")
        if type(n_components) is not int or n_components < 1:
            raise ValueError("n_components must be a positive integer")
        if n_components > min(data.shape[0], indices.size - 1):
            raise ValueError("n_components exceeds the trainable PCA rank")

        training = data[:, indices]
        mean = np.mean(training, axis=1)
        centred = training - mean[:, None]
        left, _, _ = np.linalg.svd(centred, full_matrices=False)
        components = left[:, :n_components].T

        # Fix the arbitrary SVD sign so hashes are stable across repeated runs.
        anchors = np.argmax(np.abs(components), axis=1)
        signs = np.sign(components[np.arange(n_components), anchors])
        signs[signs == 0.0] = 1.0
        components = components * signs[:, None]

        projected = components @ centred
        scales = np.sqrt(np.mean(projected * projected, axis=1))
        tolerance = np.finfo(np.float64).eps * max(data.shape) * scales.max()
        if np.any(~np.isfinite(scales)) or np.any(scales <= tolerance):
            raise ValueError("training data do not support the requested PCA rank")

        index_hash = _sha256_array(indices, "<i8")
        state_hash = _sha256_state(mean, components, scales, index_hash)
        return cls(
            channel_mean=np.asarray(mean, dtype=np.float64),
            components=np.asarray(components, dtype=np.float64),
            pc_scales=np.asarray(scales, dtype=np.float64),
            n_training_samples=int(indices.size),
            training_index_sha256=index_hash,
            state_sha256=state_hash,
        )

    @property
    def n_components(self) -> int:
        return int(self.components.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.components.shape[1])

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply the fixed training transform without estimating new state."""
        data = _validate_matrix(x, "x")
        if data.shape[0] != self.n_channels:
            raise ValueError("x has a different channel count from the fitted state")
        centred = data - self.channel_mean[:, None]
        return (self.components @ centred) / self.pc_scales[:, None]


@dataclass(frozen=True)
class PreparedHeldoutFold:
    """Training and evaluation arrays with auditable sample identities.

    Both arrays are in the fixed training-only PCA coordinate system.  They
    must be passed to AMICA without any further learned centring, whitening,
    sphering, or PCA transformation.
    """

    fold: BlockedFold
    sampling_rate_hz: float
    n_recording_samples: int
    transform: TrainingOnlyPCATransform
    train_data: np.ndarray
    test_data: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_index_sha256: str
    test_index_sha256: str
    evaluation_sample_sha256: str


def prepare_heldout_fold(
    x: np.ndarray,
    fold: BlockedFold,
    *,
    n_components: int,
    sampling_rate_hz: float,
) -> PreparedHeldoutFold:
    """Create one fold using only guarded training samples to fit transforms."""
    data = _validate_matrix(x, "x")
    if not math.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be finite and positive")
    expected_guard_samples = int(
        round(EXPECTED_GUARD_SECONDS * sampling_rate_hz)
    )
    if fold.guard_samples != expected_guard_samples:
        raise ValueError(
            "fold does not use the prespecified five-second guard"
        )
    n_samples = data.shape[1]
    if fold.test.stop > n_samples:
        raise ValueError("fold test span extends beyond x")
    excluded_start = max(0, fold.test.start - fold.guard_samples)
    excluded_stop = min(n_samples, fold.test.stop + fold.guard_samples)
    expected_train = []
    if excluded_start > 0:
        expected_train.append(SampleSpan(0, excluded_start))
    if excluded_stop < n_samples:
        expected_train.append(SampleSpan(excluded_stop, n_samples))
    expected_guard = []
    if excluded_start < fold.test.start:
        expected_guard.append(SampleSpan(excluded_start, fold.test.start))
    if fold.test.stop < excluded_stop:
        expected_guard.append(SampleSpan(fold.test.stop, excluded_stop))
    if fold.train != tuple(expected_train) or fold.guard != tuple(expected_guard):
        raise ValueError("fold spans are not the exact guarded complement")
    train_indices = spans_to_indices(fold.train)
    test_indices = spans_to_indices((fold.test,))
    transform = TrainingOnlyPCATransform.fit(
        data,
        train_indices,
        n_components=n_components,
    )
    train_data = transform.transform(data[:, train_indices])
    test_data = transform.transform(data[:, test_indices])
    return PreparedHeldoutFold(
        fold=fold,
        sampling_rate_hz=float(sampling_rate_hz),
        n_recording_samples=int(n_samples),
        transform=transform,
        train_data=train_data,
        test_data=test_data,
        train_indices=train_indices,
        test_indices=test_indices,
        train_index_sha256=transform.training_index_sha256,
        test_index_sha256=_sha256_array(test_indices, "<i8"),
        evaluation_sample_sha256=_sha256_array(data[:, test_indices], "<f8"),
    )


@dataclass(frozen=True)
class PredictiveDensityScore:
    """Log predictive density on one held-out sample set."""

    total_nats: float
    mean_nats_per_sample: float
    n_samples: int


def score_log_joint(log_joint: np.ndarray) -> PredictiveDensityScore:
    """Score log joint terms ``log pi_m + log p_m(x_t)`` stably.

    Parameters
    ----------
    log_joint
        Array with shape ``(n_models, n_samples)``.
    """
    values = np.asarray(log_joint, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("log_joint must have shape (n_models, n_samples)")
    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise ValueError("log_joint contains NaN or positive infinity")
    maxima = np.max(values, axis=0)
    if np.any(np.isneginf(maxima)):
        raise ValueError("at least one sample has zero density under every model")
    per_sample = maxima + np.log(
        np.sum(np.exp(values - maxima[None, :]), axis=0)
    )
    if not np.all(np.isfinite(per_sample)):
        raise ValueError("held-out log predictive density is non-finite")
    total = float(np.sum(per_sample, dtype=np.float64))
    return PredictiveDensityScore(
        total_nats=total,
        mean_nats_per_sample=total / values.shape[1],
        n_samples=int(values.shape[1]),
    )


def _spans_payload(spans: Sequence[SampleSpan]) -> list[list[int]]:
    return [[int(span.start), int(span.stop)] for span in spans]


def make_heldout_fold_result(
    prepared: PreparedHeldoutFold,
    score: PredictiveDensityScore,
    *,
    run_id: str,
    dataset: str,
    subject: str,
    fit_model_order: int,
    fit_seed: int,
    selection_rule: str | HeldoutCandidateSelectionRule,
    result_path: str,
    manifest_sha256: str,
    manifest_row_index: int,
    manifest_row_sha256: str,
    benchmark_git_sha: str,
    package_git_sha: str,
) -> dict[str, object]:
    """Build and validate a strict JSON-ready fold result record."""
    fold = prepared.fold
    if score.n_samples != prepared.test_indices.size:
        raise ValueError("score sample count does not match the held-out fold")
    record: dict[str, object] = {
        "schema_version": HELDOUT_RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "campaign": HELDOUT_CAMPAIGN,
        "dataset": dataset,
        "subject": subject,
        "fold_index": fold.fold_index,
        "n_splits": EXPECTED_N_SPLITS,
        "guard_seconds": EXPECTED_GUARD_SECONDS,
        "fit_model_order": fit_model_order,
        "fit_seed": fit_seed,
        "selection_rule": (
            selection_rule.value
            if isinstance(selection_rule, HeldoutCandidateSelectionRule)
            else selection_rule
        ),
        "n_components": prepared.transform.n_components,
        "sampling_rate_hz": prepared.sampling_rate_hz,
        "n_recording_samples": prepared.n_recording_samples,
        "guard_samples": fold.guard_samples,
        "test_start": fold.test.start,
        "test_stop": fold.test.stop,
        "train_spans": _spans_payload(fold.train),
        "guard_spans": _spans_payload(fold.guard),
        "n_train_samples": int(prepared.train_indices.size),
        "n_guard_samples": int(sum(span.size for span in fold.guard)),
        "n_test_samples": int(prepared.test_indices.size),
        "train_index_sha256": prepared.train_index_sha256,
        "test_index_sha256": prepared.test_index_sha256,
        "evaluation_sample_sha256": prepared.evaluation_sample_sha256,
        "preprocessing_state_sha256": prepared.transform.state_sha256,
        "transform_fit_scope": "training_samples_only",
        "lpd_test_sum": score.total_nats,
        "lpd_test_mean": score.mean_nats_per_sample,
        "lpd_unit": "nats per held-out sample",
        "result_path": result_path,
        "manifest_sha256": manifest_sha256,
        "manifest_row_index": manifest_row_index,
        "manifest_row_sha256": manifest_row_sha256,
        "benchmark_git_sha": benchmark_git_sha,
        "package_git_sha": package_git_sha,
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "provenance_run_id": run_id,
    }
    validate_heldout_fold_result(record)
    return record


def _exact_keys(record: Mapping[str, object]) -> None:
    missing = sorted(_RESULT_FIELDS - set(record))
    extra = sorted(set(record) - _RESULT_FIELDS)
    if missing or extra:
        raise ValueError(
            f"held-out result schema mismatch; missing={missing}, extra={extra}"
        )


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str, lower: int = 0) -> int:
    if type(value) is not int or value < lower:
        raise ValueError(f"{name} must be an integer >= {lower}")
    return value


def _finite_number(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite JSON number")
    return float(value)


def _hash(value: object, length: int, name: str) -> str:
    text = _nonempty_string(value, name)
    if len(text) != length or any(character not in _HEX40 for character in text):
        raise ValueError(f"{name} must be a {length}-character lowercase hex hash")
    return text


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a JSON boolean")
    return value


def _optional_finite_number(value: object, name: str) -> float | None:
    if value is None:
        return None
    return _finite_number(value, name)


def _exact_record_keys(
    record: Mapping[str, object],
    expected: set[str],
    name: str,
) -> None:
    missing = sorted(expected - set(record))
    extra = sorted(set(record) - expected)
    if missing or extra:
        raise ValueError(
            f"{name} schema mismatch; missing={missing}, extra={extra}"
        )


def make_heldout_fit_candidate(
    *,
    candidate_id: str,
    dataset: str,
    subject: str,
    fold_index: int,
    fit_model_order: int,
    fit_seed: int,
    training_ll_recomputed: float | None,
    fit_healthy: bool,
    stopping_reason: str,
    converged: bool,
    reached_iteration_cap: bool,
    occupancy_ok: bool,
    kish_effective_sample_size_ok: bool,
    fitted_state_sha256: str,
    output_sha256: str,
    result_path: str,
) -> dict[str, object]:
    """Create one strict, JSON-ready candidate for training-only selection.

    ``training_ll_recomputed`` is the objective evaluated from the returned
    fitted state on training samples. ``None`` represents an unavailable or
    non-finite objective in strict JSON. Health, stopping, occupancy, and Kish
    fields are audit annotations: they never replace training likelihood as
    the selection statistic.
    """
    record: dict[str, object] = {
        "schema_version": HELDOUT_FIT_CANDIDATE_SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "dataset": dataset,
        "subject": subject,
        "fold_index": fold_index,
        "fit_model_order": fit_model_order,
        "fit_seed": fit_seed,
        "training_ll_recomputed": training_ll_recomputed,
        "training_ll_unit": (
            "nats per retained component per training sample"
        ),
        "fit_healthy": fit_healthy,
        "stopping_reason": stopping_reason,
        "converged": converged,
        "reached_iteration_cap": reached_iteration_cap,
        "occupancy_ok": occupancy_ok,
        "kish_effective_sample_size_ok": kish_effective_sample_size_ok,
        "fitted_state_sha256": fitted_state_sha256,
        "output_sha256": output_sha256,
        "result_path": result_path,
    }
    validate_heldout_fit_candidate(record)
    return record


def validate_heldout_fit_candidate(record: Mapping[str, object]) -> None:
    """Validate one candidate without consulting held-out or nuisance data."""
    _exact_record_keys(record, _FIT_CANDIDATE_FIELDS, "fit candidate")
    if record["schema_version"] != HELDOUT_FIT_CANDIDATE_SCHEMA_VERSION:
        raise ValueError("unsupported held-out fit candidate schema_version")
    _nonempty_string(record["candidate_id"], "candidate_id")
    _nonempty_string(record["dataset"], "dataset")
    _nonempty_string(record["subject"], "subject")
    fold_index = _integer(record["fold_index"], "fold_index")
    if fold_index >= EXPECTED_N_SPLITS:
        raise ValueError("fold_index is outside the prespecified five folds")
    _integer(record["fit_model_order"], "fit_model_order", lower=1)
    _integer(record["fit_seed"], "fit_seed")
    training_ll = _optional_finite_number(
        record["training_ll_recomputed"],
        "training_ll_recomputed",
    )
    if (
        record["training_ll_unit"]
        != "nats per retained component per training sample"
    ):
        raise ValueError("training likelihood has the wrong unit")
    fit_healthy = _boolean(record["fit_healthy"], "fit_healthy")
    _nonempty_string(record["stopping_reason"], "stopping_reason")
    _boolean(record["converged"], "converged")
    _boolean(record["reached_iteration_cap"], "reached_iteration_cap")
    _boolean(record["occupancy_ok"], "occupancy_ok")
    _boolean(
        record["kish_effective_sample_size_ok"],
        "kish_effective_sample_size_ok",
    )
    if fit_healthy and training_ll is None:
        raise ValueError(
            "a healthy fit must have a finite recomputed training likelihood"
        )
    _hash(record["fitted_state_sha256"], 64, "fitted_state_sha256")
    _hash(record["output_sha256"], 64, "output_sha256")
    _nonempty_string(record["result_path"], "result_path")


@dataclass(frozen=True)
class HeldoutCandidateSelection:
    """Selected fit plus a complete, versioned candidate audit."""

    selected_candidate: Mapping[str, object]
    audit: Mapping[str, object]


def select_heldout_fit_candidate(
    candidates: Sequence[Mapping[str, object]],
    *,
    rule: HeldoutCandidateSelectionRule = (
        HeldoutCandidateSelectionRule.MAX_FINITE_TRAINING_LL
    ),
    expected_fit_seeds: Sequence[int] = EXPECTED_FIT_SEEDS,
) -> HeldoutCandidateSelection:
    """Select one of exactly three initialisations without held-out leakage.

    The only ranking inputs are the recomputed training likelihood and, for
    exact ties, the integer fit seed. Health and occupancy diagnostics remain
    visible in the audit so a downstream gate can stop on a selected warning
    without silently substituting a lower-likelihood initialisation.
    """
    try:
        selection_rule = HeldoutCandidateSelectionRule(rule)
    except (TypeError, ValueError) as error:
        raise ValueError("unsupported held-out candidate selection rule") from error
    if selection_rule is not HeldoutCandidateSelectionRule.MAX_FINITE_TRAINING_LL:
        raise ValueError("unsupported held-out candidate selection rule")

    seeds = tuple(expected_fit_seeds)
    if (
        len(seeds) != 3
        or len(set(seeds)) != 3
        or any(type(seed) is not int or seed < 0 for seed in seeds)
    ):
        raise ValueError("expected_fit_seeds must contain exactly three unique seeds")
    if len(candidates) != 3:
        raise ValueError("candidate selection requires exactly three records")

    normalized = [dict(candidate) for candidate in candidates]
    for candidate in normalized:
        validate_heldout_fit_candidate(candidate)
    identities = {
        (
            candidate["dataset"],
            candidate["subject"],
            candidate["fold_index"],
            candidate["fit_model_order"],
        )
        for candidate in normalized
    }
    if len(identities) != 1:
        raise ValueError(
            "candidates must describe one dataset/subject/fold/model order"
        )
    candidate_seeds = [int(candidate["fit_seed"]) for candidate in normalized]
    if len(set(candidate_seeds)) != 3 or set(candidate_seeds) != set(seeds):
        raise ValueError(
            "candidate fit seeds do not match the three expected initialisations"
        )
    candidate_ids = [str(candidate["candidate_id"]) for candidate in normalized]
    if len(set(candidate_ids)) != 3:
        raise ValueError("candidate_id values must be unique")

    finite = [
        candidate
        for candidate in normalized
        if candidate["training_ll_recomputed"] is not None
    ]
    if not finite:
        raise ValueError("no candidate has a finite recomputed training likelihood")
    ranked = sorted(
        finite,
        key=lambda candidate: (
            -float(candidate["training_ll_recomputed"]),
            int(candidate["fit_seed"]),
        ),
    )
    winner = ranked[0]
    rank_by_id = {
        str(candidate["candidate_id"]): rank
        for rank, candidate in enumerate(ranked, start=1)
    }
    selected_id = str(winner["candidate_id"])

    audit_candidates = []
    for candidate in sorted(normalized, key=lambda item: int(item["fit_seed"])):
        candidate_id = str(candidate["candidate_id"])
        finite_training_ll = candidate["training_ll_recomputed"] is not None
        audit_candidates.append(
            {
                "candidate_id": candidate_id,
                "fit_seed": int(candidate["fit_seed"]),
                "training_ll_recomputed": candidate["training_ll_recomputed"],
                "finite_training_ll": finite_training_ll,
                "selection_rank": rank_by_id.get(candidate_id),
                "selected": candidate_id == selected_id,
                "fit_healthy": candidate["fit_healthy"],
                "stopping_reason": candidate["stopping_reason"],
                "converged": candidate["converged"],
                "reached_iteration_cap": candidate["reached_iteration_cap"],
                "occupancy_ok": candidate["occupancy_ok"],
                "kish_effective_sample_size_ok": candidate[
                    "kish_effective_sample_size_ok"
                ],
                "fitted_state_sha256": candidate["fitted_state_sha256"],
                "output_sha256": candidate["output_sha256"],
                "result_path": candidate["result_path"],
            }
        )

    identity = next(iter(identities))
    selected_record = {
        "candidate_id": selected_id,
        "fit_seed": int(winner["fit_seed"]),
        "fitted_state_sha256": winner["fitted_state_sha256"],
        "output_sha256": winner["output_sha256"],
        "result_path": winner["result_path"],
    }
    audit: dict[str, object] = {
        "schema_version": HELDOUT_SELECTION_AUDIT_SCHEMA_VERSION,
        "selection_rule": selection_rule.value,
        "dataset": identity[0],
        "subject": identity[1],
        "fold_index": identity[2],
        "fit_model_order": identity[3],
        "expected_fit_seeds": list(seeds),
        "selected_candidate_id": selected_id,
        "selected_fit_seed": int(winner["fit_seed"]),
        "selected_record": selected_record,
        "candidates": audit_candidates,
    }
    validate_heldout_candidate_selection_audit(audit)
    return HeldoutCandidateSelection(
        selected_candidate=dict(winner),
        audit=audit,
    )


def validate_heldout_candidate_selection_audit(
    audit: Mapping[str, object],
) -> None:
    """Validate and independently replay a training-only selection audit."""
    _exact_record_keys(audit, _SELECTION_AUDIT_FIELDS, "selection audit")
    if audit["schema_version"] != HELDOUT_SELECTION_AUDIT_SCHEMA_VERSION:
        raise ValueError("unsupported held-out selection audit schema_version")
    try:
        rule = HeldoutCandidateSelectionRule(audit["selection_rule"])
    except (TypeError, ValueError) as error:
        raise ValueError("unsupported held-out candidate selection rule") from error
    if rule is not HeldoutCandidateSelectionRule.MAX_FINITE_TRAINING_LL:
        raise ValueError("unsupported held-out candidate selection rule")
    _nonempty_string(audit["dataset"], "dataset")
    _nonempty_string(audit["subject"], "subject")
    fold_index = _integer(audit["fold_index"], "fold_index")
    if fold_index >= EXPECTED_N_SPLITS:
        raise ValueError("fold_index is outside the prespecified five folds")
    _integer(audit["fit_model_order"], "fit_model_order", lower=1)

    expected_seeds = audit["expected_fit_seeds"]
    if (
        isinstance(expected_seeds, (str, bytes))
        or not isinstance(expected_seeds, Sequence)
        or len(expected_seeds) != 3
        or len(set(expected_seeds)) != 3
        or any(type(seed) is not int or seed < 0 for seed in expected_seeds)
    ):
        raise ValueError("expected_fit_seeds must contain exactly three unique seeds")

    rows = audit["candidates"]
    if (
        isinstance(rows, (str, bytes))
        or not isinstance(rows, Sequence)
        or len(rows) != 3
    ):
        raise ValueError("selection audit must contain exactly three candidates")
    selected_id = _nonempty_string(
        audit["selected_candidate_id"],
        "selected_candidate_id",
    )
    selected_seed = _integer(audit["selected_fit_seed"], "selected_fit_seed")
    seen_ids = set()
    seen_seeds = set()
    selected_rows = []
    finite_rows = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"candidates[{index}] must be a mapping")
        _exact_record_keys(
            row,
            _SELECTION_AUDIT_CANDIDATE_FIELDS,
            f"candidates[{index}]",
        )
        candidate_id = _nonempty_string(
            row["candidate_id"],
            f"candidates[{index}].candidate_id",
        )
        fit_seed = _integer(
            row["fit_seed"],
            f"candidates[{index}].fit_seed",
        )
        if candidate_id in seen_ids or fit_seed in seen_seeds:
            raise ValueError("selection audit candidate IDs and seeds must be unique")
        seen_ids.add(candidate_id)
        seen_seeds.add(fit_seed)
        training_ll = _optional_finite_number(
            row["training_ll_recomputed"],
            f"candidates[{index}].training_ll_recomputed",
        )
        finite_flag = _boolean(
            row["finite_training_ll"],
            f"candidates[{index}].finite_training_ll",
        )
        if finite_flag != (training_ll is not None):
            raise ValueError("finite_training_ll does not match the training objective")
        selection_rank = row["selection_rank"]
        if selection_rank is not None:
            _integer(
                selection_rank,
                f"candidates[{index}].selection_rank",
                lower=1,
            )
        selected = _boolean(
            row["selected"],
            f"candidates[{index}].selected",
        )
        _boolean(row["fit_healthy"], f"candidates[{index}].fit_healthy")
        _nonempty_string(
            row["stopping_reason"],
            f"candidates[{index}].stopping_reason",
        )
        for field in (
            "converged",
            "reached_iteration_cap",
            "occupancy_ok",
            "kish_effective_sample_size_ok",
        ):
            _boolean(row[field], f"candidates[{index}].{field}")
        _hash(
            row["fitted_state_sha256"],
            64,
            f"candidates[{index}].fitted_state_sha256",
        )
        _hash(
            row["output_sha256"],
            64,
            f"candidates[{index}].output_sha256",
        )
        _nonempty_string(
            row["result_path"],
            f"candidates[{index}].result_path",
        )
        if finite_flag:
            finite_rows.append(row)
        if selected:
            selected_rows.append(row)

    if seen_seeds != set(expected_seeds):
        raise ValueError("selection audit seeds do not match expected_fit_seeds")
    if len(selected_rows) != 1:
        raise ValueError("selection audit must mark exactly one candidate selected")
    if not finite_rows:
        raise ValueError("selection audit has no finite training objective")
    replayed = sorted(
        finite_rows,
        key=lambda row: (
            -float(row["training_ll_recomputed"]),
            int(row["fit_seed"]),
        ),
    )
    expected_ranks = {
        str(row["candidate_id"]): rank
        for rank, row in enumerate(replayed, start=1)
    }
    for row in rows:
        expected_rank = expected_ranks.get(str(row["candidate_id"]))
        if row["selection_rank"] != expected_rank:
            raise ValueError("selection_rank is inconsistent with the declared rule")
    winner = replayed[0]
    if (
        selected_rows[0]["candidate_id"] != winner["candidate_id"]
        or selected_id != winner["candidate_id"]
        or selected_seed != winner["fit_seed"]
    ):
        raise ValueError("selected candidate does not follow the declared rule")

    link = audit["selected_record"]
    if not isinstance(link, Mapping):
        raise ValueError("selected_record must be a mapping")
    _exact_record_keys(link, _SELECTED_RECORD_LINK_FIELDS, "selected_record")
    expected_link = {
        "candidate_id": winner["candidate_id"],
        "fit_seed": winner["fit_seed"],
        "fitted_state_sha256": winner["fitted_state_sha256"],
        "output_sha256": winner["output_sha256"],
        "result_path": winner["result_path"],
    }
    if dict(link) != expected_link:
        raise ValueError("selected_record does not link to the selected candidate")


def _decode_spans(value: object, name: str) -> Tuple[SampleSpan, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    spans = []
    for index, pair in enumerate(value):
        if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence):
            raise ValueError(f"{name}[{index}] must be [start, stop]")
        if len(pair) != 2:
            raise ValueError(f"{name}[{index}] must have two values")
        start = _integer(pair[0], f"{name}[{index}][0]")
        stop = _integer(pair[1], f"{name}[{index}][1]", lower=1)
        spans.append(SampleSpan(start, stop))
    spans_to_indices(spans)
    return tuple(spans)


def validate_heldout_fold_result(
    record: Mapping[str, object],
    *,
    provenance: Mapping[str, object] | None = None,
) -> None:
    """Validate result structure and optionally cross-check full provenance."""
    _exact_keys(record)
    if record["schema_version"] != HELDOUT_RESULT_SCHEMA_VERSION:
        raise ValueError("unsupported held-out result schema_version")
    if record["campaign"] != HELDOUT_CAMPAIGN:
        raise ValueError("held-out result has the wrong campaign")
    run_id = _nonempty_string(record["run_id"], "run_id")
    _nonempty_string(record["dataset"], "dataset")
    _nonempty_string(record["subject"], "subject")
    fold_index = _integer(record["fold_index"], "fold_index")
    if fold_index >= EXPECTED_N_SPLITS:
        raise ValueError("fold_index is outside the prespecified five folds")
    if record["n_splits"] != EXPECTED_N_SPLITS:
        raise ValueError("held-out evaluation requires five folds")
    if not math.isclose(
        _finite_number(record["guard_seconds"], "guard_seconds"),
        EXPECTED_GUARD_SECONDS,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("held-out evaluation requires five-second guards")
    _integer(record["fit_model_order"], "fit_model_order", lower=1)
    _integer(record["fit_seed"], "fit_seed")
    try:
        selection_rule = HeldoutCandidateSelectionRule(
            _nonempty_string(record["selection_rule"], "selection_rule")
        )
    except ValueError as error:
        raise ValueError(
            "held-out result has an unsupported selection_rule"
        ) from error
    if (
        selection_rule
        is not HeldoutCandidateSelectionRule.MAX_FINITE_TRAINING_LL
    ):
        raise ValueError("held-out result has an unsupported selection_rule")
    _integer(record["n_components"], "n_components", lower=1)
    sampling_rate_hz = _finite_number(
        record["sampling_rate_hz"],
        "sampling_rate_hz",
    )
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be finite and positive")
    n_recording_samples = _integer(
        record["n_recording_samples"],
        "n_recording_samples",
        lower=2,
    )
    guard_samples = _integer(record["guard_samples"], "guard_samples")
    test_start = _integer(record["test_start"], "test_start")
    test_stop = _integer(record["test_stop"], "test_stop", lower=1)
    if test_stop <= test_start:
        raise ValueError("test_stop must be greater than test_start")
    if test_stop > n_recording_samples:
        raise ValueError("test span extends beyond n_recording_samples")
    train_spans = _decode_spans(record["train_spans"], "train_spans")
    guard_spans = _decode_spans(record["guard_spans"], "guard_spans")
    n_train = _integer(record["n_train_samples"], "n_train_samples", lower=1)
    n_guard = _integer(record["n_guard_samples"], "n_guard_samples")
    n_test = _integer(record["n_test_samples"], "n_test_samples", lower=1)
    if sum(span.size for span in train_spans) != n_train:
        raise ValueError("n_train_samples does not match train_spans")
    if sum(span.size for span in guard_spans) != n_guard:
        raise ValueError("n_guard_samples does not match guard_spans")
    if test_stop - test_start != n_test:
        raise ValueError("n_test_samples does not match the test span")
    expected_guard_samples = int(
        round(EXPECTED_GUARD_SECONDS * sampling_rate_hz)
    )
    if guard_samples != expected_guard_samples:
        raise ValueError("guard_samples does not represent five seconds")
    excluded_start = max(0, test_start - guard_samples)
    excluded_stop = min(n_recording_samples, test_stop + guard_samples)
    expected_train = []
    if excluded_start > 0:
        expected_train.append(SampleSpan(0, excluded_start))
    if excluded_stop < n_recording_samples:
        expected_train.append(SampleSpan(excluded_stop, n_recording_samples))
    expected_guard = []
    if excluded_start < test_start:
        expected_guard.append(SampleSpan(excluded_start, test_start))
    if test_stop < excluded_stop:
        expected_guard.append(SampleSpan(test_stop, excluded_stop))
    if train_spans != tuple(expected_train):
        raise ValueError("train_spans are not the exact guarded complement")
    if guard_spans != tuple(expected_guard):
        raise ValueError("guard_spans do not match the five-second exclusion")
    for field in (
        "train_index_sha256",
        "test_index_sha256",
        "evaluation_sample_sha256",
        "preprocessing_state_sha256",
        "manifest_sha256",
        "manifest_row_sha256",
    ):
        _hash(record[field], 64, field)
    _hash(record["benchmark_git_sha"], 40, "benchmark_git_sha")
    _hash(record["package_git_sha"], 40, "package_git_sha")
    if record["transform_fit_scope"] != "training_samples_only":
        raise ValueError("transform_fit_scope must be training_samples_only")
    lpd_sum = _finite_number(record["lpd_test_sum"], "lpd_test_sum")
    lpd_mean = _finite_number(record["lpd_test_mean"], "lpd_test_mean")
    if not math.isclose(
        lpd_mean,
        lpd_sum / n_test,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("lpd_test_mean is inconsistent with sum and sample count")
    if record["lpd_unit"] != "nats per held-out sample":
        raise ValueError("held-out LPD has the wrong unit")
    result_path = _nonempty_string(record["result_path"], "result_path")
    manifest_row_index = _integer(
        record["manifest_row_index"], "manifest_row_index"
    )
    if record["provenance_schema_version"] != PROVENANCE_SCHEMA_VERSION:
        raise ValueError("result points to an unsupported provenance schema")
    if record["provenance_run_id"] != run_id:
        raise ValueError("provenance_run_id does not match run_id")

    if provenance is None:
        return
    validate_provenance(provenance)
    if provenance["campaign"] != HELDOUT_CAMPAIGN:
        raise ValueError("provenance campaign does not describe held-out real EEG")
    matches = {
        "run_id": (provenance["run_id"], run_id),
        "manifest_sha256": (
            provenance["manifest_sha256"],
            record["manifest_sha256"],
        ),
        "manifest_row_index": (
            provenance["manifest_row_index"],
            manifest_row_index,
        ),
        "manifest_row_sha256": (
            provenance["manifest_row_sha256"],
            record["manifest_row_sha256"],
        ),
        "benchmark_git_sha": (
            provenance["software"]["benchmark_git_sha"],
            record["benchmark_git_sha"],
        ),
        "package_git_sha": (
            provenance["software"]["package_git_sha"],
            record["package_git_sha"],
        ),
    }
    mismatches = {
        key: values for key, values in matches.items() if values[0] != values[1]
    }
    if mismatches:
        raise ValueError(f"result/provenance mismatch: {mismatches}")
    output_paths = {output["path"] for output in provenance["outputs"]}
    if result_path not in output_paths:
        raise ValueError("result_path is absent from provenance outputs")


def validate_common_evaluation_samples(
    records: Sequence[Mapping[str, object]],
    *,
    expected_model_orders: Sequence[int] = DEFAULT_MODEL_ORDERS,
) -> None:
    """Require one common five-fold evaluation set for every fitted order."""
    if not records:
        raise ValueError("at least one held-out record is required")
    orders = tuple(expected_model_orders)
    if not orders or len(set(orders)) != len(orders):
        raise ValueError("expected_model_orders must be non-empty and unique")
    for record in records:
        validate_heldout_fold_result(record)

    identity = {
        (
            record["dataset"],
            record["subject"],
            record["n_components"],
            record["sampling_rate_hz"],
            record["n_recording_samples"],
            record["benchmark_git_sha"],
            record["package_git_sha"],
        )
        for record in records
    }
    if len(identity) != 1:
        raise ValueError("records do not describe one subject and software state")

    by_key: dict[tuple[int, int], Mapping[str, object]] = {}
    for record in records:
        key = (int(record["fold_index"]), int(record["fit_model_order"]))
        if key in by_key:
            raise ValueError(f"duplicate held-out result for fold/model {key}")
        by_key[key] = record
    expected_keys = {
        (fold_index, model_order)
        for fold_index in range(EXPECTED_N_SPLITS)
        for model_order in orders
    }
    if set(by_key) != expected_keys:
        missing = sorted(expected_keys - set(by_key))
        extra = sorted(set(by_key) - expected_keys)
        raise ValueError(
            f"incomplete fold/model grid; missing={missing}, extra={extra}"
        )

    common_fields = (
        "test_start",
        "test_stop",
        "n_train_samples",
        "n_guard_samples",
        "n_test_samples",
        "guard_samples",
        "train_index_sha256",
        "test_index_sha256",
        "evaluation_sample_sha256",
        "preprocessing_state_sha256",
    )
    expected_test_start = 0
    for fold_index in range(EXPECTED_N_SPLITS):
        reference = by_key[(fold_index, orders[0])]
        if int(reference["test_start"]) != expected_test_start:
            raise ValueError("the five test folds are not one contiguous partition")
        expected_test_start = int(reference["test_stop"])
        for model_order in orders[1:]:
            candidate = by_key[(fold_index, model_order)]
            differing = [
                field
                for field in common_fields
                if candidate[field] != reference[field]
            ]
            if differing:
                raise ValueError(
                    f"fold {fold_index} uses different evaluation/preprocessing "
                    f"state for M={model_order}: {differing}"
                )
    if expected_test_start != int(records[0]["n_recording_samples"]):
        raise ValueError("the five test folds do not cover the full recording")


@dataclass(frozen=True)
class SubjectHeldoutSummary:
    """Fold-size-weighted predictive-density curve for one subject."""

    dataset: str
    subject: str
    model_orders: Tuple[int, ...]
    mean_lpd_nats_per_sample: Tuple[float, ...]
    delta_lpd_vs_m1: Tuple[float, ...]
    total_test_samples: int


def aggregate_subject_lpd(
    records: Sequence[Mapping[str, object]],
    *,
    expected_model_orders: Sequence[int] = DEFAULT_MODEL_ORDERS,
) -> SubjectHeldoutSummary:
    """Aggregate folds by sample count, with ``M=1`` as the subject baseline."""
    orders = tuple(expected_model_orders)
    if 1 not in orders:
        raise ValueError("expected_model_orders must include M=1")
    validate_common_evaluation_samples(
        records,
        expected_model_orders=orders,
    )
    first = records[0]
    means = []
    totals = []
    for model_order in orders:
        selected = [
            record
            for record in records
            if int(record["fit_model_order"]) == model_order
        ]
        total_samples = sum(int(record["n_test_samples"]) for record in selected)
        total_lpd = math.fsum(float(record["lpd_test_sum"]) for record in selected)
        means.append(total_lpd / total_samples)
        totals.append(total_samples)
    if len(set(totals)) != 1:
        raise ValueError("model orders do not have identical held-out sample counts")
    baseline = means[orders.index(1)]
    return SubjectHeldoutSummary(
        dataset=str(first["dataset"]),
        subject=str(first["subject"]),
        model_orders=orders,
        mean_lpd_nats_per_sample=tuple(means),
        delta_lpd_vs_m1=tuple(value - baseline for value in means),
        total_test_samples=totals[0],
    )
