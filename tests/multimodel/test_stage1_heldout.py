from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "multimodel"
sys.path.insert(0, str(MODULE_DIR))

from stage1.folds import make_guarded_blocked_folds  # noqa: E402
from stage1.heldout import (  # noqa: E402
    HeldoutCandidateSelectionRule,
    PredictiveDensityScore,
    aggregate_subject_lpd,
    make_heldout_fit_candidate,
    make_heldout_fold_result,
    prepare_heldout_fold,
    score_log_joint,
    select_heldout_fit_candidate,
    validate_common_evaluation_samples,
    validate_heldout_candidate_selection_audit,
    validate_heldout_fit_candidate,
    validate_heldout_fold_result,
)
from stage1.provenance import PROVENANCE_SCHEMA_VERSION  # noqa: E402


BENCHMARK_SHA = "c" * 40
PACKAGE_SHA = "d" * 40
MANIFEST_SHA = "a" * 64
ROW_SHA = "b" * 64


def _prepared_folds(n_samples=1_003, sampling_rate_hz=10.0):
    rng = np.random.default_rng(20260726)
    x = rng.standard_normal((6, n_samples))
    folds = make_guarded_blocked_folds(
        n_samples=n_samples,
        sampling_rate_hz=sampling_rate_hz,
        n_splits=5,
        guard_seconds=5.0,
    )
    return x, tuple(
        prepare_heldout_fold(
            x,
            fold,
            n_components=4,
            sampling_rate_hz=sampling_rate_hz,
        )
        for fold in folds
    )


def _result(prepared, model_order, mean_lpd, row_index):
    n_test = prepared.test_indices.size
    score = PredictiveDensityScore(
        total_nats=float(mean_lpd * n_test),
        mean_nats_per_sample=float(mean_lpd),
        n_samples=int(n_test),
    )
    run_id = f"ds-test-sub-01-fold{prepared.fold.fold_index}-M{model_order}"
    return make_heldout_fold_result(
        prepared,
        score,
        run_id=run_id,
        dataset="ds-test",
        subject="sub-01",
        fit_model_order=model_order,
        fit_seed=17,
        selection_rule=HeldoutCandidateSelectionRule.MAX_FINITE_TRAINING_LL,
        result_path=f"results/{run_id}.json",
        manifest_sha256=MANIFEST_SHA,
        manifest_row_index=row_index,
        manifest_row_sha256=ROW_SHA,
        benchmark_git_sha=BENCHMARK_SHA,
        package_git_sha=PACKAGE_SHA,
    )


def _fit_candidate(
    seed,
    training_ll,
    *,
    fit_healthy=True,
    occupancy_ok=True,
    kish_ok=True,
):
    hash_char = "abcdef"[seed % 6]
    output_hash_char = "fedcba"[seed % 6]
    return make_heldout_fit_candidate(
        candidate_id=f"ds-test-sub-01-fold2-M3-seed{seed}",
        dataset="ds-test",
        subject="sub-01",
        fold_index=2,
        fit_model_order=3,
        fit_seed=seed,
        training_ll_recomputed=training_ll,
        fit_healthy=fit_healthy,
        stopping_reason="max_iter",
        converged=False,
        reached_iteration_cap=True,
        occupancy_ok=occupancy_ok,
        kish_effective_sample_size_ok=kish_ok,
        fitted_state_sha256=hash_char * 64,
        output_sha256=output_hash_char * 64,
        result_path=f"results/candidate-seed{seed}.json",
    )


def _provenance(record):
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "run_id": record["run_id"],
        "campaign": "real_eeg_heldout",
        "condition_id": (
            f"{record['dataset']}-{record['subject']}-"
            f"fold{record['fold_index']}-M{record['fit_model_order']}"
        ),
        "manifest_path": "manifests/real_eeg_heldout.csv",
        "manifest_sha256": record["manifest_sha256"],
        "manifest_row_index": record["manifest_row_index"],
        "manifest_row_sha256": record["manifest_row_sha256"],
        "command": ["python", "run_heldout.py", "--row", "0"],
        "started_at_utc": "2026-07-26T12:00:00+00:00",
        "completed_at_utc": "2026-07-26T12:05:00+00:00",
        "execution": {
            "mode": "local",
            "hostname": "test-host",
            "slurm_job_id": None,
            "slurm_array_task_id": None,
            "account": None,
            "partition": None,
            "slurm_cpus_per_task": None,
            "slurm_memory_bytes": None,
            "slurm_time_limit": None,
            "slurm_gpu_request": None,
            "cuda_visible_devices": None,
        },
        "software": {
            "benchmark_git_sha": record["benchmark_git_sha"],
            "package_git_sha": record["package_git_sha"],
            "python": "3.11.9",
            "numpy": "2.0.1",
            "scipy": "1.14.0",
            "amica": "0.0.1",
            "jax": None,
            "jaxlib": None,
            "mne": "1.10.0",
            "scikit_learn": "1.7.0",
            "blas": "OpenBLAS 0.3.27",
            "cuda": None,
            "driver": None,
            "os": "Linux 6.8",
            "benchmark_worktree_clean": True,
            "package_worktree_clean": True,
        },
        "hardware": {
            "cpu_model": "Test CPU",
            "physical_cpus": 4,
            "logical_cpus": 8,
            "memory_bytes": 16 * 1024**3,
            "gpu_model": None,
            "gpu_uuid": None,
        },
        "runtime": {
            "backend": "numpy-cpu",
            "precision": "float64",
            "omp_num_threads": 1,
            "mkl_num_threads": 1,
            "openblas_num_threads": 1,
            "jax_enable_x64": None,
            "jax_platform": None,
            "jax_default_matmul_precision": None,
            "xla_flags": None,
            "xla_preallocate": None,
            "xla_memory_fraction": None,
        },
        "inputs": [
            {
                "path": "manifests/real_eeg_heldout.csv",
                "sha256": record["manifest_sha256"],
                "bytes": 5678,
            }
        ],
        "outputs": [
            {
                "path": record["result_path"],
                "sha256": "e" * 64,
                "bytes": 1234,
            }
        ],
    }


def test_preprocessing_is_fitted_only_on_guarded_training_samples():
    x, prepared_folds = _prepared_folds()
    prepared = prepared_folds[2]
    np.testing.assert_allclose(
        np.mean(prepared.train_data, axis=1),
        0.0,
        atol=2e-14,
    )
    np.testing.assert_allclose(
        np.mean(prepared.train_data**2, axis=1),
        1.0,
        atol=2e-14,
    )

    changed = x.copy()
    changed[:, prepared.test_indices] += 1e6
    for span in prepared.fold.guard:
        changed[:, span.start : span.stop] -= 1e6
    refit = prepare_heldout_fold(
        changed,
        prepared.fold,
        n_components=4,
        sampling_rate_hz=10.0,
    )
    assert refit.transform.state_sha256 == prepared.transform.state_sha256
    np.testing.assert_array_equal(
        refit.transform.channel_mean,
        prepared.transform.channel_mean,
    )
    np.testing.assert_array_equal(
        refit.transform.components,
        prepared.transform.components,
    )
    assert refit.evaluation_sample_sha256 != prepared.evaluation_sample_sha256

    with pytest.raises(ValueError, match="five-second guard"):
        prepare_heldout_fold(
            x,
            prepared.fold,
            n_components=4,
            sampling_rate_hz=20.0,
        )


def test_log_joint_scoring_uses_stable_mixture_logsumexp():
    log_joint = np.log(
        np.array(
            [
                [0.20, 0.10, 0.05],
                [0.30, 0.40, 0.15],
            ]
        )
    )
    expected = np.log(np.array([0.50, 0.50, 0.20]))
    score = score_log_joint(log_joint)
    assert score.n_samples == 3
    assert score.total_nats == pytest.approx(expected.sum())
    assert score.mean_nats_per_sample == pytest.approx(expected.mean())

    invalid = log_joint.copy()
    invalid[:, 1] = -np.inf
    with pytest.raises(ValueError, match="zero density"):
        score_log_joint(invalid)


def test_all_model_orders_must_use_identical_evaluation_samples_and_state():
    _, prepared_folds = _prepared_folds()
    records = []
    for fold_index, prepared in enumerate(prepared_folds):
        for model_order in (1, 2, 3):
            records.append(
                _result(
                    prepared,
                    model_order,
                    mean_lpd=-10.0 + model_order / 10.0,
                    row_index=fold_index * 3 + model_order - 1,
                )
            )
    validate_common_evaluation_samples(
        records,
        expected_model_orders=(1, 2, 3),
    )

    mismatched = copy.deepcopy(records)
    mismatched[4]["evaluation_sample_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="different evaluation/preprocessing"):
        validate_common_evaluation_samples(
            mismatched,
            expected_model_orders=(1, 2, 3),
        )

    missing = records[:-1]
    with pytest.raises(ValueError, match="incomplete fold/model grid"):
        validate_common_evaluation_samples(
            missing,
            expected_model_orders=(1, 2, 3),
        )


def test_subject_lpd_is_weighted_by_fold_sample_count():
    _, prepared_folds = _prepared_folds(n_samples=1_007)
    records = []
    fold_means = (-5.0, -4.0, -3.0, -2.0, -1.0)
    for fold_index, (prepared, baseline) in enumerate(
        zip(prepared_folds, fold_means)
    ):
        records.append(_result(prepared, 1, baseline, fold_index * 2))
        records.append(_result(prepared, 2, baseline + 0.25, fold_index * 2 + 1))

    summary = aggregate_subject_lpd(
        records,
        expected_model_orders=(1, 2),
    )
    sizes = np.array(
        [prepared.test_indices.size for prepared in prepared_folds],
        dtype=float,
    )
    expected_m1 = np.average(np.asarray(fold_means), weights=sizes)
    assert summary.mean_lpd_nats_per_sample[0] == pytest.approx(expected_m1)
    assert summary.delta_lpd_vs_m1 == pytest.approx((0.0, 0.25))
    assert summary.total_test_samples == 1_007


def test_result_schema_cross_checks_full_provenance():
    _, prepared_folds = _prepared_folds()
    record = _result(prepared_folds[0], 1, -3.0, 0)
    validate_heldout_fold_result(record, provenance=_provenance(record))

    wrong_commit = _provenance(record)
    wrong_commit["software"]["package_git_sha"] = "f" * 40
    with pytest.raises(ValueError, match="result/provenance mismatch"):
        validate_heldout_fold_result(record, provenance=wrong_commit)

    missing_output = _provenance(record)
    missing_output["outputs"][0]["path"] = "results/another.json"
    with pytest.raises(ValueError, match="absent from provenance outputs"):
        validate_heldout_fold_result(record, provenance=missing_output)

    extra_field = dict(record, unexpected=True)
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_heldout_fold_result(extra_field)

    coerced_number = dict(record, lpd_test_mean=str(record["lpd_test_mean"]))
    with pytest.raises(ValueError, match="finite JSON number"):
        validate_heldout_fold_result(coerced_number)

    free_text_rule = dict(record, selection_rule="best looking fit")
    with pytest.raises(ValueError, match="unsupported selection_rule"):
        validate_heldout_fold_result(free_text_rule)


def test_candidate_selection_uses_only_recomputed_training_likelihood():
    candidates = [
        _fit_candidate(0, -10.0),
        _fit_candidate(
            1,
            -8.0,
            fit_healthy=False,
            occupancy_ok=False,
            kish_ok=False,
        ),
        _fit_candidate(2, -9.0),
    ]
    selection = select_heldout_fit_candidate(
        candidates,
        rule=HeldoutCandidateSelectionRule.MAX_FINITE_TRAINING_LL,
    )

    assert selection.selected_candidate["fit_seed"] == 1
    assert selection.audit["selection_rule"] == (
        HeldoutCandidateSelectionRule.MAX_FINITE_TRAINING_LL.value
    )
    assert selection.audit["selected_fit_seed"] == 1
    assert selection.audit["expected_fit_seeds"] == [0, 1, 2]
    assert len(selection.audit["candidates"]) == 3
    selected_row = next(
        row for row in selection.audit["candidates"] if row["selected"]
    )
    assert selected_row["fit_healthy"] is False
    assert selected_row["occupancy_ok"] is False
    assert selected_row["kish_effective_sample_size_ok"] is False
    assert selection.audit["selected_record"] == {
        "candidate_id": selection.selected_candidate["candidate_id"],
        "fit_seed": 1,
        "fitted_state_sha256": selection.selected_candidate[
            "fitted_state_sha256"
        ],
        "output_sha256": selection.selected_candidate["output_sha256"],
        "result_path": selection.selected_candidate["result_path"],
    }
    validate_heldout_candidate_selection_audit(selection.audit)


def test_candidate_selection_tie_breaks_by_smallest_seed():
    selection = select_heldout_fit_candidate(
        [
            _fit_candidate(2, -7.5),
            _fit_candidate(0, -7.5),
            _fit_candidate(1, -8.0),
        ]
    )
    assert selection.selected_candidate["fit_seed"] == 0
    assert [
        row["fit_seed"]
        for row in selection.audit["candidates"]
        if row["selection_rank"] in (1, 2)
    ] == [0, 2]


def test_candidate_selection_requires_exactly_three_declared_seeds():
    with pytest.raises(ValueError, match="exactly three records"):
        select_heldout_fit_candidate(
            [_fit_candidate(0, -3.0), _fit_candidate(1, -2.0)]
        )

    with pytest.raises(ValueError, match="three expected initialisations"):
        select_heldout_fit_candidate(
            [
                _fit_candidate(0, -3.0),
                _fit_candidate(1, -2.0),
                _fit_candidate(3, -1.0),
            ]
        )

    mismatched = _fit_candidate(2, -1.0)
    mismatched["subject"] = "sub-02"
    with pytest.raises(ValueError, match="one dataset/subject/fold/model order"):
        select_heldout_fit_candidate(
            [_fit_candidate(0, -3.0), _fit_candidate(1, -2.0), mismatched]
        )


def test_nonfinite_candidate_is_excluded_and_all_nonfinite_is_rejected():
    selection = select_heldout_fit_candidate(
        [
            _fit_candidate(0, -3.0),
            _fit_candidate(1, None, fit_healthy=False),
            _fit_candidate(2, -2.0),
        ]
    )
    assert selection.selected_candidate["fit_seed"] == 2
    nonfinite_row = next(
        row for row in selection.audit["candidates"] if row["fit_seed"] == 1
    )
    assert nonfinite_row["finite_training_ll"] is False
    assert nonfinite_row["selection_rank"] is None

    with pytest.raises(ValueError, match="no candidate has a finite"):
        select_heldout_fit_candidate(
            [
                _fit_candidate(0, None, fit_healthy=False),
                _fit_candidate(1, None, fit_healthy=False),
                _fit_candidate(2, None, fit_healthy=False),
            ]
        )


def test_selection_schema_rejects_heldout_or_nuisance_inputs_and_tampering():
    candidate = _fit_candidate(0, -3.0)
    candidate["lpd_test_mean"] = -1.0
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_heldout_fit_candidate(candidate)

    candidate = _fit_candidate(0, -3.0)
    candidate["nuisance_score"] = 0.99
    with pytest.raises(ValueError, match="schema mismatch"):
        validate_heldout_fit_candidate(candidate)

    selection = select_heldout_fit_candidate(
        [
            _fit_candidate(0, -3.0),
            _fit_candidate(1, -2.0),
            _fit_candidate(2, -1.0),
        ]
    )
    tampered = copy.deepcopy(selection.audit)
    tampered["selected_record"]["output_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not link"):
        validate_heldout_candidate_selection_audit(tampered)

    tampered = copy.deepcopy(selection.audit)
    tampered["candidates"][0]["selection_rank"] = 1
    with pytest.raises(ValueError, match="selection_rank is inconsistent"):
        validate_heldout_candidate_selection_audit(tampered)


def test_fold_result_accepts_the_versioned_selection_rule_enum():
    _, prepared_folds = _prepared_folds()
    prepared = prepared_folds[0]
    n_test = prepared.test_indices.size
    record = make_heldout_fold_result(
        prepared,
        PredictiveDensityScore(
            total_nats=-2.0 * n_test,
            mean_nats_per_sample=-2.0,
            n_samples=n_test,
        ),
        run_id="ds-test-sub-01-fold0-M3",
        dataset="ds-test",
        subject="sub-01",
        fit_model_order=3,
        fit_seed=2,
        selection_rule=HeldoutCandidateSelectionRule.MAX_FINITE_TRAINING_LL,
        result_path="results/ds-test-sub-01-fold0-M3.json",
        manifest_sha256=MANIFEST_SHA,
        manifest_row_index=0,
        manifest_row_sha256=ROW_SHA,
        benchmark_git_sha=BENCHMARK_SHA,
        package_git_sha=PACKAGE_SHA,
    )
    assert record["selection_rule"] == (
        HeldoutCandidateSelectionRule.MAX_FINITE_TRAINING_LL.value
    )
