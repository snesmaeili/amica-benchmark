from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "multimodel"
sys.path.insert(0, str(MODULE_DIR))

from audit_harmonized_pilot import audit_rows, read_results  # noqa: E402


def _manifest_row(fit_seed=0):
    return {
        "dataset": "ds004505",
        "subject": 1,
        "num_models": 2,
        "fit_seed": fit_seed,
        "surrogate": "none",
        "surrogate_seed": 0,
    }


def _result_row(fit_seed=0, *, recomputed=None, prior=(0.5, 0.5)):
    row = {
        **_manifest_row(fit_seed),
        "schema_version": 2,
        "n_components": 2,
        "n_samples": 500,
        "n_iter": 20,
        "max_iter": 20,
        "converged": False,
        "ll_final": -2.0,
        "ll_history_final": -2.0,
        "package_name": "amica",
        "package_commit": "7c223685",
        "package_git_dirty": False,
        "gm": list(prior),
        "cmir": {
            "soft_100bins": {
                "models": [
                    {"effective_n": 300.0, "posterior_mass": 300.0},
                    {"effective_n": 200.0, "posterior_mass": 200.0},
                ]
            }
        },
        "_json_path": "fit.json",
        "_npz_path": "fit.npz",
    }
    if recomputed is not None:
        row["ll_final_recomputed"] = recomputed
        row["ll_final"] = recomputed
    return row


def test_audit_blocks_legacy_archive_without_recomputed_final_ll(monkeypatch):
    monkeypatch.setattr(
        "audit_harmonized_pilot._npz_diagnostics",
        lambda path: {
            "npz_present": True,
            "ll_history_length": 20,
            "ll_nonfinite_count": 0,
            "ll_history_last": -2.0,
        },
    )
    manifest = [_manifest_row(seed) for seed in (0, 1, 2)]
    results = [_result_row(seed) for seed in (0, 1, 2)]
    _, summary = audit_rows(manifest, results)
    assert summary["gate_status"] == "blocked_final_objective_verification"
    assert summary["rows_without_final_recomputed_likelihood"] == 3


def test_audit_passes_complete_verified_three_seed_group(monkeypatch):
    monkeypatch.setattr(
        "audit_harmonized_pilot._npz_diagnostics",
        lambda path: {
            "npz_present": True,
            "ll_history_length": 20,
            "ll_nonfinite_count": 0,
            "ll_history_last": -2.0,
        },
    )
    manifest = [_manifest_row(seed) for seed in (0, 1, 2)]
    results = [_result_row(seed, recomputed=-1.99) for seed in (0, 1, 2)]
    _, summary = audit_rows(manifest, results)
    assert summary["gate_status"] == "pass"
    assert summary["complete_rows"] == 3
    assert summary["incomplete_seed_groups"] == 0


def test_audit_compares_npz_history_with_reported_history_endpoint(monkeypatch):
    monkeypatch.setattr(
        "audit_harmonized_pilot._npz_diagnostics",
        lambda path: {
            "npz_present": True,
            "ll_history_length": 20,
            "ll_nonfinite_count": 0,
            "ll_history_last": -2.0,
        },
    )
    manifest = [_manifest_row(seed) for seed in (0, 1, 2)]
    results = [_result_row(seed, recomputed=-1.99) for seed in (0, 1, 2)]
    audit, summary = audit_rows(manifest, results)
    assert summary["gate_status"] == "pass"
    assert all(row["ll_final"] == -1.99 for row in audit)
    assert all(row["ll_history_final_reported"] == -2.0 for row in audit)


def test_audit_rejects_inconsistent_returned_state_objective_alias(monkeypatch):
    monkeypatch.setattr(
        "audit_harmonized_pilot._npz_diagnostics",
        lambda path: {
            "npz_present": True,
            "ll_history_length": 20,
            "ll_nonfinite_count": 0,
            "ll_history_last": -2.0,
        },
    )
    manifest = [_manifest_row(seed) for seed in (0, 1, 2)]
    results = [_result_row(seed, recomputed=-1.99) for seed in (0, 1, 2)]
    results[0]["ll_final"] = -1.98
    _, summary = audit_rows(manifest, results)
    assert summary["gate_status"] == "fail"
    assert summary["rows_with_final_objective_alias_mismatch"] == 1


def test_audit_reports_low_occupancy_and_missing_rows(monkeypatch):
    monkeypatch.setattr(
        "audit_harmonized_pilot._npz_diagnostics",
        lambda path: {
            "npz_present": True,
            "ll_history_length": 20,
            "ll_nonfinite_count": 0,
            "ll_history_last": -2.0,
        },
    )
    manifest = [_manifest_row(seed) for seed in (0, 1, 2)]
    results = [_result_row(0, recomputed=-1.99, prior=(0.99, 0.01))]
    audit, summary = audit_rows(manifest, results)
    assert summary["gate_status"] == "fail"
    assert summary["missing_rows"] == 2
    assert summary["fits_with_prior_below_0_02"] == 1
    assert sum(row["coverage_status"] == "missing" for row in audit) == 2


def test_read_results_attaches_archive_paths(tmp_path):
    json_path = tmp_path / "mmc_test.json"
    npz_path = json_path.with_suffix(".npz")
    json_path.write_text(json.dumps(_result_row()), encoding="utf-8")
    np.savez_compressed(npz_path, ll_history=np.asarray([-3.0, -2.0]))
    rows = read_results(tmp_path)
    assert rows[0]["_json_path"] == str(json_path)
    assert rows[0]["_npz_path"] == str(npz_path)


def test_audit_rejects_nonfinite_cmir_and_nonstandard_json(monkeypatch):
    monkeypatch.setattr(
        "audit_harmonized_pilot._npz_diagnostics",
        lambda path: {
            "npz_present": True,
            "ll_history_length": 20,
            "ll_nonfinite_count": 0,
            "ll_history_last": -2.0,
        },
    )
    manifest = [_manifest_row(seed) for seed in (0, 1, 2)]
    results = [_result_row(seed, recomputed=-1.99) for seed in (0, 1, 2)]
    results[0]["cmir"]["soft_100bins"]["models"][0]["effective_n"] = float("nan")
    results[1]["_nonstandard_json_constants"] = ["NaN"]
    _, summary = audit_rows(manifest, results)
    assert summary["gate_status"] == "fail"
    assert summary["rows_with_nonfinite_cmir"] == 1
    assert summary["rows_with_nonstandard_json_constants"] == 1
