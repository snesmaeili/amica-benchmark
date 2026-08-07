from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "multimodel"
sys.path.insert(0, str(MODULE_DIR))

from aggregate_harmonized_multimodel import (  # noqa: E402
    _blocked_splits,
    _require_passing_audit,
    _score_decoder,
    select_best_seed,
)


def test_select_best_seed_is_scoped_to_surrogate_realisation():
    base = {
        "dataset": "ds004505",
        "subject": 1,
        "num_models": 3,
        "surrogate": "phase",
        "surrogate_seed": 12,
        "schema_version": 2,
        "package_name": "amica",
        "package_commit": "abc123",
        "package_git_dirty": False,
        "runner_sha256": "runner",
        "manifest_sha256": "manifest",
        "gm": [0.6, 0.3, 0.1],
        "cmir_finite": True,
    }
    runs = [
        {**base, "fit_seed": 0, "ll_final_recomputed": -3.0},
        {**base, "fit_seed": 1, "ll_final_recomputed": -2.0},
        {**base, "fit_seed": 2, "ll_final_recomputed": -4.0},
    ]
    selected, audit = select_best_seed(
        runs, expected_package_commit="abc123", return_audit=True
    )
    assert len(selected) == 1
    assert selected[0]["fit_seed"] == 1
    assert selected[0]["n_fit_seeds_available"] == 3
    assert selected[0]["ll_final"] == -2.0
    assert sum(row["selected"] for row in audit) == 1


def test_select_best_seed_rejects_stale_or_unprovenanced_objective():
    row = {
        "dataset": "ds004505",
        "subject": 1,
        "num_models": 3,
        "surrogate": "none",
        "surrogate_seed": 0,
        "fit_seed": 0,
        "ll_final": -2.0,
    }
    with np.testing.assert_raises_regex(RuntimeError, "provenance/final-objective"):
        select_best_seed([row])


def test_low_occupancy_is_flagged_but_does_not_change_seed_selection():
    base = {
        "dataset": "ds004505",
        "subject": 1,
        "num_models": 2,
        "surrogate": "none",
        "surrogate_seed": 0,
        "schema_version": 2,
        "package_name": "amica",
        "package_commit": "abc123",
        "package_git_dirty": False,
        "runner_sha256": "runner",
        "manifest_sha256": "manifest",
        "cmir_finite": True,
    }
    runs = [
        {
            **base,
            "fit_seed": 0,
            "ll_final_recomputed": -2.0,
            "gm": [0.99, 0.01],
        },
        {
            **base,
            "fit_seed": 1,
            "ll_final_recomputed": -3.0,
            "gm": [0.5, 0.5],
        },
    ]
    selected = select_best_seed(runs)
    assert selected[0]["fit_seed"] == 0
    assert selected[0]["prior_below_threshold"]


def test_select_best_seed_enforces_expected_archive_hashes():
    row = {
        "dataset": "ds004505",
        "subject": 1,
        "num_models": 2,
        "surrogate": "none",
        "surrogate_seed": 0,
        "fit_seed": 0,
        "schema_version": 2,
        "package_name": "amica",
        "package_commit": "abc123",
        "package_git_dirty": False,
        "runner_sha256": "runner-a",
        "manifest_sha256": "manifest-a",
        "ll_final_recomputed": -2.0,
        "gm": [0.5, 0.5],
    }
    with np.testing.assert_raises_regex(RuntimeError, "manifest_sha256_mismatch"):
        select_best_seed(
            [row],
            expected_manifest_sha256="manifest-b",
            expected_runner_sha256="runner-a",
        )


def test_aggregation_requires_passing_pinned_audit(tmp_path):
    path = tmp_path / "pilot_gate_summary.json"
    summary = {
        "gate_status": "pass",
        "expected_package_name": "amica",
        "expected_package_commit": "abc123",
        "expected_manifest_sha256": "manifest",
    }
    path.write_text(json.dumps(summary), encoding="utf-8")
    assert (
        _require_passing_audit(path, expected_package_commit="abc123")
        == summary
    )

    summary["gate_status"] = "review_low_occupancy"
    path.write_text(json.dumps(summary), encoding="utf-8")
    with np.testing.assert_raises_regex(RuntimeError, "requires a passing"):
        _require_passing_audit(path, expected_package_commit="abc123")


def test_phase_stratified_blocked_decoder_uses_both_classes():
    labels = np.r_[np.zeros(50, dtype=int), np.ones(50, dtype=int)]
    features = np.column_stack([labels, 1 - labels]).astype(float)
    splits = _blocked_splits(labels, n_splits=5, buffer_windows=1)
    assert len(splits) == 5
    for train, test in splits:
        assert set(labels[train]) == {0, 1}
        assert set(labels[test]) == {0, 1}
        assert not set(train).intersection(test)
    assert _score_decoder(features, labels, splits) > 0.99
