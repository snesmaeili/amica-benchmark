from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


MODULE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "multimodel"
sys.path.insert(0, str(MODULE_DIR))

from run_harmonized_multimodel import (  # noqa: E402
    _sanitize_json,
    _sha256_array,
    _validate_manifest_identity,
    phase_surrogate,
    posterior_windows,
)


def test_common_phase_surrogate_preserves_cross_spectrum():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((4, 4096))
    y = phase_surrogate(x, seed=11)
    xf = np.fft.rfft(x, axis=1)
    yf = np.fft.rfft(y, axis=1)
    np.testing.assert_allclose(np.abs(yf), np.abs(xf), atol=1e-10, rtol=1e-10)
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            np.testing.assert_allclose(
                yf[i] * np.conj(yf[j]),
                xf[i] * np.conj(xf[j]),
                atol=1e-9,
                rtol=1e-10,
            )


def test_posterior_windows_sum_to_one_and_exclude_transition():
    gamma = np.vstack(
        [np.linspace(1.0, 0.0, 6000), np.linspace(0.0, 1.0, 6000)]
    )
    features, labels, starts = posterior_windows(
        gamma,
        10.0,
        window_sec=5.0,
        task_onset_sec=300.0,
        transition_buffer_sec=30.0,
    )
    assert features.shape == (120, 2)
    np.testing.assert_allclose(features.sum(axis=1), 1.0)
    assert np.all(labels[(starts >= 270.0) & (starts < 330.0)] == -1)
    assert np.all(labels[starts < 265.0] == 0)
    assert np.all(labels[starts >= 330.0] == 1)


def test_json_sanitizer_replaces_nonfinite_values_with_null():
    clean, fields = _sanitize_json(
        {"metric": np.asarray([1.0, np.nan]), "task_onset": float("inf")}
    )
    assert clean == {"metric": [1.0, None], "task_onset": None}
    assert fields == ["metric[1]", "task_onset"]


def test_array_checksum_is_shape_and_precision_specific():
    array = np.arange(6, dtype=np.float64).reshape(2, 3)
    assert _sha256_array(array, dtype="<f8") == _sha256_array(
        array.copy(), dtype="<f8"
    )
    assert _sha256_array(array, dtype="<f8") != _sha256_array(
        array, dtype="<f4"
    )


def test_manifest_identity_is_checked_before_fit(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "dataset,subject,num_models,fit_seed,surrogate,surrogate_seed\n"
        "ds004505,1,3,2,none,0\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        manifest_path=manifest,
        manifest_row_index=0,
        dataset="ds004505",
        subject=1,
        num_models=3,
        fit_seed=2,
        surrogate="none",
        surrogate_seed=0,
    )
    assert _validate_manifest_identity(args)["num_models"] == "3"
    args.fit_seed = 1
    with pytest.raises(RuntimeError, match="do not match manifest"):
        _validate_manifest_identity(args)
