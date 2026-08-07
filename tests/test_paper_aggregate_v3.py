from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_aggregate():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "paper" / "aggregate.py"
    spec = importlib.util.spec_from_file_location("paper_aggregate", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_per_subject_results_skips_underscore_metadata(tmp_path):
    aggregate = load_aggregate()
    write_json(
        tmp_path / "benchmark_sub-01_hp1.0hz.json",
        {
            "_schema_version": "3.0",
            "_data": {"subject": "sub-01", "n_samples": 1000},
            "_iclean": {"mode": "none"},
            "amica": {
                "backend": "jax",
                "device": "gpu",
                "time": 12.0,
                "n_iter": 2,
                "n_components": 2,
                "converged": False,
                "iclabel": {"brain": 1, "brain_50pct": 1},
                "kurtosis": {"brain_like_kurtosis": 1, "kurtosis_mean": 2.0},
                "psd_alpha": {"alpha_peaked_ics": 1},
                "mir": {"mir": -3.0},
                "reconstruction_error": 1e-12,
            },
        },
    )

    df = aggregate.load_per_subject_results(tmp_path)

    assert df["method"].tolist() == ["amica"]
    row = df.iloc[0]
    assert row["backend"] == "jax"
    assert row["device"] == "gpu"
    assert bool(row["converged"]) is False
    assert row["time"] == 12.0
    assert row["iclabel_brain"] == 1


def test_load_single_method_results_accepts_backend_suffix(tmp_path):
    aggregate = load_aggregate()
    write_json(
        tmp_path / "benchmark_sub-02_hp1.0hz_jax_cpu.json",
        {
            "_schema_version": "3.0",
            "_data": {"subject": "sub-02"},
            "amica": {
                "backend": "jax",
                "device": "cpu",
                "time": 45.0,
                "n_iter": 10,
                "n_components": 3,
            },
        },
    )

    df = aggregate.load_single_method_results(tmp_path)

    assert df[["subject", "hp", "method", "backend", "device", "time"]].to_dict("records") == [
        {
            "subject": "sub-02",
            "hp": 1.0,
            "method": "amica",
            "backend": "jax",
            "device": "cpu",
            "time": 45.0,
        }
    ]
