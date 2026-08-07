#!/usr/bin/env python
"""Extract compact empirical-density data for Figure 1.

This is a read-only analysis of the archived ds004505 sub-01 fit.  It loads the
open BIDS recording, applies the benchmark preprocessing, projects it through
the archived ICA object, and stores only binned activation densities and the
corresponding fitted AMICA mixture densities.  No model is fitted here.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import mne
import numpy as np
import pandas as pd
from scipy.special import gammaln


from _paths import DATA_ROOT as WORKSPACE

HERE = Path(__file__).resolve().parent
CAPSULE = WORKSPACE / "figdata/synth/amica-capsule"
RESULT_DIR = WORKSPACE / "ablation_results/m1_reject0"
JSON_PATH = RESULT_DIR / "benchmark_sub-01_hp1.0hz_jax_gpu.json"
ICA_PATH = RESULT_DIR / "benchmark_sub-01_hp1.0hz_jax_gpu_ica.fif"
RAW_PATH = (
    WORKSPACE
    / "datasets/ds004505/raw_bids/sub-01/eeg"
    / "sub-01_task-TableTennis_eeg.fdt"
)
CSV_PATH = HERE / "fig1_empirical_densities.csv"
AUDIT_PATH = HERE / "fig1_empirical_densities_audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mixture_pdf(
    values: np.ndarray,
    alpha: np.ndarray,
    mu: np.ndarray,
    beta: np.ndarray,
    rho: np.ndarray,
) -> np.ndarray:
    output = np.zeros_like(values, dtype=float)
    for k in range(len(alpha)):
        scaled = beta[k] * (values - mu[k])
        log_density = (
            np.log(beta[k])
            - np.maximum(np.abs(scaled), 1e-300) ** rho[k]
            - gammaln(1.0 + 1.0 / rho[k])
            - np.log(2.0)
        )
        output += alpha[k] * np.exp(log_density)
    return output


def main() -> None:
    for path in (JSON_PATH, ICA_PATH, RAW_PATH):
        if not path.exists():
            raise FileNotFoundError(path)

    sys.path.insert(0, str(CAPSULE))
    os.environ["BIDS_ROOT_DS4505"] = str(
        WORKSPACE / "datasets/ds004505/raw_bids"
    )
    from amica_python.benchmark.runner import load_data, preprocess

    mne.set_log_level("ERROR")
    raw = load_data("ds004505", 1)
    preprocess(raw)
    ica = mne.preprocessing.read_ica(ICA_PATH)
    if abs(raw.info["sfreq"] - ica.info["sfreq"]) > 1e-6:
        raw.resample(ica.info["sfreq"])
    sources = ica.get_sources(raw).get_data()

    with JSON_PATH.open(encoding="utf-8") as handle:
        record = json.load(handle)
    params = record["amica"]["pdf_params"]
    alpha = np.asarray(params["alpha"], dtype=float)
    mu = np.asarray(params["mu"], dtype=float)
    beta = np.asarray(params["beta"], dtype=float)
    rho = np.asarray(params["rho"], dtype=float)
    mean_rho = (alpha * rho).sum(axis=0) / alpha.sum(axis=0)

    targets = [1.04, 1.42, 1.79]
    components = [int(np.abs(mean_rho - target).argmin()) for target in targets]
    rows: list[dict] = []
    for component in components:
        activation = sources[component]
        scale = float(np.std(activation))
        lo, hi = -5.0 * scale, 5.0 * scale
        hist, edges = np.histogram(
            activation,
            bins=160,
            range=(lo, hi),
            density=True,
        )
        centres = (edges[:-1] + edges[1:]) / 2.0
        fitted = mixture_pdf(
            centres,
            alpha[:, component],
            mu[:, component],
            beta[:, component],
            rho[:, component],
        )
        for x, empirical, model in zip(centres, hist, fitted):
            rows.append(
                {
                    "dataset": "ds004505",
                    "subject": "sub-01",
                    "component": component,
                    "mean_rho": float(mean_rho[component]),
                    "activation": float(x),
                    "empirical_density": float(empirical),
                    "fitted_density": float(model),
                }
            )

    pd.DataFrame(rows).to_csv(CSV_PATH, index=False, float_format="%.17g")
    audit = {
        "purpose": "Figure 1 empirical adaptive-density miniatures",
        "model_fit": "archived m1_reject0 ds004505 sub-01 JAX-GPU fit; no refit",
        "selection_rule": "nearest components to prespecified mean-rho targets 1.04, 1.42, and 1.79",
        "components": components,
        "mean_rho": [float(mean_rho[index]) for index in components],
        "histogram": "160 equal-width bins over +/-5 source standard deviations; density normalisation",
        "inputs": {
            "result_json": {"path": str(JSON_PATH), "sha256": sha256(JSON_PATH)},
            "ica_fif": {"path": str(ICA_PATH), "sha256": sha256(ICA_PATH)},
            "raw_fdt": {"path": str(RAW_PATH), "sha256": sha256(RAW_PATH)},
        },
        "output": {"path": str(CSV_PATH), "sha256": sha256(CSV_PATH)},
    }
    AUDIT_PATH.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {CSV_PATH}")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
