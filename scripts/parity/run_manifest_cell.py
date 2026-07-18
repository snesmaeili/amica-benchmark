#!/usr/bin/env python
"""Run and audit one prespecified Python/Fortran AMICA parity cell."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from scripts.parity.adapters import AmicaPythonAdapter, FortranAdapter
from scripts.parity.metrics import (
    as_model_array,
    match_models,
    normalized_rmse,
    rejection_metrics,
)
from scripts.validation.provenance import collect_provenance


THRESHOLDS = {
    "single_ll_relative": 1e-3,
    "single_worst_row_correlation": 0.999,
    "density_nrmse": 0.01,
    "rejection_jaccard": 0.99,
    "rejected_count_difference": 1,
    "multi_ll_relative": 1e-3,
    "multi_worst_row_correlation": 0.99,
    "model_weight_mean_absolute": 0.01,
    "posterior_mean_absolute": 0.01,
}


DEFAULT_PARAMS = {
    "lrate": 0.01,
    "newtrate": 1.0,
    "newt_start": 50,
    "newt_ramp": 10,
    "rho0": 1.5,
    "pdftype": 0,
    "minrho": 1.0,
    "maxrho": 2.0,
    "rholrate": 0.05,
    "invsigmin": 1e-8,
    "invsigmax": 100.0,
    "max_decs": 3,
    "min_dll": 1e-9,
    "use_min_dll": False,
    "minlrate": 1e-8,
    "lratefact": 0.5,
    "rholratefact": 0.5,
    "fix_init": True,
    "doscaling": True,
    "sphere_type": "pca",
    "use_grad_norm": False,
    "rejsig": 3.0,
    "rejstart": 2,
    "rejint": 3,
    "numrej": 5,
    "threads": 4,
}


def load_cell(path: Path, task_index: int):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if task_index < 1 or task_index > len(rows):
        raise IndexError(f"task index {task_index} outside 1..{len(rows)}")
    row = rows[task_index - 1]
    for key in ("num_models", "num_mix", "do_newton", "do_reject", "seed", "max_iter"):
        row[key] = int(row[key])
    row["do_newton"] = bool(row["do_newton"])
    row["do_reject"] = bool(row["do_reject"])
    return row


def make_fixture(cell):
    """Generate deterministic non-degenerate mixtures and rejection outliers."""

    rng = np.random.default_rng(int(cell["seed"]) + 20260716)
    n_channels, n_samples = 6, 12_000
    regimes = np.array_split(np.arange(n_samples), int(cell["num_models"]))
    data = np.empty((n_channels, n_samples), dtype=np.float64)
    for regime_index, indices in enumerate(regimes):
        sources = np.vstack(
            [
                rng.laplace(size=len(indices)),
                rng.standard_t(df=5, size=len(indices)),
                rng.normal(size=len(indices)),
                rng.logistic(size=len(indices)),
                rng.uniform(-np.sqrt(3), np.sqrt(3), size=len(indices)),
                rng.laplace(scale=0.6, size=len(indices)),
            ]
        )
        mixing = rng.normal(size=(n_channels, n_channels))
        mixing += (1.5 + 0.2 * regime_index) * np.eye(n_channels)
        data[:, indices] = mixing @ sources
    data -= data.mean(axis=1, keepdims=True)
    data /= data.std(axis=1, keepdims=True)
    outlier_indices = np.array([], dtype=int)
    if cell["do_reject"]:
        outlier_indices = np.linspace(200, n_samples - 201, 36, dtype=int)
        data[:, outlier_indices] += rng.normal(
            0.0, 35.0, size=(n_channels, len(outlier_indices))
        )
    return data, outlier_indices


def align_python_iteration_params(params, cell):
    """Map Fortran's one-based controls to Python's zero-based iteration loop."""

    aligned = dict(params)
    if cell["do_newton"]:
        aligned["newt_start"] = max(0, int(params["newt_start"]) - 1)
    if cell["do_reject"]:
        aligned["rejstart"] = max(1, int(params["rejstart"]) - 1)
    return aligned


def _sensor_models(result):
    models = as_model_array(result["W"])
    sphere = np.asarray(result["sphere"], dtype=float)
    return np.stack([matrix @ sphere for matrix in models])


def _row_assignment(reference, candidate):
    corr = np.corrcoef(reference, candidate)[: reference.shape[0], reference.shape[0] :]
    rows, columns = linear_sum_assignment(-np.abs(corr))
    signs = np.sign(corr[rows, columns])
    signs[signs == 0] = 1
    return columns, signs, np.abs(corr[rows, columns])


def _density_nrmse(reference, candidate, model_assignment):
    metrics = {name: [] for name in ("alpha", "mu", "beta", "rho")}
    ref_w = as_model_array(reference["W"])
    cand_w = as_model_array(candidate["W"])
    for ref_model, cand_model in enumerate(model_assignment):
        components, signs, _ = _row_assignment(
            ref_w[ref_model], cand_w[cand_model]
        )
        for ref_component, cand_component in enumerate(components):
            ref_vectors = {
                name: np.asarray(reference[name])[ref_model, :, ref_component]
                if np.asarray(reference[name]).ndim == 3
                else np.asarray(reference[name])[:, ref_component]
                for name in metrics
            }
            cand_vectors = {
                name: np.asarray(candidate[name])[cand_model, :, cand_component]
                if np.asarray(candidate[name]).ndim == 3
                else np.asarray(candidate[name])[:, cand_component]
                for name in metrics
            }
            cand_vectors["mu"] = cand_vectors["mu"] * signs[ref_component]
            scale = {
                name: max(float(np.ptp(ref_vectors[name])), 1e-6)
                for name in metrics
            }
            cost = np.zeros((len(ref_vectors["mu"]), len(cand_vectors["mu"])))
            for left in range(cost.shape[0]):
                for right in range(cost.shape[1]):
                    cost[left, right] = sum(
                        abs(ref_vectors[name][left] - cand_vectors[name][right])
                        / scale[name]
                        for name in metrics
                    )
            rows, columns = linear_sum_assignment(cost)
            for name in metrics:
                metrics[name].append(
                    normalized_rmse(
                        ref_vectors[name][rows], cand_vectors[name][columns]
                    )
                )
    return {f"{name}_nrmse": float(np.mean(values)) for name, values in metrics.items()}


def compare(reference, candidate, cell):
    ref_sensor = _sensor_models(reference)
    cand_sensor = _sensor_models(candidate)
    model_assignment, model_scores = match_models(ref_sensor, cand_sensor)
    row_correlations = []
    for ref_model, cand_model in enumerate(model_assignment):
        _, _, correlations = _row_assignment(
            ref_sensor[ref_model], cand_sensor[cand_model]
        )
        row_correlations.extend(correlations.tolist())
    ref_ll = float(reference["ll_history"][-1])
    cand_ll = float(candidate["ll_history"][-1])
    ll_relative = abs(ref_ll - cand_ll) / max(abs(ref_ll), np.finfo(float).eps)
    output = {
        "reference_final_ll": ref_ll,
        "candidate_final_ll": cand_ll,
        "final_ll_relative_difference": ll_relative,
        "model_assignment": model_assignment.tolist(),
        "model_assignment_correlations": model_scores.tolist(),
        "worst_matched_row_correlation": float(np.min(row_correlations)),
        "mean_matched_row_correlation": float(np.mean(row_correlations)),
    }
    output.update(_density_nrmse(reference, candidate, model_assignment))
    if cell["num_models"] > 1:
        ref_gm = np.asarray(reference["gm"])
        cand_gm = np.asarray(candidate["gm"])[model_assignment]
        output["model_weight_mean_absolute_difference"] = float(
            np.mean(np.abs(ref_gm - cand_gm))
        )
        ref_post = np.asarray(reference["model_posteriors"])
        cand_post = np.asarray(candidate["model_posteriors"])[model_assignment]
        valid = np.all(np.isfinite(ref_post), axis=0) & np.all(
            np.isfinite(cand_post), axis=0
        )
        output["posterior_mean_absolute_difference"] = float(
            np.mean(np.abs(ref_post[:, valid] - cand_post[:, valid]))
        )
    if cell["do_reject"]:
        output.update(
            rejection_metrics(reference["sample_mask"], candidate["sample_mask"])
        )
    return output


def pass_fail(metrics, cell):
    multi = cell["num_models"] > 1
    checks = {
        "final_ll": metrics["final_ll_relative_difference"]
        <= THRESHOLDS["multi_ll_relative" if multi else "single_ll_relative"],
        "unmixing": metrics["worst_matched_row_correlation"]
        >= THRESHOLDS[
            "multi_worst_row_correlation" if multi else "single_worst_row_correlation"
        ],
        "density": all(
            metrics[f"{name}_nrmse"] <= THRESHOLDS["density_nrmse"]
            for name in ("alpha", "mu", "beta", "rho")
        ),
    }
    if multi:
        checks.update(
            {
                "model_weights": metrics["model_weight_mean_absolute_difference"]
                <= THRESHOLDS["model_weight_mean_absolute"],
                "posteriors": metrics["posterior_mean_absolute_difference"]
                <= THRESHOLDS["posterior_mean_absolute"],
            }
        )
    if cell["do_reject"]:
        checks.update(
            {
                "rejection_jaccard": metrics["rejection_jaccard"]
                >= THRESHOLDS["rejection_jaccard"],
                "rejection_count": metrics["rejected_count_difference"]
                <= THRESHOLDS["rejected_count_difference"],
            }
        )
    return checks


def run(args):
    task_index = args.task_index or int(os.environ.get("SLURM_ARRAY_TASK_ID", "1"))
    cell = load_cell(args.manifest, task_index)
    data, planted_outliers = make_fixture(cell)
    # Fortran reads float32 FDT data.  Up-casting those exact values makes the
    # two solvers see bit-identical observations.
    data = np.asarray(data, dtype=np.float32).astype(np.float64)
    params = {
        **DEFAULT_PARAMS,
        **cell,
        "pcakeep": data.shape[0],
    }
    fortran = FortranAdapter(args.fortran_binary)
    if not fortran.available:
        raise FileNotFoundError(f"Fortran binary unavailable: {fortran._binary}")
    fortran_result = fortran.run(data, params, cell["max_iter"])
    # Fortran evaluates these controls in a one-based loop, while Python uses
    # ``range(max_iter)``. Align the control points without changing either
    # solver's public semantics.
    python_params = align_python_iteration_params(params, cell)
    python_result = AmicaPythonAdapter().run(
        data,
        python_params,
        cell["max_iter"],
        shared_sphere=fortran_result["sphere"],
        shared_mean=fortran_result["mean"],
    )
    metrics = compare(fortran_result, python_result, cell)
    checks = pass_fail(metrics, cell)
    payload = {
        "schema_version": 1,
        "cell": cell,
        "fixture": {
            "n_channels": int(data.shape[0]),
            "n_samples": int(data.shape[1]),
            "planted_outlier_indices": planted_outliers.tolist(),
            "input_dtype_seen_by_both_solvers": "float32 values up-cast to float64",
            "fixed_initialisation": True,
            "shared_fortran_mean_and_sphere": True,
            "python_effective_rejstart": python_params["rejstart"],
            "fortran_newt_start": params["newt_start"],
            "python_effective_newt_start": python_params["newt_start"],
        },
        "thresholds": THRESHOLDS,
        "metrics": metrics,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "provenance": collect_provenance(
            command=sys.argv,
            repositories=[
                path
                for path in (
                    Path(__file__).resolve().parents[2],
                    Path(os.environ["AMICA_PACKAGE_REPO"])
                    if os.environ.get("AMICA_PACKAGE_REPO")
                    else None,
                )
                if path is not None
            ],
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{cell['cell_id']}.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "passed": payload["passed"]}))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--fortran-binary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
