#!/usr/bin/env python
"""Instrument one Python/Fortran parity cell at prespecified checkpoints.

This diagnostic reruns the same deterministic fixture from a clean solver
initialisation at each checkpoint.  It records the complete likelihood
histories and final-state agreement so that the first divergent update can be
located without changing either solver.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from scripts.parity.adapters import AmicaPythonAdapter, FortranAdapter
from scripts.parity.run_manifest_cell import (
    DEFAULT_PARAMS,
    align_python_iteration_params,
    compare,
    load_cell,
    make_fixture,
)
from scripts.validation.provenance import collect_provenance


DEFAULT_CHECKPOINTS = (1, 2, 3, 10, 20, 49, 50, 51, 60, 100, 200)


def trajectory_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    """Return aligned likelihood diagnostics without hiding index offsets."""

    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    n_common = min(reference.size, candidate.size)
    if n_common == 0:
        raise ValueError("likelihood histories must be non-empty")
    delta = candidate[:n_common] - reference[:n_common]
    finite = np.isfinite(reference[:n_common]) & np.isfinite(candidate[:n_common])
    first_nonfinite = np.flatnonzero(~finite)
    return {
        "reference_n_values": int(reference.size),
        "candidate_n_values": int(candidate.size),
        "n_common": int(n_common),
        "reference": reference.tolist(),
        "candidate": candidate.tolist(),
        "candidate_minus_reference": delta.tolist(),
        "initial_absolute_difference": float(abs(delta[0])),
        "final_absolute_difference": float(abs(delta[-1])),
        "maximum_absolute_difference": float(np.max(np.abs(delta[finite])))
        if np.any(finite)
        else None,
        "first_nonfinite_index": int(first_nonfinite[0])
        if first_nonfinite.size
        else None,
    }


def parse_checkpoints(value: str) -> tuple[int, ...]:
    checkpoints = tuple(sorted({int(item) for item in value.split(",")}))
    if not checkpoints or checkpoints[0] < 1:
        raise argparse.ArgumentTypeError("checkpoints must be positive integers")
    return checkpoints


def state_snapshot(result: dict) -> dict:
    """Persist the small fixture state needed to localise an update mismatch."""

    return {
        name: np.asarray(result[name], dtype=float).tolist()
        for name in ("W", "alpha", "mu", "beta", "rho", "c")
    }


def run(args) -> dict:
    task_index = args.task_index or int(os.environ.get("SLURM_ARRAY_TASK_ID", "1"))
    cell = load_cell(args.manifest, task_index)
    data, planted_outliers = make_fixture(cell)
    data = np.asarray(data, dtype=np.float32).astype(np.float64)
    base_params = {**DEFAULT_PARAMS, **cell, "pcakeep": data.shape[0]}
    fortran = FortranAdapter(args.fortran_binary)
    if not fortran.available:
        raise FileNotFoundError(f"Fortran binary unavailable: {fortran._binary}")

    checkpoint_results = []
    for n_iters in args.checkpoints:
        checkpoint_cell = {**cell, "max_iter": int(n_iters)}
        params = {**base_params, "max_iter": int(n_iters)}
        reference = fortran.run(data, params, n_iters)
        python_params = align_python_iteration_params(params, cell)
        candidate = AmicaPythonAdapter().run(
            data,
            python_params,
            n_iters,
            shared_sphere=reference["sphere"],
            shared_mean=reference["mean"],
        )
        checkpoint_results.append(
            {
                "n_iters": int(n_iters),
                "trajectory": trajectory_metrics(
                    reference["ll_history"], candidate["ll_history"]
                ),
                "final_state": compare(reference, candidate, checkpoint_cell),
                "reference_state": state_snapshot(reference),
                "candidate_state": state_snapshot(candidate),
            }
        )

    payload = {
        "schema_version": 1,
        "analysis": "instrumented Python/Fortran parity trajectory",
        "cell": cell,
        "fixture": {
            "n_channels": int(data.shape[0]),
            "n_samples": int(data.shape[1]),
            "planted_outlier_indices": planted_outliers.tolist(),
            "input_dtype_seen_by_both_solvers": "float32 values up-cast to float64",
            "fixed_initialisation": True,
            "shared_fortran_mean_and_sphere": True,
            "fortran_newt_start": base_params["newt_start"],
            "python_effective_newt_start": align_python_iteration_params(
                base_params, cell
            )["newt_start"],
        },
        "checkpoints": checkpoint_results,
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "checkpoints": list(args.checkpoints)}))
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-index", type=int, default=1)
    parser.add_argument("--fortran-binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoints",
        type=parse_checkpoints,
        default=DEFAULT_CHECKPOINTS,
        help="comma-separated iteration checkpoints",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
