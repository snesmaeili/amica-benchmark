#!/usr/bin/env python
"""Run one subject of five-fold, guard-banded held-out MIR validation.

Filtering and resampling are deterministic recording-level operations.  Every
learned transformation (centering, pre-whitening, PCA, ICA, and density state)
is fitted on the training samples only.  A guard around each test block removes
adjacent filtered samples from training.  The same explicit evaluation indices
are used for every method and histogram resolution within a fold.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from scripts.heldout.core import (
    complete_mir_from_fitted_ica,
    contiguous_folds,
    evaluation_indices,
)
from scripts.validation.provenance import collect_provenance


COMPARATORS = ("picard", "infomax", "fastica")
LINE_FREQUENCY = {"ds004505": 60.0, "ds004504": 50.0, "ds004621": 50.0}
EXPECTED_SUBJECTS = {
    "ds004505": tuple(range(1, 26)),
    "ds004504": tuple(range(37, 66)),
    "ds004621": tuple(range(1, 43)),
}


def _benchmark_runner():
    """Resolve the archived dataset loader used by the main benchmark."""

    errors = []
    for package in ("amica_python", "amica"):
        try:
            module = __import__(f"{package}.benchmark.runner", fromlist=["runner"])
            return module, package
        except (ImportError, ModuleNotFoundError) as exc:
            errors.append(f"{package}: {exc}")
    raise ImportError(
        "The paper benchmark runner is unavailable. Activate the archived paper "
        "environment containing amica_python.benchmark.runner. " + "; ".join(errors)
    )


def _amica_fit(train_raw, args):
    try:
        from amica import fit_ica

        package = "amica"
    except ImportError:
        from amica_python import fit_ica

        package = "amica_python"
    fit_params = {
        "dtype": "float64",
        "do_reject": False,
        "do_newton": True,
        "chunk_size": int(args.chunk_size),
    }
    started = time.perf_counter()
    ica = fit_ica(
        train_raw,
        n_components=args.n_components,
        max_iter=args.amica_max_iter,
        num_mix=3,
        random_state=args.random_state,
        fit_params=fit_params,
        verbose="WARNING",
    )
    # Materialising a result array enforces completion of asynchronous JAX work.
    np.asarray(ica.amica_result_.unmixing_matrix_white_)
    return ica, time.perf_counter() - started, package, fit_params


def _comparator_fit(train_raw, method, args):
    import mne

    if method == "picard":
        fit_params = {"ortho": False, "extended": True, "tol": 1e-6}
    elif method == "infomax":
        fit_params = {"extended": True, "w_change": 1e-7}
    elif method == "fastica":
        fit_params = {"fun": "logcosh", "tol": 1e-6}
    else:  # pragma: no cover - guarded by COMPARATORS
        raise ValueError(method)
    ica = mne.preprocessing.ICA(
        n_components=args.n_components,
        method=method,
        fit_params=fit_params,
        max_iter=args.comparator_max_iter,
        random_state=args.random_state,
        verbose="WARNING",
    )
    started = time.perf_counter()
    ica.fit(train_raw, verbose="WARNING")
    return ica, time.perf_counter() - started, "mne", fit_params


def _atomic_write(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _score(raw, ica, indices, bins):
    return {
        str(n_bins): complete_mir_from_fitted_ica(
            raw, ica, indices=indices, n_bins=n_bins
        )
        for n_bins in bins
    }


def run(args):
    if args.subject not in EXPECTED_SUBJECTS[args.dataset]:
        raise ValueError(f"subject {args.subject} is not in {args.dataset}")

    # Select the JAX platform before importing the solver.
    os.environ.setdefault("AMICA_NO_JAX", "0")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ["JAX_PLATFORM_NAME"] = args.device

    import mne

    mne.set_log_level("ERROR")
    runner, runner_package = _benchmark_runner()
    raw, load_metadata = runner.load_data(
        args.dataset, args.subject, input_level="bids", return_metadata=True
    )
    target_sfreq = getattr(runner, "DATASET_RESAMPLE", {}).get(args.dataset)
    window_metadata = runner.apply_analysis_window(
        raw, duration_sec=None, resample_sfreq=target_sfreq
    )
    load_metadata.update(window_metadata)
    raw = runner.preprocess(raw, line_freq=LINE_FREQUENCY[args.dataset])
    raw.pick("eeg")
    data = np.asarray(raw.get_data(), dtype=np.float64)
    guard_samples = int(round(args.guard_seconds * float(raw.info["sfreq"])))
    folds = contiguous_folds(data.shape[1], args.folds, guard_samples)

    repositories = [Path(__file__).resolve().parents[2]]
    package_repo = os.environ.get("AMICA_PACKAGE_REPO")
    if package_repo:
        repositories.append(Path(package_repo))
    payload = {
        "schema_version": 1,
        "analysis": "post-hoc guard-banded held-out complete MIR",
        "status": "partial",
        "dataset": args.dataset,
        "subject": args.subject,
        "n_components": args.n_components,
        "n_channels": int(data.shape[0]),
        "n_times": int(data.shape[1]),
        "sfreq_hz": float(raw.info["sfreq"]),
        "n_folds": args.folds,
        "guard_seconds": args.guard_seconds,
        "guard_samples": guard_samples,
        "max_evaluation_samples": args.max_evaluation_samples,
        "histogram_bins": list(args.histogram_bins),
        "random_state": args.random_state,
        "amica_max_iter": args.amica_max_iter,
        "comparator_max_iter": args.comparator_max_iter,
        "amica_num_mix": 3,
        "amica_chunk_size": args.chunk_size,
        "preprocessing_scope": (
            "recording-level deterministic filtering/resampling before splitting; "
            "all learned transforms fitted on training samples only"
        ),
        "load_metadata": load_metadata,
        "runner_package": runner_package,
        "folds": [],
        "provenance": collect_provenance(
            command=sys.argv, repositories=repositories
        ),
    }
    _atomic_write(args.output, payload)

    info = raw.info.copy()
    for fold in folds:
        train_raw = mne.io.RawArray(
            data[:, fold.train_indices], info, verbose="ERROR"
        )
        test_eval = evaluation_indices(
            len(fold.test_indices), args.max_evaluation_samples, args.random_state + fold.fold
        )
        test_eval_global = fold.test_indices[test_eval]
        train_eval = evaluation_indices(
            len(fold.train_indices), args.max_evaluation_samples, 1000 + args.random_state + fold.fold
        )
        fold_record = {
            "fold": fold.fold,
            "test_start": fold.test_start,
            "test_stop": fold.test_stop,
            "excluded_start": fold.excluded_start,
            "excluded_stop": fold.excluded_stop,
            "n_train": int(len(fold.train_indices)),
            "n_test": int(len(fold.test_indices)),
            "methods": [],
        }
        for method in ("amica",) + COMPARATORS:
            if method == "amica":
                ica, elapsed, package, fit_params = _amica_fit(train_raw, args)
                max_iter = args.amica_max_iter
            else:
                ica, elapsed, package, fit_params = _comparator_fit(
                    train_raw, method, args
                )
                max_iter = args.comparator_max_iter
            record = {
                "method": method,
                "package": package,
                "fit_seconds": float(elapsed),
                "n_iter": int(
                    getattr(ica, "n_iter_", None)
                    if getattr(ica, "n_iter_", None) is not None
                    else getattr(getattr(ica, "amica_result_", None), "n_iter", 0)
                ),
                "max_iter": int(max_iter),
                "fit_params": fit_params,
                "heldout_mir": _score(
                    raw, ica, test_eval_global, args.histogram_bins
                ),
                "training_mir": _score(
                    train_raw, ica, train_eval, args.histogram_bins
                ),
            }
            fold_record["methods"].append(record)
            payload["folds"] = [
                item for item in payload["folds"] if item["fold"] != fold.fold
            ] + [fold_record]
            payload["folds"].sort(key=lambda item: item["fold"])
            _atomic_write(args.output, payload)

    payload["status"] = "complete"
    _atomic_write(args.output, payload)
    print(args.output)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(EXPECTED_SUBJECTS), required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--n-components", type=int, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--guard-seconds", type=float, default=5.0)
    parser.add_argument("--max-evaluation-samples", type=int, default=20_000)
    parser.add_argument("--histogram-bins", type=int, nargs="+", default=(50, 100, 200))
    parser.add_argument("--amica-max-iter", type=int, default=3000)
    parser.add_argument("--comparator-max-iter", type=int, default=5000)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
