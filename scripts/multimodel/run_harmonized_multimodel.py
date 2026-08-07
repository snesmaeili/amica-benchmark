"""Run one harmonized real-EEG multi-model AMICA fit on a Slurm compute node.

This is benchmark code, not package code.  It fits one dataset/subject/model
order/seed combination, computes the exploratory posterior-weighted cMIR
diagnostics in-process, and writes a compact, auditable NPZ plus JSON sidecar.

The script is intentionally one-fit-per-process so JAX allocator state and
random initialization cannot leak between manifest rows.  It must be launched
with Slurm on Compute Canada; do not run the real-data path on a login node.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from conditional_mir import (  # noqa: E402
    arrays_from_amica_result,
    conditional_mir,
)


TENTWENTY = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
    "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2",
)
ALIASES = {
    "T7": ("T7", "T3"),
    "T8": ("T8", "T4"),
    "P7": ("P7", "T5"),
    "P8": ("P8", "T6"),
}
LINE_FREQ = {"ds004505": 60.0, "ds004504": 50.0, "ds004621": 50.0}
IGNORED_EVENT_PREFIXES = ("status", "boundary", "bad")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot JSON-encode {type(value)!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray, *, dtype: str) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    digest = hashlib.sha256(
        f"{array.dtype.str}|{array.shape}".encode("ascii")
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sanitize_json(value, path=""):
    """Return strict-JSON data plus paths whose non-finite values became null."""
    nonfinite = []
    if isinstance(value, dict):
        output = {}
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            output[key], child_nonfinite = _sanitize_json(child, child_path)
            nonfinite.extend(child_nonfinite)
        return output, nonfinite
    if isinstance(value, (list, tuple, np.ndarray)):
        output = []
        for index, child in enumerate(list(value)):
            child_path = f"{path}[{index}]"
            clean, child_nonfinite = _sanitize_json(child, child_path)
            output.append(clean)
            nonfinite.extend(child_nonfinite)
        return output, nonfinite
    if isinstance(value, (np.integer,)):
        return int(value), nonfinite
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(float(value)):
            return None, [path]
        return float(value), nonfinite
    if isinstance(value, Path):
        return str(value), nonfinite
    return value, nonfinite


def _git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _git_dirty(path: Path) -> bool | None:
    try:
        status = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(status.strip())
    except Exception:
        return None


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _cpu_model() -> str:
    processor = platform.processor().strip()
    if processor:
        return processor
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[-1].strip()
    return "unknown"


def _memory_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        return None


def _numpy_blas_configuration() -> str:
    stream = io.StringIO()
    with redirect_stdout(stream):
        np.__config__.show()
    return stream.getvalue().strip() or "unknown"


def _nvidia_smi() -> str | None:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _import_amica(expected_commit: str):
    import amica as amica_module
    from amica import Amica, AmicaConfig
    from amica.benchmark import runner
    from amica.benchmark.metrics import complete_mir

    module_path = Path(amica_module.__file__).resolve()
    package_root = module_path.parent.parent
    package_commit = _git_commit(package_root)
    if package_commit == "unknown":
        raise RuntimeError(
            f"resolved amica at {module_path}, but its Git commit is unavailable"
        )
    if package_commit != expected_commit:
        raise RuntimeError(
            "resolved amica commit does not match the pinned deployment: "
            f"{package_commit} != {expected_commit} ({module_path})"
        )
    package_dirty = _git_dirty(package_root)
    if package_dirty is not False:
        raise RuntimeError(
            "resolved amica worktree must be a clean, committed deployment; "
            f"dirty_state={package_dirty!r} ({package_root})"
        )
    try:
        package_version = importlib.metadata.version("amica")
    except importlib.metadata.PackageNotFoundError:
        package_version = getattr(amica_module, "__version__", "unknown")
    package = {
        "name": "amica",
        "version": package_version,
        "module_path": str(module_path),
        "root": str(package_root),
        "commit": package_commit,
        "dirty": package_dirty,
    }
    return Amica, AmicaConfig, runner, complete_mir, package


def _manifest_row(path: Path, zero_based_index: int) -> dict:
    with path.open(encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if index == zero_based_index:
                return row
    raise IndexError(f"manifest row {zero_based_index} not found in {path}")


def _validate_manifest_identity(args) -> dict:
    row = _manifest_row(args.manifest_path, args.manifest_row_index)
    expected = {
        "dataset": str(args.dataset),
        "subject": str(args.subject),
        "num_models": str(args.num_models),
        "fit_seed": str(args.fit_seed),
        "surrogate": str(args.surrogate),
        "surrogate_seed": str(args.surrogate_seed),
    }
    mismatches = {
        key: (row.get(key), value)
        for key, value in expected.items()
        if str(row.get(key)) != value
    }
    if mismatches:
        raise RuntimeError(
            f"CLI fields do not match manifest row {args.manifest_row_index}: "
            f"{mismatches}"
        )
    return row


def _pick_exact_tentwenty(raw):
    """Pick the canonical 19 channels in canonical order or fail explicitly."""

    have = {name.upper(): name for name in raw.info["ch_names"]}
    actual = []
    missing = []
    for canonical in TENTWENTY:
        found = None
        for alias in ALIASES.get(canonical, (canonical,)):
            found = have.get(alias.upper())
            if found is not None:
                break
        if found is None:
            missing.append(canonical)
        else:
            actual.append(found)
    if missing or len(set(actual)) != len(TENTWENTY):
        raise RuntimeError(
            "common-montage requirement failed: "
            f"found={actual!r}, missing={missing!r}"
        )
    raw.pick(actual)
    return actual


def phase_surrogate(x: np.ndarray, seed: int) -> np.ndarray:
    """Apply one common Fourier phase rotation per frequency to all PCs."""

    arr = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    spectrum = np.fft.rfft(arr, axis=1)
    phase = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, spectrum.shape[1]))
    phase[0] = 1.0
    if arr.shape[1] % 2 == 0:
        phase[-1] = 1.0
    return np.fft.irfft(spectrum * phase[None, :], n=arr.shape[1], axis=1)


def _read_ds004505_events(subject: int, duration_sec: float):
    root = os.environ.get("BIDS_ROOT_DS4505")
    if not root:
        return np.array([]), np.array([]), np.array([], dtype="U1")
    path = (
        Path(root)
        / f"sub-{subject:02d}"
        / "eeg"
        / f"sub-{subject:02d}_task-TableTennis_events.tsv"
    )
    if not path.exists():
        return np.array([]), np.array([]), np.array([], dtype="U1")
    import csv

    onsets, durations, labels = [], [], []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            try:
                onset = float(row.get("onset", "nan"))
            except ValueError:
                continue
            if not np.isfinite(onset) or onset > duration_sec:
                continue
            try:
                event_duration = float(row.get("duration", 0.0))
            except ValueError:
                event_duration = 0.0
            onsets.append(onset)
            durations.append(event_duration)
            labels.append(str(row.get("trial_type", "")))
    return (
        np.asarray(onsets, dtype=float),
        np.asarray(durations, dtype=float),
        np.asarray(labels, dtype="U128"),
    )


def _task_onset(onsets: np.ndarray, labels: np.ndarray) -> float:
    candidates = []
    for onset, label in zip(onsets, labels):
        normalized = str(label).strip().lower()
        if normalized and not normalized.startswith(IGNORED_EVENT_PREFIXES):
            candidates.append(float(onset))
    return min(candidates) if candidates else float("nan")


def posterior_windows(
    posteriors: np.ndarray,
    sfreq: float,
    *,
    window_sec: float,
    task_onset_sec: float,
    transition_buffer_sec: float,
):
    """Average posteriors in fixed windows and attach independent task labels."""

    gamma = np.asarray(posteriors, dtype=float)
    width = int(round(float(window_sec) * float(sfreq)))
    if width < 1:
        raise ValueError("window_sec produces an empty window")
    n_windows = gamma.shape[1] // width
    gamma = gamma[:, : n_windows * width]
    features = gamma.reshape(gamma.shape[0], n_windows, width).mean(axis=2).T
    starts = np.arange(n_windows, dtype=float) * float(window_sec)
    stops = starts + float(window_sec)
    labels = np.full(n_windows, -1, dtype=np.int8)
    if np.isfinite(task_onset_sec):
        labels[stops <= task_onset_sec - transition_buffer_sec] = 0
        labels[starts >= task_onset_sec + transition_buffer_sec] = 1
    return features, labels, starts


def preprocess_harmonized(
    dataset: str,
    subject: int,
    *,
    duration_sec: float,
    sfreq: float,
    n_components: int,
    surrogate: str,
    surrogate_seed: int,
):
    """Load, harmonize, filter, PCA-reduce, normalize, and optionally surrogate."""

    if n_components != 15:
        raise ValueError("the primary harmonized analysis is prespecified at 15 PCs")
    if duration_sec != 600.0 or sfreq != 250.0:
        raise ValueError("the primary harmonized analysis requires 600 s at 250 Hz")
    _, _, runner, _, _ = _import_amica()
    from sklearn.decomposition import PCA

    raw, metadata = runner.load_data(
        dataset, subject, input_level="bids", return_metadata=True
    )
    source_channels = list(raw.info["ch_names"])
    picked_channels = _pick_exact_tentwenty(raw)
    runner.apply_analysis_window(
        raw, duration_sec=float(duration_sec), resample_sfreq=float(sfreq)
    )
    raw = runner.preprocess(raw, line_freq=LINE_FREQ[dataset])
    actual_sfreq = float(raw.info["sfreq"])
    data = np.asarray(raw.get_data(), dtype=np.float64)
    if data.shape[1] != int(round(duration_sec * sfreq)):
        raise RuntimeError(
            f"expected {int(round(duration_sec * sfreq))} samples, got {data.shape[1]}"
        )
    pca = PCA(n_components=n_components, whiten=False, svd_solver="full")
    projected = pca.fit_transform(data.T).T
    pc_stds = np.std(projected, axis=1, keepdims=True)
    if np.any(~np.isfinite(pc_stds)) or np.any(pc_stds <= 0):
        raise RuntimeError("non-finite or zero PCA scale")
    x = projected / pc_stds
    if surrogate == "phase":
        x = phase_surrogate(x, surrogate_seed)
    elif surrogate != "none":
        raise ValueError(f"unknown surrogate type: {surrogate}")

    event_onsets, event_durations, event_labels = (
        _read_ds004505_events(subject, duration_sec)
        if dataset == "ds004505"
        else (np.array([]), np.array([]), np.array([], dtype="U1"))
    )
    return {
        "x": np.asarray(x, dtype=np.float64),
        "sfreq": actual_sfreq,
        "source_channel_count": len(source_channels),
        "picked_channels": picked_channels,
        "canonical_channels": list(TENTWENTY),
        "pca_components": np.asarray(pca.components_, dtype=np.float32),
        "pca_stds": np.asarray(pc_stds[:, 0], dtype=np.float32),
        "pca_explained_variance_ratio": np.asarray(
            pca.explained_variance_ratio_, dtype=np.float64
        ),
        "metadata": metadata,
        "event_onsets": event_onsets,
        "event_durations": event_durations,
        "event_labels": event_labels,
        "task_onset_sec": _task_onset(event_onsets, event_labels),
    }


def _flatten_cmir(result):
    row = result.to_dict()
    row["models"] = list(row["models"])
    return row


def run(args):
    manifest_row = _validate_manifest_identity(args)
    Amica, AmicaConfig, _, complete_mir, package = _import_amica(
        args.expected_package_commit
    )
    pre = preprocess_harmonized(
        args.dataset,
        args.subject,
        duration_sec=args.duration_sec,
        sfreq=args.sfreq,
        n_components=args.n_components,
        surrogate=args.surrogate,
        surrogate_seed=args.surrogate_seed,
    )
    x = pre["x"]
    n_components, n_samples = x.shape
    kappa_eff = n_samples / (args.num_models * n_components * n_components)

    config = AmicaConfig(
        num_models=args.num_models,
        num_mix_comps=args.num_mix,
        pcakeep=n_components,
        dtype="float64",
        max_iter=args.max_iter,
        do_reject=False,
        do_newton=True,
        do_sphere=True,
        do_mean=True,
        chunk_size=args.chunk_size,
    )
    started = time.time()
    result = Amica(config, random_state=args.fit_seed).fit(x)
    elapsed_sec = time.time() - started

    x_internal, y_models, w_models, posteriors = arrays_from_amica_result(result, x)
    cmir_variants = {}
    for n_bins in (50, 100, 200):
        key = f"soft_{n_bins}bins"
        cmir_variants[key] = _flatten_cmir(
            conditional_mir(
                x_internal,
                y_models,
                w_models,
                posteriors,
                pre["sfreq"],
                n_bins=n_bins,
                assignment="soft",
                min_effective_n=args.min_effective_n,
                min_posterior_mass=args.min_posterior_mass,
            )
        )
    for assignment in ("hard", "time_permuted"):
        cmir_variants[f"{assignment}_100bins"] = _flatten_cmir(
            conditional_mir(
                x_internal,
                y_models,
                w_models,
                posteriors,
                pre["sfreq"],
                n_bins=100,
                assignment=assignment,
                assignment_random_state=args.assignment_seed,
                min_effective_n=args.min_effective_n,
                min_posterior_mass=args.min_posterior_mass,
            )
        )

    ordinary_mir = None
    m1_identity_error = None
    if args.num_models == 1:
        ordinary = complete_mir(
            x_internal,
            y_models[0],
            w_models[0],
            pre["sfreq"],
            n_bins=100,
            clip_sd=5.0,
            max_samples=None,
            subspace_mode=True,
        )
        ordinary_mir = ordinary.to_dict()
        m1_identity_error = float(
            cmir_variants["soft_100bins"]["bits_per_sample"]
            - ordinary.bits_per_sample
        )

    task_features, task_labels, task_window_starts = posterior_windows(
        posteriors,
        pre["sfreq"],
        window_sec=args.posterior_window_sec,
        task_onset_sec=pre["task_onset_sec"],
        transition_buffer_sec=args.transition_buffer_sec,
    )
    ll_history = np.asarray(result.log_likelihood, dtype=np.float64)
    final_ll = getattr(result, "final_log_likelihood_", None)
    if final_ll is None or not np.isfinite(float(final_ll)):
        raise RuntimeError(
            "current-package run did not produce a finite returned-state "
            "final_log_likelihood_"
        )
    final_ll = float(final_ll)
    gm = np.atleast_1d(np.asarray(result.gm_, dtype=np.float64))
    device = "unknown"
    try:
        import jax

        devices = jax.devices()
        device = ",".join(
            f"{d.platform}:{getattr(d, 'device_kind', '')}" for d in devices
        )
        jax_version = jax.__version__
        jaxlib_version = _package_version("jaxlib")
        platform_version = (
            str(getattr(devices[0].client, "platform_version", "unknown"))
            if devices
            else "unknown"
        )
    except Exception:
        jax_version = "unknown"
        jaxlib_version = "unknown"
        platform_version = "unknown"

    scipy_version = _package_version("scipy")
    sklearn_version = _package_version("scikit-learn")
    mne_version = _package_version("mne")

    workflow_root = SCRIPT_DIR.parents[1]
    metadata = {
        "schema_version": 2,
        "metric_status": "exploratory posterior-weighted conditional MIR",
        "dataset": args.dataset,
        "subject": args.subject,
        "surrogate": args.surrogate,
        "surrogate_seed": args.surrogate_seed,
        "fit_seed": args.fit_seed,
        "assignment_seed": args.assignment_seed,
        "num_models": args.num_models,
        "num_mix": args.num_mix,
        "n_components": n_components,
        "n_samples": n_samples,
        "duration_sec": args.duration_sec,
        "sfreq": pre["sfreq"],
        "kappa_eff_per_model": kappa_eff,
        "max_iter": args.max_iter,
        "n_iter": int(result.n_iter),
        "converged": bool(result.converged),
        "do_reject": False,
        "dtype": "float64",
        "backend": "JAX-GPU",
        "device": device,
        "elapsed_sec": elapsed_sec,
        "ll_unit": "nats per retained component per sample",
        "ll_final": final_ll,
        "ll_final_recomputed": final_ll,
        "ll_history_final": float(ll_history[-1]),
        "ll_initial": float(ll_history[0]),
        "gm": gm,
        "model_posteriors_shape": list(np.asarray(posteriors).shape),
        "model_posteriors_sha256_float64": _sha256_array(
            posteriors, dtype="<f8"
        ),
        "cmir": cmir_variants,
        "ordinary_mir_m1": ordinary_mir,
        "m1_identity_error_bits_per_sample": m1_identity_error,
        "source_channel_count": pre["source_channel_count"],
        "picked_channel_count": len(pre["picked_channels"]),
        "picked_channels": pre["picked_channels"],
        "canonical_channels": pre["canonical_channels"],
        "pca_explained_variance_ratio": pre["pca_explained_variance_ratio"],
        "task_onset_sec": pre["task_onset_sec"],
        "task_window_sec": args.posterior_window_sec,
        "transition_buffer_sec": args.transition_buffer_sec,
        "n_baseline_windows": int(np.sum(task_labels == 0)),
        "n_task_windows": int(np.sum(task_labels == 1)),
        "python": sys.version,
        "platform": platform.platform(),
        "package_name": package["name"],
        "package_version": package["version"],
        "package_module_path": package["module_path"],
        "package_root": package["root"],
        "package_commit": package["commit"],
        "package_git_dirty": package["dirty"],
        "expected_package_commit": args.expected_package_commit,
        "jax_version": jax_version,
        "jaxlib_version": jaxlib_version,
        "jax_platform_version": platform_version,
        "numpy_version": np.__version__,
        "scipy_version": scipy_version,
        "sklearn_version": sklearn_version,
        "mne_version": mne_version,
        "numpy_blas_configuration": _numpy_blas_configuration(),
        "workflow_commit": _git_commit(workflow_root),
        "workflow_git_dirty": _git_dirty(workflow_root),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "manifest_path": str(args.manifest_path.resolve()),
        "manifest_sha256": _sha256(args.manifest_path),
        "manifest_row_index": args.manifest_row_index,
        "manifest_row": manifest_row,
        "command": [sys.executable, *sys.argv],
        "node": platform.node(),
        "cpu_model": _cpu_model(),
        "logical_cpu_count": os.cpu_count(),
        "physical_memory_bytes": _memory_bytes(),
        "nvidia_smi": _nvidia_smi(),
        "loaded_modules": os.environ.get("LOADEDMODULES", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "jax_platforms": os.environ.get("JAX_PLATFORMS", ""),
        "jax_enable_x64": os.environ.get("JAX_ENABLE_X64", ""),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "xla_python_client_preallocate": os.environ.get(
            "XLA_PYTHON_CLIENT_PREALLOCATE", ""
        ),
        "xla_python_client_mem_fraction": os.environ.get(
            "XLA_PYTHON_CLIENT_MEM_FRACTION", ""
        ),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", ""),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", ""),
        "slurm_account": os.environ.get("SLURM_JOB_ACCOUNT", ""),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", ""),
        "slurm_mem_per_node": os.environ.get("SLURM_MEM_PER_NODE", ""),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST", ""),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
    }

    kind = "real" if args.surrogate == "none" else f"phase{args.surrogate_seed}"
    stem = (
        f"mmc_{args.dataset}_sub-{args.subject:02d}_M{args.num_models:02d}_"
        f"fitseed{args.fit_seed}_{kind}"
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{stem}.npz"
    json_path = out_dir / f"{stem}.json"
    metadata, nonfinite_fields = _sanitize_json(metadata)
    metadata["nonfinite_fields"] = nonfinite_fields
    metadata["cmir_finite"] = not any(
        field.startswith("cmir.") for field in nonfinite_fields
    )
    metadata_json = json.dumps(
        metadata,
        default=_json_default,
        sort_keys=True,
        allow_nan=False,
    )
    np.savez_compressed(
        npz_path,
        ll_history=ll_history,
        gm=gm,
        task_features=np.asarray(task_features, dtype=np.float32),
        task_labels=task_labels,
        task_window_starts_sec=task_window_starts,
        event_onsets=pre["event_onsets"],
        event_durations=pre["event_durations"],
        event_labels=pre["event_labels"],
        pca_components=pre["pca_components"],
        pca_stds=pre["pca_stds"],
        unmixing_matrix_white=np.asarray(result.unmixing_matrix_white_, dtype=np.float64),
        whitener=np.asarray(result.whitener_, dtype=np.float64),
        fitted_mean=np.asarray(result.mean_, dtype=np.float64),
        fitted_data_scale=np.asarray(result.data_scale, dtype=np.float64),
        model_centres=np.asarray(result.c_, dtype=np.float64),
        alpha=np.asarray(result.alpha_, dtype=np.float64),
        mu=np.asarray(result.mu_, dtype=np.float64),
        sbeta=np.asarray(result.sbeta_, dtype=np.float64),
        rho=np.asarray(result.rho_, dtype=np.float64),
        sample_mask=(
            np.asarray(result.sample_mask_, dtype=bool)
            if result.sample_mask_ is not None
            else np.empty(0, dtype=bool)
        ),
        metadata_json=metadata_json,
    )
    json_path.write_text(
        json.dumps(
            metadata,
            default=_json_default,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"npz": str(npz_path), "json": str(json_path), "ll_final": metadata["ll_final"]}))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--manifest-row-index", type=int, required=True)
    parser.add_argument("--expected-package-commit", required=True)
    parser.add_argument("--dataset", required=True, choices=sorted(LINE_FREQ))
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--num-models", type=int, required=True, choices=range(1, 11))
    parser.add_argument("--fit-seed", type=int, required=True)
    parser.add_argument("--surrogate", choices=("none", "phase"), default="none")
    parser.add_argument("--surrogate-seed", type=int, default=0)
    parser.add_argument("--assignment-seed", type=int, default=20260715)
    parser.add_argument("--n-components", type=int, default=15)
    parser.add_argument("--num-mix", type=int, default=3)
    parser.add_argument("--duration-sec", type=float, default=600.0)
    parser.add_argument("--sfreq", type=float, default=250.0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--min-effective-n", type=float, default=2000.0)
    parser.add_argument("--min-posterior-mass", type=float, default=2000.0)
    parser.add_argument("--posterior-window-sec", type=float, default=5.0)
    parser.add_argument("--transition-buffer-sec", type=float, default=30.0)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
