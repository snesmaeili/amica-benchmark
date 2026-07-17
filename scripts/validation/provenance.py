"""Capture machine-readable software, hardware, and Slurm provenance.

The functions in this module are intentionally side-effect free apart from the
optional CLI write.  They are used by every new validation job so that a result
can be tied to a command, node, software environment, and repository revision.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import io
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


TRACKED_ENV = (
    "AMICA_NO_JAX",
    "CUDA_VISIBLE_DEVICES",
    "JAX_ENABLE_X64",
    "JAX_PLATFORM_NAME",
    "JAX_PLATFORMS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "XLA_FLAGS",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
)

SLURM_ENV = (
    "SLURM_JOB_ID",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_CLUSTER_NAME",
    "SLURM_JOB_ACCOUNT",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_NODELIST",
    "SLURM_CPUS_PER_TASK",
    "SLURM_MEM_PER_NODE",
    "SLURM_GPUS",
)

PACKAGES = (
    "amica",
    "amica-python",
    "jax",
    "jaxlib",
    "mne",
    "mne-bids",
    "numpy",
    "pandas",
    "picard",
    "python-picard",
    "scikit-learn",
    "scipy",
    "threadpoolctl",
)


def _run(command: Iterable[str]) -> dict:
    exe = shutil.which(next(iter(command)))
    if exe is None:
        return {"status": "unavailable", "command": list(command)}
    try:
        completed = subprocess.run(
            list(command), capture_output=True, text=True, timeout=30, check=False
        )
    except Exception as exc:  # pragma: no cover - platform dependent
        return {"status": "error", "command": list(command), "error": str(exc)}
    return {
        "status": "ok" if completed.returncode == 0 else "error",
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def git_revision(path: str | Path) -> dict:
    """Return commit and dirty status for a repository without modifying it."""

    root = Path(path).resolve()
    commit = _run(("git", "-C", str(root), "rev-parse", "HEAD"))
    status = _run(("git", "-C", str(root), "status", "--porcelain"))
    return {
        "path": str(root),
        "commit": commit.get("stdout", "unknown"),
        "dirty": bool(status.get("stdout", "")),
        "status": status.get("status", "error"),
    }


def _package_versions() -> dict[str, str]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _numpy_configuration() -> str:
    try:
        import numpy as np

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            np.show_config()
        return buffer.getvalue().strip()
    except Exception as exc:  # pragma: no cover - optional runtime state
        return f"unavailable: {exc}"


def _threadpools() -> list[dict]:
    try:
        from threadpoolctl import threadpool_info

        return threadpool_info()
    except Exception:  # pragma: no cover - optional dependency
        return []


def _jax_state() -> dict:
    try:
        import jax

        return {
            "version": getattr(jax, "__version__", "unknown"),
            "x64_enabled": bool(jax.config.jax_enable_x64),
            "devices": [
                {
                    "platform": device.platform,
                    "kind": getattr(device, "device_kind", "unknown"),
                    "id": getattr(device, "id", None),
                }
                for device in jax.devices()
            ],
        }
    except Exception as exc:  # pragma: no cover - optional dependency
        return {"status": "unavailable", "error": str(exc)}


def collect_provenance(
    *,
    command: Iterable[str] | None = None,
    repositories: Iterable[str | Path] = (),
) -> dict:
    """Collect the provenance required for benchmark result sidecars."""

    payload = {
        "schema_version": 1,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(command if command is not None else sys.argv),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "node": platform.node(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "environment": {key: os.environ.get(key, "") for key in TRACKED_ENV},
        "slurm": {key: os.environ.get(key, "") for key in SLURM_ENV},
        "packages": _package_versions(),
        "numpy_configuration": _numpy_configuration(),
        "threadpools": _threadpools(),
        "jax": _jax_state(),
        "lscpu": _run(("lscpu", "--json")),
        "nvidia_smi": _run(
            (
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            )
        ),
        "modules": _run(("bash", "-lc", "module -t list 2>&1")),
        "repositories": [git_revision(path) for path in repositories],
    }
    return payload


def validate_provenance(payload: dict) -> list[str]:
    """Return missing required provenance fields; an empty list is valid."""

    required = (
        "schema_version",
        "captured_utc",
        "command",
        "python",
        "platform",
        "environment",
        "slurm",
        "packages",
        "repositories",
    )
    return [field for field in required if field not in payload]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", action="append", default=[])
    args = parser.parse_args(argv)
    payload = collect_provenance(repositories=args.repo)
    missing = validate_provenance(payload)
    if missing:
        raise RuntimeError(f"incomplete provenance: {missing}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
