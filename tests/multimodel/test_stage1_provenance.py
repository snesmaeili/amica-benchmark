from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


MODULE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "multimodel"
sys.path.insert(0, str(MODULE_DIR))

from stage1.provenance import (  # noqa: E402
    PROVENANCE_SCHEMA_VERSION,
    ProvenanceValidationError,
    validate_provenance,
)


def _record():
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "run_id": "core_joint_m3-seed0-M3",
        "campaign": "core",
        "condition_id": "core_joint_m3",
        "manifest_path": "manifests/stage1_core.csv",
        "manifest_sha256": "a" * 64,
        "manifest_row_index": 0,
        "manifest_row_sha256": "b" * 64,
        "command": ["python", "run_stage1.py", "--manifest-row", "0"],
        "started_at_utc": "2026-07-26T12:00:00+00:00",
        "completed_at_utc": "2026-07-26T12:05:00+00:00",
        "execution": {
            "mode": "local",
            "hostname": "test-host",
            "slurm_job_id": None,
            "slurm_array_task_id": None,
            "account": None,
            "partition": None,
            "slurm_cpus_per_task": None,
            "slurm_memory_bytes": None,
            "slurm_time_limit": None,
            "slurm_gpu_request": None,
            "cuda_visible_devices": None,
        },
        "software": {
            "benchmark_git_sha": "c" * 40,
            "package_git_sha": "d" * 40,
            "python": "3.11.9",
            "numpy": "2.0.1",
            "scipy": "1.14.0",
            "amica": "0.0.1",
            "jax": None,
            "jaxlib": None,
            "mne": "1.10.0",
            "scikit_learn": "1.7.0",
            "blas": "OpenBLAS 0.3.27",
            "cuda": None,
            "driver": None,
            "os": "Linux 6.8",
            "benchmark_worktree_clean": True,
            "package_worktree_clean": True,
        },
        "hardware": {
            "cpu_model": "Test CPU",
            "physical_cpus": 4,
            "logical_cpus": 8,
            "memory_bytes": 16 * 1024**3,
            "gpu_model": None,
            "gpu_uuid": None,
        },
        "runtime": {
            "backend": "numpy-cpu",
            "precision": "float64",
            "omp_num_threads": 1,
            "mkl_num_threads": 1,
            "openblas_num_threads": 1,
            "jax_enable_x64": None,
            "jax_platform": None,
            "jax_default_matmul_precision": None,
            "xla_flags": None,
            "xla_preallocate": None,
            "xla_memory_fraction": None,
        },
        "inputs": [
            {
                "path": "manifests/stage1_core.csv",
                "sha256": "a" * 64,
                "bytes": 5678,
            }
        ],
        "outputs": [
            {
                "path": "results/core_joint_m3-seed0-M3.json",
                "sha256": "e" * 64,
                "bytes": 1234,
            }
        ],
    }


def test_complete_local_numpy_record_is_valid():
    validate_provenance(_record())


def test_unknown_or_missing_fields_are_rejected():
    record = _record()
    record["unexpected"] = "value"
    with pytest.raises(ProvenanceValidationError, match="schema mismatch"):
        validate_provenance(record)

    record = _record()
    del record["software"]["blas"]
    with pytest.raises(ProvenanceValidationError, match="schema mismatch"):
        validate_provenance(record)


def test_jax_gpu_requires_resolved_gpu_and_cuda_provenance():
    record = _record()
    record["runtime"].update(
        {
            "backend": "jax-gpu",
            "precision": "float64",
            "jax_enable_x64": True,
            "jax_platform": "gpu",
            "xla_flags": "--xla_gpu_deterministic_ops=true",
            "xla_preallocate": False,
        }
    )
    with pytest.raises(ProvenanceValidationError, match="software.jax"):
        validate_provenance(record)

    record["software"].update(
        {
            "jax": "0.4.35",
            "jaxlib": "0.4.35",
            "cuda": "12.6",
            "driver": "560.35",
        }
    )
    record["hardware"]["gpu_model"] = "NVIDIA H100 80GB HBM3"
    record["hardware"]["gpu_uuid"] = "GPU-test-uuid"
    record["execution"]["cuda_visible_devices"] = "0"
    validate_provenance(record)


def test_slurm_mode_requires_scheduler_identity():
    record = _record()
    record["execution"]["mode"] = "slurm"
    with pytest.raises(ProvenanceValidationError, match="slurm_job_id"):
        validate_provenance(record)

    record["execution"].update(
        {
            "slurm_job_id": "49000001",
            "slurm_array_task_id": "7",
            "account": "def-kjerbi_gpu",
            "partition": "gpubase_bygpu_b1",
            "slurm_cpus_per_task": 6,
            "slurm_memory_bytes": 16 * 1024**3,
            "slurm_time_limit": "01:00:00",
            "slurm_gpu_request": None,
        }
    )
    validate_provenance(record)


def test_mutated_hash_and_backwards_timestamp_are_rejected():
    record = copy.deepcopy(_record())
    record["manifest_sha256"] = "not-a-hash"
    with pytest.raises(ProvenanceValidationError, match="hash format"):
        validate_provenance(record)

    record = _record()
    record["completed_at_utc"] = "2026-07-26T11:59:59+00:00"
    with pytest.raises(ProvenanceValidationError, match="precedes"):
        validate_provenance(record)


def test_float64_jax_requires_x64_and_platform_consistency():
    record = _record()
    record["runtime"].update(
        {
            "backend": "jax-cpu",
            "precision": "float64",
            "jax_enable_x64": False,
            "jax_platform": "cpu",
        }
    )
    record["software"].update({"jax": "0.4.35", "jaxlib": "0.4.35"})
    with pytest.raises(ProvenanceValidationError, match="jax_enable_x64=true"):
        validate_provenance(record)

    record["runtime"]["jax_enable_x64"] = True
    record["runtime"]["jax_platform"] = "gpu"
    with pytest.raises(ProvenanceValidationError, match="jax_platform='cpu'"):
        validate_provenance(record)


def test_dirty_worktree_and_manifest_input_mismatch_are_rejected():
    record = _record()
    record["software"]["package_worktree_clean"] = False
    with pytest.raises(ProvenanceValidationError, match="must be true"):
        validate_provenance(record)

    record = _record()
    record["inputs"][0]["sha256"] = "f" * 64
    with pytest.raises(ProvenanceValidationError, match="manifest input checksum"):
        validate_provenance(record)
