"""Strict provenance-schema validation for completed Stage I runs."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Mapping, Sequence


PROVENANCE_SCHEMA_VERSION = "amica-multimodel-stage1-provenance-v2"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ProvenanceValidationError(ValueError):
    """Raised when a provenance record is incomplete or internally inconsistent."""


_TOP_LEVEL_KEYS = {
    "schema_version",
    "run_id",
    "campaign",
    "condition_id",
    "manifest_path",
    "manifest_sha256",
    "manifest_row_index",
    "manifest_row_sha256",
    "command",
    "started_at_utc",
    "completed_at_utc",
    "execution",
    "software",
    "hardware",
    "runtime",
    "inputs",
    "outputs",
}
_EXECUTION_KEYS = {
    "mode",
    "hostname",
    "slurm_job_id",
    "slurm_array_task_id",
    "account",
    "partition",
    "slurm_cpus_per_task",
    "slurm_memory_bytes",
    "slurm_time_limit",
    "slurm_gpu_request",
    "cuda_visible_devices",
}
_SOFTWARE_KEYS = {
    "benchmark_git_sha",
    "package_git_sha",
    "python",
    "numpy",
    "scipy",
    "amica",
    "jax",
    "jaxlib",
    "mne",
    "scikit_learn",
    "blas",
    "cuda",
    "driver",
    "os",
    "benchmark_worktree_clean",
    "package_worktree_clean",
}
_HARDWARE_KEYS = {
    "cpu_model",
    "physical_cpus",
    "logical_cpus",
    "memory_bytes",
    "gpu_model",
    "gpu_uuid",
}
_RUNTIME_KEYS = {
    "backend",
    "precision",
    "omp_num_threads",
    "mkl_num_threads",
    "openblas_num_threads",
    "jax_enable_x64",
    "jax_platform",
    "jax_default_matmul_precision",
    "xla_flags",
    "xla_preallocate",
    "xla_memory_fraction",
}
_ARTIFACT_KEYS = {"path", "sha256", "bytes"}


def _exact_keys(mapping: Mapping[str, object], expected: set, name: str) -> None:
    missing = sorted(expected - set(mapping))
    extra = sorted(set(mapping) - expected)
    if missing or extra:
        raise ProvenanceValidationError(
            f"{name} schema mismatch; missing={missing}, extra={extra}"
        )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProvenanceValidationError(f"{name} must be a mapping")
    return value


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProvenanceValidationError(f"{name} must be a non-empty string")
    return value


def _nullable_string(value: object, name: str) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ProvenanceValidationError(f"{name} must be null or a non-empty string")


def _positive_int(value: object, name: str, allow_zero: bool = False) -> int:
    lower = 0 if allow_zero else 1
    if type(value) is not int or value < lower:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ProvenanceValidationError(f"{name} must be a {qualifier} integer")
    return value


def _nullable_bool(value: object, name: str) -> None:
    if value is not None and type(value) is not bool:
        raise ProvenanceValidationError(f"{name} must be null or boolean")


def _nullable_positive_int(value: object, name: str) -> None:
    if value is not None:
        _positive_int(value, name)


def _timestamp(value: object, name: str) -> datetime:
    text = _nonempty_string(value, name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProvenanceValidationError(f"{name} is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ProvenanceValidationError(f"{name} must include a UTC offset")
    if parsed.utcoffset() != timedelta(0):
        raise ProvenanceValidationError(f"{name} must be expressed in UTC")
    return parsed


def _hash(value: object, pattern: re.Pattern, name: str) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ProvenanceValidationError(f"{name} has the wrong hash format")


def _nonempty_sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ProvenanceValidationError(f"{name} must be a non-empty sequence")
    return value


def _validate_artifacts(value: object, name: str) -> set[str]:
    artifacts = _nonempty_sequence(value, name)
    paths = set()
    for index, artifact_value in enumerate(artifacts):
        artifact = _mapping(artifact_value, f"{name}[{index}]")
        _exact_keys(artifact, _ARTIFACT_KEYS, f"{name}[{index}]")
        path = _nonempty_string(artifact["path"], f"{name}[{index}].path")
        if path in paths:
            raise ProvenanceValidationError(f"duplicate {name} path: {path}")
        paths.add(path)
        _hash(artifact["sha256"], _HEX64, f"{name}[{index}].sha256")
        _positive_int(
            artifact["bytes"],
            f"{name}[{index}].bytes",
            allow_zero=True,
        )
    return paths


def validate_provenance(record: Mapping[str, object]) -> None:
    """Validate a completed-run provenance record without coercing values."""
    _exact_keys(record, _TOP_LEVEL_KEYS, "provenance")
    if record["schema_version"] != PROVENANCE_SCHEMA_VERSION:
        raise ProvenanceValidationError("unsupported provenance schema_version")
    _nonempty_string(record["run_id"], "run_id")
    if record["campaign"] not in {
        "core",
        "one_factor_stress",
        "real_eeg_heldout",
    }:
        raise ProvenanceValidationError("campaign is not a Stage I campaign")
    _nonempty_string(record["condition_id"], "condition_id")
    _nonempty_string(record["manifest_path"], "manifest_path")
    _hash(record["manifest_sha256"], _HEX64, "manifest_sha256")
    _positive_int(
        record["manifest_row_index"], "manifest_row_index", allow_zero=True
    )
    _hash(record["manifest_row_sha256"], _HEX64, "manifest_row_sha256")
    command = _nonempty_sequence(record["command"], "command")
    for index, argument in enumerate(command):
        _nonempty_string(argument, f"command[{index}]")
    started = _timestamp(record["started_at_utc"], "started_at_utc")
    completed = _timestamp(record["completed_at_utc"], "completed_at_utc")
    if completed < started:
        raise ProvenanceValidationError("completed_at_utc precedes started_at_utc")

    execution = _mapping(record["execution"], "execution")
    _exact_keys(execution, _EXECUTION_KEYS, "execution")
    if execution["mode"] not in {"local", "slurm"}:
        raise ProvenanceValidationError("execution.mode must be local or slurm")
    _nonempty_string(execution["hostname"], "execution.hostname")
    for field in (
        "slurm_job_id",
        "slurm_array_task_id",
        "account",
        "partition",
        "slurm_time_limit",
        "slurm_gpu_request",
        "cuda_visible_devices",
    ):
        _nullable_string(execution[field], f"execution.{field}")
    for field in ("slurm_cpus_per_task", "slurm_memory_bytes"):
        _nullable_positive_int(execution[field], f"execution.{field}")
    if execution["mode"] == "slurm":
        for field in (
            "slurm_job_id",
            "account",
            "partition",
            "slurm_time_limit",
        ):
            _nonempty_string(execution[field], f"execution.{field}")
        for field in ("slurm_cpus_per_task", "slurm_memory_bytes"):
            _positive_int(execution[field], f"execution.{field}")
    elif any(
        execution[field] is not None
        for field in (
            "slurm_job_id",
            "slurm_array_task_id",
            "account",
            "partition",
            "slurm_cpus_per_task",
            "slurm_memory_bytes",
            "slurm_time_limit",
            "slurm_gpu_request",
        )
    ):
        raise ProvenanceValidationError(
            "local execution cannot contain Slurm identifiers"
        )

    software = _mapping(record["software"], "software")
    _exact_keys(software, _SOFTWARE_KEYS, "software")
    _hash(software["benchmark_git_sha"], _HEX40, "software.benchmark_git_sha")
    _hash(software["package_git_sha"], _HEX40, "software.package_git_sha")
    for field in (
        "python",
        "numpy",
        "scipy",
        "amica",
        "mne",
        "scikit_learn",
        "blas",
        "os",
    ):
        _nonempty_string(software[field], f"software.{field}")
    for field in ("jax", "jaxlib", "cuda", "driver"):
        _nullable_string(software[field], f"software.{field}")
    for field in ("benchmark_worktree_clean", "package_worktree_clean"):
        if type(software[field]) is not bool:
            raise ProvenanceValidationError(
                f"software.{field} must be boolean"
            )
        if not software[field]:
            raise ProvenanceValidationError(
                f"software.{field} must be true for an admissible run"
            )

    hardware = _mapping(record["hardware"], "hardware")
    _exact_keys(hardware, _HARDWARE_KEYS, "hardware")
    _nonempty_string(hardware["cpu_model"], "hardware.cpu_model")
    physical_cpus = _positive_int(
        hardware["physical_cpus"], "hardware.physical_cpus"
    )
    _positive_int(hardware["logical_cpus"], "hardware.logical_cpus")
    if physical_cpus > hardware["logical_cpus"]:
        raise ProvenanceValidationError(
            "hardware.physical_cpus cannot exceed logical_cpus"
        )
    _positive_int(hardware["memory_bytes"], "hardware.memory_bytes")
    _nullable_string(hardware["gpu_model"], "hardware.gpu_model")
    _nullable_string(hardware["gpu_uuid"], "hardware.gpu_uuid")

    runtime = _mapping(record["runtime"], "runtime")
    _exact_keys(runtime, _RUNTIME_KEYS, "runtime")
    if runtime["backend"] not in {"numpy-cpu", "jax-cpu", "jax-gpu"}:
        raise ProvenanceValidationError("runtime.backend is unsupported")
    if runtime["precision"] not in {"float32", "float64"}:
        raise ProvenanceValidationError("runtime.precision is unsupported")
    for field in ("omp_num_threads", "mkl_num_threads", "openblas_num_threads"):
        _positive_int(runtime[field], f"runtime.{field}")
    _nullable_bool(runtime["jax_enable_x64"], "runtime.jax_enable_x64")
    for field in (
        "jax_platform",
        "jax_default_matmul_precision",
        "xla_flags",
        "xla_memory_fraction",
    ):
        _nullable_string(runtime[field], f"runtime.{field}")
    _nullable_bool(runtime["xla_preallocate"], "runtime.xla_preallocate")

    if runtime["backend"].startswith("jax"):
        for field in ("jax", "jaxlib"):
            _nonempty_string(software[field], f"software.{field}")
        if type(runtime["jax_enable_x64"]) is not bool:
            raise ProvenanceValidationError(
                "JAX backends require runtime.jax_enable_x64"
            )
        _nonempty_string(runtime["jax_platform"], "runtime.jax_platform")
        if (
            runtime["precision"] == "float64"
            and runtime["jax_enable_x64"] is not True
        ):
            raise ProvenanceValidationError(
                "float64 JAX execution requires runtime.jax_enable_x64=true"
            )
    else:
        for field in (
            "jax_enable_x64",
            "jax_platform",
            "jax_default_matmul_precision",
            "xla_flags",
            "xla_preallocate",
            "xla_memory_fraction",
        ):
            if runtime[field] is not None:
                raise ProvenanceValidationError(
                    f"numpy-cpu requires runtime.{field}=null"
                )
    if runtime["backend"] == "jax-gpu":
        if runtime["precision"] != "float64":
            raise ProvenanceValidationError(
                "Stage I jax-gpu execution requires float64 precision"
            )
        if runtime["jax_platform"] != "gpu":
            raise ProvenanceValidationError(
                "jax-gpu requires runtime.jax_platform='gpu'"
            )
        for field in ("cuda", "driver"):
            _nonempty_string(software[field], f"software.{field}")
        _nonempty_string(hardware["gpu_model"], "hardware.gpu_model")
        _nonempty_string(hardware["gpu_uuid"], "hardware.gpu_uuid")
        _nonempty_string(
            execution["cuda_visible_devices"],
            "execution.cuda_visible_devices",
        )
        if type(runtime["xla_preallocate"]) is not bool:
            raise ProvenanceValidationError(
                "jax-gpu requires runtime.xla_preallocate"
            )
        if execution["mode"] == "slurm":
            _nonempty_string(
                execution["slurm_gpu_request"],
                "execution.slurm_gpu_request",
            )
    elif runtime["backend"] == "jax-cpu" and runtime["jax_platform"] != "cpu":
        raise ProvenanceValidationError(
            "jax-cpu requires runtime.jax_platform='cpu'"
        )

    input_paths = _validate_artifacts(record["inputs"], "inputs")
    if record["manifest_path"] not in input_paths:
        raise ProvenanceValidationError(
            "manifest_path is absent from provenance inputs"
        )
    manifest_inputs = [
        item
        for item in record["inputs"]
        if item["path"] == record["manifest_path"]
    ]
    if manifest_inputs[0]["sha256"] != record["manifest_sha256"]:
        raise ProvenanceValidationError(
            "manifest input checksum does not match manifest_sha256"
        )
    _validate_artifacts(record["outputs"], "outputs")
