#!/usr/bin/env python
"""Audit which environment fields survive in archived benchmark records."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


FIELDS = (
    "hostname",
    "platform",
    "cpu_model",
    "gpu_model",
    "python_version",
    "mne_version",
    "numpy_version",
    "scipy_version",
    "jax_version",
    "jaxlib_version",
    "cuda_version",
    "driver_version",
    "blas",
    "omp_num_threads",
    "mkl_num_threads",
    "openblas_num_threads",
    "xla_flags",
    "xla_preallocate",
)


def _walk(value, path=""):
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            yield child_path, child
            yield from _walk(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")


def audit(root: Path):
    evidence = defaultdict(list)
    files = 0
    unreadable = []
    for path in root.rglob("*.json"):
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            unreadable.append({"path": str(path), "error": str(exc)})
            continue
        files += 1
        for key_path, value in _walk(payload):
            normalized = key_path.rsplit(".", 1)[-1].lower()
            for field in FIELDS:
                if normalized == field and value not in (None, "", [], {}):
                    evidence[field].append(
                        {"path": str(path), "key": key_path, "value": value}
                    )
    return {
        "schema_version": 1,
        "root": str(root.resolve()),
        "json_files_read": files,
        "unreadable": unreadable,
        "fields": {
            field: {
                "available": bool(evidence[field]),
                "n_records": len(evidence[field]),
                "examples": evidence[field][:10],
            }
            for field in FIELDS
        },
        "unresolved_required_fields": [
            field for field in FIELDS if not evidence[field]
        ],
        "interpretation": (
            "A missing field cannot be reconstructed from the archived JSON alone. "
            "A later node probe must be labelled separately from historical run metadata."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = audit(args.input_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "json_files_read": payload["json_files_read"],
        "unresolved_required_fields": payload["unresolved_required_fields"],
        "output": str(args.output),
    }))


if __name__ == "__main__":
    main()
