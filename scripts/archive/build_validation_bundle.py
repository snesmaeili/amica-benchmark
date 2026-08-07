#!/usr/bin/env python
"""Build a checksummed, code-and-metrics validation archive.

Raw OpenNeuro data, fitted ICA objects, credentials, and full solver arrays are
excluded.  A preview bundle may be built while evidence is incomplete; the
``--require-complete`` gate is intended for the deposit candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


CODE_PATHS = (
    "scripts/heldout",
    "scripts/parity",
    "scripts/validation",
    "slurm/heldout",
    "slurm/parity",
    "tests/test_heldout_validation.py",
    "tests/test_parity_campaign.py",
    "tests/test_validation_provenance.py",
    "pyproject.toml",
    "environment.yaml",
)

EVIDENCE_FILES = (
    "heldout/heldout_mir_subjects.csv",
    "heldout/heldout_mir_contrasts.csv",
    "heldout/heldout_mir_summary.json",
    "parity/parity_cells.csv",
    "parity/parity_summary.json",
    "environment/environment_provenance.json",
)

FORBIDDEN_SUFFIXES = (".fif", ".set", ".fdt", ".vhdr", ".eeg", ".vmrk")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _copy(source: Path, destination: Path):
    if source.is_dir():
        shutil.copytree(
            source,
            destination,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def build(args):
    benchmark_root = args.benchmark_root.resolve()
    manuscript_root = args.manuscript_root.resolve()
    evidence_root = args.evidence_root.resolve()
    output_root = args.output_dir.resolve()
    bundle_name = f"amica-validation-{args.version}"
    staging = output_root / bundle_name
    if staging.exists():
        raise FileExistsError(
            f"refusing to replace existing bundle directory: {staging}"
        )
    staging.mkdir(parents=True)

    missing = []
    for relative in CODE_PATHS:
        source = benchmark_root / relative
        if source.exists():
            _copy(source, staging / "benchmark" / relative)
        else:
            missing.append(f"benchmark/{relative}")
    for relative in EVIDENCE_FILES:
        source = evidence_root / relative
        if source.exists():
            _copy(source, staging / "results" / relative)
        else:
            missing.append(f"results/{relative}")

    for relative in (
        "zenodo.tex",
        "results_final.tex",
        "backmatter_final.tex",
        "references.bib",
        "figures/src/make_main_figures.py",
    ):
        source = manuscript_root / relative
        if source.exists():
            _copy(source, staging / "manuscript" / relative)
        else:
            missing.append(f"manuscript/{relative}")

    forbidden = [
        str(path.relative_to(staging))
        for path in staging.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES
    ]
    if forbidden:
        raise RuntimeError(f"forbidden raw/fitted data in bundle: {forbidden}")
    if args.require_complete and missing:
        raise RuntimeError(f"deposit bundle is incomplete: {missing}")

    manifest = {
        "schema_version": 1,
        "bundle": bundle_name,
        "version": args.version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if not missing else "preview-incomplete",
        "missing": missing,
        "repositories": {
            "benchmark": _git_revision(benchmark_root),
            "manuscript": _git_revision(manuscript_root),
        },
        "exclusions": [
            "raw OpenNeuro recordings",
            "large fitted ICA objects",
            "full posterior arrays not required for displayed results",
            "credentials and remote URLs containing credentials",
        ],
    }
    (staging / "ARTIFACT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (staging / "README.md").write_text(
        "# AMICA manuscript validation bundle\n\n"
        f"Version: `{args.version}`  \n"
        f"Status: `{manifest['status']}`\n\n"
        "This archive contains benchmark code, job manifests, machine-readable "
        "summary results, provenance, and figure/manuscript sources. It excludes "
        "raw OpenNeuro data and large fitted objects. Run the included unit tests "
        "before regenerating aggregate tables or figures.\n",
        encoding="utf-8",
    )
    files = sorted(path for path in staging.rglob("*") if path.is_file())
    checksums = [
        f"{_sha256(path)}  {path.relative_to(staging).as_posix()}" for path in files
    ]
    (staging / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
    archive = shutil.make_archive(
        str(output_root / bundle_name), "zip", root_dir=output_root, base_dir=bundle_name
    )
    print(json.dumps({"directory": str(staging), "archive": archive, **manifest}))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--manuscript-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())

