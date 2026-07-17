#!/usr/bin/env python
"""Aggregate the 18-cell parity campaign and fail on missing or failed cells."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.parity.parity_manifest import manifest_rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    expected = {row["cell_id"] for row in manifest_rows()}
    records = {}
    for path in sorted(args.input_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        cell_id = payload.get("cell", {}).get("cell_id")
        if cell_id in expected:
            records[cell_id] = payload
    missing = sorted(expected - records.keys())
    if missing:
        raise RuntimeError(f"missing parity cells: {missing}")
    rows = []
    for cell_id in sorted(expected):
        payload = records[cell_id]
        rows.append(
            {
                **payload["cell"],
                **payload["metrics"],
                "passed": payload["passed"],
                "failed_checks": ",".join(
                    name for name, passed in payload["checks"].items() if not passed
                ),
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "parity_cells.csv", index=False)
    summary = {
        "schema_version": 1,
        "expected_cells": len(expected),
        "passing_cells": int(frame.passed.sum()),
        "all_passed": bool(frame.passed.all()),
        "failed_cells": frame.loc[~frame.passed, "cell_id"].tolist(),
    }
    (args.output_dir / "parity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary))
    if not summary["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
