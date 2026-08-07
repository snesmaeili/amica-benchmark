"""Write one predeclared Stage I synthetic manifest without running fits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from stage1.protocol import (
    MANIFEST_SCHEMA_VERSION,
    build_core_manifest,
    build_one_factor_stress_manifest,
    write_manifest_csv,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign",
        choices=("core", "one_factor_stress"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = (
        build_core_manifest()
        if args.campaign == "core"
        else build_one_factor_stress_manifest()
    )
    path = write_manifest_csv(rows, args.output)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "campaign": args.campaign,
                "rows": len(rows),
                "fit_rows": sum(bool(row["requires_fit"]) for row in rows),
                "reused_reference_rows": sum(
                    not bool(row["requires_fit"]) for row in rows
                ),
                "path": str(path.resolve()),
                "sha256": digest,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
