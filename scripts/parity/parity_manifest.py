#!/usr/bin/env python
"""Create the prespecified 18-cell Python/Fortran parity manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def manifest_rows():
    for num_mix in (1, 3, 5):
        for do_newton in (True, False):
            for do_reject in (False, True):
                yield {
                    "cell_id": (
                        f"single_K{num_mix}_"
                        f"{'newton' if do_newton else 'natgrad'}_"
                        f"{'reject' if do_reject else 'no-reject'}"
                    ),
                    "num_models": 1,
                    "num_mix": num_mix,
                    "do_newton": int(do_newton),
                    "do_reject": int(do_reject),
                    "seed": 42,
                    "max_iter": 200,
                }
    for num_models in (2, 3):
        for seed in (0, 1, 2):
            yield {
                "cell_id": f"multi_M{num_models}_K3_newton_seed{seed}",
                "num_models": num_models,
                "num_mix": 3,
                "do_newton": 1,
                "do_reject": 0,
                "seed": seed,
                "max_iter": 300,
            }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = list(manifest_rows())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} cells to {args.output}")


if __name__ == "__main__":
    main()
