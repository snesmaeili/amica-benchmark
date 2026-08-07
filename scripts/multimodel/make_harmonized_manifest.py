"""Create deterministic Slurm manifests for the harmonized multi-model study."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SUBJECTS = {
    "ds004505": tuple(range(1, 26)),
    # Healthy-control cohort uses the dataset's actual three-digit BIDS IDs.
    "ds004504": tuple(range(37, 66)),
    "ds004621": tuple(range(1, 43)),
}
MODEL_ORDERS = tuple(range(1, 11))
FIT_SEEDS = (0, 1, 2)


def rows(smoke: bool, pilot: bool, extra_surrogates: bool):
    for dataset, cohort in SUBJECTS.items():
        selected = cohort[:1] if smoke else (cohort[:3] if pilot else cohort)
        orders = (1, 10) if smoke else ((1, 2, 3, 5, 10) if pilot else MODEL_ORDERS)
        fit_seeds = (0,) if smoke else FIT_SEEDS
        for subject in selected:
            main_surrogate_seed = 100_000 + subject
            for num_models in orders:
                for fit_seed in fit_seeds:
                    yield {
                        "dataset": dataset,
                        "subject": subject,
                        "num_models": num_models,
                        "fit_seed": fit_seed,
                        "surrogate": "none",
                        "surrogate_seed": 0,
                    }
                    yield {
                        "dataset": dataset,
                        "subject": subject,
                        "num_models": num_models,
                        "fit_seed": fit_seed,
                        "surrogate": "phase",
                        "surrogate_seed": main_surrogate_seed,
                    }
        if extra_surrogates and not pilot and not smoke:
            # Prespecified balanced subset: first ten complete subject IDs.
            for subject in cohort[:10]:
                for realization in range(1, 5):
                    surrogate_seed = 100_000 + subject + 10_000 * realization
                    for num_models in MODEL_ORDERS:
                        for fit_seed in FIT_SEEDS:
                            yield {
                                "dataset": dataset,
                                "subject": subject,
                                "num_models": num_models,
                                "fit_seed": fit_seed,
                                "surrogate": "phase",
                                "surrogate_seed": surrogate_seed,
                            }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--extra-surrogates", action="store_true")
    parser.add_argument("--dataset", choices=tuple(SUBJECTS))
    args = parser.parse_args(argv)
    if args.smoke and args.pilot:
        parser.error("--smoke and --pilot are mutually exclusive")
    manifest_rows = list(rows(args.smoke, args.pilot, args.extra_surrogates))
    if args.dataset:
        manifest_rows = [row for row in manifest_rows if row["dataset"] == args.dataset]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"wrote {len(manifest_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
