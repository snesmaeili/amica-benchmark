#!/usr/bin/env python
"""Aggregate complete held-out MIR records with subject-level inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from scripts.heldout.run_heldout_cv import COMPARATORS, EXPECTED_SUBJECTS


def _percentile_interval(values, alpha=0.05):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return np.asarray([np.nan, np.nan])
    return np.quantile(values, (alpha / 2.0, 1.0 - alpha / 2.0))


def _bootstrap(values: np.ndarray, n_resamples: int, seed: int):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_resamples, len(values)))
    samples = values[indices]
    means = samples.mean(axis=1)
    sd = samples.std(axis=1, ddof=1)
    dz = np.divide(means, sd, out=np.full_like(means, np.nan), where=sd > 0)
    return means, dz


def _holm(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted_sorted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted.tolist()


def load_records(input_dir: Path, allow_incomplete: bool = False) -> pd.DataFrame:
    rows = []
    seen = set()
    for path in sorted(input_dir.rglob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("analysis") != "post-hoc guard-banded held-out complete MIR":
            continue
        key = (payload["dataset"], int(payload["subject"]))
        seen.add(key)
        if payload.get("status") != "complete" and not allow_incomplete:
            raise RuntimeError(f"incomplete record: {path}")
        for fold in payload.get("folds", []):
            methods = fold.get("methods", [])
            if not allow_incomplete and {item["method"] for item in methods} != {
                "amica", *COMPARATORS
            }:
                raise RuntimeError(f"missing method in {path}, fold {fold['fold']}")
            for method in methods:
                for bins, score in method["heldout_mir"].items():
                    rows.append(
                        {
                            "dataset": payload["dataset"],
                            "subject": int(payload["subject"]),
                            "fold": int(fold["fold"]),
                            "n_test": int(fold["n_test"]),
                            "method": method["method"],
                            "n_bins": int(bins),
                            "heldout_mir_kbits_s": float(score["kbits_per_sec"]),
                            "sample_index_sha256": score["sample_index_sha256"],
                            "source": str(path),
                        }
                    )
    if not rows:
        raise FileNotFoundError(f"no held-out records under {input_dir}")
    if not allow_incomplete:
        expected = {
            (dataset, subject)
            for dataset, subjects in EXPECTED_SUBJECTS.items()
            for subject in subjects
        }
        missing = sorted(expected - seen)
        if missing:
            raise RuntimeError(f"missing {len(missing)} subject records: {missing[:10]}")
    frame = pd.DataFrame(rows)
    hash_counts = frame.groupby(["dataset", "subject", "fold", "n_bins"])[
        "sample_index_sha256"
    ].nunique()
    if (hash_counts != 1).any():
        raise RuntimeError("methods did not use identical held-out sample indices")
    return frame


def aggregate(frame: pd.DataFrame, n_bootstrap: int, seed: int):
    weighted_rows = []
    for keys, group in frame.groupby(["dataset", "subject", "method", "n_bins"]):
        dataset, subject, method, n_bins = keys
        value = np.average(group.heldout_mir_kbits_s, weights=group.n_test)
        weighted_rows.append(
            {
                "dataset": dataset,
                "subject": int(subject),
                "method": method,
                "n_bins": int(n_bins),
                "heldout_mir_kbits_s": float(value),
            }
        )
    subjects = pd.DataFrame(weighted_rows)
    contrasts = []
    for (dataset, n_bins), group in subjects.groupby(["dataset", "n_bins"]):
        pivot = group.pivot(index="subject", columns="method", values="heldout_mir_kbits_s")
        for comparator in COMPARATORS:
            values = (pivot["amica"] - pivot[comparator]).dropna().to_numpy()
            boot_mean, boot_dz = _bootstrap(values, n_bootstrap, seed + int(n_bins))
            mean_ci = _percentile_interval(boot_mean)
            dz = float(values.mean() / values.std(ddof=1)) if values.std(ddof=1) > 0 else float("nan")
            dz_ci = _percentile_interval(boot_dz[np.isfinite(boot_dz)])
            try:
                wilcoxon_p = float(stats.wilcoxon(values, alternative="two-sided").pvalue)
            except ValueError:
                wilcoxon_p = float("nan")
            contrasts.append(
                {
                    "dataset": dataset,
                    "comparator": comparator,
                    "n_bins": int(n_bins),
                    "n_subjects": int(len(values)),
                    "mean_delta_mir_kbits_s": float(values.mean()),
                    "mean_ci_low": float(mean_ci[0]),
                    "mean_ci_high": float(mean_ci[1]),
                    "dz": dz,
                    "dz_ci_low": float(dz_ci[0]),
                    "dz_ci_high": float(dz_ci[1]),
                    "paired_t_p": float(stats.ttest_1samp(values, 0.0).pvalue),
                    "wilcoxon_p": wilcoxon_p,
                    "positive_n": int(np.sum(values > 0)),
                    "negative_n": int(np.sum(values < 0)),
                }
            )
    contrast_frame = pd.DataFrame(contrasts)
    primary = contrast_frame.n_bins == 100
    contrast_frame.loc[primary, "holm_p_paired_t"] = _holm(
        contrast_frame.loc[primary, "paired_t_p"].tolist()
    )
    return subjects, contrast_frame


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    frame = load_records(args.input_dir, args.allow_incomplete)
    subjects, contrasts = aggregate(frame, args.bootstrap, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects.to_csv(args.output_dir / "heldout_mir_subjects.csv", index=False)
    contrasts.to_csv(args.output_dir / "heldout_mir_contrasts.csv", index=False)
    summary = {
        "schema_version": 1,
        "analysis": "post-hoc guard-banded held-out complete MIR",
        "n_bootstrap": args.bootstrap,
        "bootstrap_seed": args.seed,
        "complete": not args.allow_incomplete,
        "primary_n_bins": 100,
        "contrasts": contrasts.to_dict(orient="records"),
    }
    (args.output_dir / "heldout_mir_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
