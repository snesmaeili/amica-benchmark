"""Aggregate harmonized multi-model runs and audit the cMIR decision rule."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _read_runs(input_dir: Path):
    runs = []
    for path in sorted(input_dir.glob("mmc_*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["json_path"] = str(path)
        row["npz_path"] = str(path.with_suffix(".npz"))
        runs.append(row)
    if not runs:
        raise RuntimeError(f"no mmc_*.json files found under {input_dir}")
    return runs


def _group_key(row):
    return (
        row["dataset"],
        int(row["subject"]),
        int(row["num_models"]),
        row["surrogate"],
        int(row["surrogate_seed"]),
    )


def _finite_float(value):
    try:
        return float(value) if np.isfinite(float(value)) else None
    except (TypeError, ValueError):
        return None


def _core_record_errors(
    row,
    expected_package_commit=None,
    expected_manifest_sha256=None,
    expected_runner_sha256=None,
):
    errors = []
    if int(row.get("schema_version", 0)) < 2:
        errors.append("schema_version<2")
    if row.get("package_name") != "amica":
        errors.append("package_name!=amica")
    commit = str(row.get("package_commit", "")).strip()
    if not commit or commit == "unknown":
        errors.append("package_commit_missing")
    if row.get("package_git_dirty") is not False:
        errors.append("package_worktree_not_clean")
    if expected_package_commit is not None and commit != expected_package_commit:
        errors.append("package_commit_mismatch")
    if not str(row.get("runner_sha256", "")).strip():
        errors.append("runner_sha256_missing")
    if not str(row.get("manifest_sha256", "")).strip():
        errors.append("manifest_sha256_missing")
    if (
        expected_manifest_sha256 is not None
        and row.get("manifest_sha256") != expected_manifest_sha256
    ):
        errors.append("manifest_sha256_mismatch")
    if (
        expected_runner_sha256 is not None
        and row.get("runner_sha256") != expected_runner_sha256
    ):
        errors.append("runner_sha256_mismatch")
    if _finite_float(row.get("ll_final_recomputed")) is None:
        errors.append("final_objective_missing_or_nonfinite")
    return errors


def _occupancy_diagnostics(row, *, prior_threshold=0.02):
    priors = np.asarray(row.get("gm", []), dtype=float)
    if priors.size == 0 or np.any(~np.isfinite(priors)):
        return {
            "minimum_model_prior": float("nan"),
            "prior_below_threshold": True,
        }
    return {
        "minimum_model_prior": float(np.min(priors)),
        "prior_below_threshold": bool(np.min(priors) < prior_threshold),
    }


def select_best_seed(
    runs,
    *,
    expected_package_commit=None,
    expected_manifest_sha256=None,
    expected_runner_sha256=None,
    prior_threshold=0.02,
    return_audit=False,
):
    """Select the largest returned-state likelihood without hiding gate failures.

    Provenance or final-objective failures invalidate the aggregate rather than
    silently falling back to a different seed. Low occupancy is retained as a
    sensitivity flag: excluding it before model selection would change the
    estimand and could favour a lower-likelihood seed.
    """
    invalid = []
    for row in runs:
        errors = _core_record_errors(
            row,
            expected_package_commit,
            expected_manifest_sha256,
            expected_runner_sha256,
        )
        if errors:
            invalid.append(
                f"{row.get('json_path', _group_key(row))}: {','.join(errors)}"
            )
    if invalid:
        preview = "\n".join(invalid[:10])
        suffix = "" if len(invalid) <= 10 else f"\n... {len(invalid) - 10} more"
        raise RuntimeError(
            "aggregate contains records that fail the provenance/final-objective "
            f"gate:\n{preview}{suffix}"
        )

    grouped = defaultdict(list)
    for row in runs:
        grouped[_group_key(row)].append(row)
    selected = []
    audit = []
    for key, candidates in grouped.items():
        winner = max(candidates, key=lambda r: float(r["ll_final_recomputed"]))
        winner_seed = int(winner["fit_seed"])
        for candidate in candidates:
            occupancy = _occupancy_diagnostics(
                candidate, prior_threshold=prior_threshold
            )
            audit.append(
                {
                    "dataset": key[0],
                    "subject": key[1],
                    "model_order": key[2],
                    "surrogate": key[3],
                    "surrogate_seed": key[4],
                    "fit_seed": int(candidate["fit_seed"]),
                    "selected": int(candidate["fit_seed"]) == winner_seed,
                    "ll_final_recomputed": float(
                        candidate["ll_final_recomputed"]
                    ),
                    "cmir_finite": bool(candidate.get("cmir_finite", False)),
                    **occupancy,
                }
            )
        winner = dict(winner)
        final_ll = float(winner["ll_final_recomputed"])
        winner["ll_final"] = final_ll
        winner["n_fit_seeds_available"] = len(candidates)
        winner["selected_fit_seed"] = winner_seed
        winner.update(
            _occupancy_diagnostics(winner, prior_threshold=prior_threshold)
        )
        winner["selected_cmir_finite"] = bool(winner.get("cmir_finite", False))
        selected.append(winner)
    return (selected, audit) if return_audit else selected


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_passing_audit(
    path: Path,
    *,
    expected_package_commit: str,
) -> dict:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("gate_status") != "pass":
        raise RuntimeError(
            "aggregation requires a passing pilot audit; "
            f"found {summary.get('gate_status')!r}"
        )
    if summary.get("expected_package_name") != "amica":
        raise RuntimeError("pilot audit did not pin package_name='amica'")
    if summary.get("expected_package_commit") != expected_package_commit:
        raise RuntimeError(
            "pilot audit package commit does not match aggregation commit"
        )
    manifest_hash = str(summary.get("expected_manifest_sha256", "")).strip()
    if not manifest_hash:
        raise RuntimeError("pilot audit does not contain a manifest SHA-256")
    return summary


def _cmir(row, variant="soft_100bins"):
    value = row.get("cmir", {}).get(variant, {}).get("kbits_per_sec")
    return float(value) if value is not None else float("nan")


def _minimum_model_diagnostics(row, variant="soft_100bins"):
    models = row["cmir"][variant]["models"]
    return (
        min(float(m["effective_n"]) for m in models),
        min(float(m["posterior_mass"]) for m in models),
        bool(row["cmir"][variant]["any_low_occupancy"]),
    )


def model_order_rows(selected):
    lookup = {_group_key(row): row for row in selected}
    output = []
    for row in selected:
        baseline_key = (
            row["dataset"],
            int(row["subject"]),
            1,
            row["surrogate"],
            int(row["surrogate_seed"]),
        )
        baseline = lookup.get(baseline_key)
        if baseline is None:
            continue
        min_teff, min_mass, low = _minimum_model_diagnostics(row)
        out = {
            "dataset": row["dataset"],
            "subject": int(row["subject"]),
            "model_order": int(row["num_models"]),
            "surrogate": row["surrogate"],
            "surrogate_seed": int(row["surrogate_seed"]),
            "selected_fit_seed": int(row["fit_seed"]),
            "n_fit_seeds_available": int(row["n_fit_seeds_available"]),
            "cmir_kbits_per_sec": _cmir(row),
            "delta_cmir_kbits_per_sec": _cmir(row) - _cmir(baseline),
            "delta_ll_nats_per_component_sample": float(row["ll_final"])
            - float(baseline["ll_final"]),
            "cmir_50bins_kbits_per_sec": _cmir(row, "soft_50bins"),
            "cmir_200bins_kbits_per_sec": _cmir(row, "soft_200bins"),
            "cmir_hard_kbits_per_sec": _cmir(row, "hard_100bins"),
            "cmir_time_permuted_kbits_per_sec": _cmir(
                row, "time_permuted_100bins"
            ),
            "minimum_effective_n": min_teff,
            "minimum_posterior_mass": min_mass,
            "low_occupancy_flag": low,
            "ll_final": float(row["ll_final"]),
            "n_iter": int(row["n_iter"]),
            "converged": bool(row["converged"]),
            "npz_path": row["npz_path"],
            "json_path": row["json_path"],
        }
        output.append(out)
    return output


def _blocked_splits(labels: np.ndarray, n_splits: int, buffer_windows: int):
    labels = np.asarray(labels)
    valid = labels >= 0
    all_valid = np.flatnonzero(valid)
    by_class = {value: np.flatnonzero(labels == value) for value in (0, 1)}
    if min(len(v) for v in by_class.values()) < n_splits:
        return []
    chunks = {value: np.array_split(indices, n_splits) for value, indices in by_class.items()}
    splits = []
    for fold in range(n_splits):
        test = np.sort(np.concatenate([chunks[0][fold], chunks[1][fold]]))
        blocked = np.zeros(labels.size, dtype=bool)
        for index in test:
            lo = max(0, index - buffer_windows)
            hi = min(labels.size, index + buffer_windows + 1)
            blocked[lo:hi] = True
        train = all_valid[~blocked[all_valid]]
        if len(np.unique(labels[train])) == 2 and len(np.unique(labels[test])) == 2:
            splits.append((train, test))
    return splits


def _score_decoder(features, labels, splits):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    scores = []
    for train, test in splits:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced", solver="liblinear", random_state=0
            ),
        )
        model.fit(features[train], labels[train])
        scores.append(balanced_accuracy_score(labels[test], model.predict(features[test])))
    return float(np.mean(scores)) if scores else float("nan")


def _block_permute(labels, rng, block_size):
    labels = np.asarray(labels)
    valid_indices = np.flatnonzero(labels >= 0)
    values = labels[valid_indices]
    blocks = [values[i : i + block_size] for i in range(0, len(values), block_size)]
    order = rng.permutation(len(blocks))
    permuted = np.concatenate([blocks[i] for i in order])
    result = labels.copy()
    result[valid_indices] = permuted
    return result


def task_decoding_rows(
    selected,
    *,
    n_splits: int,
    buffer_windows: int,
    n_permutations: int,
    permutation_block_windows: int,
):
    output = []
    for row in selected:
        if (
            row["dataset"] != "ds004505"
            or row["surrogate"] != "none"
            or int(row["num_models"]) < 2
        ):
            continue
        with np.load(row["npz_path"], allow_pickle=False) as data:
            features = np.asarray(data["task_features"], dtype=float)
            labels = np.asarray(data["task_labels"], dtype=int)
        splits = _blocked_splits(labels, n_splits, buffer_windows)
        observed = _score_decoder(features, labels, splits)
        rng = np.random.default_rng(
            1_000_000 + 1_000 * int(row["subject"]) + int(row["num_models"])
        )
        null = []
        for _ in range(n_permutations):
            permuted = _block_permute(labels, rng, permutation_block_windows)
            score = _score_decoder(features, permuted, splits)
            if np.isfinite(score):
                null.append(score)
        null = np.asarray(null, dtype=float)
        output.append(
            {
                "dataset": row["dataset"],
                "subject": int(row["subject"]),
                "model_order": int(row["num_models"]),
                "selected_fit_seed": int(row["fit_seed"]),
                "balanced_accuracy": observed,
                "permutation_mean": float(np.mean(null)) if null.size else float("nan"),
                "permutation_p95": float(np.quantile(null, 0.95)) if null.size else float("nan"),
                "permutation_p": float((1 + np.sum(null >= observed)) / (1 + null.size))
                if null.size
                else float("nan"),
                "n_permutations_valid": int(null.size),
                "n_splits_valid": len(splits),
                "n_baseline_windows": int(np.sum(labels == 0)),
                "n_task_windows": int(np.sum(labels == 1)),
                "buffer_windows": buffer_windows,
                "permutation_block_windows": permutation_block_windows,
            }
        )
    return output


def excess_auc_rows(metric_rows):
    by_key = {
        (
            row["dataset"],
            row["subject"],
            row["model_order"],
            row["surrogate"],
            row["surrogate_seed"],
        ): row
        for row in metric_rows
    }
    output = []
    subjects = sorted({(r["dataset"], r["subject"]) for r in metric_rows})
    for dataset, subject in subjects:
        real = {
            m: by_key.get((dataset, subject, m, "none", 0))
            for m in range(2, 11)
        }
        surrogate_seeds = sorted(
            {
                r["surrogate_seed"]
                for r in metric_rows
                if r["dataset"] == dataset
                and r["subject"] == subject
                and r["surrogate"] == "phase"
            }
        )
        if not surrogate_seeds or any(real[m] is None for m in real):
            continue
        excess_by_seed = []
        for seed in surrogate_seeds:
            values = []
            for model_order in range(2, 11):
                surrogate = by_key.get(
                    (dataset, subject, model_order, "phase", seed)
                )
                if surrogate is None:
                    values = []
                    break
                values.append(
                    real[model_order]["delta_cmir_kbits_per_sec"]
                    - surrogate["delta_cmir_kbits_per_sec"]
                )
            if values:
                excess_by_seed.append(values)
        if not excess_by_seed:
            continue
        excess = np.median(np.asarray(excess_by_seed, dtype=float), axis=0)
        output.append(
            {
                "dataset": dataset,
                "subject": subject,
                "auc_excess_mean_m2_m10_kbits_per_sec": float(np.mean(excess)),
                "n_complete_surrogates": len(excess_by_seed),
                **{
                    f"excess_m{model_order}_kbits_per_sec": float(value)
                    for model_order, value in zip(range(2, 11), excess)
                },
            }
        )
    return output


def validation_summary(selected, metric_rows):
    from scipy.stats import spearmanr

    m1_errors = [
        abs(float(row["m1_identity_error_bits_per_sample"]))
        for row in selected
        if int(row["num_models"]) == 1
        and row.get("m1_identity_error_bits_per_sample") is not None
    ]
    real_rows = [r for r in metric_rows if r["surrogate"] == "none" and r["model_order"] > 1]
    correlations = {}
    for alternate in (
        "cmir_50bins_kbits_per_sec",
        "cmir_200bins_kbits_per_sec",
        "cmir_hard_kbits_per_sec",
        "cmir_time_permuted_kbits_per_sec",
    ):
        primary = np.asarray([r["cmir_kbits_per_sec"] for r in real_rows])
        comparison = np.asarray([r[alternate] for r in real_rows])
        correlations[alternate] = float(spearmanr(primary, comparison).statistic)
    delta_cmir = np.asarray([r["delta_cmir_kbits_per_sec"] for r in real_rows])
    delta_ll = np.asarray([r["delta_ll_nats_per_component_sample"] for r in real_rows])
    ll_relation = (
        float(spearmanr(delta_cmir, delta_ll).statistic)
        if len(delta_cmir) >= 3
        else float("nan")
    )
    coverage = defaultdict(lambda: {"subjects": set(), "orders": set()})
    for row in metric_rows:
        key = f"{row['dataset']}:{row['surrogate']}"
        coverage[key]["subjects"].add(row["subject"])
        coverage[key]["orders"].add(row["model_order"])
    return {
        "metric_status": "exploratory; decision rule requires scientific review",
        "maximum_m1_identity_error_bits_per_sample": max(m1_errors) if m1_errors else None,
        "n_m1_identity_checks": len(m1_errors),
        "sensitivity_spearman_correlations": correlations,
        "delta_cmir_delta_ll_spearman": ll_relation,
        "low_occupancy_fit_count": int(sum(r["low_occupancy_flag"] for r in metric_rows)),
        "fit_count": len(metric_rows),
        "coverage": {
            key: {
                "n_subjects": len(value["subjects"]),
                "model_orders": sorted(value["orders"]),
            }
            for key, value in coverage.items()
        },
        "cmir_main_figure_approved": False,
        "cmir_main_figure_approval_note": (
            "Not automated. Inspect identity, bin sensitivity, occupancy, "
            "assignment sensitivity, and likelihood relationship before approval."
        ),
    }


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--buffer-windows", type=int, default=1)
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--permutation-block-windows", type=int, default=6)
    parser.add_argument("--expected-package-commit", required=True)
    parser.add_argument("--audit-summary", type=Path, required=True)
    args = parser.parse_args(argv)

    audit_summary = _require_passing_audit(
        args.audit_summary,
        expected_package_commit=args.expected_package_commit,
    )
    runs = _read_runs(args.input_dir)
    runner_path = Path(__file__).with_name("run_harmonized_multimodel.py")
    selected, selection_audit = select_best_seed(
        runs,
        expected_package_commit=args.expected_package_commit,
        expected_manifest_sha256=audit_summary["expected_manifest_sha256"],
        expected_runner_sha256=_sha256(runner_path),
        return_audit=True,
    )
    metrics = model_order_rows(selected)
    decoding = task_decoding_rows(
        selected,
        n_splits=args.n_splits,
        buffer_windows=args.buffer_windows,
        n_permutations=args.n_permutations,
        permutation_block_windows=args.permutation_block_windows,
    )
    auc = excess_auc_rows(metrics)
    validation = validation_summary(selected, metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "selected_fits.csv", selected)
    _write_csv(args.output_dir / "seed_selection_audit.csv", selection_audit)
    _write_csv(args.output_dir / "model_order_metrics.csv", metrics)
    _write_csv(args.output_dir / "task_decoding.csv", decoding)
    _write_csv(args.output_dir / "excess_auc.csv", auc)
    (args.output_dir / "validation_summary.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
