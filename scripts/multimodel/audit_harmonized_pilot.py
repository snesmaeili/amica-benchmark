"""Audit manifest coverage and numerical integrity for harmonized pilot fits.

This module deliberately separates archive completeness from scientific
approval.  In particular, legacy runs that do not contain a likelihood
recomputed from the returned final parameters cannot pass the final-objective
gate even when every manifest row is present.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


KEY_FIELDS = (
    "dataset",
    "subject",
    "num_models",
    "fit_seed",
    "surrogate",
    "surrogate_seed",
)


def _normalise_key(row: dict) -> tuple:
    return (
        str(row["dataset"]),
        int(row["subject"]),
        int(row["num_models"]),
        int(row["fit_seed"]),
        str(row["surrogate"]),
        int(row["surrogate_seed"]),
    )


def _key_dict(key: tuple) -> dict:
    return dict(zip(KEY_FIELDS, key))


def read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_results(input_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(input_dir.glob("mmc_*.json")):
        nonstandard_constants = []

        def _capture_constant(value):
            nonstandard_constants.append(value)
            return float(value)

        row = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_capture_constant,
        )
        row["_json_path"] = str(path)
        row["_npz_path"] = str(path.with_suffix(".npz"))
        row["_nonstandard_json_constants"] = nonstandard_constants
        rows.append(row)
    return rows


def _minimum_model_diagnostics(row: dict) -> tuple[float, float]:
    models = row.get("cmir", {}).get("soft_100bins", {}).get("models", [])
    if not models:
        return float("nan"), float("nan")
    effective_n = [float(model["effective_n"]) for model in models]
    posterior_mass = [float(model["posterior_mass"]) for model in models]
    return min(effective_n), min(posterior_mass)


def _npz_diagnostics(path: Path) -> dict:
    if not path.is_file():
        return {
            "npz_present": False,
            "ll_history_length": 0,
            "ll_nonfinite_count": None,
            "ll_history_last": None,
        }
    with np.load(path, allow_pickle=False) as archive:
        history = np.asarray(archive["ll_history"], dtype=float)
        # Access every member so NumPy/zipfile checks each compressed stream.
        for name in archive.files:
            np.asarray(archive[name])
    return {
        "npz_present": True,
        "ll_history_length": int(history.size),
        "ll_nonfinite_count": int(np.sum(~np.isfinite(history))),
        "ll_history_last": float(history[-1]) if history.size else None,
    }


def _nonfinite_numeric_paths(value, prefix="") -> list[str]:
    paths = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_nonfinite_numeric_paths(child, child_prefix))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            paths.extend(_nonfinite_numeric_paths(child, f"{prefix}[{index}]"))
    elif isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(float(value)):
            paths.append(prefix)
    return paths


def audit_rows(
    manifest_rows: list[dict],
    result_rows: list[dict],
    *,
    occupancy_kappa_thresholds: tuple[float, ...] = (10.0, 25.0, 50.0),
    expected_package_name: str | None = None,
    expected_package_commit: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> tuple[list[dict], dict]:
    expected_keys = [_normalise_key(row) for row in manifest_rows]
    result_keys = [_normalise_key(row) for row in result_rows]
    expected_counts = Counter(expected_keys)
    result_counts = Counter(result_keys)

    result_lookup: dict[tuple, list[dict]] = defaultdict(list)
    for row in result_rows:
        result_lookup[_normalise_key(row)].append(row)

    audit = []
    for key in sorted(set(expected_keys) | set(result_keys)):
        candidates = result_lookup.get(key, [])
        row = candidates[0] if len(candidates) == 1 else None
        out = {
            **_key_dict(key),
            "expected_count": int(expected_counts[key]),
            "result_count": int(result_counts[key]),
            "coverage_status": (
                "complete"
                if expected_counts[key] == 1 and result_counts[key] == 1
                else "missing"
                if expected_counts[key] == 1 and result_counts[key] == 0
                else "unexpected"
                if expected_counts[key] == 0
                else "duplicate"
            ),
        }
        if row is None:
            audit.append(out)
            continue

        npz = _npz_diagnostics(Path(row["_npz_path"]))
        min_effective_n, min_posterior_mass = _minimum_model_diagnostics(row)
        n_components = int(row.get("n_components", 0))
        min_effective_kappa = (
            min_effective_n / (n_components * n_components)
            if n_components > 0 and np.isfinite(min_effective_n)
            else float("nan")
        )
        gm = np.atleast_1d(np.asarray(row.get("gm", []), dtype=float))
        ll_final = float(row.get("ll_final", float("nan")))
        ll_history_last = npz["ll_history_last"]
        schema_version = int(row.get("schema_version", 0) or 0)
        reported_history_final = row.get("ll_history_final")
        if reported_history_final is None and schema_version < 2:
            # Legacy archives used ``ll_final`` for the pre-update history
            # endpoint. They are still blocked below when the independently
            # recomputed returned-state objective is absent.
            reported_history_final = row.get("ll_final")
        ll_history_metadata_error = (
            abs(float(reported_history_final) - ll_history_last)
            if (
                reported_history_final is not None
                and ll_history_last is not None
                and np.isfinite(float(reported_history_final))
            )
            else float("nan")
        )
        recomputed = row.get("ll_final_recomputed")
        recomputed_available = recomputed is not None and np.isfinite(float(recomputed))
        final_objective_alias_error = (
            abs(ll_final - float(recomputed))
            if recomputed_available and np.isfinite(ll_final)
            else float("nan")
        )
        cmir_nonfinite_paths = _nonfinite_numeric_paths(row.get("cmir", {}), "cmir")
        nonstandard_json_constants = row.get("_nonstandard_json_constants", [])
        package_name = str(row.get("package_name", ""))
        package_commit = str(row.get("package_commit", ""))
        package_git_dirty = row.get("package_git_dirty")
        manifest_sha256 = str(row.get("manifest_sha256", ""))
        prior_finite = bool(gm.size and np.all(np.isfinite(gm)))
        prior_sum_error = (
            float(abs(np.sum(gm) - 1.0)) if prior_finite else float("nan")
        )
        package_name_matches = (
            expected_package_name is None or package_name == expected_package_name
        )
        package_commit_matches = (
            expected_package_commit is None
            or package_commit == expected_package_commit
        )
        manifest_hash_matches = (
            expected_manifest_sha256 is None
            or manifest_sha256 == expected_manifest_sha256
        )

        out.update(
            {
                "json_path": row["_json_path"],
                "npz_path": row["_npz_path"],
                **npz,
                "schema_version": row.get("schema_version"),
                "n_components": n_components,
                "n_samples": int(row.get("n_samples", 0)),
                "n_iter": int(row.get("n_iter", 0)),
                "max_iter": int(row.get("max_iter", 0)),
                "reached_iteration_cap": (
                    int(row.get("n_iter", 0)) >= int(row.get("max_iter", 0))
                ),
                "reported_converged": bool(row.get("converged", False)),
                "ll_final": ll_final,
                "ll_final_recomputed": recomputed,
                "final_ll_recomputed_available": recomputed_available,
                "final_objective_alias_error": final_objective_alias_error,
                "final_objective_alias_matches": (
                    np.isfinite(final_objective_alias_error)
                    and final_objective_alias_error <= 1e-12
                ),
                "ll_history_final_reported": reported_history_final,
                "ll_history_metadata_error": ll_history_metadata_error,
                "ll_history_length_matches_n_iter": (
                    int(npz["ll_history_length"]) == int(row.get("n_iter", 0))
                ),
                "ll_history_matches_metadata": (
                    np.isfinite(ll_history_metadata_error)
                    and ll_history_metadata_error <= 1e-12
                ),
                "nonstandard_json_constant_count": len(
                    nonstandard_json_constants
                ),
                "nonfinite_cmir_field_count": len(cmir_nonfinite_paths),
                "nonfinite_cmir_fields": ";".join(cmir_nonfinite_paths),
                "minimum_effective_n": min_effective_n,
                "minimum_posterior_mass": min_posterior_mass,
                "minimum_effective_kappa": min_effective_kappa,
                "minimum_model_prior": float(np.min(gm)) if gm.size else float("nan"),
                "model_prior_finite": prior_finite,
                "model_prior_sum_error": prior_sum_error,
                "model_prior_normalized": (
                    prior_finite and prior_sum_error <= 1e-10
                ),
                "prior_below_0_02": bool(gm.size and np.min(gm) < 0.02),
                "package_name": package_name,
                "package_commit": package_commit,
                "package_git_dirty": package_git_dirty,
                "package_worktree_clean": package_git_dirty is False,
                "package_commit_known": package_commit not in ("", "unknown"),
                "package_name_matches_expected": package_name_matches,
                "package_commit_matches_expected": package_commit_matches,
                "manifest_sha256": manifest_sha256,
                "manifest_hash_matches_expected": manifest_hash_matches,
                "jax_version": row.get("jax_version", ""),
                "device": row.get("device", ""),
            }
        )
        for threshold in occupancy_kappa_thresholds:
            name = str(threshold).replace(".", "_")
            out[f"effective_kappa_below_{name}"] = bool(
                np.isfinite(min_effective_kappa)
                and min_effective_kappa < threshold
            )
        audit.append(out)

    complete_rows = [row for row in audit if row["coverage_status"] == "complete"]
    expected_group_seeds: dict[tuple, set[int]] = defaultdict(set)
    for key in expected_keys:
        group = (key[0], key[1], key[2], key[4], key[5])
        expected_group_seeds[group].add(int(key[3]))

    group_seeds: dict[tuple, set[int]] = defaultdict(set)
    for row in complete_rows:
        group = (
            row["dataset"],
            row["subject"],
            row["num_models"],
            row["surrogate"],
            row["surrogate_seed"],
        )
        group_seeds[group].add(int(row["fit_seed"]))

    coverage = defaultdict(lambda: {"subjects": set(), "orders": set(), "rows": 0})
    for row in complete_rows:
        key = f"{row['dataset']}:{row['surrogate']}"
        coverage[key]["subjects"].add(int(row["subject"]))
        coverage[key]["orders"].add(int(row["num_models"]))
        coverage[key]["rows"] += 1

    missing_count = sum(row["coverage_status"] == "missing" for row in audit)
    duplicate_count = sum(row["coverage_status"] == "duplicate" for row in audit)
    unexpected_count = sum(row["coverage_status"] == "unexpected" for row in audit)
    missing_npz_count = sum(
        row.get("npz_present") is False for row in complete_rows
    )
    nonfinite_history_count = sum(
        (row.get("ll_nonfinite_count") or 0) > 0 for row in complete_rows
    )
    missing_recomputed_count = sum(
        not row.get("final_ll_recomputed_available", False)
        for row in complete_rows
    )
    final_objective_alias_mismatch_count = sum(
        row.get("final_ll_recomputed_available", False)
        and not row.get("final_objective_alias_matches", False)
        for row in complete_rows
    )
    incomplete_seed_groups = sum(
        group_seeds.get(group, set()) != seeds
        for group, seeds in expected_group_seeds.items()
    )
    low_prior_count = sum(row.get("prior_below_0_02", False) for row in complete_rows)
    low_kappa_25_count = sum(
        row.get("effective_kappa_below_25_0", False) for row in complete_rows
    )
    nonfinite_cmir_count = sum(
        row.get("nonfinite_cmir_field_count", 0) > 0 for row in complete_rows
    )
    nonstandard_json_count = sum(
        row.get("nonstandard_json_constant_count", 0) > 0
        for row in complete_rows
    )
    history_length_mismatch_count = sum(
        not row.get("ll_history_length_matches_n_iter", False)
        for row in complete_rows
    )
    history_metadata_mismatch_count = sum(
        not row.get("ll_history_matches_metadata", False)
        for row in complete_rows
    )
    invalid_prior_count = sum(
        not row.get("model_prior_normalized", False) for row in complete_rows
    )
    invalid_schema_count = sum(
        int(row.get("schema_version", 0) or 0) < 2 for row in complete_rows
    )
    unknown_commit_count = sum(
        not row.get("package_commit_known", False) for row in complete_rows
    )
    package_name_mismatch_count = sum(
        not row.get("package_name_matches_expected", True)
        for row in complete_rows
    )
    package_commit_mismatch_count = sum(
        not row.get("package_commit_matches_expected", True)
        for row in complete_rows
    )
    unclean_package_count = sum(
        not row.get("package_worktree_clean", False) for row in complete_rows
    )
    manifest_hash_mismatch_count = sum(
        not row.get("manifest_hash_matches_expected", True)
        for row in complete_rows
    )

    low_occupancy_groups = 0
    all_seed_low_occupancy_groups = 0
    selected_seed_low_occupancy_groups = 0
    grouped_complete: dict[tuple, list[dict]] = defaultdict(list)
    for row in complete_rows:
        group = (
            row["dataset"],
            row["subject"],
            row["num_models"],
            row["surrogate"],
            row["surrogate_seed"],
        )
        grouped_complete[group].append(row)
    for candidates in grouped_complete.values():
        low = [
            bool(
                candidate.get("prior_below_0_02", False)
                or candidate.get("effective_kappa_below_25_0", False)
            )
            for candidate in candidates
        ]
        if any(low):
            low_occupancy_groups += 1
        if low and all(low):
            all_seed_low_occupancy_groups += 1
        finite = [
            candidate
            for candidate in candidates
            if np.isfinite(
                float(
                    candidate.get("ll_final_recomputed")
                    if candidate.get("final_ll_recomputed_available")
                    else candidate.get("ll_final", float("nan"))
                )
            )
        ]
        if finite:
            winner = max(
                finite,
                key=lambda candidate: float(
                    candidate.get("ll_final_recomputed")
                    if candidate.get("final_ll_recomputed_available")
                    else candidate["ll_final"]
                ),
            )
            if (
                winner.get("prior_below_0_02", False)
                or winner.get("effective_kappa_below_25_0", False)
            ):
                selected_seed_low_occupancy_groups += 1

    hard_failures = (
        missing_count
        + duplicate_count
        + unexpected_count
        + missing_npz_count
        + nonfinite_history_count
        + incomplete_seed_groups
        + nonfinite_cmir_count
        + nonstandard_json_count
        + history_length_mismatch_count
        + history_metadata_mismatch_count
        + invalid_prior_count
        + invalid_schema_count
        + unknown_commit_count
        + package_name_mismatch_count
        + package_commit_mismatch_count
        + unclean_package_count
        + manifest_hash_mismatch_count
        + final_objective_alias_mismatch_count
    )
    gate_status = (
        "fail"
        if hard_failures
        else "blocked_final_objective_verification"
        if missing_recomputed_count
        else "review_low_occupancy"
        if low_prior_count or low_kappa_25_count
        else "pass"
    )
    summary = {
        "schema_version": 1,
        "gate_status": gate_status,
        "expected_manifest_rows": len(expected_keys),
        "result_json_rows": len(result_keys),
        "complete_rows": len(complete_rows),
        "missing_rows": missing_count,
        "duplicate_rows": duplicate_count,
        "unexpected_rows": unexpected_count,
        "missing_npz_rows": missing_npz_count,
        "rows_with_nonfinite_likelihood_history": nonfinite_history_count,
        "rows_with_nonfinite_cmir": nonfinite_cmir_count,
        "rows_with_nonstandard_json_constants": nonstandard_json_count,
        "rows_with_history_length_mismatch": history_length_mismatch_count,
        "rows_with_history_metadata_mismatch": history_metadata_mismatch_count,
        "rows_with_invalid_model_priors": invalid_prior_count,
        "rows_with_legacy_or_invalid_schema": invalid_schema_count,
        "rows_with_unknown_package_commit": unknown_commit_count,
        "rows_with_package_name_mismatch": package_name_mismatch_count,
        "rows_with_package_commit_mismatch": package_commit_mismatch_count,
        "rows_without_clean_package_worktree": unclean_package_count,
        "rows_with_manifest_hash_mismatch": manifest_hash_mismatch_count,
        "rows_without_final_recomputed_likelihood": missing_recomputed_count,
        "rows_with_final_objective_alias_mismatch": (
            final_objective_alias_mismatch_count
        ),
        "incomplete_seed_groups": incomplete_seed_groups,
        "fits_reaching_iteration_cap": sum(
            row.get("reached_iteration_cap", False) for row in complete_rows
        ),
        "fits_reported_converged": sum(
            row.get("reported_converged", False) for row in complete_rows
        ),
        "fits_with_prior_below_0_02": low_prior_count,
        "fits_with_effective_kappa_below_10": sum(
            row.get("effective_kappa_below_10_0", False) for row in complete_rows
        ),
        "fits_with_effective_kappa_below_25": low_kappa_25_count,
        "fits_with_effective_kappa_below_50": sum(
            row.get("effective_kappa_below_50_0", False) for row in complete_rows
        ),
        "seed_groups_with_low_occupancy": low_occupancy_groups,
        "seed_groups_low_occupancy_in_all_seeds": all_seed_low_occupancy_groups,
        "max_likelihood_seed_groups_with_low_occupancy": (
            selected_seed_low_occupancy_groups
        ),
        "expected_package_name": expected_package_name,
        "expected_package_commit": expected_package_commit,
        "expected_manifest_sha256": expected_manifest_sha256,
        "coverage": {
            key: {
                "rows": value["rows"],
                "n_subjects": len(value["subjects"]),
                "model_orders": sorted(value["orders"]),
            }
            for key, value in sorted(coverage.items())
        },
        "interpretation": (
            "Coverage and numerical-integrity gate only. A pass does not approve "
            "conditional MIR, a real-EEG stationarity claim, or Figure 5."
        ),
    }
    return audit, summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-package-name", default="amica")
    parser.add_argument("--expected-package-commit", required=True)
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args(argv)
    manifest_sha256 = (
        args.expected_manifest_sha256
        or hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    )

    audit, summary = audit_rows(
        read_manifest(args.manifest),
        read_results(args.input_dir),
        expected_package_name=args.expected_package_name,
        expected_package_commit=args.expected_package_commit,
        expected_manifest_sha256=manifest_sha256,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "pilot_fit_audit.csv", audit)
    (args.output_dir / "pilot_gate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["gate_status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
