from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest


MODULE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "multimodel"
sys.path.insert(0, str(MODULE_DIR))

from stage1.protocol import (  # noqa: E402
    BOUNDARY_TOLERANCE_SAMPLES,
    BOUNDARY_TOLERANCE_SECONDS,
    CORE_CASES,
    CORE_DESIGNS,
    DATA_SEEDS,
    DENSITY_TERMS_K,
    FIT_BACKEND,
    FIT_CHUNK_SIZE,
    FIT_DO_REJECT,
    FIT_MAX_ITER,
    FITTED_MODEL_ORDERS,
    FIT_PRECISION,
    FIT_SEEDS,
    MANIFEST_FIELDS,
    MEDIAN_DWELL_SECONDS,
    N_CHANNELS,
    SAMPLING_RATE_HZ,
    STRESS_AXIS_LEVELS,
    SYNTHETIC_GENERATOR_SCHEMA_VERSION,
    TOTAL_SAMPLES,
    TRUE_MODEL_ORDER,
    build_core_manifest,
    build_one_factor_stress_manifest,
    one_factor_stress_designs,
    read_manifest_csv,
    validate_manifest,
    write_manifest_csv,
)
from stage1.synthetic_data import GENERATOR_SCHEMA_VERSION  # noqa: E402


ROWS_PER_CONDITION = (
    len(DATA_SEEDS) * len(FIT_SEEDS) * len(FITTED_MODEL_ORDERS)
)
CORE_ROW_COUNT = len(CORE_CASES) * ROWS_PER_CONDITION
STRESS_LEVEL_COUNT = sum(len(levels) for levels in STRESS_AXIS_LEVELS.values())
STRESS_ROW_COUNT = STRESS_LEVEL_COUNT * ROWS_PER_CONDITION


@pytest.fixture(scope="module")
def core_rows():
    return build_core_manifest()


@pytest.fixture(scope="module")
def stress_rows():
    return build_one_factor_stress_manifest()


def test_core_matches_approved_stage1_constants_and_cases(core_rows):
    assert N_CHANNELS == 16
    assert SAMPLING_RATE_HZ == 250.0
    assert TOTAL_SAMPLES == 150_000
    assert DENSITY_TERMS_K == 3
    assert TRUE_MODEL_ORDER == 3
    assert FITTED_MODEL_ORDERS == tuple(range(1, 9))
    assert DATA_SEEDS == tuple(range(30))
    assert FIT_SEEDS == (0, 1, 2)
    assert MEDIAN_DWELL_SECONDS == 10.0
    assert FIT_MAX_ITER == 2_000
    assert FIT_CHUNK_SIZE == 65_536
    assert FIT_BACKEND == "jax-gpu"
    assert FIT_PRECISION == "float64"
    assert FIT_DO_REJECT is False
    assert BOUNDARY_TOLERANCE_SECONDS == 0.5
    assert BOUNDARY_TOLERANCE_SAMPLES == 125
    assert SYNTHETIC_GENERATOR_SCHEMA_VERSION == GENERATOR_SCHEMA_VERSION
    assert tuple(design.case for design in CORE_DESIGNS) == CORE_CASES
    assert len(core_rows) == CORE_ROW_COUNT == 5_040

    for row in core_rows:
        assert row["n_channels"] == 16
        assert row["sampling_rate_hz"] == 250.0
        assert row["total_samples"] == 150_000
        assert row["density_terms_k"] == 3
        assert row["median_dwell_seconds"] == 10.0
        assert row["fit_max_iter"] == 2_000
        assert row["fit_chunk_size"] == 65_536
        assert row["fit_backend"] == "jax-gpu"
        assert row["fit_precision"] == "float64"
        assert row["fit_do_reject"] is False
        assert row["generator_schema_version"] == GENERATOR_SCHEMA_VERSION
        assert row["boundary_tolerance_seconds"] == 0.5
        assert row["boundary_tolerance_samples"] == 125
        assert row["requires_fit"] is True
        assert row["reference_condition_id"] is None
        if row["case"] == "stationary_fixed_ica":
            assert row["true_model_order"] == 1
            assert row["generator_regime_count"] == 1
            assert row["temporal_form"] == "stationary"
        elif row["case"] == "continuous_mixing_drift":
            assert row["true_model_order"] is None
            assert row["generator_regime_count"] == 3
            assert row["sample_ratio"] is None
            assert row["temporal_form"] == "continuous_drift"
        else:
            assert row["true_model_order"] == 3
            assert row["generator_regime_count"] == 3
            assert row["temporal_form"] == "recurrent_abrupt"


def test_core_crosses_all_generating_fit_seeds_and_orders(core_rows):
    expected = {
        (data_seed, fit_seed, fit_model_order)
        for data_seed in range(30)
        for fit_seed in range(3)
        for fit_model_order in range(1, 9)
    }
    for design in CORE_DESIGNS:
        observed = {
            (row["data_seed"], row["fit_seed"], row["fit_model_order"])
            for row in core_rows
            if row["condition_id"] == design.condition_id
        }
        assert observed == expected
        assert len(observed) == ROWS_PER_CONDITION == 720


def test_manifest_generation_and_row_hashes_are_deterministic(core_rows):
    repeated = build_core_manifest()
    assert repeated == core_rows


def test_stress_axes_levels_and_row_count_are_exact(stress_rows):
    assert STRESS_AXIS_LEVELS == {
        "true_model_order": (1, 2, 3, 5),
        "sample_ratio": (10.0, 25.0, 50.0, 100.0),
        "occupancy": ("balanced", "0.70-0.20-0.10", "0.90-0.09-0.01"),
        "median_dwell_seconds": (0.5, 5.0, 30.0),
        "snr_db": ("noiseless", 10.0, 0.0),
        "shared_source_fraction": (0.0, 0.5, 0.8),
        "temporal_form": (
            "single_concatenated",
            "recurrent_abrupt",
            "gradual",
        ),
    }
    assert len(one_factor_stress_designs()) == STRESS_LEVEL_COUNT == 23
    assert len(stress_rows) == STRESS_ROW_COUNT == 16_560
    assert {row["case"] for row in stress_rows} == {
        "combined_mixing_density"
    }
    counts = {}
    for row in stress_rows:
        counts[row["condition_id"]] = counts.get(row["condition_id"], 0) + 1
    assert set(counts.values()) == {ROWS_PER_CONDITION}


def test_sample_ratio_axis_sets_exact_total_sample_counts(stress_rows):
    expected_samples = {
        "10": 7_680,
        "25": 19_200,
        "50": 38_400,
        "100": 76_800,
    }
    observed = {}
    for row in stress_rows:
        if row["stress_axis"] == "sample_ratio":
            observed[row["stress_level"]] = row["total_samples"]
    assert observed == expected_samples


def test_reference_levels_are_explicit_but_change_no_control():
    reference_levels = {
        design.stress_axis: design.stress_level
        for design in one_factor_stress_designs()
        if design.is_reference_level
    }
    assert reference_levels == {
        "true_model_order": "3",
        "occupancy": "balanced",
        "snr_db": "noiseless",
        "shared_source_fraction": "0",
        "temporal_form": "recurrent_abrupt",
    }


def test_stress_reference_rows_reuse_core_instead_of_requesting_duplicate_fits(
    stress_rows,
):
    aliases = [row for row in stress_rows if not row["requires_fit"]]
    fits = [row for row in stress_rows if row["requires_fit"]]
    assert len(aliases) == 5 * ROWS_PER_CONDITION == 3_600
    assert len(fits) == STRESS_ROW_COUNT - len(aliases) == 12_960
    assert {
        row["reference_condition_id"] for row in aliases
    } == {"core_combined_mixing_density"}
    assert all(row["is_reference_level"] for row in aliases)
    assert all(row["reference_condition_id"] is None for row in fits)


def test_manifest_hash_detects_mutation(core_rows):
    mutated = [dict(row) for row in core_rows]
    mutated[0]["fit_seed"] = 2
    with pytest.raises(ValueError, match="row hash mismatch|duplicate"):
        validate_manifest(mutated)


def test_incomplete_seed_crossing_is_rejected(core_rows):
    with pytest.raises(ValueError, match="incomplete seed/order crossing"):
        validate_manifest(core_rows[:-1])


def test_manifest_writer_preserves_declared_column_order(tmp_path, core_rows):
    path = write_manifest_csv(core_rows, tmp_path / "stage1_core.csv")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        first = next(reader)
        assert tuple(reader.fieldnames or ()) == MANIFEST_FIELDS
    assert first["row_sha256"] == core_rows[0]["row_sha256"]
    assert read_manifest_csv(path) == core_rows


def _rewrite_csv(path, mutate):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    mutate(rows)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("is_reference_level", "false", "exactly True or False"),
        ("true_model_order", "null", "canonical integer"),
        ("snr_db", "NaN", "finite float"),
    ],
)
def test_manifest_reader_rejects_malformed_boolean_and_null_encodings(
    tmp_path,
    core_rows,
    field,
    value,
    match,
):
    path = write_manifest_csv(core_rows, tmp_path / f"malformed-{field}.csv")
    _rewrite_csv(path, lambda rows: rows[0].__setitem__(field, value))
    with pytest.raises(ValueError, match=match):
        read_manifest_csv(path)


def test_manifest_reader_recomputes_and_rejects_row_hash(tmp_path, core_rows):
    path = write_manifest_csv(core_rows, tmp_path / "bad-hash.csv")
    _rewrite_csv(
        path,
        lambda rows: rows[0].__setitem__("row_sha256", "f" * 64),
    )
    with pytest.raises(ValueError, match="row hash mismatch"):
        read_manifest_csv(path)
