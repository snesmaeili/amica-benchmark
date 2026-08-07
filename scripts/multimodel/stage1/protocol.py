"""Predeclared Stage I synthetic multi-model benchmark manifests.

The protocol is deliberately declarative.  It fixes the synthetic conditions,
generating seeds, fit seeds, and fitted model-order sweep without generating
data, fitting AMICA, or submitting compute jobs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


MANIFEST_SCHEMA_VERSION = "amica-multimodel-stage1-manifest-v3"
SYNTHETIC_GENERATOR_SCHEMA_VERSION = (
    "amica-multimodel-synthetic-generator-v1"
)

N_CHANNELS = 16
SAMPLING_RATE_HZ = 250.0
TOTAL_SAMPLES = 150_000
DENSITY_TERMS_K = 3
TRUE_MODEL_ORDER = 3
FITTED_MODEL_ORDERS = tuple(range(1, 9))
DATA_SEEDS = tuple(range(30))
FIT_SEEDS = tuple(range(3))
MEDIAN_DWELL_SECONDS = 10.0
FIT_MAX_ITER = 2_000
FIT_CHUNK_SIZE = 65_536
FIT_BACKEND = "jax-gpu"
FIT_PRECISION = "float64"
FIT_DO_REJECT = False
BOUNDARY_TOLERANCE_SECONDS = 0.5
BOUNDARY_TOLERANCE_SAMPLES = int(
    round(BOUNDARY_TOLERANCE_SECONDS * SAMPLING_RATE_HZ)
)

CORE_CASES = (
    "stationary_fixed_ica",
    "mixing_only",
    "density_only",
    "centre_scale_only",
    "combined_mixing_density",
    "artifact_eog_emg_bursts",
    "continuous_mixing_drift",
)

STRESS_AXIS_LEVELS = {
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


@dataclass(frozen=True)
class SyntheticDesign:
    """One fully declared synthetic data-generating condition."""

    condition_id: str
    campaign: str
    case: str
    n_channels: int
    sampling_rate_hz: float
    total_samples: int
    density_terms_k: int
    true_model_order: Optional[int]
    generator_regime_count: int
    occupancy: str
    regime_probabilities: Tuple[float, ...]
    median_dwell_seconds: float
    snr_db: Optional[float]
    shared_source_fraction: float
    temporal_form: str
    stress_axis: str = "none"
    stress_level: str = "reference"
    is_reference_level: bool = False

    @property
    def sample_ratio(self) -> Optional[float]:
        """Return ``T / (M_true * C^2)``."""
        if self.true_model_order is None:
            return None
        return self.total_samples / (
            self.true_model_order * self.n_channels**2
        )

    def validate(self) -> None:
        """Reject internally inconsistent data-generating declarations."""
        if not self.condition_id:
            raise ValueError("condition_id must be non-empty")
        if self.campaign not in {"core", "one_factor_stress"}:
            raise ValueError("campaign must be core or one_factor_stress")
        if self.case not in CORE_CASES:
            raise ValueError(f"unsupported synthetic case: {self.case}")
        if self.n_channels < 2:
            raise ValueError("n_channels must be at least 2")
        if (
            not math.isfinite(self.sampling_rate_hz)
            or self.sampling_rate_hz <= 0
        ):
            raise ValueError("sampling_rate_hz must be finite and positive")
        if self.total_samples < 1:
            raise ValueError("total_samples must be positive")
        if self.density_terms_k < 1:
            raise ValueError("density_terms_k must be positive")
        if self.generator_regime_count < 1:
            raise ValueError("generator_regime_count must be positive")
        if self.true_model_order is not None and self.true_model_order < 1:
            raise ValueError("true_model_order must be positive or null")
        if self.true_model_order is None and self.case != "continuous_mixing_drift":
            raise ValueError(
                "only continuous mixing drift may omit a discrete true order"
            )
        if len(self.regime_probabilities) != self.generator_regime_count:
            raise ValueError(
                "regime_probabilities length must equal generator_regime_count"
            )
        if any(
            (not math.isfinite(probability)) or probability <= 0
            for probability in self.regime_probabilities
        ):
            raise ValueError("regime probabilities must be finite and positive")
        if not math.isclose(
            sum(self.regime_probabilities),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("regime probabilities must sum to one")
        if self.median_dwell_seconds <= 0:
            raise ValueError("median_dwell_seconds must be positive")
        if self.snr_db is not None and not math.isfinite(self.snr_db):
            raise ValueError("snr_db must be finite or null for noiseless data")
        if not 0.0 <= self.shared_source_fraction <= 1.0:
            raise ValueError("shared_source_fraction must lie in [0, 1]")
        if self.temporal_form not in {
            "stationary",
            "single_concatenated",
            "recurrent_abrupt",
            "gradual",
            "continuous_drift",
        }:
            raise ValueError("unsupported temporal_form")


def _balanced_probabilities(model_order: int) -> Tuple[float, ...]:
    return (1.0 / model_order,) * model_order


REFERENCE_DESIGN = SyntheticDesign(
    condition_id="core_combined_mixing_density",
    campaign="core",
    case="combined_mixing_density",
    n_channels=N_CHANNELS,
    sampling_rate_hz=SAMPLING_RATE_HZ,
    total_samples=TOTAL_SAMPLES,
    density_terms_k=DENSITY_TERMS_K,
    true_model_order=TRUE_MODEL_ORDER,
    generator_regime_count=TRUE_MODEL_ORDER,
    occupancy="balanced",
    regime_probabilities=_balanced_probabilities(TRUE_MODEL_ORDER),
    median_dwell_seconds=MEDIAN_DWELL_SECONDS,
    snr_db=None,
    shared_source_fraction=0.0,
    temporal_form="recurrent_abrupt",
)

def _core_design(case: str) -> SyntheticDesign:
    updates: Dict[str, object] = {}
    if case == "stationary_fixed_ica":
        updates.update(
            true_model_order=1,
            generator_regime_count=1,
            regime_probabilities=(1.0,),
            temporal_form="stationary",
        )
    elif case == "continuous_mixing_drift":
        updates.update(
            true_model_order=None,
            temporal_form="continuous_drift",
        )
    return replace(
        REFERENCE_DESIGN,
        condition_id=f"core_{case}",
        case=case,
        **updates,
    )


CORE_DESIGNS = tuple(_core_design(case) for case in CORE_CASES)

_REFERENCE_CONTROL_VALUES = {
    "true_model_order": TRUE_MODEL_ORDER,
    "total_samples": TOTAL_SAMPLES,
    "occupancy": "balanced",
    "median_dwell_seconds": MEDIAN_DWELL_SECONDS,
    "snr_db": None,
    "shared_source_fraction": 0.0,
    "temporal_form": "recurrent_abrupt",
}
_STRESS_AXIS_TO_CONTROL = {
    "true_model_order": "true_model_order",
    "sample_ratio": "total_samples",
    "occupancy": "occupancy",
    "median_dwell_seconds": "median_dwell_seconds",
    "snr_db": "snr_db",
    "shared_source_fraction": "shared_source_fraction",
    "temporal_form": "temporal_form",
}

MANIFEST_FIELDS = (
    "schema_version",
    "campaign",
    "condition_id",
    "case",
    "stress_axis",
    "stress_level",
    "is_reference_level",
    "data_seed",
    "fit_seed",
    "fit_model_order",
    "fit_max_iter",
    "fit_chunk_size",
    "fit_backend",
    "fit_precision",
    "fit_do_reject",
    "n_channels",
    "sampling_rate_hz",
    "total_samples",
    "density_terms_k",
    "generator_schema_version",
    "true_model_order",
    "generator_regime_count",
    "sample_ratio",
    "occupancy",
    "regime_probabilities",
    "median_dwell_seconds",
    "snr_db",
    "shared_source_fraction",
    "temporal_form",
    "boundary_tolerance_seconds",
    "boundary_tolerance_samples",
    "requires_fit",
    "reference_condition_id",
    "row_sha256",
)


def _format_level(value: object) -> str:
    if value is None:
        return "noiseless"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _stress_design(axis: str, level: object) -> SyntheticDesign:
    updates: Dict[str, object] = {}
    reference_level = False
    if axis == "true_model_order":
        model_order = int(level)
        updates.update(
            true_model_order=model_order,
            generator_regime_count=model_order,
            regime_probabilities=_balanced_probabilities(model_order),
        )
        reference_level = model_order == TRUE_MODEL_ORDER
    elif axis == "sample_ratio":
        ratio = float(level)
        updates["total_samples"] = int(
            round(ratio * TRUE_MODEL_ORDER * N_CHANNELS**2)
        )
        reference_level = updates["total_samples"] == TOTAL_SAMPLES
    elif axis == "occupancy":
        occupancy = str(level)
        probabilities = {
            "balanced": _balanced_probabilities(TRUE_MODEL_ORDER),
            "0.70-0.20-0.10": (0.70, 0.20, 0.10),
            "0.90-0.09-0.01": (0.90, 0.09, 0.01),
        }[occupancy]
        updates.update(
            occupancy=occupancy,
            regime_probabilities=probabilities,
        )
        reference_level = occupancy == "balanced"
    elif axis == "median_dwell_seconds":
        updates["median_dwell_seconds"] = float(level)
        reference_level = float(level) == MEDIAN_DWELL_SECONDS
    elif axis == "snr_db":
        snr_db = None if level == "noiseless" else float(level)
        updates["snr_db"] = snr_db
        reference_level = snr_db is None
    elif axis == "shared_source_fraction":
        updates["shared_source_fraction"] = float(level)
        reference_level = float(level) == 0.0
    elif axis == "temporal_form":
        updates["temporal_form"] = str(level)
        reference_level = level == "recurrent_abrupt"
    else:
        raise ValueError(f"unsupported stress axis: {axis}")

    return replace(
        REFERENCE_DESIGN,
        condition_id=f"stress_{axis}_{_format_level(level)}",
        campaign="one_factor_stress",
        stress_axis=axis,
        stress_level=_format_level(level),
        is_reference_level=reference_level,
        **updates,
    )


def one_factor_stress_designs() -> Tuple[SyntheticDesign, ...]:
    """Return all 23 approved stress-axis levels in stable order."""
    return tuple(
        _stress_design(axis, level)
        for axis, levels in STRESS_AXIS_LEVELS.items()
        for level in levels
    )


def _canonical_row_payload(row: Mapping[str, object]) -> str:
    payload = {
        key: row[key] for key in MANIFEST_FIELDS if key != "row_sha256"
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _row_from_design(
    design: SyntheticDesign,
    data_seed: int,
    fit_seed: int,
    fit_model_order: int,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "campaign": design.campaign,
        "condition_id": design.condition_id,
        "case": design.case,
        "stress_axis": design.stress_axis,
        "stress_level": design.stress_level,
        "is_reference_level": design.is_reference_level,
        "data_seed": data_seed,
        "fit_seed": fit_seed,
        "fit_model_order": fit_model_order,
        "fit_max_iter": FIT_MAX_ITER,
        "fit_chunk_size": FIT_CHUNK_SIZE,
        "fit_backend": FIT_BACKEND,
        "fit_precision": FIT_PRECISION,
        "fit_do_reject": FIT_DO_REJECT,
        "n_channels": design.n_channels,
        "sampling_rate_hz": design.sampling_rate_hz,
        "total_samples": design.total_samples,
        "density_terms_k": design.density_terms_k,
        "generator_schema_version": SYNTHETIC_GENERATOR_SCHEMA_VERSION,
        "true_model_order": design.true_model_order,
        "generator_regime_count": design.generator_regime_count,
        "sample_ratio": design.sample_ratio,
        "occupancy": design.occupancy,
        "regime_probabilities": json.dumps(
            list(design.regime_probabilities),
            separators=(",", ":"),
        ),
        "median_dwell_seconds": design.median_dwell_seconds,
        "snr_db": design.snr_db,
        "shared_source_fraction": design.shared_source_fraction,
        "temporal_form": design.temporal_form,
        "boundary_tolerance_seconds": BOUNDARY_TOLERANCE_SECONDS,
        "boundary_tolerance_samples": BOUNDARY_TOLERANCE_SAMPLES,
        "requires_fit": not design.is_reference_level,
        "reference_condition_id": (
            REFERENCE_DESIGN.condition_id
            if design.is_reference_level
            else None
        ),
    }
    row["row_sha256"] = hashlib.sha256(
        _canonical_row_payload(row).encode("utf-8")
    ).hexdigest()
    return row


def _build_manifest(
    designs: Sequence[SyntheticDesign],
) -> List[Dict[str, object]]:
    rows = [
        _row_from_design(design, data_seed, fit_seed, fit_model_order)
        for design in designs
        for data_seed in DATA_SEEDS
        for fit_seed in FIT_SEEDS
        for fit_model_order in FITTED_MODEL_ORDERS
    ]
    validate_manifest(rows)
    return rows


def build_core_manifest() -> List[Dict[str, object]]:
    """Build the exact seven-case core campaign."""
    return _build_manifest(CORE_DESIGNS)


def build_one_factor_stress_manifest() -> List[Dict[str, object]]:
    """Build the exact seven-axis, 23-level stress campaign."""
    return _build_manifest(one_factor_stress_designs())


def _decode_design(row: Mapping[str, object]) -> SyntheticDesign:
    try:
        probabilities = tuple(json.loads(str(row["regime_probabilities"])))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            "regime_probabilities must contain a JSON array"
        ) from exc
    snr_value = row["snr_db"]
    return SyntheticDesign(
        condition_id=str(row["condition_id"]),
        campaign=str(row["campaign"]),
        case=str(row["case"]),
        n_channels=int(row["n_channels"]),
        sampling_rate_hz=float(row["sampling_rate_hz"]),
        total_samples=int(row["total_samples"]),
        density_terms_k=int(row["density_terms_k"]),
        true_model_order=(
            None
            if row["true_model_order"] is None
            else int(row["true_model_order"])
        ),
        generator_regime_count=int(row["generator_regime_count"]),
        occupancy=str(row["occupancy"]),
        regime_probabilities=probabilities,
        median_dwell_seconds=float(row["median_dwell_seconds"]),
        snr_db=None if snr_value is None else float(snr_value),
        shared_source_fraction=float(row["shared_source_fraction"]),
        temporal_form=str(row["temporal_form"]),
        stress_axis=str(row["stress_axis"]),
        stress_level=str(row["stress_level"]),
        is_reference_level=bool(row["is_reference_level"]),
    )


def _validate_design_against_protocol(design: SyntheticDesign) -> None:
    design.validate()
    if design.n_channels != N_CHANNELS:
        raise ValueError("Stage I requires C=16")
    if design.sampling_rate_hz != SAMPLING_RATE_HZ:
        raise ValueError("Stage I requires sfreq=250 Hz")
    if design.density_terms_k != DENSITY_TERMS_K:
        raise ValueError("Stage I requires K=3 density terms")

    if design.campaign == "core":
        expected = next(
            (
                candidate
                for candidate in CORE_DESIGNS
                if candidate.condition_id == design.condition_id
            ),
            None,
        )
        if design != expected:
            raise ValueError(
                f"{design.condition_id} does not match its predeclared core design"
            )
        return

    if design.case != "combined_mixing_density":
        raise ValueError("stress conditions must use the combined case")
    if design.stress_axis not in STRESS_AXIS_LEVELS:
        raise ValueError("unsupported stress_axis")
    expected = _stress_design(design.stress_axis, design.stress_level)
    if design != expected:
        raise ValueError(
            f"{design.condition_id} does not match its predeclared stress level"
        )

    changed_controls = []
    for control, reference_value in _REFERENCE_CONTROL_VALUES.items():
        if getattr(design, control) != reference_value:
            changed_controls.append(control)
    expected_control = _STRESS_AXIS_TO_CONTROL[design.stress_axis]
    if design.is_reference_level:
        if changed_controls:
            raise ValueError(
                f"reference stress level changes controls: {changed_controls}"
            )
    elif changed_controls != [expected_control]:
        raise ValueError(
            f"{design.condition_id} changes {changed_controls}; "
            f"expected only {expected_control}"
        )


def validate_manifest(rows: Sequence[Mapping[str, object]]) -> None:
    """Validate schema, hashes, declared cells, and the complete seed crossing."""
    if not rows:
        raise ValueError("manifest must contain at least one row")

    identities = set()
    conditions: Dict[str, set] = {}
    campaign = None
    for row in rows:
        if set(row) != set(MANIFEST_FIELDS):
            missing = sorted(set(MANIFEST_FIELDS) - set(row))
            extra = sorted(set(row) - set(MANIFEST_FIELDS))
            raise ValueError(
                f"manifest schema mismatch; missing={missing}, extra={extra}"
            )
        if row["schema_version"] != MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported manifest schema_version")
        if type(row["is_reference_level"]) is not bool:
            raise ValueError("is_reference_level must be boolean")
        if type(row["fit_do_reject"]) is not bool:
            raise ValueError("fit_do_reject must be boolean")
        if type(row["requires_fit"]) is not bool:
            raise ValueError("requires_fit must be boolean")
        design = _decode_design(row)
        _validate_design_against_protocol(design)
        if campaign is None:
            campaign = design.campaign
        elif design.campaign != campaign:
            raise ValueError("one manifest cannot mix core and stress campaigns")

        if type(row["data_seed"]) is not int or row["data_seed"] not in DATA_SEEDS:
            raise ValueError("data_seed is outside the predeclared 30 seeds")
        if type(row["fit_seed"]) is not int or row["fit_seed"] not in FIT_SEEDS:
            raise ValueError("fit_seed is outside the predeclared three seeds")
        if (
            type(row["fit_model_order"]) is not int
            or row["fit_model_order"] not in FITTED_MODEL_ORDERS
        ):
            raise ValueError("fit_model_order must lie in 1..8")
        expected_fit_values = {
            "fit_max_iter": FIT_MAX_ITER,
            "fit_chunk_size": FIT_CHUNK_SIZE,
            "fit_backend": FIT_BACKEND,
            "fit_precision": FIT_PRECISION,
            "fit_do_reject": FIT_DO_REJECT,
            "generator_schema_version": SYNTHETIC_GENERATOR_SCHEMA_VERSION,
            "boundary_tolerance_seconds": BOUNDARY_TOLERANCE_SECONDS,
            "boundary_tolerance_samples": BOUNDARY_TOLERANCE_SAMPLES,
        }
        mismatched_fit_values = {
            field: (row[field], expected)
            for field, expected in expected_fit_values.items()
            if row[field] != expected
        }
        if mismatched_fit_values:
            raise ValueError(
                "row differs from the frozen Stage I fit/evaluation "
                f"configuration: {mismatched_fit_values}"
            )
        expected_reference = (
            REFERENCE_DESIGN.condition_id
            if design.campaign == "one_factor_stress"
            and design.is_reference_level
            else None
        )
        if row["reference_condition_id"] != expected_reference:
            raise ValueError("reference_condition_id is inconsistent")
        expected_requires_fit = expected_reference is None
        if row["requires_fit"] is not expected_requires_fit:
            raise ValueError("requires_fit is inconsistent with reference reuse")
        row_ratio = row["sample_ratio"]
        ratio_matches = (
            row_ratio is None
            if design.sample_ratio is None
            else row_ratio is not None
            and math.isclose(
                float(row_ratio),
                design.sample_ratio,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        if not ratio_matches:
            raise ValueError("sample_ratio is inconsistent with T, M_true, and C")

        expected_hash = hashlib.sha256(
            _canonical_row_payload(row).encode("utf-8")
        ).hexdigest()
        if row["row_sha256"] != expected_hash:
            raise ValueError(f"row hash mismatch for {design.condition_id}")

        identity = (
            design.condition_id,
            row["data_seed"],
            row["fit_seed"],
            row["fit_model_order"],
        )
        if identity in identities:
            raise ValueError(f"duplicate manifest row identity: {identity}")
        identities.add(identity)
        conditions.setdefault(design.condition_id, set()).add(identity[1:])

    expected_cross = {
        (data_seed, fit_seed, fit_model_order)
        for data_seed in DATA_SEEDS
        for fit_seed in FIT_SEEDS
        for fit_model_order in FITTED_MODEL_ORDERS
    }
    expected_conditions = (
        {design.condition_id for design in CORE_DESIGNS}
        if campaign == "core"
        else {design.condition_id for design in one_factor_stress_designs()}
    )
    if set(conditions) != expected_conditions:
        missing = sorted(expected_conditions - set(conditions))
        extra = sorted(set(conditions) - expected_conditions)
        raise ValueError(
            f"condition set mismatch; missing={missing}, extra={extra}"
        )
    for condition_id, observed_cross in conditions.items():
        if observed_cross != expected_cross:
            missing = len(expected_cross - observed_cross)
            extra = len(observed_cross - expected_cross)
            raise ValueError(
                f"{condition_id} has incomplete seed/order crossing; "
                f"missing={missing}, extra={extra}"
            )


def write_manifest_csv(
    rows: Sequence[Mapping[str, object]],
    output_path: Path,
) -> Path:
    """Validate and write a stable, UTF-8 CSV manifest."""
    validate_manifest(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


_CSV_STRING_FIELDS = {
    "schema_version",
    "campaign",
    "condition_id",
    "case",
    "stress_axis",
    "stress_level",
    "fit_backend",
    "fit_precision",
    "generator_schema_version",
    "occupancy",
    "regime_probabilities",
    "temporal_form",
    "row_sha256",
}
_CSV_INTEGER_FIELDS = {
    "data_seed",
    "fit_seed",
    "fit_model_order",
    "fit_max_iter",
    "fit_chunk_size",
    "n_channels",
    "total_samples",
    "density_terms_k",
    "generator_regime_count",
    "boundary_tolerance_samples",
}
_CSV_OPTIONAL_INTEGER_FIELDS = {"true_model_order"}
_CSV_FLOAT_FIELDS = {
    "sampling_rate_hz",
    "median_dwell_seconds",
    "shared_source_fraction",
    "boundary_tolerance_seconds",
}
_CSV_OPTIONAL_FLOAT_FIELDS = {"sample_ratio", "snr_db"}
_CSV_BOOLEAN_FIELDS = {
    "is_reference_level",
    "fit_do_reject",
    "requires_fit",
}
_CSV_OPTIONAL_STRING_FIELDS = {"reference_condition_id"}


def _parse_csv_boolean(value: str, field: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"{field} must be exactly True or False")


def _parse_csv_integer(value: str, field: str) -> int:
    if not value or value.strip() != value:
        raise ValueError(f"{field} must be a canonical integer")
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a canonical integer") from exc
    if str(parsed) != value:
        raise ValueError(f"{field} must be a canonical integer")
    return parsed


def _parse_csv_float(value: str, field: str) -> float:
    if not value or value.strip() != value:
        raise ValueError(f"{field} must be a finite float")
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a finite float") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be a finite float")
    return parsed


def _typed_csv_row(row: Mapping[str, str]) -> Dict[str, object]:
    typed: Dict[str, object] = {}
    for field in MANIFEST_FIELDS:
        value = row[field]
        if field in _CSV_STRING_FIELDS:
            if not value:
                raise ValueError(f"{field} must be non-empty")
            typed[field] = value
        elif field in _CSV_INTEGER_FIELDS:
            typed[field] = _parse_csv_integer(value, field)
        elif field in _CSV_OPTIONAL_INTEGER_FIELDS:
            typed[field] = (
                None if value == "" else _parse_csv_integer(value, field)
            )
        elif field in _CSV_FLOAT_FIELDS:
            typed[field] = _parse_csv_float(value, field)
        elif field in _CSV_OPTIONAL_FLOAT_FIELDS:
            typed[field] = (
                None if value == "" else _parse_csv_float(value, field)
            )
        elif field in _CSV_BOOLEAN_FIELDS:
            typed[field] = _parse_csv_boolean(value, field)
        elif field in _CSV_OPTIONAL_STRING_FIELDS:
            typed[field] = value or None
        else:  # pragma: no cover - MANIFEST_FIELDS and converters are coupled
            raise RuntimeError(f"no CSV converter declared for {field}")
    return typed


def read_manifest_csv(input_path: Path) -> List[Dict[str, object]]:
    """Read, type, hash-check, and validate a complete Stage I manifest."""
    input_path = Path(input_path)
    with input_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
            raise ValueError("manifest CSV header does not match MANIFEST_FIELDS")
        rows = []
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(
                    f"manifest CSV row {row_number} has extra columns"
                )
            try:
                rows.append(_typed_csv_row(row))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid manifest CSV row {row_number}: {exc}"
                ) from exc
    if not rows:
        raise ValueError("manifest CSV contains no data rows")
    validate_manifest(rows)
    return rows
