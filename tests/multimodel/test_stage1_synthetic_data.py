from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from scripts.multimodel.stage1.protocol import REFERENCE_DESIGN
from scripts.multimodel.stage1.synthetic_data import (
    _recurrent_labels,
    _run_lengths,
    generate_synthetic_data,
)


def _design(
    case: str = "combined_mixing_density",
    *,
    temporal_form: str = "recurrent_abrupt",
    total_samples: int = 4_000,
    sampling_rate_hz: float = 100.0,
    true_model_order: int | None = 3,
    generator_regime_count: int = 3,
    probabilities: tuple[float, ...] = (0.4, 0.35, 0.25),
    median_dwell_seconds: float = 0.5,
    snr_db: float | None = None,
    shared_source_fraction: float = 0.0,
):
    return replace(
        REFERENCE_DESIGN,
        condition_id=f"test_{case}_{temporal_form}",
        case=case,
        n_channels=4,
        sampling_rate_hz=sampling_rate_hz,
        total_samples=total_samples,
        true_model_order=true_model_order,
        generator_regime_count=generator_regime_count,
        regime_probabilities=probabilities,
        median_dwell_seconds=median_dwell_seconds,
        snr_db=snr_db,
        shared_source_fraction=shared_source_fraction,
        temporal_form=temporal_form,
    )


@pytest.mark.parametrize(
    ("case", "true_order", "regime_count", "probabilities", "temporal_form"),
    [
        ("stationary_fixed_ica", 1, 1, (1.0,), "stationary"),
        ("mixing_only", 3, 3, (0.4, 0.35, 0.25), "recurrent_abrupt"),
        ("density_only", 3, 3, (0.4, 0.35, 0.25), "recurrent_abrupt"),
        ("centre_scale_only", 3, 3, (0.4, 0.35, 0.25), "recurrent_abrupt"),
        (
            "combined_mixing_density",
            3,
            3,
            (0.4, 0.35, 0.25),
            "recurrent_abrupt",
        ),
        (
            "artifact_eog_emg_bursts",
            3,
            3,
            (0.4, 0.35, 0.25),
            "recurrent_abrupt",
        ),
        ("continuous_mixing_drift", None, 3, (1 / 3,) * 3, "continuous_drift"),
    ],
)
def test_all_declared_cases_generate_finite_ground_truth(
    case,
    true_order,
    regime_count,
    probabilities,
    temporal_form,
):
    design = _design(
        case,
        true_model_order=true_order,
        generator_regime_count=regime_count,
        probabilities=probabilities,
        temporal_form=temporal_form,
    )
    generated = generate_synthetic_data(design, data_seed=17)

    assert generated.data.shape == (4, 4_000)
    assert generated.sources.shape == (4, 4_000)
    assert np.isfinite(generated.data).all()
    assert generated.mixing_matrices.shape == (regime_count, 4, 4)
    np.testing.assert_allclose(
        generated.density.sbeta * generated.density.beta,
        1.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        generated.unmixing_matrices @ generated.mixing_matrices,
        np.broadcast_to(np.eye(4), (regime_count, 4, 4)),
        atol=1e-10,
    )
    np.testing.assert_allclose(generated.regime_weights.sum(axis=0), 1.0)
    assert generated.regime_recovery_applicable is (true_order not in {None, 1})


def test_named_streams_make_generation_bitwise_deterministic():
    design = _design()
    first = generate_synthetic_data(design, data_seed=123)
    second = generate_synthetic_data(design, data_seed=123)
    different = generate_synthetic_data(design, data_seed=124)

    assert np.array_equal(first.data, second.data)
    assert np.array_equal(first.sources, second.sources)
    assert np.array_equal(first.regime_labels, second.regime_labels)
    assert not np.array_equal(first.data, different.data)


def test_stationary_fixture_is_fixed_and_sample_order_invariant():
    design = _design(
        "stationary_fixed_ica",
        temporal_form="stationary",
        true_model_order=1,
        generator_regime_count=1,
        probabilities=(1.0,),
    )
    generated = generate_synthetic_data(design, data_seed=5)
    permutation = np.random.default_rng(9).permutation(design.total_samples)

    assert np.all(generated.regime_labels == 0)
    assert generated.transition_boundaries.size == 0
    assert generated.mechanisms == ("stationary_control",)
    np.testing.assert_allclose(
        generated.data,
        generated.mixing_matrices[0] @ generated.sources,
        atol=1e-12,
    )
    sample_indices = np.array([0, 17, design.total_samples - 1])
    np.testing.assert_allclose(
        generated.unmixing_at(sample_indices) @ generated.mixing_at(sample_indices),
        np.broadcast_to(np.eye(4), (sample_indices.size, 4, 4)),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.cov(generated.data[:, permutation]),
        np.cov(generated.data),
        atol=1e-12,
    )


def test_only_declared_mechanisms_change_across_regimes():
    seed = 71
    mixing = generate_synthetic_data(_design("mixing_only"), seed)
    density = generate_synthetic_data(_design("density_only"), seed)
    centre = generate_synthetic_data(_design("centre_scale_only"), seed)
    combined = generate_synthetic_data(_design("combined_mixing_density"), seed)
    artifact = generate_synthetic_data(_design("artifact_eog_emg_bursts"), seed)

    assert not np.allclose(mixing.mixing_matrices[0], mixing.mixing_matrices[1])
    np.testing.assert_array_equal(mixing.density.alpha[0], mixing.density.alpha[1])
    np.testing.assert_array_equal(mixing.density.mu[0], mixing.density.mu[1])

    np.testing.assert_array_equal(
        density.mixing_matrices[0], density.mixing_matrices[1]
    )
    assert not np.allclose(density.density.alpha[0], density.density.alpha[1])
    assert not np.allclose(density.density.rho[0], density.density.rho[1])
    np.testing.assert_array_equal(density.density.mu[0], density.density.mu[1])
    np.testing.assert_array_equal(density.density.beta[0], density.density.beta[1])

    np.testing.assert_array_equal(centre.mixing_matrices[0], centre.mixing_matrices[1])
    np.testing.assert_array_equal(centre.density.alpha[0], centre.density.alpha[1])
    np.testing.assert_array_equal(centre.density.rho[0], centre.density.rho[1])
    assert not np.allclose(centre.density.mu[0], centre.density.mu[1])
    assert not np.allclose(centre.density.beta[0], centre.density.beta[1])

    assert not np.allclose(combined.mixing_matrices[0], combined.mixing_matrices[1])
    assert not np.allclose(combined.density.alpha[0], combined.density.alpha[1])
    assert not np.allclose(combined.density.mu[0], combined.density.mu[1])

    np.testing.assert_array_equal(
        artifact.mixing_matrices[0], artifact.mixing_matrices[1]
    )
    np.testing.assert_array_equal(artifact.density.alpha[0], artifact.density.alpha[1])
    assert artifact.artifact_components is not None


def test_artifact_case_differs_only_by_reconstructable_artifact():
    seed = 31
    artifact_design = _design("artifact_eog_emg_bursts")
    fixed_design = _design("stationary_fixed_ica")
    planted = generate_synthetic_data(artifact_design, seed)
    fixed = generate_synthetic_data(fixed_design, seed)

    assert planted.artifact_components is not None
    np.testing.assert_array_equal(planted.sources, fixed.sources)
    np.testing.assert_array_equal(planted.mixing_matrices, fixed.mixing_matrices)
    np.testing.assert_array_equal(planted.density.alpha, fixed.density.alpha)
    expected = planted.artifact_components.reconstruct(planted.regime_weights)
    np.testing.assert_allclose(planted.data - fixed.data, expected, atol=1e-12)


def test_single_concatenated_has_exact_occupancy_and_one_block_per_regime():
    design = _design(
        temporal_form="single_concatenated",
        total_samples=1_000,
        probabilities=(0.5, 0.3, 0.2),
    )
    generated = generate_synthetic_data(design, data_seed=12)

    assert generated.transition_boundaries.size == 2
    assert generated.dwell_lengths_samples.size == 3
    np.testing.assert_allclose(
        generated.achieved_regime_probabilities,
        design.regime_probabilities,
        atol=1 / design.total_samples,
    )


def test_recurrent_schedule_targets_occupancy_and_median_dwell():
    design = _design(
        total_samples=40_000,
        probabilities=(0.5, 0.3, 0.2),
        median_dwell_seconds=0.25,
    )
    generated = generate_synthetic_data(design, data_seed=44)

    assert generated.transition_boundaries.size > 100
    np.testing.assert_allclose(
        generated.achieved_regime_probabilities,
        design.regime_probabilities,
        atol=0.06,
    )
    assert generated.achieved_median_dwell_seconds == pytest.approx(
        design.median_dwell_seconds,
        abs=0.08,
    )
    np.testing.assert_array_equal(
        generated.transition_boundaries,
        np.flatnonzero(np.diff(generated.regime_labels) != 0) + 1,
    )


def test_core_schedule_has_exact_balance_across_all_declared_seeds():
    for data_seed in range(30):
        labels = _recurrent_labels(REFERENCE_DESIGN, data_seed)
        realised = np.bincount(labels, minlength=3) / labels.size
        np.testing.assert_allclose(
            realised,
            REFERENCE_DESIGN.regime_probabilities,
            atol=1 / labels.size,
        )
        _, dwell_lengths = _run_lengths(labels)
        realised_median = (
            np.median(dwell_lengths) / REFERENCE_DESIGN.sampling_rate_hz
        )
        # The geometric duration is a stochastic target, not a hard dwell
        # constraint; preserve a broad predeclared sanity range.
        assert 0.65 <= realised_median / REFERENCE_DESIGN.median_dwell_seconds <= 1.7


def test_gradual_and_continuous_forms_expose_soft_weights():
    gradual = generate_synthetic_data(_design(temporal_form="gradual"), data_seed=6)
    drift = generate_synthetic_data(
        _design(
            "continuous_mixing_drift",
            temporal_form="continuous_drift",
            true_model_order=None,
            probabilities=(1 / 3,) * 3,
        ),
        data_seed=6,
    )

    assert gradual.regime_labels is not None
    assert np.any((gradual.regime_weights > 0) & (gradual.regime_weights < 1))
    assert drift.regime_labels is None
    assert drift.transition_boundaries.size == 0
    assert not drift.regime_recovery_applicable
    assert np.all(np.sum(drift.regime_weights > 0, axis=0) <= 2)
    assert np.any((drift.regime_weights > 0) & (drift.regime_weights < 1))


def test_shared_sources_retain_maps_and_densities_across_regimes():
    design = _design(shared_source_fraction=0.5)
    generated = generate_synthetic_data(design, data_seed=19)
    shared = generated.shared_source_mask

    assert shared.sum() == 2
    for regime in range(1, design.generator_regime_count):
        np.testing.assert_array_equal(
            generated.mixing_matrices[regime, :, shared],
            generated.mixing_matrices[0, :, shared],
        )
        np.testing.assert_array_equal(
            generated.density.alpha[regime, shared],
            generated.density.alpha[0, shared],
        )
        np.testing.assert_array_equal(
            generated.density.mu[regime, shared],
            generated.density.mu[0, shared],
        )


@pytest.mark.parametrize(
    ("case", "temporal_form"),
    [
        ("combined_mixing_density", "gradual"),
        ("continuous_mixing_drift", "continuous_drift"),
    ],
)
def test_interpolated_mixing_is_nonsingular_for_all_declared_seeds(
    case,
    temporal_form,
):
    sample_indices = np.arange(401, dtype=np.int64)
    for data_seed in range(30):
        generated = generate_synthetic_data(
            _design(
                case,
                temporal_form=temporal_form,
                total_samples=401,
                true_model_order=(
                    None if case == "continuous_mixing_drift" else 3
                ),
            ),
            data_seed=data_seed,
        )
        singular_values = np.linalg.svd(
            generated.mixing_at(sample_indices),
            compute_uv=False,
        )
        assert np.min(singular_values) >= 1.0 - 1e-12


def test_declared_snr_adds_independent_sensor_noise_at_target_power():
    clean_design = _design(snr_db=None)
    noisy_design = _design(snr_db=10.0)
    clean = generate_synthetic_data(clean_design, data_seed=90)
    noisy = generate_synthetic_data(noisy_design, data_seed=90)
    realised_noise_power = float(np.mean((noisy.data - clean.data) ** 2))

    np.testing.assert_array_equal(clean.sources, noisy.sources)
    assert noisy.noise_power == pytest.approx(clean.clean_sensor_power / 10.0)
    assert realised_noise_power == pytest.approx(
        noisy.noise_power,
        rel=0.05,
    )
    assert noisy.mechanisms[-1] == "additive_sensor_noise"


def test_invalid_seed_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        generate_synthetic_data(_design(), data_seed=-1)
