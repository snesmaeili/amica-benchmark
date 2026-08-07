r"""Deterministic Stage I synthetic data generation.

The generator implements the data-generating side of the predeclared
multi-model protocol.  It does not fit AMICA or infer a model order.

For a discrete regime ``m`` the noiseless sensor sample is

.. math::

    x_t = A_m s_t,

where each source follows a finite mixture of generalised Gaussian terms,

.. math::

    p(s_i) = \sum_k \alpha_{mik}
      \frac{\rho_{mik}}{2\beta_{mik}\Gamma(1/\rho_{mik})}
      \exp\left[-\left|\frac{s_i-\mu_{mik}}{\beta_{mik}}\right|^{\rho_{mik}}\right].

Gradual conditions interpolate the declared density parameters and mixing
matrices. Mixing anchors use a common orthogonal base multiplied by
``I + S_m``, where each ``S_m`` is skew-symmetric. Any convex interpolation is
therefore ``A_0(I + \sum_m w_m S_m)`` and remains nonsingular. Continuous
drift interpolates these anchors but deliberately has no finite discrete true
order. All random streams are independently derived from ``data_seed`` and a
stable stream name so that mechanism-specific conditions remain controlled
counterfactuals.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.special import gammaincinv

from .protocol import SyntheticDesign


GENERATOR_SCHEMA_VERSION = "amica-multimodel-synthetic-generator-v1"
_MIX_CHUNK_SAMPLES = 16_384


@dataclass(frozen=True)
class DensityParameters:
    """True generalised-Gaussian mixture parameters.

    Every array has shape ``(generator_regime_count, C, K)``.
    ``beta`` is the generative scale in the density equation above. The AMICA
    implementation stores the inverse scale as ``sbeta``; comparisons must use
    the conversion exposed by the corresponding property.
    """

    alpha: np.ndarray
    mu: np.ndarray
    beta: np.ndarray
    rho: np.ndarray

    @property
    def sbeta(self) -> np.ndarray:
        """Return the AMICA/Fortran inverse-scale convention."""
        return 1.0 / self.beta


@dataclass(frozen=True)
class ArtifactComponents:
    """Compact reconstruction metadata for planted EOG/EMG-like artifacts.

    ``sensor_projection @ (time_courses * amplitudes)`` reconstructs the
    artifact contribution, where ``amplitudes = regime_weights.T @
    regime_amplitudes``.
    """

    names: Tuple[str, str]
    sensor_projection: np.ndarray
    time_courses: np.ndarray
    regime_amplitudes: np.ndarray

    def reconstruct(self, regime_weights: np.ndarray) -> np.ndarray:
        """Reconstruct the planted sensor-space artifact contribution."""
        amplitudes = regime_weights.T @ self.regime_amplitudes
        return self.sensor_projection @ (self.time_courses * amplitudes.T)


@dataclass(frozen=True)
class SyntheticDataset:
    """Generated observations and complete ground-truth metadata."""

    schema_version: str
    design: SyntheticDesign
    data_seed: int
    data: np.ndarray
    sources: np.ndarray
    regime_labels: Optional[np.ndarray]
    regime_weights: np.ndarray
    regime_recovery_applicable: bool
    transition_boundaries: np.ndarray
    dwell_lengths_samples: np.ndarray
    achieved_regime_probabilities: np.ndarray
    achieved_median_dwell_seconds: Optional[float]
    mixing_matrices: np.ndarray
    unmixing_matrices: np.ndarray
    density: DensityParameters
    shared_source_mask: np.ndarray
    artifact_components: Optional[ArtifactComponents]
    clean_sensor_power: float
    noise_power: float
    mechanisms: Tuple[str, ...]

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[1])

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[0])

    def mixing_at(self, sample_indices: np.ndarray) -> np.ndarray:
        """Return effective mixing matrices without storing a ``T x C x C`` cube.

        The result has shape ``(len(sample_indices), C, C)``.  For abrupt
        regimes it selects a planted regime matrix; for gradual or drift
        conditions it reconstructs the declared interpolation.
        """
        indices = np.asarray(sample_indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("sample_indices must be a one-dimensional integer array")
        if np.any(indices < 0) or np.any(indices >= self.n_samples):
            raise IndexError("sample index is outside the generated recording")
        return np.einsum(
            "mn,mij->nij",
            self.regime_weights[:, indices],
            self.mixing_matrices,
            optimize=True,
        )

    def unmixing_at(self, sample_indices: np.ndarray) -> np.ndarray:
        """Return inverses of the effective mixing matrices at selected samples."""
        return np.linalg.inv(self.mixing_at(sample_indices))


def _named_rng(data_seed: int, stream_name: str) -> np.random.Generator:
    """Return a stable independent RNG stream keyed by seed and name."""
    if type(data_seed) is not int or data_seed < 0:
        raise ValueError("data_seed must be a non-negative integer")
    digest = hashlib.sha256(
        f"{GENERATOR_SCHEMA_VERSION}:{data_seed}:{stream_name}".encode("utf-8")
    ).digest()
    words = np.frombuffer(digest[:16], dtype="<u4").astype(np.uint32)
    seed_sequence = np.random.SeedSequence([np.uint32(data_seed), *words])
    return np.random.default_rng(seed_sequence)


def _orthogonal_matrix(rng: np.random.Generator, n_channels: int) -> np.ndarray:
    matrix = rng.standard_normal((n_channels, n_channels))
    q_matrix, r_matrix = np.linalg.qr(matrix)
    signs = np.where(np.diag(r_matrix) < 0.0, -1.0, 1.0)
    return q_matrix * signs[np.newaxis, :]


def _shared_source_mask(
    n_channels: int,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    count = int(round(n_channels * fraction))
    count = min(max(count, 0), n_channels)
    mask = np.zeros(n_channels, dtype=bool)
    if count:
        mask[rng.permutation(n_channels)[:count]] = True
    return mask


def _mixing_matrices(
    design: SyntheticDesign,
    data_seed: int,
    shared_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    regime_count = design.generator_regime_count
    n_channels = design.n_channels
    base_rng = _named_rng(data_seed, "mixing-base")
    base = _orthogonal_matrix(base_rng, n_channels)
    matrices = np.repeat(base[np.newaxis, :, :], regime_count, axis=0)

    changes_mixing = design.case in {
        "mixing_only",
        "combined_mixing_density",
        "continuous_mixing_drift",
    }
    if changes_mixing and regime_count > 1:
        changing_indices = np.flatnonzero(~shared_mask)
        for regime in range(1, regime_count):
            candidate_rng = _named_rng(data_seed, f"mixing-regime-{regime}")
            if changing_indices.size == 0:
                continue
            raw = candidate_rng.standard_normal(
                (changing_indices.size, changing_indices.size)
            )
            skew_subspace = raw - raw.T
            spectral_norm = np.linalg.norm(skew_subspace, ord=2)
            if spectral_norm > 0.0:
                skew_subspace *= 0.75 / spectral_norm
            generator = np.zeros((n_channels, n_channels), dtype=np.float64)
            generator[np.ix_(changing_indices, changing_indices)] = (
                skew_subspace
            )
            # Every convex combination of these anchors is the orthogonal
            # base times I plus a real skew-symmetric matrix, whose singular
            # values are bounded away from zero.
            matrices[regime] = base @ (
                np.eye(n_channels, dtype=np.float64) + generator
            )

    unmixing = np.linalg.inv(matrices)
    return matrices, unmixing


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=-1, keepdims=True)


def _density_parameters(
    design: SyntheticDesign,
    data_seed: int,
    shared_mask: np.ndarray,
) -> DensityParameters:
    regimes = design.generator_regime_count
    channels = design.n_channels
    terms = design.density_terms_k
    base_rng = _named_rng(data_seed, "density-base")

    concentration = np.linspace(2.5, 1.0, terms)
    alpha_base = base_rng.dirichlet(concentration, size=channels)
    rho_grid = np.linspace(1.05, 1.95, terms)
    rho_base = np.clip(
        rho_grid[np.newaxis, :] + base_rng.normal(0.0, 0.025, size=(channels, terms)),
        1.01,
        1.99,
    )
    mu_base = np.zeros((channels, terms), dtype=float)
    beta_base = np.linspace(0.90, 1.10, terms)[np.newaxis, :]
    beta_base = np.repeat(beta_base, channels, axis=0)

    alpha = np.repeat(alpha_base[np.newaxis, :, :], regimes, axis=0)
    mu = np.repeat(mu_base[np.newaxis, :, :], regimes, axis=0)
    beta = np.repeat(beta_base[np.newaxis, :, :], regimes, axis=0)
    rho = np.repeat(rho_base[np.newaxis, :, :], regimes, axis=0)

    changes_shape = design.case in {
        "density_only",
        "combined_mixing_density",
    }
    changes_centre_scale = design.case in {
        "centre_scale_only",
        "combined_mixing_density",
    }
    source_phase = np.linspace(0.0, 2.0 * np.pi, channels, endpoint=False)
    term_phase = np.linspace(0.0, np.pi, terms, endpoint=True)

    for regime in range(regimes):
        regime_phase = 2.0 * np.pi * regime / max(regimes, 1)
        if changes_shape:
            logits = np.log(alpha_base)
            logits += 0.65 * np.sin(
                source_phase[:, np.newaxis] + term_phase[np.newaxis, :] + regime_phase
            )
            alpha[regime] = _softmax(logits)
            rho[regime] = np.clip(
                rho_base
                + 0.32
                * np.sin(
                    0.7 * source_phase[:, np.newaxis]
                    + term_phase[np.newaxis, :]
                    + regime_phase
                ),
                1.01,
                1.99,
            )
        if changes_centre_scale:
            source_shift = 0.40 * np.sin(source_phase + regime_phase)
            mu[regime] = source_shift[:, np.newaxis]
            source_scale = np.exp(0.30 * np.cos(1.3 * source_phase + regime_phase))
            beta[regime] = beta_base * source_scale[:, np.newaxis]

    if shared_mask.any():
        alpha[:, shared_mask] = alpha[0, shared_mask]
        mu[:, shared_mask] = mu[0, shared_mask]
        beta[:, shared_mask] = beta[0, shared_mask]
        rho[:, shared_mask] = rho[0, shared_mask]

    return DensityParameters(alpha=alpha, mu=mu, beta=beta, rho=rho)


def _largest_remainder_counts(
    probabilities: np.ndarray,
    total_samples: int,
) -> np.ndarray:
    raw = probabilities * total_samples
    counts = np.floor(raw).astype(int)
    remaining = total_samples - int(counts.sum())
    if remaining:
        order = np.argsort(-(raw - counts), kind="stable")
        counts[order[:remaining]] += 1
    return counts


def _recurrent_labels(
    design: SyntheticDesign,
    data_seed: int,
) -> np.ndarray:
    """Generate recurrent episodes with exact declared sample occupancy.

    Episode durations are geometric with the declared target median, while a
    remaining-sample quota guarantees that realised occupancy equals the
    largest-remainder allocation of the declared probabilities. The duration
    is therefore a target rather than a hard per-regime constraint, especially
    for severely imbalanced occupancy stress cases.
    """
    probabilities = np.asarray(design.regime_probabilities, dtype=float)
    sample_count = design.total_samples
    dwell_samples = max(
        1, int(round(design.median_dwell_seconds * design.sampling_rate_hz))
    )
    exit_probability = 1.0 - 0.5 ** (1.0 / dwell_samples)
    rng = _named_rng(data_seed, "regime-schedule")
    remaining = _largest_remainder_counts(probabilities, sample_count)
    labels = np.empty(sample_count, dtype=np.int16)
    state = int(rng.choice(probabilities.size, p=remaining / remaining.sum()))
    cursor = 0
    while cursor < sample_count:
        if remaining[state] <= 0:
            available = np.flatnonzero(remaining > 0)
            state = int(rng.choice(available))
        duration = min(
            int(rng.geometric(exit_probability)),
            int(remaining[state]),
        )
        stop = cursor + duration
        labels[cursor:stop] = state
        remaining[state] -= duration
        cursor = stop
        if cursor == sample_count:
            break
        next_weights = remaining.astype(float)
        next_weights[state] = 0.0
        if np.sum(next_weights) == 0.0:
            # Only the current state's quota remains. Continuing the same
            # episode is the only way to satisfy the exact occupancy target.
            continue
        next_weights /= np.sum(next_weights)
        state = int(rng.choice(probabilities.size, p=next_weights))
    if np.any(remaining != 0):
        raise RuntimeError("recurrent schedule did not exhaust occupancy quotas")
    return labels


def _single_concatenated_labels(
    design: SyntheticDesign,
    data_seed: int,
) -> np.ndarray:
    probabilities = np.asarray(design.regime_probabilities, dtype=float)
    counts = _largest_remainder_counts(probabilities, design.total_samples)
    if np.any(counts == 0):
        raise ValueError("single_concatenated requires at least one sample per regime")
    order = _named_rng(data_seed, "regime-schedule").permutation(counts.size)
    return np.concatenate(
        [np.full(counts[regime], regime, dtype=np.int16) for regime in order]
    )


def _smooth_regime_weights(
    labels: np.ndarray,
    regime_count: int,
    design: SyntheticDesign,
) -> np.ndarray:
    weights = np.eye(regime_count, dtype=float)[labels].T
    boundaries = np.flatnonzero(np.diff(labels) != 0) + 1
    if boundaries.size == 0:
        return weights
    target_half_width = max(
        1,
        min(
            int(round(0.5 * design.sampling_rate_hz)),
            int(round(0.20 * design.median_dwell_seconds * design.sampling_rate_hz)),
        ),
    )
    for boundary_index, boundary in enumerate(boundaries):
        previous_boundary = int(boundaries[boundary_index - 1]) if boundary_index else 0
        next_boundary = (
            int(boundaries[boundary_index + 1])
            if boundary_index + 1 < boundaries.size
            else labels.size
        )
        half_width = min(
            target_half_width,
            max(1, (boundary - previous_boundary) // 2),
            max(1, (next_boundary - boundary) // 2),
        )
        start = boundary - half_width
        stop = boundary + half_width
        previous_regime = int(labels[boundary - 1])
        next_regime = int(labels[boundary])
        phase = np.linspace(0.0, 1.0, stop - start, endpoint=False)
        previous_weight = np.cos(0.5 * np.pi * phase) ** 2
        weights[:, start:stop] = 0.0
        weights[previous_regime, start:stop] = previous_weight
        weights[next_regime, start:stop] = 1.0 - previous_weight
    return weights


def _continuous_drift_weights(design: SyntheticDesign) -> np.ndarray:
    regimes = design.generator_regime_count
    samples_per_anchor = max(
        2, int(round(design.median_dwell_seconds * design.sampling_rate_hz))
    )
    position = np.arange(design.total_samples, dtype=float) / samples_per_anchor
    lower = np.floor(position).astype(int) % regimes
    upper = (lower + 1) % regimes
    phase = position - np.floor(position)
    upper_weight = np.sin(0.5 * np.pi * phase) ** 2
    weights = np.zeros((regimes, design.total_samples), dtype=float)
    sample_indices = np.arange(design.total_samples)
    weights[lower, sample_indices] = 1.0 - upper_weight
    weights[upper, sample_indices] += upper_weight
    return weights


def _regime_schedule(
    design: SyntheticDesign,
    data_seed: int,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    regimes = design.generator_regime_count
    if design.temporal_form == "continuous_drift":
        return None, _continuous_drift_weights(design)
    if design.temporal_form == "stationary":
        labels = np.zeros(design.total_samples, dtype=np.int16)
    elif design.temporal_form == "single_concatenated":
        labels = _single_concatenated_labels(design, data_seed)
    else:
        labels = _recurrent_labels(design, data_seed)
    if design.temporal_form == "gradual":
        weights = _smooth_regime_weights(labels, regimes, design)
    else:
        weights = np.eye(regimes, dtype=float)[labels].T
    return labels, weights


def _sample_sources(
    design: SyntheticDesign,
    data_seed: int,
    regime_weights: np.ndarray,
    density: DensityParameters,
) -> np.ndarray:
    channels = design.n_channels
    samples = design.total_samples
    sources = np.empty((channels, samples), dtype=float)
    tiny = np.finfo(float).eps

    for source in range(channels):
        rng = _named_rng(data_seed, f"source-{source}")
        component_uniform = rng.random(samples)
        magnitude_uniform = np.clip(rng.random(samples), tiny, 1.0 - tiny)
        signs = np.where(rng.random(samples) < 0.5, -1.0, 1.0)

        alpha_t = regime_weights.T @ density.alpha[:, source, :]
        cumulative = np.cumsum(alpha_t, axis=1)
        component = np.sum(
            component_uniform[:, np.newaxis] > cumulative,
            axis=1,
        )
        component = np.minimum(component, design.density_terms_k - 1)

        def selected(parameter: np.ndarray) -> np.ndarray:
            by_regime = np.vstack(
                [
                    parameter[regime, source, component]
                    for regime in range(design.generator_regime_count)
                ]
            )
            return np.einsum("mt,mt->t", regime_weights, by_regime, optimize=True)

        mu_t = selected(density.mu)
        beta_t = selected(density.beta)
        rho_t = selected(density.rho)
        gamma_quantile = gammaincinv(1.0 / rho_t, magnitude_uniform)
        sources[source] = mu_t + signs * beta_t * np.power(gamma_quantile, 1.0 / rho_t)
    return sources


def _mix_sources(
    sources: np.ndarray,
    regime_weights: np.ndarray,
    mixing_matrices: np.ndarray,
) -> np.ndarray:
    channels, samples = sources.shape
    data = np.zeros((channels, samples), dtype=float)
    for start in range(0, samples, _MIX_CHUNK_SAMPLES):
        stop = min(samples, start + _MIX_CHUNK_SAMPLES)
        source_chunk = sources[:, start:stop]
        for regime, matrix in enumerate(mixing_matrices):
            data[:, start:stop] += matrix @ (
                source_chunk * regime_weights[regime, start:stop][np.newaxis, :]
            )
    return data


def _standardise(values: np.ndarray) -> np.ndarray:
    centred = values - np.mean(values)
    scale = float(np.std(centred))
    return centred if scale == 0.0 else centred / scale


def _artifact_components(
    design: SyntheticDesign,
    data_seed: int,
    sensor_rms: float,
) -> ArtifactComponents:
    channels = design.n_channels
    samples = design.total_samples
    sfreq = design.sampling_rate_hz
    rng = _named_rng(data_seed, "artifact-drivers")

    event_rate_hz = 0.20
    event_count = max(1, int(round(samples / sfreq * event_rate_hz)))
    impulses = np.zeros(samples, dtype=float)
    locations = rng.integers(0, samples, size=event_count)
    np.add.at(impulses, locations, rng.normal(0.0, 1.0, event_count))
    half_width = min(
        max(2, int(round(0.40 * sfreq))),
        max(0, (samples - 1) // 2),
    )
    kernel_x = np.arange(-half_width, half_width + 1) / sfreq
    kernel = np.exp(-0.5 * (kernel_x / 0.12) ** 2)
    kernel /= kernel.max()
    eog = _standardise(np.convolve(impulses, kernel, mode="same"))

    raw_emg = rng.standard_normal(samples)
    moving_width = min(samples, max(2, int(round(0.04 * sfreq))))
    moving = np.convolve(
        raw_emg,
        np.ones(moving_width) / moving_width,
        mode="same",
    )
    emg = _standardise(raw_emg - moving)
    time_courses = np.vstack([2.0 * sensor_rms * eog, 0.8 * sensor_rms * emg])

    projection = np.zeros((channels, 2), dtype=float)
    frontal_count = max(1, channels // 4)
    temporal_count = max(1, channels // 4)
    projection[:frontal_count, 0] = np.linspace(1.0, 0.4, frontal_count)
    projection[-temporal_count:, 1] = np.linspace(0.4, 1.0, temporal_count)
    projection /= np.linalg.norm(projection, axis=0, keepdims=True)

    amplitudes = np.zeros((design.generator_regime_count, 2), dtype=float)
    for regime in range(1, design.generator_regime_count):
        if regime % 2:
            amplitudes[regime, 0] = 1.0 + 0.15 * (regime - 1)
        else:
            amplitudes[regime, 1] = 1.0 + 0.15 * (regime - 2)

    components = ArtifactComponents(
        names=("EOG-like", "EMG-like"),
        sensor_projection=projection,
        time_courses=time_courses,
        regime_amplitudes=amplitudes,
    )
    if (
        components.sensor_projection.shape != (channels, 2)
        or components.time_courses.shape != (2, samples)
        or components.regime_amplitudes.shape != (design.generator_regime_count, 2)
    ):
        raise RuntimeError("artifact reconstruction shape mismatch")
    return components


def _run_lengths(labels: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if labels is None:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    boundaries = np.flatnonzero(np.diff(labels) != 0).astype(np.int64) + 1
    edges = np.concatenate(([0], boundaries, [labels.size]))
    lengths = np.diff(edges).astype(np.int64)
    return boundaries, lengths


def _mechanisms(design: SyntheticDesign) -> Tuple[str, ...]:
    mapping = {
        "stationary_fixed_ica": ("stationary_control",),
        "mixing_only": ("mixing_matrix",),
        "density_only": ("density_shape_weight",),
        "centre_scale_only": ("density_centre_scale",),
        "combined_mixing_density": (
            "mixing_matrix",
            "density_shape_weight",
            "density_centre_scale",
        ),
        "artifact_eog_emg_bursts": ("sensor_artifact",),
        "continuous_mixing_drift": ("continuous_mixing",),
    }
    mechanisms = list(mapping[design.case])
    if design.snr_db is not None:
        mechanisms.append("additive_sensor_noise")
    return tuple(mechanisms)


def _validate_generated(dataset: SyntheticDataset) -> None:
    design = dataset.design
    channels = design.n_channels
    samples = design.total_samples
    regimes = design.generator_regime_count
    terms = design.density_terms_k
    if dataset.data.shape != (channels, samples):
        raise RuntimeError("generated data shape mismatch")
    if dataset.sources.shape != (channels, samples):
        raise RuntimeError("generated source shape mismatch")
    if dataset.regime_weights.shape != (regimes, samples):
        raise RuntimeError("regime-weight shape mismatch")
    if not np.allclose(dataset.regime_weights.sum(axis=0), 1.0, atol=1e-12):
        raise RuntimeError("regime weights do not sum to one")
    if np.any(dataset.regime_weights < -1e-12):
        raise RuntimeError("regime weights contain negative values")
    if dataset.mixing_matrices.shape != (regimes, channels, channels):
        raise RuntimeError("mixing-matrix shape mismatch")
    if dataset.unmixing_matrices.shape != (regimes, channels, channels):
        raise RuntimeError("unmixing-matrix shape mismatch")
    for parameter in (
        dataset.density.alpha,
        dataset.density.mu,
        dataset.density.beta,
        dataset.density.rho,
    ):
        if parameter.shape != (regimes, channels, terms):
            raise RuntimeError("density-parameter shape mismatch")
    if not np.allclose(dataset.density.alpha.sum(axis=2), 1.0, atol=1e-12):
        raise RuntimeError("density weights do not sum to one")
    if np.any(dataset.density.beta <= 0.0) or np.any(dataset.density.rho <= 0.0):
        raise RuntimeError("density scales and shapes must be positive")
    if not np.isfinite(dataset.data).all() or not np.isfinite(dataset.sources).all():
        raise RuntimeError("generated arrays contain non-finite values")
    identities = dataset.unmixing_matrices @ dataset.mixing_matrices
    expected = np.broadcast_to(np.eye(channels), identities.shape)
    if not np.allclose(identities, expected, atol=1e-10):
        raise RuntimeError("stored mixing and unmixing matrices disagree")
    if dataset.regime_labels is None:
        if dataset.transition_boundaries.size:
            raise RuntimeError("continuous drift cannot have hard boundaries")
    elif dataset.regime_labels.shape != (samples,):
        raise RuntimeError("regime-label shape mismatch")


def generate_synthetic_data(
    design: SyntheticDesign,
    data_seed: int,
) -> SyntheticDataset:
    """Generate one deterministic Stage I synthetic dataset.

    The returned arrays use channel-by-sample orientation.  No inference,
    fitting, or result selection is performed.
    """
    design.validate()
    if type(data_seed) is not int or data_seed < 0:
        raise ValueError("data_seed must be a non-negative integer")

    shared_rng = _named_rng(data_seed, "shared-source-mask")
    shared_mask = _shared_source_mask(
        design.n_channels,
        design.shared_source_fraction,
        shared_rng,
    )
    mixing, unmixing = _mixing_matrices(design, data_seed, shared_mask)
    density = _density_parameters(design, data_seed, shared_mask)
    labels, regime_weights = _regime_schedule(design, data_seed)
    sources = _sample_sources(design, data_seed, regime_weights, density)
    data = _mix_sources(sources, regime_weights, mixing)

    artifact_components = None
    if design.case == "artifact_eog_emg_bursts":
        sensor_rms = math.sqrt(float(np.mean(data**2)))
        artifact_components = _artifact_components(
            design,
            data_seed,
            sensor_rms,
        )
        for start in range(0, design.total_samples, _MIX_CHUNK_SAMPLES):
            stop = min(design.total_samples, start + _MIX_CHUNK_SAMPLES)
            amplitudes = (
                regime_weights[:, start:stop].T @ artifact_components.regime_amplitudes
            )
            data[:, start:stop] += artifact_components.sensor_projection @ (
                artifact_components.time_courses[:, start:stop] * amplitudes.T
            )

    clean_sensor_power = float(np.mean(data**2))
    noise_power = 0.0
    if design.snr_db is not None:
        noise_power = clean_sensor_power / (10.0 ** (design.snr_db / 10.0))
        noise_scale = math.sqrt(noise_power)
        noise_rng = _named_rng(data_seed, "sensor-noise")
        for channel in range(design.n_channels):
            data[channel] += noise_scale * noise_rng.standard_normal(
                design.total_samples
            )

    boundaries, dwell_lengths = _run_lengths(labels)
    if labels is None:
        achieved_probabilities = np.mean(regime_weights, axis=1)
        achieved_median_dwell_seconds = None
    else:
        achieved_probabilities = np.bincount(
            labels,
            minlength=design.generator_regime_count,
        ).astype(float)
        achieved_probabilities /= labels.size
        achieved_median_dwell_seconds = (
            float(np.median(dwell_lengths)) / design.sampling_rate_hz
        )

    true_order = design.true_model_order
    dataset = SyntheticDataset(
        schema_version=GENERATOR_SCHEMA_VERSION,
        design=design,
        data_seed=data_seed,
        data=data,
        sources=sources,
        regime_labels=labels,
        regime_weights=regime_weights,
        regime_recovery_applicable=(
            true_order is not None
            and true_order > 1
            and design.case != "stationary_fixed_ica"
        ),
        transition_boundaries=boundaries,
        dwell_lengths_samples=dwell_lengths,
        achieved_regime_probabilities=achieved_probabilities,
        achieved_median_dwell_seconds=achieved_median_dwell_seconds,
        mixing_matrices=mixing,
        unmixing_matrices=unmixing,
        density=density,
        shared_source_mask=shared_mask,
        artifact_components=artifact_components,
        clean_sensor_power=clean_sensor_power,
        noise_power=noise_power,
        mechanisms=_mechanisms(design),
    )
    _validate_generated(dataset)
    return dataset
