"""Posterior-weighted conditional MIR for exploratory multi-model audits.

This module intentionally lives in the benchmark repository.  It does not
change AMICA's fitting objective or public package API.  Standard complete
MIR assumes one global square transform.  Multi-model AMICA instead provides
one transform per model and a posterior responsibility for every sample.  The
quantity implemented here is therefore a newly defined conditional diagnostic,
not the standard MIR used in the single-model benchmark.

For model ``m`` with posterior responsibilities ``gamma[m, t]``::

    cMIR_m = sum_i h_w(X_i) - sum_i h_w(Y_i_m) + log2(abs(det(W_m)))
    cMIR   = sum_m pi_m * cMIR_m

where ``w`` is the normalized posterior weight within a model and
``pi_m = mean_t gamma[m, t]``.  Entropies are weighted differential histogram
estimates and include the bin-width term required for scale invariance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np


AssignmentMode = Literal["soft", "hard", "time_permuted"]


@dataclass(frozen=True)
class ModelConditionalMIR:
    """One model's contribution to the conditional MIR diagnostic."""

    model: int
    occupancy: float
    posterior_mass: float
    effective_n: float
    h_input_bits: float
    h_sources_bits: float
    log2_abs_det_w: float
    bits_per_sample: float
    kbits_per_sec: float
    low_occupancy: bool


@dataclass(frozen=True)
class ConditionalMIR:
    """Posterior-weighted conditional MIR and its audit metadata."""

    bits_per_sample: float
    kbits_per_sec: float
    n_models: int
    n_components: int
    n_samples: int
    n_bins: int
    clip_sd: float | None
    assignment: AssignmentMode
    min_effective_n: float
    min_posterior_mass: float
    any_low_occupancy: bool
    models: tuple[ModelConditionalMIR, ...]
    note: str = (
        "Exploratory posterior-weighted conditional MIR; not standard global MIR."
    )

    def to_dict(self) -> dict:
        return asdict(self)


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size for non-negative sample weights."""

    w = np.asarray(weights, dtype=float).ravel()
    w = w[np.isfinite(w) & (w >= 0)]
    denom = float(np.sum(w * w))
    if denom <= 0:
        return 0.0
    total = float(np.sum(w))
    return total * total / denom


def weighted_entropy_histogram(
    x: np.ndarray,
    weights: np.ndarray,
    *,
    n_bins: int = 100,
    clip_sd: float | None = 5.0,
) -> float:
    """Weighted one-dimensional differential entropy in bits.

    The histogram probabilities are weighted by ``weights``.  The returned
    entropy includes ``log2(bin_width)`` so a reciprocal source/unmixing-row
    rescaling cancels analytically in complete MIR.
    """

    x = np.asarray(x, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    if x.shape != w.shape:
        raise ValueError(f"x and weights must have the same shape, got {x.shape} and {w.shape}")
    if int(n_bins) < 2:
        raise ValueError("n_bins must be at least 2")

    valid = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    x = x[valid]
    w = w[valid]
    total_w = float(np.sum(w))
    if x.size < 2 or total_w <= 0:
        return float("nan")
    w = w / total_w

    mean = float(np.sum(w * x))
    variance = float(np.sum(w * (x - mean) ** 2))
    sd = float(np.sqrt(max(variance, 0.0)))
    if not np.isfinite(sd) or sd <= 0:
        return float("nan")

    if clip_sd is None:
        lo, hi = float(np.min(x)), float(np.max(x))
    else:
        lo, hi = mean - float(clip_sd) * sd, mean + float(clip_sd) * sd
        x = np.clip(x, lo, hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float("nan")

    edges = np.linspace(lo, hi, int(n_bins) + 1)
    counts, _ = np.histogram(x, bins=edges, weights=w)
    counts = np.asarray(counts, dtype=float)
    mass = float(np.sum(counts))
    if mass <= 0:
        return float("nan")
    p = counts / mass
    nz = p > 0
    h_discrete = -float(np.sum(p[nz] * np.log2(p[nz])))
    bin_width = (hi - lo) / float(n_bins)
    return h_discrete + float(np.log2(bin_width))


def assignment_weights(
    posteriors: np.ndarray,
    mode: AssignmentMode,
    *,
    random_state: int = 0,
) -> np.ndarray:
    """Return soft, hard, or occupancy-preserving time-permuted assignments."""

    gamma = np.asarray(posteriors, dtype=float)
    if gamma.ndim != 2:
        raise ValueError("posteriors must have shape (n_models, n_samples)")
    if np.any(~np.isfinite(gamma)) or np.any(gamma < 0):
        raise ValueError("posteriors must be finite and non-negative")
    colsum = gamma.sum(axis=0, keepdims=True)
    if np.any(colsum <= 0):
        raise ValueError("every sample must have positive total posterior mass")
    gamma = gamma / colsum

    if mode == "soft":
        return gamma
    if mode == "hard":
        hard = np.zeros_like(gamma)
        hard[np.argmax(gamma, axis=0), np.arange(gamma.shape[1])] = 1.0
        return hard
    if mode == "time_permuted":
        rng = np.random.default_rng(random_state)
        return gamma[:, rng.permutation(gamma.shape[1])]
    raise ValueError(f"unknown assignment mode: {mode}")


def conditional_mir(
    x_input: np.ndarray,
    y_models: np.ndarray,
    w_models: np.ndarray,
    posteriors: np.ndarray,
    sfreq_hz: float,
    *,
    n_bins: int = 100,
    clip_sd: float | None = 5.0,
    assignment: AssignmentMode = "soft",
    assignment_random_state: int = 0,
    min_effective_n: float = 2_000.0,
    min_posterior_mass: float = 2_000.0,
) -> ConditionalMIR:
    """Compute posterior-weighted conditional MIR from aligned model arrays.

    Parameters
    ----------
    x_input
        Input data in the retained square analysis space, shape ``(N, T)``.
    y_models
        Model-specific source representations, shape ``(M, N, T)``.
    w_models
        Effective transforms satisfying ``Y_m = W_m X + translation``, shape
        ``(M, N, N)``.  Translation does not affect entropy or determinant.
    posteriors
        Model responsibilities, shape ``(M, T)``.
    sfreq_hz
        Sampling rate used to convert bits/sample to kbits/s.
    """

    x = np.asarray(x_input, dtype=float)
    y = np.asarray(y_models, dtype=float)
    w = np.asarray(w_models, dtype=float)
    gamma = assignment_weights(
        posteriors, assignment, random_state=assignment_random_state
    )

    if x.ndim != 2 or y.ndim != 3 or w.ndim != 3:
        raise ValueError("expected X (N,T), Y (M,N,T), and W (M,N,N)")
    n_components, n_samples = x.shape
    n_models = y.shape[0]
    if y.shape != (n_models, n_components, n_samples):
        raise ValueError(f"Y shape {y.shape} is incompatible with X shape {x.shape}")
    if w.shape != (n_models, n_components, n_components):
        raise ValueError(f"W shape {w.shape} is incompatible with X shape {x.shape}")
    if gamma.shape != (n_models, n_samples):
        raise ValueError(
            f"posterior shape {gamma.shape} must be {(n_models, n_samples)}"
        )

    model_rows: list[ModelConditionalMIR] = []
    for model in range(n_models):
        weights = gamma[model]
        occupancy = float(np.mean(weights))
        posterior_mass = float(np.sum(weights))
        n_eff = effective_sample_size(weights)
        h_input = float(
            sum(
                weighted_entropy_histogram(
                    x[i], weights, n_bins=n_bins, clip_sd=clip_sd
                )
                for i in range(n_components)
            )
        )
        h_sources = float(
            sum(
                weighted_entropy_histogram(
                    y[model, i], weights, n_bins=n_bins, clip_sd=clip_sd
                )
                for i in range(n_components)
            )
        )
        sign, logabsdet = np.linalg.slogdet(w[model])
        log2_abs_det = (
            float(logabsdet / np.log(2.0)) if sign != 0 else float("nan")
        )
        bits = h_input - h_sources + log2_abs_det
        model_rows.append(
            ModelConditionalMIR(
                model=model,
                occupancy=occupancy,
                posterior_mass=posterior_mass,
                effective_n=n_eff,
                h_input_bits=h_input,
                h_sources_bits=h_sources,
                log2_abs_det_w=log2_abs_det,
                bits_per_sample=float(bits),
                kbits_per_sec=float(bits * float(sfreq_hz) / 1_000.0),
                low_occupancy=bool(
                    n_eff < float(min_effective_n)
                    or posterior_mass < float(min_posterior_mass)
                ),
            )
        )

    values = np.asarray([row.bits_per_sample for row in model_rows], dtype=float)
    occupancies = np.asarray([row.occupancy for row in model_rows], dtype=float)
    total = float(np.sum(occupancies * values)) if np.all(np.isfinite(values)) else float("nan")
    return ConditionalMIR(
        bits_per_sample=total,
        kbits_per_sec=float(total * float(sfreq_hz) / 1_000.0),
        n_models=n_models,
        n_components=n_components,
        n_samples=n_samples,
        n_bins=int(n_bins),
        clip_sd=None if clip_sd is None else float(clip_sd),
        assignment=assignment,
        min_effective_n=float(min_effective_n),
        min_posterior_mass=float(min_posterior_mass),
        any_low_occupancy=any(row.low_occupancy for row in model_rows),
        models=tuple(model_rows),
    )


def arrays_from_amica_result(result, x_input: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build ``X``, per-model ``Y``, effective ``W``, and posteriors.

    ``x_input`` must be the exact retained-space array passed to ``Amica.fit``.
    The helper reproduces the solver's scaling, global centering, sphering, and
    model-specific centering conventions.
    """

    x = np.asarray(x_input, dtype=float)
    scale = float(getattr(result, "data_scale", 1.0))
    x_internal = x * scale
    mean = np.asarray(result.mean_, dtype=float).reshape(-1)
    sphere = np.asarray(result.whitener_, dtype=float)
    data_white = sphere @ (x_internal - mean[:, None])

    w_white = np.asarray(result.unmixing_matrix_white_, dtype=float)
    if w_white.ndim == 2:
        w_white = w_white[None, ...]
    n_models, n_components, _ = w_white.shape
    centers = np.asarray(result.c_, dtype=float)
    if centers.ndim == 1:
        centers = centers[None, ...]
    if centers.shape != (n_models, n_components):
        raise ValueError(
            f"model centers have shape {centers.shape}, expected {(n_models, n_components)}"
        )

    y_models = np.stack(
        [w_white[m] @ (data_white - centers[m, :, None]) for m in range(n_models)],
        axis=0,
    )
    w_effective = np.stack([w_white[m] @ sphere for m in range(n_models)], axis=0)
    posterior = getattr(result, "model_posteriors_", None)
    if posterior is None:
        posterior = np.ones((1, x.shape[1]), dtype=float)
    posterior = np.asarray(posterior, dtype=float)
    return x_internal, y_models, w_effective, posterior


def conditional_mir_from_amica_result(
    result,
    x_input: np.ndarray,
    sfreq_hz: float,
    **kwargs,
) -> ConditionalMIR:
    """Convenience wrapper using a fitted ``AmicaResult``."""

    arrays = arrays_from_amica_result(result, x_input)
    return conditional_mir(*arrays, sfreq_hz=sfreq_hz, **kwargs)
