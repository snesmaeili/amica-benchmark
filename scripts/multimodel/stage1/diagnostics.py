"""Posterior occupancy and Kish effective-sample-size diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class OccupancyDiagnostics:
    """Fitted priors, posterior occupancy, and effective sample size."""

    fitted_priors: np.ndarray
    posterior_mass: np.ndarray
    posterior_occupancy: np.ndarray
    kish_effective_n: np.ndarray
    kish_per_c_squared: np.ndarray
    hard_occupancy: np.ndarray
    low_fitted_prior: np.ndarray
    low_posterior_occupancy: np.ndarray
    low_kish_effective_n: np.ndarray
    kish_below_10: np.ndarray
    kish_below_25: np.ndarray
    kish_below_50: np.ndarray
    total_weight: float
    n_samples: int

    @property
    def any_degenerate(self) -> bool:
        """Whether a fitted-prior or primary Kish gate fails."""
        return bool(
            np.any(self.low_fitted_prior | self.low_kish_effective_n)
        )

    def as_records(self) -> List[Mapping[str, object]]:
        """Return JSON-ready records using zero-based model indices."""
        return [
            {
                "model_index": model_index,
                "fitted_prior": float(self.fitted_priors[model_index]),
                "posterior_mass": float(self.posterior_mass[model_index]),
                "posterior_occupancy": float(
                    self.posterior_occupancy[model_index]
                ),
                "kish_effective_n": float(
                    self.kish_effective_n[model_index]
                ),
                "kish_per_c_squared": float(
                    self.kish_per_c_squared[model_index]
                ),
                "hard_occupancy": float(self.hard_occupancy[model_index]),
                "low_fitted_prior": bool(
                    self.low_fitted_prior[model_index]
                ),
                "low_posterior_occupancy": bool(
                    self.low_posterior_occupancy[model_index]
                ),
                "low_kish_effective_n": bool(
                    self.low_kish_effective_n[model_index]
                ),
                "kish_per_c_squared_below_10": bool(
                    self.kish_below_10[model_index]
                ),
                "kish_per_c_squared_below_25": bool(
                    self.kish_below_25[model_index]
                ),
                "kish_per_c_squared_below_50": bool(
                    self.kish_below_50[model_index]
                ),
            }
            for model_index in range(self.posterior_occupancy.size)
        ]


def occupancy_kish_diagnostics(
    posteriors: np.ndarray,
    *,
    fitted_priors: np.ndarray,
    n_components: int,
    sample_weights: Optional[np.ndarray] = None,
    min_fitted_prior: float = 0.02,
    min_posterior_occupancy: float = 0.02,
    primary_kish_per_c_squared: float = 25.0,
    posterior_sum_atol: float = 1e-6,
) -> OccupancyDiagnostics:
    """Calculate fitted-prior, occupancy, and Kish diagnostics.

    For model ``m``, the fractional contribution of sample ``t`` is
    ``a_mt = sample_weight_t * posterior_mt`` and
    ``Kish_m = sum(a_mt)^2 / sum(a_mt^2)``.  The fitted prior ``pi_m`` and
    posterior occupancy are retained separately: they agree only under the
    corresponding EM update/evaluation conditions.  The primary 0.02 gate
    applies to the returned fitted prior, not to a post-hoc occupancy estimate.
    """
    values = np.asarray(posteriors, dtype=float)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("posteriors must have shape (n_models, n_samples)")
    if not np.all(np.isfinite(values)):
        raise ValueError("posteriors contain non-finite values")
    if np.any(values < 0):
        raise ValueError("posteriors cannot be negative")
    column_sums = values.sum(axis=0)
    if not np.allclose(column_sums, 1.0, rtol=0.0, atol=posterior_sum_atol):
        raise ValueError("posterior probabilities must sum to one per sample")
    priors = np.asarray(fitted_priors, dtype=float)
    if priors.shape != (values.shape[0],):
        raise ValueError("fitted_priors must have shape (n_models,)")
    if not np.all(np.isfinite(priors)) or np.any(priors < 0.0):
        raise ValueError("fitted_priors must be finite and non-negative")
    if not np.isclose(priors.sum(), 1.0, rtol=0.0, atol=posterior_sum_atol):
        raise ValueError("fitted_priors must sum to one")
    if not 0.0 <= min_fitted_prior < 1.0:
        raise ValueError("min_fitted_prior must lie in [0, 1)")
    if not 0.0 <= min_posterior_occupancy < 1.0:
        raise ValueError("min_posterior_occupancy must lie in [0, 1)")
    if type(n_components) is not int or n_components < 1:
        raise ValueError("n_components must be a positive integer")
    if primary_kish_per_c_squared < 0:
        raise ValueError("primary_kish_per_c_squared cannot be negative")

    n_samples = values.shape[1]
    if sample_weights is None:
        weights = np.ones(n_samples, dtype=float)
    else:
        weights = np.asarray(sample_weights, dtype=float)
        if weights.shape != (n_samples,):
            raise ValueError("sample_weights must have shape (n_samples,)")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("sample_weights must be finite and non-negative")
    total_weight = float(weights.sum())
    if total_weight <= 0:
        raise ValueError("sample_weights must contain positive total weight")

    contributions = values * weights[np.newaxis, :]
    posterior_mass = contributions.sum(axis=1)
    posterior_occupancy = posterior_mass / total_weight
    squared_mass = np.square(contributions).sum(axis=1)
    kish = np.divide(
        np.square(posterior_mass),
        squared_mass,
        out=np.zeros_like(posterior_mass),
        where=squared_mass > 0,
    )
    kish_per_c_squared = kish / float(n_components**2)

    hard_assignments = np.argmax(values, axis=0)
    hard_mass = np.bincount(
        hard_assignments, weights=weights, minlength=values.shape[0]
    )
    hard_occupancy = hard_mass / total_weight
    return OccupancyDiagnostics(
        fitted_priors=priors,
        posterior_mass=posterior_mass,
        posterior_occupancy=posterior_occupancy,
        kish_effective_n=kish,
        kish_per_c_squared=kish_per_c_squared,
        hard_occupancy=hard_occupancy,
        low_fitted_prior=priors < min_fitted_prior,
        low_posterior_occupancy=(
            posterior_occupancy < min_posterior_occupancy
        ),
        low_kish_effective_n=(
            kish_per_c_squared < primary_kish_per_c_squared
        ),
        kish_below_10=kish_per_c_squared < 10.0,
        kish_below_25=kish_per_c_squared < 25.0,
        kish_below_50=kish_per_c_squared < 50.0,
        total_weight=total_weight,
        n_samples=n_samples,
    )
