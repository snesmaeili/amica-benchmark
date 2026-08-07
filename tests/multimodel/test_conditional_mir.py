from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "multimodel"
sys.path.insert(0, str(MODULE_DIR))

from conditional_mir import (  # noqa: E402
    assignment_weights,
    conditional_mir,
    effective_sample_size,
    weighted_entropy_histogram,
)


def _fixture(seed=0, n=4, t=30_000):
    rng = np.random.default_rng(seed)
    x = rng.standard_t(df=5, size=(n, t))
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    y = (q @ x)[None, ...]
    return x, y, q[None, ...], np.ones((1, t))


def test_uniform_weight_entropy_matches_unweighted_formula():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(50_000)
    h_weighted = weighted_entropy_histogram(x, np.ones(x.size), n_bins=100)

    mean = float(np.mean(x))
    sd = float(np.std(x))
    lo, hi = mean - 5 * sd, mean + 5 * sd
    counts, _ = np.histogram(np.clip(x, lo, hi), bins=np.linspace(lo, hi, 101))
    p = counts[counts > 0] / counts.sum()
    h_reference = -np.sum(p * np.log2(p)) + np.log2((hi - lo) / 100)
    np.testing.assert_allclose(h_weighted, h_reference, atol=1e-12, rtol=0)


def test_m1_conditional_mir_is_ordinary_mir():
    x, y, w, gamma = _fixture()
    result = conditional_mir(x, y, w, gamma, 250.0, min_effective_n=2_000)
    h_x = sum(weighted_entropy_histogram(row, gamma[0]) for row in x)
    h_y = sum(weighted_entropy_histogram(row, gamma[0]) for row in y[0])
    reference = h_x - h_y + np.linalg.slogdet(w[0])[1] / np.log(2)
    np.testing.assert_allclose(result.bits_per_sample, reference, atol=1e-12, rtol=0)
    assert not result.any_low_occupancy


def test_sign_permutation_and_reciprocal_scale_invariance():
    x, y, w, gamma = _fixture(seed=2)
    base = conditional_mir(x, y, w, gamma, 250.0).bits_per_sample

    perm = np.array([2, 0, 3, 1])
    scale = np.array([-2.0, 0.5, -1.5, 3.0])
    y2 = y[:, perm, :] * scale[None, :, None]
    w2 = w[:, perm, :] * scale[None, :, None]
    changed = conditional_mir(x, y2, w2, gamma, 250.0).bits_per_sample
    np.testing.assert_allclose(changed, base, atol=5e-12, rtol=0)


def test_assignment_modes_preserve_probability_mass_and_permutation_occupancy():
    rng = np.random.default_rng(3)
    gamma = rng.dirichlet(np.ones(3), size=10_000).T
    soft = assignment_weights(gamma, "soft")
    hard = assignment_weights(gamma, "hard")
    perm = assignment_weights(gamma, "time_permuted", random_state=9)
    np.testing.assert_allclose(soft.sum(axis=0), 1.0)
    np.testing.assert_allclose(hard.sum(axis=0), 1.0)
    np.testing.assert_allclose(perm.sum(axis=0), 1.0)
    np.testing.assert_allclose(perm.mean(axis=1), soft.mean(axis=1), atol=1e-15)


def test_effective_sample_size_and_low_occupancy_flag():
    assert effective_sample_size(np.ones(100)) == 100
    x, y1, w1, _ = _fixture(seed=4, t=10_000)
    y = np.concatenate([y1, y1], axis=0)
    w = np.concatenate([w1, w1], axis=0)
    gamma = np.vstack([np.full(10_000, 0.9999), np.full(10_000, 0.0001)])
    result = conditional_mir(x, y, w, gamma, 250.0)
    assert result.any_low_occupancy
    assert not result.models[0].low_occupancy
    assert result.models[1].low_occupancy
    np.testing.assert_allclose(result.models[1].effective_n, 10_000)
    np.testing.assert_allclose(result.models[1].posterior_mass, 1.0)
