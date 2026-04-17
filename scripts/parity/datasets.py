"""Dataset loading for parity tests."""

import numpy as np


def make_synthetic_laplacian(n_channels=6, n_samples=5000, seed=42):
    """Deterministic synthetic data with Laplacian sources.

    Returns (data, A_true, S_true) where data = A_true @ S_true.
    data shape: (n_channels, n_samples)
    """
    rng = np.random.default_rng(seed)
    S = rng.laplace(0, 1, (n_channels, n_samples))
    # Random orthogonal mixing matrix
    A_true = np.linalg.qr(rng.standard_normal((n_channels, n_channels)))[0]
    X = A_true @ S
    return X, A_true, S
