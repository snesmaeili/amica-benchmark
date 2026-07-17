"""Pure helpers for contiguous block cross-validation and complete MIR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemporalFold:
    fold: int
    test_start: int
    test_stop: int
    excluded_start: int
    excluded_stop: int
    train_indices: np.ndarray
    test_indices: np.ndarray


def contiguous_folds(n_times: int, n_folds: int, guard_samples: int) -> list[TemporalFold]:
    """Create contiguous test blocks with guard samples removed from training."""

    if n_times < 2:
        raise ValueError("n_times must be at least 2")
    if n_folds < 2 or n_folds > n_times:
        raise ValueError("n_folds must be between 2 and n_times")
    if guard_samples < 0:
        raise ValueError("guard_samples must be non-negative")
    bounds = np.linspace(0, n_times, n_folds + 1, dtype=int)
    all_indices = np.arange(n_times, dtype=np.int64)
    folds = []
    for fold, (start, stop) in enumerate(zip(bounds[:-1], bounds[1:])):
        excluded_start = max(0, int(start) - int(guard_samples))
        excluded_stop = min(n_times, int(stop) + int(guard_samples))
        train = np.concatenate(
            (all_indices[:excluded_start], all_indices[excluded_stop:])
        )
        test = all_indices[start:stop]
        if not len(train) or not len(test):
            raise ValueError(
                f"fold {fold} is empty after a {guard_samples}-sample guard"
            )
        if np.intersect1d(train, test).size:
            raise AssertionError("training and test samples overlap")
        folds.append(
            TemporalFold(
                fold=fold,
                test_start=int(start),
                test_stop=int(stop),
                excluded_start=excluded_start,
                excluded_stop=excluded_stop,
                train_indices=train,
                test_indices=test,
            )
        )
    return folds


def evaluation_indices(n_samples: int, max_samples: int | None, seed: int) -> np.ndarray:
    """Return deterministic sample indices shared by every method in a fold."""

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if max_samples is None or max_samples >= n_samples:
        return np.arange(n_samples, dtype=np.int64)
    if max_samples < 2:
        raise ValueError("max_samples must be at least 2")
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(n_samples, int(max_samples), replace=False))


def project_with_fitted_ica(raw, ica, indices: np.ndarray | None = None):
    """Project data using only transformations stored in a training-fitted ICA."""

    pca_components = np.asarray(ica.pca_components_, dtype=float)
    unmixing = np.asarray(ica.unmixing_matrix_, dtype=float)
    if unmixing.ndim != 2 or unmixing.shape[0] != unmixing.shape[1]:
        raise ValueError("complete MIR requires a square retained-rank unmixing matrix")
    data = np.asarray(raw.get_data(), dtype=float)
    if indices is not None:
        data = data[:, np.asarray(indices, dtype=np.int64)]
    # Reuse the training-fitted MNE projection and pre-whitener exactly.  The
    # public transform path ultimately calls this helper; the explicit fallback
    # keeps the pure unit tests independent of MNE.
    if hasattr(ica, "_pre_whiten"):
        data = np.asarray(ica._pre_whiten(data.copy()), dtype=float)
    else:
        pre_whitener = getattr(ica, "pre_whitener_", None)
        if pre_whitener is not None:
            whitener = np.asarray(pre_whitener, dtype=float)
            if whitener.ndim == 2 and whitener.shape[0] == whitener.shape[1]:
                data = whitener @ data
            else:
                data = data / whitener.reshape(-1, 1)
    pca_mean = getattr(ica, "pca_mean_", None)
    if pca_mean is None:
        raise ValueError("training-fitted ICA does not expose pca_mean_")
    data = data - np.asarray(pca_mean, dtype=float).reshape(-1, 1)
    x_pca = pca_components[: unmixing.shape[0]] @ data
    return x_pca, unmixing


def entropy_histogram(x: np.ndarray, n_bins: int, clip_sd: float = 5.0) -> float:
    """Histogram differential entropy in bits, matching the manuscript estimator."""

    values = np.asarray(x, dtype=float).ravel()
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    if values.size < 2 or not np.isfinite(std) or std <= 0:
        return float("nan")
    low, high = mean - clip_sd * std, mean + clip_sd * std
    values = np.clip(values, low, high)
    counts, _ = np.histogram(values, bins=np.linspace(low, high, n_bins + 1))
    probabilities = counts.astype(float) / float(counts.sum())
    nonzero = probabilities > 0
    bin_width = (high - low) / n_bins
    return float(
        -np.sum(probabilities[nonzero] * np.log2(probabilities[nonzero]))
        + np.log2(bin_width)
    )


def complete_mir_from_fitted_ica(
    raw,
    ica,
    *,
    indices: np.ndarray,
    n_bins: int,
) -> dict:
    """Compute retained-rank complete MIR on explicit, shared sample indices."""

    x_pca, w_raw = project_with_fitted_ica(raw, ica, indices)
    y_raw = w_raw @ x_pca
    scale = y_raw.std(axis=1, keepdims=True)
    scale = np.where(scale > 0, scale, 1.0)
    y = y_raw / scale
    w = w_raw / scale
    sign, logabsdet = np.linalg.slogdet(w)
    if sign == 0:
        raise ValueError("singular unmixing matrix")
    h_input = sum(entropy_histogram(row, n_bins) for row in x_pca)
    h_sources = sum(entropy_histogram(row, n_bins) for row in y)
    bits_per_sample = float(h_input - h_sources + logabsdet / np.log(2.0))
    return {
        "bits_per_sample": bits_per_sample,
        "kbits_per_sec": bits_per_sample * float(raw.info["sfreq"]) / 1000.0,
        "h_input_bits": float(h_input),
        "h_sources_bits": float(h_sources),
        "log2_abs_det_w": float(logabsdet / np.log(2.0)),
        "n_samples_used": int(len(indices)),
        "n_bins": int(n_bins),
        "sample_index_sha256": __import__("hashlib").sha256(
            np.asarray(indices, dtype=np.int64).tobytes()
        ).hexdigest(),
        "space": "retained training-fitted PCA rank",
    }
