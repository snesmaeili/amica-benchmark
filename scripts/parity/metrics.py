"""Comparison metrics for AMICA parity testing."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def sign_align_rows(M):
    """Align row signs so largest-magnitude element is positive."""
    out = M.copy()
    for i in range(out.shape[0]):
        if out[i, np.argmax(np.abs(out[i]))] < 0:
            out[i] *= -1
    return out


def match_and_align(M1, M2):
    """Match rows of M2 to M1 by correlation, then sign-align.

    Returns M2 with rows permuted and sign-flipped to best match M1.
    """
    n = M1.shape[0]
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            r = np.corrcoef(M1[i], M2[j])[0, 1]
            corr[i, j] = r if np.isfinite(r) else 0.0

    row_ind, col_ind = linear_sum_assignment(-np.abs(corr))
    M2_aligned = M2[col_ind].copy()
    signs = np.sign(corr[row_ind, col_ind])
    M2_aligned *= signs[:, None]

    row_corrs = np.abs(corr[row_ind, col_ind])
    return M2_aligned, row_corrs


def compare_matrices(M1, M2, name="matrix"):
    """Compare two matrices after permutation/sign alignment."""
    M2a, row_corrs = match_and_align(M1, M2)
    diff = M1 - M2a
    return {
        f"{name}_max_abs_diff": float(np.max(np.abs(diff))),
        f"{name}_frobenius": float(np.linalg.norm(diff)),
        f"{name}_min_row_corr": float(np.min(row_corrs)),
        f"{name}_mean_row_corr": float(np.mean(row_corrs)),
    }


def compare_ll_trajectories(ll1, ll2):
    """Compare two LL trajectories of potentially different lengths."""
    n = min(len(ll1), len(ll2))
    ll1, ll2 = ll1[:n], ll2[:n]
    diff = np.abs(ll1 - ll2)
    return {
        "ll_max_abs_diff": float(np.max(diff)),
        "ll_mean_abs_diff": float(np.mean(diff)),
        "ll_final_diff": float(abs(ll1[-1] - ll2[-1])),
        "ll_final_rel_diff": float(abs(ll1[-1] - ll2[-1]) / (abs(ll1[-1]) + 1e-30)),
        "n_compared": n,
    }


def compare_params(res1, res2):
    """Compare PDF parameters (alpha, mu, beta, rho) after alignment."""
    metrics = {}
    for key in ("alpha", "mu", "beta", "rho"):
        p1, p2 = res1[key], res2[key]
        if p1.shape != p2.shape:
            metrics[f"{key}_shape_mismatch"] = True
            continue
        diff = np.abs(p1 - p2)
        metrics[f"{key}_max_diff"] = float(np.max(diff))
        metrics[f"{key}_mean_diff"] = float(np.mean(diff))
    return metrics


def compare_spheres(S1, S2):
    """Compare sphering matrices with sign alignment per row."""
    S1a = sign_align_rows(S1)
    S2a = sign_align_rows(S2)
    diff = S1a - S2a
    return {
        "sphere_max_abs_diff": float(np.max(np.abs(diff))),
        "sphere_frobenius": float(np.linalg.norm(diff)),
    }


def as_model_array(value):
    """Normalise a single-model or multi-model matrix to (M, C, C)."""

    array = np.asarray(value)
    return array[None, ...] if array.ndim == 2 else array


def match_models(reference, candidate):
    """Align candidate models by mean Hungarian-matched row correlation."""

    ref = as_model_array(reference)
    cand = as_model_array(candidate)
    if ref.shape != cand.shape:
        raise ValueError(f"model matrix shape mismatch: {ref.shape} vs {cand.shape}")
    score = np.zeros((ref.shape[0], cand.shape[0]))
    for left in range(ref.shape[0]):
        for right in range(cand.shape[0]):
            _, correlations = match_and_align(ref[left], cand[right])
            score[left, right] = correlations.mean()
    rows, columns = linear_sum_assignment(-score)
    if not np.array_equal(rows, np.arange(ref.shape[0])):
        raise AssertionError("unexpected model assignment order")
    return columns, score[rows, columns]


def rejection_metrics(reference_mask, candidate_mask):
    """Compare rejection masks using counts and Jaccard overlap."""

    reference = np.asarray(reference_mask, dtype=bool)
    candidate = np.asarray(candidate_mask, dtype=bool)
    if reference.shape != candidate.shape:
        raise ValueError("rejection mask shape mismatch")
    ref_rejected = ~reference
    cand_rejected = ~candidate
    union = np.sum(ref_rejected | cand_rejected)
    intersection = np.sum(ref_rejected & cand_rejected)
    jaccard = 1.0 if union == 0 else float(intersection / union)
    return {
        "reference_rejected": int(ref_rejected.sum()),
        "candidate_rejected": int(cand_rejected.sum()),
        "rejected_count_difference": int(abs(ref_rejected.sum() - cand_rejected.sum())),
        "rejection_jaccard": jaccard,
    }


def normalized_rmse(reference, candidate):
    """RMSE divided by reference IQR, with a stable zero-IQR fallback."""

    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    rmse = float(np.sqrt(np.mean((reference - candidate) ** 2)))
    scale = float(np.subtract(*np.percentile(reference, (75, 25))))
    if scale <= np.finfo(float).eps:
        scale = max(float(np.std(reference)), 1.0)
    return rmse / scale
