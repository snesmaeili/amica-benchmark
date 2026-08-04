"""Supplementary Figure S2 -- pairwise mutual-information matrices (illustrative).

Input scalp channels vs AMICA component activations on one ds004505 subject (sub-01). ICA reduces
the off-diagonal pairwise MI: the input channels carry structured cross-channel dependence, while
the AMICA components are closer to the finite-sample MI floor. Absolute values include the
histogram estimator's positive bias; the contrast (input > components) is the illustrative point.
Reads the committed pairwise_mi_ds004505_sub01.npz (computed on the cluster by compute_pairwise_mi.py).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import style

HERE = Path(__file__).resolve().parent


def _offdiag(M):
    M = M.astype(float).copy()
    np.fill_diagonal(M, np.nan)
    return M


def main():
    style.set_paper_style()
    d = np.load(HERE / "pairwise_mi_ds004505_sub01.npz")
    I, A = _offdiag(d["mi_input"]), _offdiag(d["mi_amica"])
    vmax = float(np.nanpercentile(I, 99))

    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.5))
    for ax, M, lab, n in [(axs[0], I, "a  Input scalp channels", int(d["n_ch"])),
                          (axs[1], A, "b  AMICA components", int(d["n_comp"]))]:
        im = ax.imshow(M, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{lab} ($N{{=}}{n}$)\noff-diagonal mean "
                     f"{np.nanmean(M):.2f} bits", loc="left", fontweight="bold", fontsize=9.0)
        ax.set_xlabel("index $j$"); ax.set_ylabel("index $i$")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="pairwise MI (bits)")

    fig.suptitle("Pairwise mutual information collapses from input channels to AMICA components "
                 "(ds004505 sub-01, illustrative)", fontsize=9.6, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = style.save_vector(fig, HERE / "out" / "figS2_pairwise_mi.pdf")
    print(f"wrote {out}; input off-diag {np.nanmean(I):.3f} bits, amica off-diag {np.nanmean(A):.3f} bits")


if __name__ == "__main__":
    main()
