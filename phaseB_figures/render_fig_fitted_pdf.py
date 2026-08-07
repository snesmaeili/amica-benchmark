"""B-local figure: AMICA's adaptive per-component source densities over the
empirical activation histograms (ds004505 sub-01).

The mixture-of-K generalized-Gaussian density that AMICA fits per component is
reconstructed from the committed pdf_params (alpha, mu, beta=sbeta, rho; shape
(K, n_comp)) and overlaid on the histogram of that component's activation
(recovered via the committed AMICA _ica.fif). Demonstrates the adaptive
source-density modelling that is AMICA's defining feature.

GGD log-density (amica_python/pdf.py convention):
    logp(y) = log(sbeta) - |sbeta*(y-mu)|**rho - gammaln(1+1/rho) - log(2)

Outputs: fig_fitted_pdf.{pdf,png}
"""
import os
os.environ.setdefault("BIDS_ROOT_DS4505", r"D:\amica-validation-workspace\datasets\ds004505\raw_bids")
import json
from pathlib import Path
import numpy as np
from scipy.special import gammaln
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
mne.set_log_level("ERROR")
from amica_python.benchmark.runner import load_data, preprocess

# sub-01 fits from the published v3 run. Pulled from fir
# /scratch/sesma/amica_python_validation_v3, identified as the source of
# v3_paper_stage1_cluster/benchmark_results.csv by exact fit_runtime_s match on all
# six methods. The old repos/amica-python/scripts/... path is stale (scripts/ -> benchmark/).
RES = Path(r"D:\amica-validation-workspace\figdata\v3_paper_stage1_sub01")
OUT = Path(r"D:\amica-validation-workspace\phaseB_figures")
SUBJ = 1


def ggd_logpdf(y, mu, sbeta, rho):
    ys = sbeta * (y - mu)
    safe = np.maximum(np.abs(ys), 1e-300)
    return np.log(sbeta) - safe ** rho - gammaln(1.0 + 1.0 / rho) - np.log(2.0)


def mixture_pdf(y, alpha, mu, sbeta, rho):
    """alpha,mu,sbeta,rho: arrays (K,) for one component."""
    out = np.zeros_like(y)
    for k in range(len(alpha)):
        out = out + alpha[k] * np.exp(ggd_logpdf(y, mu[k], sbeta[k], rho[k]))
    return out


print("loading + preprocessing ds004505 sub-01 ...")
raw = load_data("ds004505", SUBJ)
preprocess(raw)
ica = mne.preprocessing.read_ica(RES / f"benchmark_sub-{SUBJ:02d}_hp1.0hz_jax_gpu_ica.fif")
rr = raw.copy()
if abs(rr.info["sfreq"] - ica.info["sfreq"]) > 1e-6:
    rr.resample(ica.info["sfreq"])
src = ica.get_sources(rr).get_data()  # (n_comp, T)
print(f"  sources: {src.shape}")

pp = json.load(open(RES / f"benchmark_sub-{SUBJ:02d}_hp1.0hz_jax_gpu.json"))["amica"]["pdf_params"]
alpha = np.asarray(pp["alpha"], float)   # (K, n_comp)
mu = np.asarray(pp["mu"], float)
sbeta = np.asarray(pp["beta"], float)    # JSON 'beta' == sbeta (inverse scale)
rho = np.asarray(pp["rho"], float)
K, ncomp = alpha.shape
print(f"  pdf_params: K={K} mixtures, n_comp={ncomp}")

# pick representative components spanning the fitted shape rho (alpha-weighted mean)
mean_rho = (alpha * rho).sum(0) / alpha.sum(0)
order = np.argsort(mean_rho)
pick = [order[0], order[1], order[ncomp // 3], order[ncomp // 2],
        order[2 * ncomp // 3], order[-1]]

plt.rcParams.update({"font.size": 9})
fig, axes = plt.subplots(2, 3, figsize=(11, 6))
for ax, i in zip(axes.ravel(), pick):
    y = src[i]
    s = np.std(y)
    lo, hi = -5 * s, 5 * s
    ax.hist(y, bins=160, range=(lo, hi), density=True, color="0.8",
            edgecolor="0.6", linewidth=0.3, label="activation histogram")
    xg = np.linspace(lo, hi, 800)
    ax.plot(xg, mixture_pdf(xg, alpha[:, i], mu[:, i], sbeta[:, i], rho[:, i]),
            color="#d62728", lw=1.8, label="AMICA fitted density")
    ax.set_title(f"IC {i}  ($\\bar\\rho$={mean_rho[i]:.2f})", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, None)
    ax.set_xticks([])
axes[0, 0].legend(fontsize=7, frameon=False)
fig.suptitle("AMICA adaptive per-component source densities over empirical activations "
             "(Table tennis sub-01; log scale)", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "fig_fitted_pdf.pdf")
fig.savefig(OUT / "fig_fitted_pdf.png", dpi=150)
print("wrote fig_fitted_pdf.pdf/.png")
