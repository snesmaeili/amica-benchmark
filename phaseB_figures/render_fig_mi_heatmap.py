"""B-local figure: pairwise mutual-information matrices, sensors vs ICA methods.

Loads ds004505 sub-01 with the benchmark's own load_data/preprocess, applies each
committed _ica.fif (AMICA, Picard, Infomax, FastICA) to recover sources, and shows
the pairwise-MI matrix of the scalp channels next to each method's source MI matrix.
Mean off-diagonal MI (bits) annotated per panel. Uses the tested
amica_python.benchmark.metrics.pairwise_mi_matrix (no recompute of fits).

Outputs: fig_mi_heatmap.{pdf,png} + mi_offdiag.csv
"""
import os
os.environ["AMICA_NO_JAX"] = "1"          # these figures need only mne + numpy; avoid jax/GPU init (hangs under RDP)
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("BIDS_ROOT_DS4505", r"D:\amica-validation-workspace\datasets\ds004505\raw_bids")
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
mne.set_log_level("ERROR")
from amica_python.benchmark.runner import load_data, preprocess
from amica_python.benchmark.metrics import pairwise_mi_matrix

# sub-01 fits from the published v3 run. Pulled from fir
# /scratch/sesma/amica_python_validation_v3, identified as the source of
# v3_paper_stage1_cluster/benchmark_results.csv by exact fit_runtime_s match on all
# six methods. The old repos/amica-python/scripts/... path is stale (scripts/ -> benchmark/).
RES = Path(r"D:\amica-validation-workspace\figdata\v3_paper_stage1_sub01")
OUT = Path(r"D:\amica-validation-workspace\phaseB_figures")
SUBJ = 1
N = 32  # components / channels shown (keeps the 5 MI matrices tractable)
MI_KW = dict(n_bins=64, max_samples=8000)  # illustrative figure; full 20k/100-bin is far slower
METHODS = [("jax_gpu", "AMICA"), ("picard", "Picard"), ("infomax", "ext. Infomax"), ("fastica", "FastICA")]

print("loading + preprocessing ds004505 sub-01 ...", flush=True)
raw = load_data("ds004505", SUBJ)
preprocess(raw)
raw.crop(tmax=min(200.0, raw.times[-1]))  # MI subsamples to 20k anyway; a window keeps it fast
print(f"  raw: {len(raw.ch_names)} ch @ {raw.info['sfreq']} Hz, {raw.n_times} samples", flush=True)

panels = []  # (label, mi_matrix)

# Sensor baseline: first N scalp EEG channels
sens = raw.get_data()[:N]
print(f"  computing sensor MI on {sens.shape} ...", flush=True)
panels.append(("Scalp channels", pairwise_mi_matrix(sens, **MI_KW)))
print("  sensor MI done", flush=True)

for key, label in METHODS:
    if key == "jax_gpu":
        fif = RES / f"benchmark_sub-{SUBJ:02d}_hp1.0hz_jax_gpu_ica.fif"
    else:
        fif = RES / f"benchmark_sub-{SUBJ:02d}_hp1.0hz_{key}_cpu_ica.fif"
    if not fif.exists():
        print(f"  MISSING {fif.name}; skipping {label}", flush=True)
        continue
    try:
        ica = mne.preprocessing.read_ica(fif)
        print(f"  {label}: ICA n_comp={ica.n_components_} ch={len(ica.ch_names)} sfreq={ica.info['sfreq']}", flush=True)
        rr = raw.copy()
        if abs(rr.info["sfreq"] - ica.info["sfreq"]) > 1e-6:
            rr.resample(ica.info["sfreq"])
        src = ica.get_sources(rr).get_data()  # (n_comp, T)
        mi = pairwise_mi_matrix(src[:N], **MI_KW)
        panels.append((label, mi))
        print(f"  {label}: sources {src.shape}  mean off-diag MI = {mi[~np.eye(mi.shape[0],dtype=bool)].mean():.4f} bits", flush=True)
    except Exception as e:
        import traceback
        print(f"  {label}: FAILED -> {e!r}", flush=True)
        traceback.print_exc()

# --- CSV ---
with open(OUT / "mi_offdiag.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["panel", "mean_offdiag_MI_bits", "median_offdiag_MI_bits"])
    for label, mi in panels:
        off = mi[~np.eye(mi.shape[0], dtype=bool)]
        w.writerow([label, f"{off.mean():.5f}", f"{np.median(off):.5f}"])
print("wrote mi_offdiag.csv")

# --- figure ---
vmax = max(np.percentile(mi[~np.eye(mi.shape[0], dtype=bool)], 99) for _, mi in panels)
fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.4))
for ax, (label, mi) in zip(axes, panels):
    im = ax.imshow(mi, vmin=0, vmax=vmax, cmap="magma")
    off = mi[~np.eye(mi.shape[0], dtype=bool)]
    ax.set_title(f"{label}\nmean off-diag {off.mean():.3f} bits", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01, label="pairwise MI (bits)")
fig.suptitle("Pairwise mutual information: scalp channels vs recovered ICA sources (Table tennis sub-01)", fontsize=10)
fig.savefig(OUT / "fig_mi_heatmap.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_mi_heatmap.png", dpi=150, bbox_inches="tight")
print("wrote fig_mi_heatmap.pdf/.png")
