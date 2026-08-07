"""B-cluster aggregation: AMICA seed-robustness on ds004505.

Reads the per-seed v3 JSONs produced by the seed-robustness array
(runner.py --random-state, one output dir per seed) and reports the
cross-seed dispersion of MIR per subject — i.e. whether the AMICA MIR
ranking is stable across random initialisations. Optionally adds a
cross-seed decomposition-stability panel (Hungarian-matched |r| between
each seed's unmixing and seed 0) from the saved _ica.fif files.

Run AFTER rsync of /scratch/sesma/amica_seed_robustness/seed* into ROOT.

Outputs: fig_seed_robustness.{pdf,png} + seed_robustness.csv
"""
import os
os.environ["AMICA_NO_JAX"] = "1"            # _ica.fif read needs only mne; avoid jax/GPU init
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import json
import glob
import re
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"D:\amica-validation-workspace\repos\amica-mm\results\seed_robustness")
OUT = Path(r"D:\amica-validation-workspace\phaseB_figures")
SEEDS = [0, 1, 2, 3, 4]
COMPARATOR_GAP = 0.5  # approx AMICA-vs-comparator MIR gap (kbits/s) from the paper, for context


def mir_of(path):
    d = json.load(open(path))
    a = d.get("amica", d)
    m = a.get("complete_mir")
    if isinstance(m, dict):
        return float(m.get("kbits_per_sec") or m.get("kbits_s") or m.get("bits_per_sample"))
    return float(a.get("mir")) if a.get("mir") is not None else None


# --- collect MIR[subject][seed] ---
data = {}
for seed in SEEDS:
    for f in glob.glob(str(ROOT / f"seed{seed}" / "*jax_gpu.json")):
        mm = re.search(r"sub-(\d+)", os.path.basename(f))
        if not mm:
            continue
        subj = int(mm.group(1))
        v = mir_of(f)
        if v is not None:
            data.setdefault(subj, {})[seed] = v

subjects = sorted(data)
if not subjects:
    raise SystemExit(f"No seed JSONs under {ROOT} (rsync first).")

means = np.array([np.mean(list(data[s].values())) for s in subjects])
stds = np.array([np.std(list(data[s].values()), ddof=1) if len(data[s]) > 1 else 0.0 for s in subjects])
ncomplete = sum(len(data[s]) == len(SEEDS) for s in subjects)
all_mir = np.array([v for s in subjects for v in data[s].values()])
print(f"{len(subjects)} subjects ({ncomplete} with all {len(SEEDS)} seeds); "
      f"cohort MIR mean={all_mir.mean():.3f} kbits/s")
print(f"median within-subject across-seed SD = {np.median(stds):.4f} kbits/s "
      f"(max {stds.max():.4f}); cf. AMICA-vs-comparator gap ~{COMPARATOR_GAP} kbits/s")

# --- CSV ---
with open(OUT / "seed_robustness.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["subject", "n_seeds", "mir_mean", "mir_sd", "mir_min", "mir_max"]
               + [f"mir_seed{sd}" for sd in SEEDS])
    for s in subjects:
        vals = data[s]
        arr = np.array(list(vals.values()))
        w.writerow([f"sub-{s:02d}", len(vals), f"{arr.mean():.5f}",
                    f"{arr.std(ddof=1) if len(arr) > 1 else 0:.5f}",
                    f"{arr.min():.5f}", f"{arr.max():.5f}"]
                   + [f"{vals.get(sd, float('nan')):.5f}" for sd in SEEDS])
print("wrote seed_robustness.csv")

# --- optional: cross-seed unmixing stability (Hungarian |r| vs seed 0) ---
wstab = {}
try:
    import mne
    from scipy.optimize import linear_sum_assignment
    mne.set_log_level("ERROR")

    def matched_mean_r(Wa, Wb):
        C = np.abs(np.corrcoef(Wa, Wb)[:Wa.shape[0], Wa.shape[0]:])
        r, c = linear_sum_assignment(-C)
        return float(np.mean(C[r, c]))

    for s in subjects:
        f0 = glob.glob(str(ROOT / "seed0" / f"*sub-{s:02d}*jax_gpu_ica.fif"))
        if not f0:
            continue
        W0 = mne.preprocessing.read_ica(f0[0]).unmixing_matrix_
        rs = []
        for seed in SEEDS[1:]:
            fk = glob.glob(str(ROOT / f"seed{seed}" / f"*sub-{s:02d}*jax_gpu_ica.fif"))
            if fk:
                Wk = mne.preprocessing.read_ica(fk[0]).unmixing_matrix_
                rs.append(matched_mean_r(W0, Wk))
        if rs:
            wstab[s] = float(np.mean(rs))
    if wstab:
        vals = np.array(list(wstab.values()))
        print(f"cross-seed matched |r| vs seed0: mean={vals.mean():.4f} min={vals.min():.4f}")
except Exception as e:
    print(f"(W-stability panel skipped: {e!r})")

# --- figure ---
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
npanel = 2 if wstab else 1
fig, axes = plt.subplots(1, npanel, figsize=(6.2 * npanel, 4.2))
axes = np.atleast_1d(axes)

ax = axes[0]
order = np.argsort(means)[::-1]
x = np.arange(len(subjects))
ax.errorbar(x, means[order], yerr=stds[order], fmt="o", ms=4, capsize=2, color="#1f77b4")
ax.set_xlabel("subject (sorted by mean MIR)")
ax.set_ylabel("MIR (kbits/s)")
ax.set_title(f"(A) AMICA MIR across {len(SEEDS)} seeds per subject")
ax.text(0.02, 0.04,
        f"median across-seed SD = {np.median(stds):.4f} kbits/s\n"
        f"(AMICA–comparator gap ≈ {COMPARATOR_GAP} kbits/s)",
        transform=ax.transAxes, fontsize=8, va="bottom")

if wstab:
    ax = axes[1]
    sw = sorted(wstab)
    ax.bar(np.arange(len(sw)), [wstab[s] for s in sw], color="#3182bd")
    ax.set_ylim(0.9, 1.0)
    ax.set_xlabel("subject")
    ax.set_ylabel("cross-seed matched |r| (vs seed 0)")
    ax.set_title("(B) Decomposition reproducibility across seeds")

fig.tight_layout()
fig.savefig(OUT / "fig_seed_robustness.pdf")
fig.savefig(OUT / "fig_seed_robustness.png", dpi=150)
print("wrote fig_seed_robustness.pdf/.png")
