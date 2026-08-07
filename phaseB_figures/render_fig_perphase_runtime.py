"""B-local figure: per-phase / per-iteration runtime decomposition for AMICA backends.

Derived entirely from the committed v3 benchmark JSONs (no recompute):
each fit's convergence.iteration_times gives the one-time JIT/compile overhead
(first iteration) and the steady-state per-iteration cost; natural-gradient
(early) vs Newton (late) iterations are compared from the same trace.

Outputs: fig_perphase_runtime.{pdf,png} + perphase_runtime.csv
"""
import json
import glob
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(r"D:\amica-validation-workspace\repos\amica-python\scripts\cc_benchmark\results\v3_paper_stage1_cluster")
OUT = Path(r"D:\amica-validation-workspace\phaseB_figures")
OUT.mkdir(exist_ok=True)
NEWTON_ONSET = 50  # natural-gradient -> Newton handover (Frank et al. 2023; AMICA default)

BACKENDS = [
    ("jax_gpu", "JAX-GPU (H100)", "#1f77b4"),
    ("jax_cpu", "JAX-CPU", "#ff7f0e"),
    ("numpy_cpu", "NumPy-CPU", "#2ca02c"),
]


def load(backend):
    rows = []
    for f in sorted(glob.glob(str(RES / f"benchmark_sub-*_{backend}.json"))):
        d = json.load(open(f))["amica"]
        it = np.asarray(d["convergence"]["iteration_times"], float)
        if it.size >= 60:
            rows.append(it)
    return rows


data = {b: load(b) for b, _, _ in BACKENDS}
for b, _, _ in BACKENDS:
    print(f"{b}: n_subj={len(data[b])}")

# --- summary table ---
summary = []
for b, label, _ in BACKENDS:
    rows = data[b]
    if not rows:
        continue
    jit = np.array([it[0] - np.median(it[1:]) for it in rows])
    steady = np.array([np.median(it[1:]) for it in rows])
    natg = np.array([np.median(it[1:NEWTON_ONSET]) for it in rows])
    newton = np.array([np.median(it[NEWTON_ONSET:]) for it in rows])
    total = np.array([it.sum() for it in rows])
    n = len(rows)
    sem = lambda a: a.std() / np.sqrt(n)
    summary.append(dict(backend=b, label=label, n=n,
                        jit=jit.mean(), jit_sem=sem(jit),
                        steady=steady.mean(), steady_sem=sem(steady),
                        natg=natg.mean(), newton=newton.mean(),
                        total=total.mean(), total_sem=sem(total)))

with open(OUT / "perphase_runtime.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["backend", "n_subj", "jit_overhead_s", "jit_sem",
                "steady_iter_s", "steady_sem", "natgrad_iter_s", "newton_iter_s",
                "total_fit_s", "total_sem"])
    for s in summary:
        w.writerow([s["backend"], s["n"], f"{s['jit']:.4f}", f"{s['jit_sem']:.4f}",
                    f"{s['steady']:.5f}", f"{s['steady_sem']:.5f}",
                    f"{s['natg']:.5f}", f"{s['newton']:.5f}",
                    f"{s['total']:.1f}", f"{s['total_sem']:.1f}"])
print("wrote perphase_runtime.csv")
for s in summary:
    print(f"  {s['label']:14s} JIT={s['jit']:.3f}s steady={s['steady']*1000:.2f}ms/it "
          f"natg={s['natg']*1000:.2f} newton={s['newton']*1000:.2f} total={s['total']:.1f}s")

# --- figure ---
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# Panel A: per-iteration wall-time vs iteration index (mean over subjects)
ax = axes[0]
for b, label, c in BACKENDS:
    rows = data[b]
    if not rows:
        continue
    L = min(len(it) for it in rows)
    M = np.vstack([it[:L] for it in rows])
    ax.plot(np.arange(1, L + 1), M.mean(0), label=label, lw=1.1, color=c)
ax.set_yscale("log")
ax.set_xlabel("iteration")
ax.set_ylabel("wall-time per iteration (s, log)")
ax.set_title("(A) Per-iteration cost (mean over subjects)")
ax.axvline(NEWTON_ONSET, color="gray", ls=":", lw=0.9)
ax.text(NEWTON_ONSET + 40, ax.get_ylim()[0] * 2.2,
        "nat-grad | Newton", fontsize=7.5, color="gray")
ax.legend(fontsize=8, frameon=False)

# Panel B: natural-gradient vs Newton per-iteration cost per backend
ax = axes[1]
xs = np.arange(len(summary))
w = 0.36
natg = [s["natg"] for s in summary]
newton = [s["newton"] for s in summary]
ax.bar(xs - w / 2, natg, w, label="natural-gradient iter", color="#9ecae1")
ax.bar(xs + w / 2, newton, w, label="Newton iter", color="#3182bd")
ax.set_yscale("log")
ax.set_xticks(xs)
ax.set_xticklabels([s["label"] for s in summary], rotation=12, fontsize=8)
ax.set_ylabel("s / iteration (log)")
ax.set_title("(B) Natural-gradient vs Newton per-iteration cost")
ax.legend(fontsize=8, frameon=False)
ax.text(0, max(natg[0], newton[0]) * 1.6, f"+{summary[0]['jit']:.2f}s one-time JIT (GPU)",
        ha="center", fontsize=7, color="#1f77b4")

fig.tight_layout()
fig.savefig(OUT / "fig_perphase_runtime.pdf")
fig.savefig(OUT / "fig_perphase_runtime.png", dpi=150)
print("wrote fig_perphase_runtime.pdf/.png")
