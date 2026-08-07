#!/usr/bin/env python
"""Regenerate the coordinated final figures for the AMICA preprint.

The script is intentionally local and analysis-only. It reads completed benchmark
artifacts from the validation workspace, performs no model fitting, and writes the
five main and two supplementary vector PDFs used by ``zenodo.tex`` plus PNG
previews.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy import stats as scipy_stats

# NumPy 2 archives refer to ``numpy._core`` when they contain object arrays.
# Alias that private module path when regenerating figures under NumPy 1.x.
if "numpy._core" not in sys.modules:
    sys.modules["numpy._core"] = np.core
    sys.modules["numpy._core.multiarray"] = np.core.multiarray
    sys.modules["numpy._core.numeric"] = np.core.numeric
    sys.modules["numpy._core.umath"] = np.core.umath


from _paths import DATA_ROOT, OUT_ROOT, REPO_ROOT

HERE = Path(__file__).resolve().parent
FIG_DIR = OUT_ROOT
WORKSPACE = DATA_ROOT
REFERENCE_EVIDENCE = HERE / "fig2_reference_evidence.json"
REFERENCE_DENSITY = HERE / "fig2_reference_density.csv"
REFERENCE_BACKEND = HERE / "fig2_backend_vector_audit.csv"
FIG2_TOPOGRAPHY_ROOT = HERE / "fig2_topography_data"
FIG2_TOPOGRAPHY_MATCHED = FIG2_TOPOGRAPHY_ROOT / "matched_topographies.csv"
FIG2_TOPOGRAPHY_SUMMARY = FIG2_TOPOGRAPHY_ROOT / "topography_recovery_summary.csv"
FIG2_TOPOGRAPHY_SELECTED = FIG2_TOPOGRAPHY_ROOT / "panel_d_selected_sources.csv"
FIG2_TOPOGRAPHY_CONFIG = FIG2_TOPOGRAPHY_ROOT / "method_configuration.csv"
FIG2_TOPOGRAPHY_MAPS = FIG2_TOPOGRAPHY_ROOT / "figure2_topography_maps.npz"
FIG2_TOPOGRAPHY_MANIFEST = FIG2_TOPOGRAPHY_ROOT / "figure2_topography_manifest.json"
FIG1_EMPIRICAL_DENSITIES = HERE / "fig1_empirical_densities.csv"
FIG1_EMPIRICAL_AUDIT = HERE / "fig1_empirical_densities_audit.json"
FIG3_SYNTHETIC_AUDIT = HERE / "fig3_synthetic_recovery_audit.csv"
FIG5_TWO_REGIME = HERE / "fig5_two_regime_alignment.csv"

BENCH_505 = (
    WORKSPACE
    / "figdata/synth/amica-capsule/benchmark/cc_benchmark/results"
    / "v3_paper_stage1_cluster/benchmark_results.csv"
)
BENCH_504 = (
    WORKSPACE
    / "figdata/synth/amica-capsule/benchmark/cc_benchmark/results"
    / "ds004504_v3/benchmark_results.csv"
)
BENCH_621 = (
    WORKSPACE
    / "figdata/synth/amica-capsule/benchmark/cc_benchmark/results"
    / "ds004621_v3/benchmark_results.csv"
)
ITER_TRACE = BENCH_505.with_name("iteration_trace.csv.gz")
CONVERGENCE_AUDIT_CSV = WORKSPACE / "results/audit/fig5_convergence_subject_audit.csv"
CONVERGENCE_AUDIT_JSON = WORKSPACE / "results/audit/fig5_convergence_integrity.json"
AMICA_CONFIG_SOURCE = WORKSPACE / "figdata/synth/amica-capsule/amica_python/config.py"
AMICA_SOLVER_SOURCE = WORKSPACE / "figdata/synth/amica-capsule/amica_python/solver.py"
AMICA_LIKELIHOOD_SOURCE = WORKSPACE / "figdata/synth/amica-capsule/amica_python/likelihood.py"
AMICA_GPU_SUBMISSION = (
    WORKSPACE
    / "figdata/synth/amica-capsule/benchmark/cc_benchmark/submit_jax_gpu_v3.sh"
)
MEMORY_CSV = WORKSPACE / "results/mem_compare/mem_comparison_table.csv"
MEMORY_JSON_ROOT = WORKSPACE / "results/mem_compare"
MEMORY_MULTISUBJECT_ROOT = WORKSPACE / "results/mem_multisubj"
RUNTIME_GPU_ROOTS = {
    100: WORKSPACE / "results/rt_gpu_100",
    600: WORKSPACE / "results/rt_gpu_600",
}
FIG4_RUNTIME_AUDIT = HERE / "fig4_fixed_workload_runtime_audit.csv"
SCALING_ROOT = WORKSPACE / "results/scaling/cpu"
RUNTIME_RUNNER = (
    WORKSPACE / "figdata/synth/amica-capsule/amica_python/benchmark/runner.py"
)
MNE_INTEGRATION = (
    WORKSPACE / "figdata/synth/amica-capsule/amica_python/mne_integration.py"
)
MEMORY_MEASUREMENT = (
    WORKSPACE
    / "figdata/synth/amica-capsule/benchmark/comparator/runners/_common.py"
)
MEMORY_AMICA_RUNNER = (
    WORKSPACE
    / "figdata/synth/amica-capsule/benchmark/comparator/runners/run_amica_python.py"
)
MEMORY_SCOTT_RUNNER = (
    WORKSPACE
    / "figdata/synth/amica-capsule/benchmark/comparator/runners/run_scott_huberty.py"
)
MEMORY_PYAMICA_RUNNER = (
    WORKSPACE
    / "figdata/synth/amica-capsule/benchmark/comparator/runners/run_pyamica.py"
)
SCALING_SUBMISSION = (
    WORKSPACE
    / "figdata/synth/amica-capsule/benchmark/cc_benchmark/submit_scaling_cpu.sh"
)
SCALING_RUNNER = (
    WORKSPACE / "figdata/synth/amica-python/amica_python/benchmark/runner.py"
)
SYNTHETIC_JSON = WORKSPACE / "figdata/multimodel_synthetic_2000/synthetic_summary.json"
MULTIMODEL_AUDIT_CSV = WORKSPACE / "results/audit/fig6_multimodel_integrity.csv"
MULTIMODEL_AUDIT_JSON = WORKSPACE / "results/audit/fig6_multimodel_integrity.json"
MULTIMODEL_RUNNER = (
    WORKSPACE
    / "figdata/synth/amica-mm/scripts/cc_benchmark/run_multimodel_benchmark.py"
)
MULTIMODEL_SUBMISSION = (
    WORKSPACE
    / "figdata/synth/amica-mm/scripts/cc_benchmark/submit_multimodel_extra.sh"
)
MULTIMODEL_SYNTHETIC_RUNNER = (
    WORKSPACE
    / "figdata/synth/amica-mm/scripts/cc_benchmark/run_synthetic_multimodel.py"
)
MULTIMODEL_DEMO_SUBMISSION = (
    WORKSPACE
    / "figdata/synth/amica-mm/scripts/cc_benchmark/submit_multimodel_demo.sh"
)
MM_ROOTS = {
    "ds004505 task (120 ch)": WORKSPACE / "figdata/mmbench_ds004505",
    "ds004505 task (19 ch)": WORKSPACE / "figdata/mmbench_ds004505_ch19",
    "ds004504 rest (19 ch)": WORKSPACE / "figdata/mmbench_ds004504",
    "ds004621 rest (127 ch)": WORKSPACE / "figdata/mmbench_ds004621",
}
SURR_ROOTS = {
    "ds004505 phase surrogate": WORKSPACE / "figdata/mmbench_ds004505_surr",
    "ds004504 phase surrogate": WORKSPACE / "figdata/mmbench_ds004504_surr",
}
DEMO_NPZ = (
    WORKSPACE
    / "repos/amica-python/results/multimodel_demo/mm_demo_sub-04_M3.npz"
)
SINGLE_MODEL_SYNTHETIC_ROOTS = {
    ("Homogeneous Laplacian", "3,000"): WORKSPACE / "figdata/synth/amica_python_synthetic_v1",
    ("Homogeneous Laplacian", "10,000"): WORKSPACE / "figdata/synth/amica_python_synthetic_v1_lap_amica10k",
    ("Heterogeneous mixture", "3,000"): WORKSPACE / "figdata/synth/amica_python_synthetic_v1_mixed",
    ("Heterogeneous mixture", "10,000"): WORKSPACE / "figdata/synth/amica_python_synthetic_v1_mixed_amica10k",
}
SEED_ROBUSTNESS_CSV = WORKSPACE / "phaseB_figures/seed_robustness.csv"
PERPHASE_RUNTIME_CSV = WORKSPACE / "phaseB_figures/perphase_runtime.csv"
COMPONENT_METRICS = {
    "ds004505": BENCH_505.with_name("component_metrics.csv"),
    "ds004504": WORKSPACE / "results/ds004504_v3_50hz/component_metrics.csv",
    "ds004621": WORKSPACE / "results/ds004621_v3_50hz/component_metrics.csv",
}
GPU_SCALING_ROOT = WORKSPACE / "results/scaling/gpu"


# Okabe-Ito-based method palette, held constant across the figure set.
BLUE = "#0072B2"
LIGHT_BLUE = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
MAGENTA = "#CC79A7"
GREY = "#7A7A7A"
LIGHT_GREY = "#D0D0D0"
VERY_LIGHT_GREY = "#F4F5F6"
INK = "#202124"

COMPARATOR_COLORS = {"Picard": GREEN, "Infomax": ORANGE, "FastICA": MAGENTA}
COMPARATOR_MARKERS = {"Picard": "o", "Infomax": "s", "FastICA": "D"}
COMPARATOR_LABELS = {"Picard": "Picard", "Infomax": "Ext. Infomax", "FastICA": "FastICA"}
DATASET_ORDER = ["ds004505", "ds004504", "ds004621"]
# Accessions stay the join key everywhere; these task-descriptive names are the only
# thing a reader ever sees, so no figure has to be decoded against the config table.
DATASET_DISPLAY = {
    "ds004505": "Table tennis",
    "ds004504": "Eyes-closed rest",
    "ds004621": "Eyes-open rest",
}


def dataset_display(text: str) -> str:
    """Swap accessions for reader-facing names in a string that is about to be drawn.

    Render-time only: archived audit/provenance records keep their accessions.
    """
    for accession, name in DATASET_DISPLAY.items():
        text = text.replace(accession, name)
    return text
COMPARATOR_ORDER = ["Picard", "Infomax", "FastICA"]
REAL_CONTRAST_Y = np.asarray([0.0, 1.0, 2.0, 3.8, 4.8, 5.8, 7.6, 8.6, 9.6])
AMICA_BENCHMARK_METHOD = "AMICA-Python (JAX-GPU)"


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.85,
            "lines.linewidth": 1.9,
            "lines.markersize": 5.2,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def finish_axes(ax: mpl.axes.Axes, grid_axis: str | None = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#D9DCE0", linewidth=0.6, alpha=0.9)
        ax.set_axisbelow(True)


def panel_title(ax: mpl.axes.Axes, label: str, title: str, y: float = 1.04) -> None:
    ax.text(
        0.0,
        y,
        label,
        transform=ax.transAxes,
        fontsize=11.5,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=INK,
    )
    ax.text(
        0.13,
        y,
        title,
        transform=ax.transAxes,
        fontsize=9.5,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=INK,
    )


def save_figure(
    fig: mpl.figure.Figure,
    stem: str,
    *,
    svg: bool = False,
    png_dpi: int = 240,
) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=png_dpi, bbox_inches="tight", pad_inches=0.04)
    if svg:
        fig.savefig(HERE / f"{stem}.svg", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def _box(
    ax: mpl.axes.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    fc: str = "white",
    ec: str = INK,
    fontsize: float = 7.5,
    weight: str = "normal",
    radius: float = 0.025,
    text_color: str = INK,
) -> FancyBboxPatch:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=0.9,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=text_color,
        linespacing=1.18,
    )
    return patch


def _arrow(
    ax: mpl.axes.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = INK,
    lw: float = 1.0,
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=lw,
            color=color,
            connectionstyle=connectionstyle,
        )
    )


# ---------------------------------------------------------------------------
# Figure 1: conceptual model, workflow, implementation, validation
# ---------------------------------------------------------------------------
def make_figure1() -> None:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    fig.subplots_adjust(
        left=0.035,
        right=0.985,
        bottom=0.055,
        top=0.88,
        wspace=0.10,
    )
    fig.suptitle(
        "AMICA model and optimisation workflow",
        fontsize=10.8,
        fontweight="bold",
        y=0.985,
    )
    for ax in axes.ravel():
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(
            FancyBboxPatch(
                (0.005, 0.005),
                0.99,
                0.99,
                boxstyle="round,pad=0.008,rounding_size=0.02",
                facecolor="#FCFCFD",
                edgecolor="#DADDE1",
                linewidth=0.8,
                zorder=-10,
            )
        )

    # A: generative model and default density range
    ax = axes[0]
    panel_title(ax, "A", "Generative AMICA model", y=0.92)
    ax.add_patch(Circle((0.075, 0.68), 0.050, facecolor="white", edgecolor=INK, linewidth=1.0))
    ax.text(0.075, 0.68, r"$z_t$", ha="center", va="center", fontsize=8.7)
    ax.text(0.075, 0.595, r"$z_t\sim\mathrm{Cat}(\pi)$", ha="center", va="top", fontsize=7.3, color=GREY)
    for dx, dy, alpha in [(0.018, 0.025, 0.45), (0.009, 0.012, 0.7), (0, 0, 1.0)]:
        patch = FancyBboxPatch(
            (0.19 + dx, 0.605 + dy),
            0.225,
            0.145,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor="white",
            edgecolor=BLUE,
            linewidth=0.9,
            alpha=alpha,
        )
        ax.add_patch(patch)
    ax.text(0.302, 0.702, "model h", ha="center", va="center", fontsize=8.2, fontweight="bold")
    ax.text(0.302, 0.647, r"$W^{(h)}, c^{(h)}, \pi^{(h)}$", ha="center", va="center", fontsize=7.4)
    _arrow(ax, (0.128, 0.68), (0.18, 0.68))
    _arrow(ax, (0.43, 0.68), (0.475, 0.68))
    rng = np.random.default_rng(7)
    t = np.linspace(0, 1, 45)
    for offset in (0.735, 0.68, 0.625):
        y = offset + 0.018 * np.cumsum(rng.normal(size=t.size)) / np.sqrt(t.size)
        ax.plot(0.475 + 0.13 * t, y, color="#2A7F80", linewidth=0.9)
    ax.text(0.54, 0.785, "sources", ha="center", fontsize=7.2, color=GREY)
    _box(ax, (0.635, 0.60), 0.10, 0.16, r"$A^{(h)}$", fc="#E7EBF0", fontsize=9.5, weight="bold")
    _arrow(ax, (0.61, 0.68), (0.625, 0.68))
    _arrow(ax, (0.75, 0.68), (0.79, 0.68))
    x1 = rng.normal(size=120)
    x2 = 0.60 * x1 + rng.normal(scale=0.55, size=120)
    ax.scatter(0.885 + 0.034 * x1, 0.68 + 0.034 * x2, s=3.0, color=BLUE, alpha=0.38, linewidths=0)
    ax.text(0.885, 0.79, r"observed $x_t$", ha="center", fontsize=7.5, fontweight="bold")
    ax.text(0.69, 0.535, r"$x_t=A^{(h)}s_t+c^{(h)}$", ha="center", fontsize=8.0)
    ax.text(0.055, 0.435, "Empirical adaptive densities", fontsize=8.2, fontweight="bold")
    ax.text(
        0.945,
        0.397,
        r"Table tennis sub-01; default $1\leq\rho\leq2$",
        ha="right",
        fontsize=6.7,
        color=GREY,
    )
    empirical = pd.read_csv(FIG1_EMPIRICAL_DENSITIES)
    components = (
        empirical[["component", "mean_rho"]]
        .drop_duplicates()
        .sort_values("mean_rho")
    )
    for index, row in enumerate(components.itertuples(index=False)):
        inset = ax.inset_axes([0.055 + 0.31 * index, 0.080, 0.27, 0.235])
        values = empirical[empirical.component == row.component]
        width = float(np.median(np.diff(values.activation)))
        inset.bar(
            values.activation,
            np.clip(values.empirical_density, 1e-4, None),
            width=width,
            color="#D7D9DC",
            edgecolor="none",
            rasterized=True,
        )
        inset.plot(values.activation, values.fitted_density, color=BLUE, linewidth=1.25)
        inset.set_yscale("log")
        inset.set_ylim(1e-4, max(values.fitted_density.max(), values.empirical_density.max()) * 1.35)
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_title(rf"IC {int(row.component)}  $\bar{{\rho}}={row.mean_rho:.2f}$", fontsize=6.5, pad=1.5)
        for spine in inset.spines.values():
            spine.set_color("#B8BCC1")
            spine.set_linewidth(0.55)
    ax.text(
        0.50,
        0.030,
        "grey: empirical activations   blue: fitted mixture density",
        ha="center",
        fontsize=6.4,
        color=GREY,
    )

    # B: optimisation path
    ax = axes[1]
    panel_title(ax, "B", "Optimisation path", y=0.92)
    rng = np.random.default_rng(13)
    heat = rng.normal(size=(8, 10))
    ax.imshow(heat, extent=(0.030, 0.135, 0.60, 0.79), cmap="GnBu", aspect="auto", interpolation="nearest")
    ax.add_patch(Rectangle((0.030, 0.60), 0.105, 0.19, fill=False, edgecolor=INK, linewidth=0.8))
    ax.text(0.0825, 0.82, "EEG", ha="center", fontsize=7.6, fontweight="bold")
    _arrow(ax, (0.145, 0.695), (0.163, 0.695))
    _box(
        ax,
        (0.173, 0.615),
        0.295,
        0.14,
        "centre + sphere\nPCA rank $N$",
        fc="white",
        fontsize=6.6,
        weight="bold",
    )
    _arrow(ax, (0.488, 0.695), (0.506, 0.695))
    outer = FancyBboxPatch(
        (0.516, 0.515),
        0.462,
        0.31,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor="#FBFCFE",
        edgecolor=INK,
        linestyle="--",
        linewidth=0.9,
    )
    ax.add_patch(outer)
    ax.text(0.747, 0.785, "EM loop", ha="center", fontsize=8.1, fontweight="bold")
    _box(ax, (0.545, 0.625), 0.165, 0.105, "E-step\n$p(k,h\\mid x)$", fc="#F2F6FA", fontsize=7.0, weight="bold")
    _box(ax, (0.762, 0.61), 0.19, 0.13, "M-step\nupdate $W$\nand density", fc="#F2F6FA", fontsize=6.3, weight="bold")
    _arrow(ax, (0.726, 0.678), (0.748, 0.678))
    ax.text(0.747, 0.555, "natural-gradient / Newton", ha="center", fontsize=6.9, color=GREY)

    _arrow(ax, (0.747, 0.50), (0.747, 0.445))
    _box(
        ax,
        (0.39, 0.335),
        0.58,
        0.105,
        "iterate until\ntolerance or iteration cap",
        fc="#E9F1F8",
        ec=BLUE,
        fontsize=7.4,
        weight="bold",
    )
    _box(
        ax,
        (0.045, 0.335),
        0.255,
        0.075,
        "optional rejection",
        fc="#F7F9FB",
        ec="#B9C2CA",
        fontsize=6.9,
        text_color=GREY,
    )
    ax.plot([0.30, 0.38], [0.373, 0.373], color="#8A8A8A", linewidth=0.8, linestyle=":")
    _arrow(ax, (0.68, 0.325), (0.68, 0.248))
    # _box() inflates its rectangle by the boxstyle pad (0.012) on every side, so the
    # drawn top of these boxes is y + height + 0.012; the arrows must stop above that.
    # FancyArrowPatch sizes its head in points, independently of the shaft, so a
    # short drop is consumed almost entirely by the head and reads as a floating
    # arrowhead rather than an arrow. These two boxes sit lower than the drop
    # from the EM loop needs, which leaves the branch arrows room to match it:
    # each now spans 0.078 against the 0.077 of the arrow above. The boxes are
    # drawn to y+height+0.012 = 0.165, so the shafts stop just above that.
    _box(ax, (0.24, 0.048), 0.26, 0.105, "stopping\nstatus", fc="#F2F6FA", fontsize=7.2, weight="bold")
    _box(ax, (0.58, 0.048), 0.30, 0.105, "full likelihood\ntrajectory", fc="#F2F6FA", fontsize=7.2, weight="bold")
    ax.plot([0.37, 0.73], [0.245, 0.245], color=INK, linewidth=0.8)
    _arrow(ax, (0.37, 0.245), (0.37, 0.167), lw=0.8)
    _arrow(ax, (0.73, 0.245), (0.73, 0.167), lw=0.8)

    save_figure(fig, "fig_amica_workflow", svg=True)


# ---------------------------------------------------------------------------
# Figure 3: single-model real-EEG benchmark
# ---------------------------------------------------------------------------
def load_real_benchmarks() -> pd.DataFrame:
    frames = []
    for ds, path in zip(DATASET_ORDER, [BENCH_505, BENCH_504, BENCH_621]):
        d = pd.read_csv(path)
        d["dataset"] = ds
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def bootstrap_ci(values: np.ndarray, statistic: str, seed: int = 0) -> tuple[float, float]:
    """Return a deterministic percentile-bootstrap interval over subjects."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, values.size, size=(10_000, values.size))]
    if statistic == "mean":
        boot = samples.mean(axis=1)
    elif statistic == "dz":
        sd = samples.std(axis=1, ddof=1)
        boot = np.divide(samples.mean(axis=1), sd, out=np.full_like(sd, np.nan), where=sd > 0)
    else:
        raise ValueError(statistic)
    return tuple(np.nanpercentile(boot, [2.5, 97.5]).tolist())


def real_contrasts(df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    required_metrics = ["mir_kbits_s", "remnant_pmi_percent", "nd_5_percent", "nd_10_percent"]
    for ds in DATASET_ORDER:
        d = df[df["dataset"] == ds]
        amica = d[d["method"] == AMICA_BENCHMARK_METHOD].set_index("subject")
        if amica.empty or not amica.index.is_unique:
            raise ValueError(f"{ds}: expected one {AMICA_BENCHMARK_METHOD} row per subject")
        for comparator in COMPARATOR_ORDER:
            comp = d[d["method"] == comparator].set_index("subject")
            if comp.empty or not comp.index.is_unique:
                raise ValueError(f"{ds}: expected one {comparator} row per subject")
            subject_mismatch = amica.index.symmetric_difference(comp.index)
            if len(subject_mismatch):
                raise ValueError(f"{ds}/{comparator}: unmatched subjects: {subject_mismatch.tolist()}")
            # Uniqueness is checked above; omit the newer pandas ``validate``
            # keyword so the archived figure environment can render the set.
            paired = amica.join(comp, lsuffix="_a", rsuffix="_c", how="inner")
            required = [f"{metric}_{suffix}" for metric in required_metrics for suffix in ("a", "c")]
            if not np.isfinite(paired[required].to_numpy(float)).all():
                raise ValueError(f"{ds}/{comparator}: non-finite Figure 3 metric")
            mir = (paired["mir_kbits_s_a"] - paired["mir_kbits_s_c"]).to_numpy(float)
            pmi = (paired["remnant_pmi_percent_a"] - paired["remnant_pmi_percent_c"]).to_numpy(float)
            rv5 = (paired["nd_5_percent_a"] - paired["nd_5_percent_c"]).to_numpy(float)
            rv10 = (paired["nd_10_percent_a"] - paired["nd_10_percent_c"]).to_numpy(float)
            dz = float(np.mean(mir) / np.std(mir, ddof=1))
            row = {
                "dataset": ds,
                "comparator": comparator,
                "n": len(mir),
                "same_sign": int(np.sum(mir > 0)),
                "paired_t_p": float(scipy_stats.ttest_1samp(mir, popmean=0.0).pvalue),
                "wilcoxon_p": float(scipy_stats.wilcoxon(mir, alternative="two-sided").pvalue),
                "mir": mir,
                "mir_mean": float(np.mean(mir)),
                "mir_ci": bootstrap_ci(mir, "mean"),
                "dz": dz,
                "dz_ci": bootstrap_ci(mir, "dz"),
                "pmi": pmi,
                "pmi_mean": float(np.mean(pmi)),
                "pmi_ci": bootstrap_ci(pmi, "mean"),
                "rv5": rv5,
                "rv5_mean": float(np.mean(rv5)),
                "rv5_ci": bootstrap_ci(rv5, "mean"),
                "rv10": rv10,
                "rv10_mean": float(np.mean(rv10)),
                "rv10_ci": bootstrap_ci(rv10, "mean"),
            }
            rows.append(row)
    if len(rows) != len(DATASET_ORDER) * len(COMPARATOR_ORDER):
        raise ValueError("Figure 3 requires nine dataset-by-comparator contrasts")
    p_values = np.asarray([row["paired_t_p"] for row in rows], dtype=float)
    order = np.argsort(p_values)
    running_max = 0.0
    adjusted = np.empty_like(p_values)
    for rank, index in enumerate(order):
        running_max = max(running_max, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(running_max, 1.0)
    for row, p_holm in zip(rows, adjusted):
        row["holm_adjusted_p"] = float(p_holm)
    return rows


def add_dataset_bands(
    ax: mpl.axes.Axes,
    contrasts: list[dict],
    *,
    label_mode: str | None,
    label_offset: float = 0.95,
    label_x: float = 0.98,
    label_ha: str = "right",
    label_fontsize: float = 6.7,
) -> None:
    dataset_ns = {row["dataset"]: row["n"] for row in contrasts}
    for idx, ds in enumerate(DATASET_ORDER):
        group_y = REAL_CONTRAST_Y[idx * len(COMPARATOR_ORDER) : (idx + 1) * len(COMPARATOR_ORDER)]
        if idx % 2 == 0:
            ax.axhspan(group_y[0] - 0.45, group_y[-1] + 0.45, color="#F1F3F5", zorder=-20)
        # Two lines: the task-descriptive name is wider than the old accession and
        # would otherwise overrun these four narrow panels.
        ax.text(
            label_x,
            group_y[0] - label_offset,
            f"{DATASET_DISPLAY[ds]}\n" + rf"$n={dataset_ns[ds]}$",
            transform=ax.get_yaxis_transform(),
            ha=label_ha,
            va="center",
            linespacing=1.25,
            fontsize=label_fontsize,
            fontweight="bold",
            color="#5E6369",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.6},
        )
    ax.set_ylim(REAL_CONTRAST_Y[-1] + 0.55, -1.65)
    ax.set_yticks(REAL_CONTRAST_Y)
    if label_mode == "mir_counts":
        labels = [
            f"{COMPARATOR_LABELS[row['comparator']]}  {row['same_sign']}/{row['n']}"
            for row in contrasts
        ]
    elif label_mode == "methods":
        labels = [COMPARATOR_LABELS[row["comparator"]] for row in contrasts]
    elif label_mode is None:
        labels = [""] * len(contrasts)
    else:
        raise ValueError(f"Unknown label_mode={label_mode!r}")
    ax.set_yticklabels(labels)


def _figure3_integrity_stats(df: pd.DataFrame, contrasts: list[dict]) -> dict:
    input_paths = {"ds004505": BENCH_505, "ds004504": BENCH_504, "ds004621": BENCH_621}
    configs: dict[str, dict] = {}
    for ds in DATASET_ORDER:
        d = df[df["dataset"] == ds]
        amica = d[d["method"] == AMICA_BENCHMARK_METHOD]
        configs[ds] = {
            "n_subjects": int(amica["subject"].nunique()),
            "subjects": sorted(amica["subject"].astype(str).tolist()),
            "input_channel_count_range": [int(d["n_channels_input"].min()), int(d["n_channels_input"].max())],
            "analysis_channel_counts": sorted(d["n_channels_ica"].astype(int).unique().tolist()),
            "retained_component_counts": sorted(d["n_components"].astype(int).unique().tolist()),
            "rank_metadata": sorted(d["rank"].astype(int).unique().tolist()),
            "iclabel_complete_rows": int(d["iclabel_brain_percent"].notna().sum()),
            "total_method_rows": int(len(d)),
        }
    reports = []
    for row in contrasts:
        reports.append(
            {
                "dataset": row["dataset"],
                "comparator": row["comparator"],
                "n": int(row["n"]),
                "positive_delta_mir": int(row["same_sign"]),
                "paired_t_p": row["paired_t_p"],
                "holm_adjusted_p_across_nine_mir_contrasts": row["holm_adjusted_p"],
                "wilcoxon_p": row["wilcoxon_p"],
                "delta_mir_mean_kbits_s": row["mir_mean"],
                "delta_mir_bootstrap_95_ci": list(row["mir_ci"]),
                "cohen_dz": row["dz"],
                "cohen_dz_bootstrap_95_ci": list(row["dz_ci"]),
                "delta_remnant_pmi_mean_percentage_points": row["pmi_mean"],
                "delta_remnant_pmi_bootstrap_95_ci": list(row["pmi_ci"]),
                "delta_near_dipolar_rv5_mean_percentage_points": row["rv5_mean"],
                "delta_near_dipolar_rv5_bootstrap_95_ci": list(row["rv5_ci"]),
                "delta_near_dipolar_rv10_mean_percentage_points": row["rv10_mean"],
                "delta_near_dipolar_rv10_bootstrap_95_ci": list(row["rv10_ci"]),
            }
        )
    n_subjects = sum(configs[ds]["n_subjects"] for ds in DATASET_ORDER)
    n_subject_comparator = sum(row["n"] for row in contrasts)
    n_positive = sum(row["same_sign"] for row in contrasts)
    return {
        "input_files": {
            ds: {
                "path": path.relative_to(WORKSPACE).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for ds, path in input_paths.items()
        },
        "amica_reference_row": AMICA_BENCHMARK_METHOD,
        "comparators": COMPARATOR_ORDER,
        "dataset_configuration": configs,
        "n_subjects": int(n_subjects),
        "n_dataset_by_comparator_contrasts": int(len(contrasts)),
        "n_subject_comparator_contrasts": int(n_subject_comparator),
        "positive_subject_comparator_contrasts": int(n_positive),
        "all_mir_contrasts_positive": bool(n_positive == n_subject_comparator),
        "sign_convention": "amica minus comparator in every panel",
        "endpoint_scope": (
            "Complete MIR was evaluated in-sample on the recordings used to fit "
            "each decomposition, under the prespecified method configurations and "
            "optimisation budgets."
        ),
        "units": {
            "mir": "kbits/s",
            "remnant_pmi": "percentage points (difference of percentages)",
            "near_dipolar_fraction": "percentage points (difference of percentages)",
        },
        "metric_definitions": {
            "complete_mir": (
                "sum marginal input entropies - sum marginal source entropies + "
                "log2|det(W)| in retained PCA-rank space; 100-bin histograms, "
                "+/-5 SD clipping, 20,000 samples, seed 42; converted from "
                "bits/sample to kbits/s with the sampling frequency"
            ),
            "remnant_pmi": (
                "100 * mean off-diagonal source pairwise MI / mean off-diagonal "
                "input pairwise MI; 32-bin two-dimensional histograms on row-wise "
                "z-scored data clipped to +/-5 SD, 20,000 samples, seed 42"
            ),
            "dipolarity": (
                "percentage of fitted component maps with equivalent-dipole residual "
                "variance <=5% or <=10%, using the Frank et al. four-shell sphere"
            ),
        },
        "bootstrap": {
            "resampling_unit": "paired subjects within each dataset-by-comparator contrast",
            "resamples": 10_000,
            "interval": "2.5th and 97.5th percentiles",
            "rng_seed": 0,
        },
        "contrasts": reports,
    }


def make_figure2_reference() -> dict:
    """Build reference-parity and clean known-topography recovery panels."""
    set_style()
    evidence = json.loads(REFERENCE_EVIDENCE.read_text(encoding="utf-8"))
    density = pd.read_csv(REFERENCE_DENSITY)
    matched = pd.read_csv(FIG2_TOPOGRAPHY_MATCHED)
    recovery_summary = pd.read_csv(FIG2_TOPOGRAPHY_SUMMARY)
    selected = pd.read_csv(FIG2_TOPOGRAPHY_SELECTED).sort_values("rank")
    method_configuration = pd.read_csv(FIG2_TOPOGRAPHY_CONFIG)
    topography_manifest = json.loads(
        FIG2_TOPOGRAPHY_MANIFEST.read_text(encoding="utf-8")
    )
    topography_maps = np.load(FIG2_TOPOGRAPHY_MAPS, allow_pickle=False)

    fig = plt.figure(figsize=(7.2, 6.95))
    outer = fig.add_gridspec(
        2,
        1,
        left=0.105,
        right=0.985,
        bottom=0.075,
        top=0.965,
        height_ratios=[0.93, 1.07],
        hspace=0.58,
    )
    top = outer[0, 0].subgridspec(1, 2, wspace=0.31)
    bottom = outer[1, 0].subgridspec(
        1, 2, width_ratios=[0.38, 0.62], wspace=0.27
    )

    # a: two aligned logarithmic axes expose both objective and matrix agreement.
    a_grid = top[0, 0].subgridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.22)
    ax_a_ll = fig.add_subplot(a_grid[0, 0])
    ax_a_w = fig.add_subplot(a_grid[0, 1], sharey=ax_a_ll)
    parity = evidence["panel_a"]
    y = np.arange(len(parity))[::-1]
    ll = np.asarray([row["relative_ll_difference_pct"] for row in parity], dtype=float)
    mismatch = 1.0 - np.asarray([row["worst_row_correlation"] for row in parity], dtype=float)
    labels = [dataset_display(row["label"]) for row in parity]
    ax_a_ll.hlines(y, 1e-8, ll, color=LIGHT_GREY, linewidth=1.0)
    ax_a_ll.scatter(ll, y, s=33, c=BLUE, edgecolors=INK, linewidths=0.6, zorder=4)
    ax_a_w.hlines(y, 1e-12, mismatch, color=LIGHT_GREY, linewidth=1.0)
    ax_a_w.scatter(mismatch, y, s=33, c=LIGHT_BLUE, edgecolors=INK, linewidths=0.6, zorder=4)
    ax_a_ll.set_xscale("log")
    ax_a_w.set_xscale("log")
    ax_a_ll.set_xlim(8e-8, 2e-1)
    ax_a_w.set_xlim(8e-12, 2e-3)
    ax_a_ll.set_yticks(y, labels)
    ax_a_w.tick_params(labelleft=False)
    ax_a_ll.set_xlabel("Relative final LL\ndifference (%)", fontsize=7.8)
    ax_a_w.set_xlabel("Worst-row mismatch\n" + r"$1-r_W^{\min}$", fontsize=7.8)
    ax_a_w.text(0.98, 0.985, "lower is better", transform=ax_a_w.transAxes, fontsize=6.6, color=GREY, ha="right", va="top")
    panel_title(ax_a_ll, "A", "Fortran reference parity", y=1.08)
    finish_axes(ax_a_ll, "x")
    finish_axes(ax_a_w, "x")
    # Header band above the top row so the "lower is better" note clears every marker.
    ax_a_ll.set_ylim(-0.35, 3.85)

    # b: aligned density terms with both correlation and scale-sensitive error.
    b_grid = top[0, 1].subgridspec(2, 2, wspace=0.34, hspace=0.40)
    density_axes = [fig.add_subplot(b_grid[i, j]) for i in range(2) for j in range(2)]
    parameter_specs = [
        ("alpha", r"Weight, $\alpha$"),
        ("mu", r"Location, $\mu$"),
        ("beta", r"Scale, $\beta$"),
        ("rho", r"Shape, $\rho$"),
    ]
    density_stats: dict[str, dict[str, float]] = {}
    for idx, (key, title) in enumerate(parameter_specs):
        ax = density_axes[idx]
        x = density[f"{key}_fortran"].to_numpy(float)
        yy = density[f"{key}_python"].to_numpy(float)
        lo = float(min(x.min(), yy.min()))
        hi = float(max(x.max(), yy.max()))
        pad = max((hi - lo) * 0.09, 1e-4)
        corr = float(np.corrcoef(x, yy)[0, 1])
        iqr = float(np.subtract(*np.percentile(x, [75, 25])))
        nrmse = float(np.sqrt(np.mean((yy - x) ** 2)) / iqr)
        medae = float(np.median(np.abs(yy - x)))
        density_stats[key] = {"pearson_r": corr, "nrmse_over_iqr": nrmse, "median_absolute_error": medae}
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=GREY, linestyle="--", linewidth=0.9)
        ax.scatter(x, yy, s=18, color=BLUE, alpha=0.78, edgecolors=INK, linewidths=0.35)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(title, fontsize=8.2, pad=2)
        ax.text(
            0.04,
            0.94,
            f"n=18\nr={corr:.6f}\nnRMSE/IQR={nrmse:.1e}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.8,
            color=INK,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.35},
        )
        if idx >= 2:
            ax.set_xlabel("Fortran", labelpad=1)
        if idx % 2 == 0:
            ax.set_ylabel("Python", labelpad=1)
        finish_axes(ax, None)
    panel_title(density_axes[0], "B", "Density-parameter agreement", y=1.20)

    # c: descriptive recovery across all 32 planted maps in one clean fixture.
    ax_c = fig.add_subplot(bottom[0, 0])
    method_order = [
        "amica_3000",
        "amica_10000",
        "picard",
        "extended_infomax",
        "fastica",
    ]
    method_labels = ["AMICA\n3,000", "AMICA\n10,000", "Picard", "Ext.\nInfomax", "FastICA"]
    method_colours = [BLUE, "#004B73", GREEN, ORANGE, MAGENTA]
    source_order = np.sort(matched["planted_source_index"].unique())
    source_jitter = np.linspace(-0.15, 0.15, len(source_order))
    jitter_lookup = dict(zip(source_order, source_jitter))

    for source in source_order:
        source_rows = matched.loc[matched.planted_source_index == source].set_index("method")
        values = source_rows.loc[method_order, "abs_r"].to_numpy(float)
        x_values = np.arange(len(method_order), dtype=float) + jitter_lookup[source]
        ax_c.plot(x_values, values, color=GREY, linewidth=0.45, alpha=0.11, zorder=1)

    recovery_stats: dict[str, dict[str, float | int | str]] = {}
    config_lookup = method_configuration.set_index("method")
    for index, (method, colour) in enumerate(zip(method_order, method_colours)):
        rows = matched.loc[matched.method == method].sort_values("planted_source_index")
        values = rows["abs_r"].to_numpy(float)
        x_values = index + rows["planted_source_index"].map(jitter_lookup).to_numpy(float)
        ax_c.scatter(
            x_values,
            values,
            s=14,
            color=colour,
            alpha=0.55,
            edgecolors="white",
            linewidths=0.25,
            zorder=3,
        )
        q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
        ax_c.vlines(index, q1, q3, color=INK, linewidth=2.0, zorder=6)
        ax_c.scatter(
            index,
            median,
            s=42,
            marker="D",
            color=colour,
            edgecolors=INK,
            linewidths=0.7,
            zorder=7,
        )
        recovery_stats[method] = {
            "display_name": str(config_lookup.loc[method, "display_name"]),
            "n": int(values.size),
            "median_abs_r": float(median),
            "q1_abs_r": float(q1),
            "q3_abs_r": float(q3),
            "minimum_abs_r": float(values.min()),
            "actual_n_iter": int(config_lookup.loc[method, "actual_n_iter"]),
            "stopping_reason": str(config_lookup.loc[method, "stopping_reason"]),
        }
    ax_c.set_xlim(-0.45, len(method_order) - 0.55)
    ax_c.set_ylim(0.0, 1.02)
    ax_c.set_xticks(np.arange(len(method_order)), method_labels, rotation=24, ha="right")
    ax_c.set_ylabel(r"Matched topographic correlation, $|r|$")
    ax_c.text(
        0.02,
        0.04,
        "one point per planted map\ndiamonds: median; bars: IQR",
        transform=ax_c.transAxes,
        fontsize=6.2,
        color=GREY,
        va="bottom",
    )
    panel_title(ax_c, "C", "Known-topography recovery", y=1.10)
    finish_axes(ax_c, "y")

    # d: the same three prespecified minimax sources across all configurations.
    import mne

    ax_d_header = fig.add_subplot(bottom[0, 1], frame_on=False)
    ax_d_header.set_axis_off()
    ax_d_header.patch.set_visible(False)
    panel_title(ax_d_header, "D", "Common difficult topographies", y=1.10)
    d_grid = bottom[0, 1].subgridspec(3, 6, wspace=0.03, hspace=0.22)
    d_axes = [[fig.add_subplot(d_grid[row, col]) for col in range(6)] for row in range(3)]
    column_titles = ["Ground\ntruth", "AMICA\n3,000", "AMICA\n10,000", "Picard", "Ext.\nInfomax", "FastICA"]
    map_keys = [
        "A_true",
        "A_est_aligned_amica_3000",
        "A_est_aligned_amica_10000",
        "A_est_aligned_picard",
        "A_est_aligned_extended_infomax",
        "A_est_aligned_fastica",
    ]
    sensor_positions = np.asarray(topography_maps["sensor_positions"], dtype=float)
    matched_lookup = matched.set_index(["method", "planted_source_index"])
    selected_records: list[dict[str, object]] = []
    for row, selected_row in enumerate(selected.itertuples(index=False)):
        source = int(selected_row.planted_source_index)
        selected_record = {
            "rank": int(selected_row.rank),
            "planted_source_index": source,
            "source_label": str(selected_row.source_label),
            "source_hemi": str(selected_row.source_hemi),
            "q_best_method": float(selected_row.q_best_method),
        }
        selected_records.append(selected_record)
        for col, (title, key) in enumerate(zip(column_titles, map_keys)):
            ax = d_axes[row][col]
            values = np.asarray(topography_maps[key][:, source], dtype=float)
            values = values - values.mean()
            scale = float(np.max(np.abs(values)))
            if not np.isfinite(scale) or scale <= np.finfo(float).eps:
                raise ValueError(f"invalid topography scale for {key}, source {source}")
            values /= scale
            mne.viz.plot_topomap(
                values,
                sensor_positions,
                axes=ax,
                show=False,
                contours=0,
                cmap="RdBu_r",
                vlim=(-1.0, 1.0),
                sensors=False,
                outlines="head",
                extrapolate="head",
                sphere=None,
            )
            if row == 0:
                ax.set_title(title, fontsize=6.8, fontweight="bold", pad=3)
            if col == 0:
                ax.text(
                    -0.15,
                    0.5,
                    f"Difficult {row + 1}\nsource {source + 1}",
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=6.0,
                    color=INK,
                )
            else:
                method = method_order[col - 1]
                correlation = float(matched_lookup.loc[(method, source), "abs_r"])
                ax.text(
                    0.5,
                    -0.02,
                    rf"$|r|={correlation:.3f}$",
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=5.8,
                    color=INK,
                )
    d_axes[-1][-1].text(
        1.0,
        -0.28,
        "centred and independently scaled for display",
        transform=d_axes[-1][-1].transAxes,
        ha="right",
        va="top",
        fontsize=5.5,
        color=GREY,
    )

    save_figure(fig, "fig2_reference_agreement", png_dpi=600)
    return {
        "inputs": {
            "evidence": str(REFERENCE_EVIDENCE.resolve()),
            "density": str(REFERENCE_DENSITY.resolve()),
            "topography_manifest": str(FIG2_TOPOGRAPHY_MANIFEST.resolve()),
            "topography_matches": str(FIG2_TOPOGRAPHY_MATCHED.resolve()),
            "topography_maps": str(FIG2_TOPOGRAPHY_MAPS.resolve()),
        },
        "panel_a": parity,
        "density_parameter_agreement": density_stats,
        "known_topography_recovery": recovery_stats,
        "selected_common_difficult_sources": selected_records,
        "topography_simulation": topography_manifest["simulation"],
        "topography_recovery_summary": recovery_summary.to_dict(orient="records"),
        "limitations": evidence["limitations"],
    }


def _summary_errorbar(ax, x: float, ci: tuple[float, float], y: float, color: str, marker: str, *, size: float = 5.8) -> None:
    ax.errorbar(
        x,
        y,
        xerr=[[x - ci[0]], [ci[1] - x]],
        fmt=marker,
        markersize=size,
        markerfacecolor=color,
        markeredgecolor="black",
        markeredgewidth=0.65,
        ecolor="black",
        elinewidth=1.1,
        capsize=0,
        zorder=8,
    )


SYNTHETIC_CONDITIONS = ["clean", "noise", "noise_eog"]
SYNTHETIC_METHODS = ["amica 3k", "amica 10k", "Picard", "Infomax", "FastICA"]
SYNTHETIC_METHOD_STYLE = {
    "amica 3k": (BLUE, "o", "-"),
    "amica 10k": ("#004B73", "s", "--"),
    "Picard": (GREEN, "^", "-"),
    "Infomax": (ORANGE, "D", "-"),
    "FastICA": (MAGENTA, "P", "-"),
}


def _load_single_model_synthetic() -> pd.DataFrame:
    rows: list[dict] = []
    method_names = {
        "jax_gpu": "amica 3k",
        "picard": "Picard",
        "infomax": "Infomax",
        "fastica": "FastICA",
    }
    for (regime, budget), root in SINGLE_MODEL_SYNTHETIC_ROOTS.items():
        for path in sorted(root.glob("synth_*.json")):
            with path.open(encoding="utf-8") as handle:
                record = json.load(handle)
            metadata = record["_synthetic"]
            condition = str(metadata["condition_id"])
            method_tag = str(metadata["method_tag"])
            if condition not in SYNTHETIC_CONDITIONS:
                continue
            if budget == "10,000" and method_tag != "jax_gpu":
                continue
            if budget == "3,000" and method_tag not in method_names:
                continue
            method = "amica 10k" if budget == "10,000" else method_names[method_tag]
            method_record = record[method_tag]
            truth = method_record["ground_truth"]
            rows.append(
                {
                    "regime": regime,
                    "condition": condition,
                    "method": method,
                    "seed": int(metadata["seed"]),
                    "iteration_budget": int(method_record["max_iter"]),
                    "r_topo_median": float(truth["r_topo_median"]),
                    "r_source_median": float(truth["r_source_median"]),
                    "amari_index": float(truth["amari_index"]),
                    "source_file": str(path.relative_to(WORKSPACE)).replace("\\", "/"),
                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    data = pd.DataFrame(rows)
    expected = pd.MultiIndex.from_product(
        [
            ["Homogeneous Laplacian", "Heterogeneous mixture"],
            SYNTHETIC_CONDITIONS,
            SYNTHETIC_METHODS,
        ],
        names=["regime", "condition", "method"],
    )
    counts = data.groupby(["regime", "condition", "method"]).size().reindex(expected)
    if counts.isna().any() or not (counts == 10).all():
        raise ValueError(f"Incomplete single-model synthetic archive:\n{counts}")
    data.to_csv(FIG3_SYNTHETIC_AUDIT, index=False, float_format="%.17g")
    return data


def _plot_synthetic_conditions(
    ax: mpl.axes.Axes,
    data: pd.DataFrame,
    metric: str,
    *,
    show_ylabel: bool,
) -> None:
    labels = ["clean", "noise", "noise +\nEOG"]
    offsets = np.linspace(-0.22, 0.22, len(SYNTHETIC_METHODS))
    for method, offset in zip(SYNTHETIC_METHODS, offsets):
        color, marker, linestyle = SYNTHETIC_METHOD_STYLE[method]
        medians = []
        for x, condition in enumerate(SYNTHETIC_CONDITIONS):
            values = data[(data.method == method) & (data.condition == condition)][metric].to_numpy(float)
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            rng = np.random.default_rng(3100 + x + SYNTHETIC_METHODS.index(method) * 11)
            ax.scatter(
                x + offset + rng.normal(0, 0.012, len(values)),
                values,
                s=5.5,
                color=color,
                alpha=0.18,
                linewidths=0,
                rasterized=True,
            )
            ax.errorbar(
                x + offset,
                median,
                yerr=[[median - q1], [q3 - median]],
                fmt=marker,
                color=color,
                markeredgecolor=INK,
                markeredgewidth=0.35,
                markersize=3.8,
                elinewidth=0.9,
                capsize=0,
                zorder=5,
            )
            medians.append(median)
        ax.plot(np.arange(3) + offset, medians, color=color, linestyle=linestyle, linewidth=1.0, alpha=0.88)
    ax.set_xticks(range(3), labels)
    ax.set_xlim(-0.48, 2.48)
    ax.set_ylim(0.10 if metric == "r_source_median" else 0.28, 1.015)
    if not show_ylabel:
        ax.tick_params(labelleft=False)
    finish_axes(ax, "y")


def make_figure3_combined_archive() -> dict:
    set_style()
    benchmark_df = load_real_benchmarks()
    contrasts = real_contrasts(benchmark_df)
    integrity = _figure3_integrity_stats(benchmark_df, contrasts)
    synthetic = _load_single_model_synthetic()
    seed = pd.read_csv(SEED_ROBUSTNESS_CSV)
    median_seed_sd = float(seed.mir_sd.median())

    fig = plt.figure(figsize=(7.2, 7.20))
    gs = fig.add_gridspec(
        2,
        14,
        left=0.075,
        right=0.988,
        bottom=0.075,
        top=0.82,
        wspace=1.75,
        hspace=0.68,
        height_ratios=[0.96, 1.04],
    )
    outer_a = fig.add_subplot(gs[0, 0:7])
    outer_b = fig.add_subplot(gs[0, 7:14])
    ax_c = fig.add_subplot(gs[1, 0:6])
    ax_d = fig.add_subplot(gs[1, 6:10])
    ax_e = fig.add_subplot(gs[1, 10:14])
    fig.suptitle("Single-model source separation in simulations and real EEG", fontsize=10.5, fontweight="bold", y=0.985)
    fig.text(0.075, 0.93, "KNOWN-SOURCE SIMULATIONS", fontsize=7.5, color=GREY, fontweight="bold")
    fig.text(0.075, 0.475, "REAL EEG: PRIMARY ENDPOINT AND DESCRIPTIVE QUALIFICATIONS", fontsize=7.5, color=GREY, fontweight="bold")
    fig.text(
        0.988,
        0.446,
        rf"{integrity['n_subject_comparator_contrasts']}/{integrity['n_subject_comparator_contrasts']} positive MIR contrasts; "
        rf"five-seed median SD {median_seed_sd:.4f} kbits/s",
        ha="right",
        fontsize=6.2,
        color=GREY,
    )

    outer_a.axis("off")
    panel_title(outer_a, "A", "Topographic recovery", y=1.04)
    for index, regime in enumerate(["Homogeneous Laplacian", "Heterogeneous mixture"]):
        inset = outer_a.inset_axes([0.02 + 0.50 * index, 0.03, 0.46, 0.84])
        _plot_synthetic_conditions(
            inset,
            synthetic[synthetic.regime == regime],
            "r_topo_median",
            show_ylabel=index == 0,
        )
        inset.set_title(regime, fontsize=7.1, fontweight="bold", pad=4)
        if index == 0:
            inset.set_ylabel("Matched topography $|r|$", fontsize=7.2)

    outer_b.axis("off")
    panel_title(outer_b, "B", "Source-time-course recovery", y=1.04)
    ax_b = outer_b.inset_axes([0.0, 0.42, 1.0, 0.53])
    _plot_synthetic_conditions(
        ax_b,
        synthetic[synthetic.regime == "Heterogeneous mixture"],
        "r_source_median",
        show_ylabel=True,
    )
    ax_b.set_ylabel("Matched source time-course $|r|$", fontsize=7.2)
    ax_b.text(0.02, 0.97, "heterogeneous source-density mixture", transform=ax_b.transAxes, va="top", fontsize=6.5, color=GREY)
    inset = outer_b.inset_axes([0.26, 0.02, 0.72, 0.22])
    inset.set_facecolor("white")
    paired = synthetic[synthetic.method.isin(["amica 3k", "amica 10k"])].pivot_table(
        index=["regime", "condition", "seed"],
        columns="method",
        values="r_topo_median",
    )
    paired["delta"] = paired["amica 10k"] - paired["amica 3k"]
    regime_specs = [
        ("Homogeneous Laplacian", LIGHT_BLUE, "o", -0.12, "homog."),
        ("Heterogeneous mixture", BLUE, "s", 0.12, "heterog."),
    ]
    for regime, color, marker, offset, label in regime_specs:
        for y, condition in enumerate(SYNTHETIC_CONDITIONS):
            values = paired.loc[(regime, condition), "delta"].to_numpy(float)
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            inset.plot([q1, q3], [y + offset, y + offset], color=color, linewidth=1.15)
            inset.plot(
                median,
                y + offset,
                marker,
                color=color,
                markeredgecolor=INK,
                markeredgewidth=0.35,
                markersize=3.4,
                label=label if y == 0 else None,
            )
    inset.axvline(0, color=INK, linewidth=0.6)
    inset.set_yticks([])
    inset.set_ylim(2.45, -0.75)
    for y, label in enumerate(["clean", "noise", "+EOG"]):
        inset.text(
            0.02,
            y,
            label,
            transform=inset.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=4.8,
            color=INK,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 0.25},
        )
    inset.text(
        0.02,
        0.98,
        r"10k $-$ 3k topography gain",
        transform=inset.transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
        fontweight="bold",
    )
    inset.tick_params(axis="x", labelsize=5.2)
    inset.legend(
        loc="upper right",
        frameon=False,
        fontsize=4.7,
        ncol=2,
        handletextpad=0.25,
        columnspacing=0.55,
        borderaxespad=0.1,
    )
    finish_axes(inset, "x")

    panel_title(ax_c, "C", "Subject-level MIR advantage", y=1.04)
    for y, row in zip(REAL_CONTRAST_Y, contrasts):
        color = COMPARATOR_COLORS[row["comparator"]]
        marker = COMPARATOR_MARKERS[row["comparator"]]
        rng = np.random.default_rng(500 + int(round(10 * y)))
        ax_c.scatter(row["mir"], y + rng.normal(0, 0.075, size=row["n"]), s=7, color=color, alpha=0.34, linewidths=0, rasterized=True)
        _summary_errorbar(ax_c, row["mir_mean"], row["mir_ci"], y, color, marker, size=5.4)
    ax_c.axvline(0, color=INK, linewidth=0.8)
    ax_c.set_xlabel(r"$\Delta$MIR = amica $-$ comparator (kbits/s)")
    add_dataset_bands(ax_c, contrasts, label_mode="methods")
    mir_max = max(max(np.max(row["mir"]), row["mir_ci"][1]) for row in contrasts)
    ax_c.set_xlim(-0.08, mir_max + 0.08)
    finish_axes(ax_c)

    panel_title(ax_d, "D", "Residual pairwise MI", y=1.04)
    for y, row in zip(REAL_CONTRAST_Y, contrasts):
        color = COMPARATOR_COLORS[row["comparator"]]
        marker = COMPARATOR_MARKERS[row["comparator"]]
        rng = np.random.default_rng(700 + int(round(10 * y)))
        ax_d.scatter(row["pmi"], y + rng.normal(0, 0.075, size=row["n"]), s=6, color=color, alpha=0.26, linewidths=0, rasterized=True)
        _summary_errorbar(ax_d, row["pmi_mean"], row["pmi_ci"], y, color, marker, size=4.8)
    ax_d.axvline(0, color=INK, linewidth=0.8)
    ax_d.set_xlabel("$\\Delta$ remnant PMI\n(percentage points)")
    ax_d.text(0.97, 0.025, "negative favours amica", transform=ax_d.transAxes, ha="right", fontsize=5.7, color=GREY)
    add_dataset_bands(ax_d, contrasts, label_mode=None)
    finish_axes(ax_d)

    panel_title(ax_e, "E", "Dipolarity", y=1.04)
    for y, row in zip(REAL_CONTRAST_Y, contrasts):
        color = COMPARATOR_COLORS[row["comparator"]]
        rng = np.random.default_rng(900 + int(round(10 * y)))
        jit = rng.normal(0, 0.045, size=row["n"])
        ax_e.scatter(row["rv5"], y - 0.12 + jit, s=5.5, color=color, alpha=0.22, marker="o", linewidths=0, rasterized=True)
        ax_e.scatter(row["rv10"], y + 0.12 + jit, s=5.5, color=color, alpha=0.22, marker="D", linewidths=0, rasterized=True)
        _summary_errorbar(ax_e, row["rv5_mean"], row["rv5_ci"], y - 0.12, color, "o", size=4.2)
        _summary_errorbar(ax_e, row["rv10_mean"], row["rv10_ci"], y + 0.12, color, "D", size=4.0)
    ax_e.axvline(0, color=INK, linewidth=0.8)
    ax_e.set_xlabel("$\\Delta$ near-dipolar fraction\n(percentage points)")
    ax_e.text(0.97, 0.025, "positive = more near-dipolar maps", transform=ax_e.transAxes, ha="right", fontsize=5.5, color=GREY)
    add_dataset_bands(ax_e, contrasts, label_mode=None)
    finish_axes(ax_e)
    ax_e.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=INK, label="RV <5%", markersize=4.2),
            Line2D([0], [0], marker="D", color="none", markerfacecolor=INK, label="RV <10%", markersize=4.0),
        ],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.06),
        frameon=False,
        ncol=2,
        fontsize=6.1,
        handletextpad=0.3,
        columnspacing=0.6,
    )

    method_handles = [
        Line2D([0], [0], color=SYNTHETIC_METHOD_STYLE[m][0], marker=SYNTHETIC_METHOD_STYLE[m][1], linestyle=SYNTHETIC_METHOD_STYLE[m][2], linewidth=1.25, markersize=3.8, label=m.replace("3k", "3,000").replace("10k", "10,000"))
        for m in SYNTHETIC_METHODS
    ]
    fig.legend(handles=method_handles, loc="upper center", bbox_to_anchor=(0.57, 0.89), ncol=5, frameon=False, fontsize=6.6, columnspacing=0.9, handlelength=1.8)

    save_figure(fig, "fig3_single_model_source_separation", png_dpi=600)
    integrity["synthetic"] = {
        "audit_csv": str(FIG3_SYNTHETIC_AUDIT.resolve()),
        "n_runs_displayed": int(len(synthetic)),
        "regimes": sorted(synthetic.regime.unique().tolist()),
        "conditions": SYNTHETIC_CONDITIONS,
        "methods": SYNTHETIC_METHODS,
    }
    integrity["seed_robustness"] = {
        "source": str(SEED_ROBUSTNESS_CSV.resolve()),
        "median_mir_sd_kbits_s": median_seed_sd,
    }
    return integrity


def make_figure3() -> dict:
    """Build the real-EEG benchmark figure used by the current manuscript."""
    set_style()
    benchmark_df = load_real_benchmarks()
    contrasts = real_contrasts(benchmark_df)
    integrity = _figure3_integrity_stats(benchmark_df, contrasts)

    fig, axes = plt.subplots(1, 4, figsize=(7.15, 3.65), sharey=False)
    fig.subplots_adjust(
        left=0.115,
        right=0.992,
        bottom=0.205,
        top=0.79,
        wspace=0.42,
    )
    fig.text(
        0.335,
        0.965,
        rf"PRIMARY ENDPOINT — "
        rf"{integrity['n_subject_comparator_contrasts']}/"
        rf"{integrity['n_subject_comparator_contrasts']} contrasts positive",
        ha="center",
        fontsize=7.6,
        color=BLUE,
        fontweight="bold",
    )
    fig.text(
        0.775,
        0.965,
        "SECONDARY DESCRIPTIVE OUTCOMES",
        ha="center",
        fontsize=7.5,
        color=GREY,
        fontweight="bold",
    )

    def stacked_panel_title(
        ax: mpl.axes.Axes,
        label: str,
        title: str,
    ) -> None:
        """Keep 1x4 panel headers legible without horizontal collisions."""
        ax.text(
            0.0,
            1.145,
            label,
            transform=ax.transAxes,
            fontsize=11.5,
            fontweight="bold",
            ha="left",
            va="bottom",
            color=INK,
        )
        ax.text(
            0.0,
            1.065,
            title,
            transform=ax.transAxes,
            fontsize=9.0,
            fontweight="bold",
            ha="left",
            va="bottom",
            color=INK,
        )

    # A: subject-level MIR contrasts.
    ax = axes[0]
    stacked_panel_title(ax, "A", "MIR advantage")
    for y, row in zip(REAL_CONTRAST_Y, contrasts):
        color = COMPARATOR_COLORS[row["comparator"]]
        marker = COMPARATOR_MARKERS[row["comparator"]]
        rng = np.random.default_rng(500 + int(round(10 * y)))
        ax.scatter(
            row["mir"],
            y + rng.normal(0, 0.075, size=row["n"]),
            s=8,
            color=color,
            alpha=0.38,
            linewidths=0,
            rasterized=True,
            zorder=3,
        )
        _summary_errorbar(
            ax, row["mir_mean"], row["mir_ci"], y, color, marker, size=6.2
        )
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_xlabel(
        r"$\Delta$MIR = amica $-$ comparator" "\n(kbits/s)",
        fontsize=8.0,
    )
    add_dataset_bands(
        ax,
        contrasts,
        label_mode="methods",
        label_offset=0.72,
        label_fontsize=7.2,
    )
    mir_max = max(max(np.max(row["mir"]), row["mir_ci"][1]) for row in contrasts)
    ax.set_xlim(-0.08, mir_max + 0.08)
    finish_axes(ax)

    # B: standardised paired effects. Dataset labels sit in the enlarged
    # inter-group whitespace so they cannot collide with the top estimate.
    ax = axes[1]
    stacked_panel_title(ax, "B", "Standardised effect")
    for y, row in zip(REAL_CONTRAST_Y, contrasts):
        color = COMPARATOR_COLORS[row["comparator"]]
        marker = COMPARATOR_MARKERS[row["comparator"]]
        _summary_errorbar(ax, row["dz"], row["dz_ci"], y, color, marker, size=5.8)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_xlabel(r"Cohen's $d_z$" "\n(95% CI)", fontsize=8.0)
    add_dataset_bands(
        ax,
        contrasts,
        label_mode=None,
        label_offset=0.72,
        label_fontsize=7.2,
    )
    finish_axes(ax)

    # C: remnant pairwise MI.
    ax = axes[2]
    stacked_panel_title(ax, "C", "Pairwise dependence")
    for y, row in zip(REAL_CONTRAST_Y, contrasts):
        color = COMPARATOR_COLORS[row["comparator"]]
        marker = COMPARATOR_MARKERS[row["comparator"]]
        rng = np.random.default_rng(700 + int(round(10 * y)))
        ax.scatter(
            row["pmi"],
            y + rng.normal(0, 0.075, size=row["n"]),
            s=8,
            color=color,
            alpha=0.30,
            linewidths=0,
            rasterized=True,
            zorder=3,
        )
        _summary_errorbar(
            ax, row["pmi_mean"], row["pmi_ci"], y, color, marker, size=5.8
        )
    ax.axvline(0, color=INK, linewidth=0.8)
    # Remnant PMI is a residual-dependence measure, so LOWER is better and a
    # negative difference favours amica -- the opposite sign convention to
    # panels A and D. Without this cue a reader cannot tell which side wins,
    # and this is the one panel where amica loses to Picard on Table tennis.
    ax.set_xlabel(
        "$\\Delta$ remnant PMI\n(percentage points)\nnegative favours amica",
        fontsize=8.0,
    )
    add_dataset_bands(
        ax,
        contrasts,
        label_mode=None,
        label_offset=0.72,
        label_fontsize=7.2,
    )
    finish_axes(ax)

    # D: paired differences in near-dipolar fractions.
    ax = axes[3]
    stacked_panel_title(ax, "D", "Dipolarity")
    for y, row in zip(REAL_CONTRAST_Y, contrasts):
        color = COMPARATOR_COLORS[row["comparator"]]
        rng = np.random.default_rng(900 + int(round(10 * y)))
        jitter = rng.normal(0, 0.045, size=row["n"])
        ax.scatter(
            row["rv5"],
            y - 0.12 + jitter,
            s=6.5,
            color=color,
            alpha=0.24,
            marker="o",
            linewidths=0,
            rasterized=True,
        )
        ax.scatter(
            row["rv10"],
            y + 0.12 + jitter,
            s=6.5,
            color=color,
            alpha=0.24,
            marker="D",
            linewidths=0,
            rasterized=True,
        )
        _summary_errorbar(
            ax, row["rv5_mean"], row["rv5_ci"], y - 0.12, color, "o", size=4.9
        )
        _summary_errorbar(
            ax, row["rv10_mean"], row["rv10_ci"], y + 0.12, color, "D", size=4.7
        )
    ax.axvline(0, color=INK, linewidth=0.8)
    # Near-dipolarity runs the other way from panel C: a higher fraction is
    # better, so a positive difference favours amica.
    ax.set_xlabel(
        "$\\Delta$ near-dipolar fraction\n(percentage points)\npositive favours amica",
        fontsize=8.0,
    )
    add_dataset_bands(
        ax,
        contrasts,
        label_mode=None,
        label_offset=0.72,
        label_fontsize=7.2,
    )
    finish_axes(ax)
    # The caption defines the RV-threshold marker shapes. Keeping the key out
    # of the narrow 1x4 header prevents a collision with the panel-D label.

    save_figure(fig, "fig3_realeeg_benchmark", png_dpi=600)
    return integrity


# ---------------------------------------------------------------------------
# Figure 4: runtime feasibility and memory control
# ---------------------------------------------------------------------------
def load_memory_rows() -> pd.DataFrame:
    """Load the one-subject memory fixture from per-implementation JSONs.

    The compact CSV is retained as a cross-check, but the plotted values come
    from the archived JSON records so fixture metadata and unrounded values are
    verified together.
    """
    rows = []
    for path in sorted(MEMORY_JSON_ROOT.glob("*/*/*.json")):
        with path.open(encoding="utf-8") as fh:
            record = json.load(fh)
        device = "gpu" if record.get("device") in {"gpu", "cuda"} else "cpu"
        rows.append(
            {
                "implementation": record["implementation"],
                "device": device,
                "n_components": int(record["n_components"]),
                "n_samples": int(record["n_samples"]),
                "max_iter": int(record["max_iter"]),
                "n_iter": int(record["n_iter"]),
                "peak_rss_gib": float(record["peak_rss_gb"]),
                "baseline_rss_gib": float(record.get("baseline_rss_gb", np.nan)),
                "delta_rss_gib": float(record.get("delta_rss_gb", np.nan)),
                "peak_vram_gib": float(record["peak_vram_gb"])
                if record.get("peak_vram_gb") is not None
                else np.nan,
                "source_path": path,
            }
        )
    memory = pd.DataFrame(rows)

    table = pd.read_csv(MEMORY_CSV)
    table["device"] = table["device"].replace({"cuda": "gpu"})
    for row in memory.itertuples(index=False):
        match = table[
            (table.implementation == row.implementation)
            & (table.device == row.device)
        ]
        if len(match) != 1:
            raise ValueError(
                f"Memory table does not uniquely match {row.implementation}/{row.device}"
            )
        tab = match.iloc[0]
        if not np.isclose(row.peak_rss_gib, float(tab.peak_rss_gb), atol=0.011):
            raise ValueError(f"Peak RSS mismatch for {row.implementation}/{row.device}")
        if np.isfinite(row.peak_vram_gib) and not np.isclose(
            row.peak_vram_gib, float(tab.peak_vram_gb), atol=0.011
        ):
            raise ValueError(f"Peak VRAM mismatch for {row.implementation}/{row.device}")
    return memory


def load_scaling_rows() -> pd.DataFrame:
    rows = []
    for path in sorted(SCALING_ROOT.glob("*/*.json")):
        with path.open(encoding="utf-8") as fh:
            a = json.load(fh)["amica"]
        config = path.parent.name
        if config.startswith("chunk-"):
            chunk = int(config.split("_")[0].split("-")[1])
            mode = "chunked"
        elif "fullbatch" in config:
            chunk = 0
            mode = "full"
        else:
            continue
        rows.append(
            {
                "config": config,
                "mode": mode,
                "chunk": chunk,
                "n_samples": int(a["n_samples"]),
                "peak_rss_gib": float(a["peak_rss_gb"]),
                "n_components": int(a["n_components"]),
                "n_channels": int(a["n_channels"]),
                "n_iter": int(a["actual_n_iter"]),
                "hostname": str(a.get("hostname", "")),
                "source_path": path,
            }
        )
    return pd.DataFrame(rows)


def _matched_unmixing_summary(reference: np.ndarray, candidate: np.ndarray) -> tuple[float, float]:
    """Median and minimum unsigned row correlation after Hungarian matching."""
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    reference = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    candidate = candidate / np.linalg.norm(candidate, axis=1, keepdims=True)
    correlation = np.abs(reference @ candidate.T)
    row, col = linear_sum_assignment(-correlation)
    matched = correlation[row, col]
    return float(np.median(matched)), float(np.min(matched))


def load_fixed_workload_runtime_audit(*, write_output: bool = True) -> pd.DataFrame:
    """Load the archived ds004505 sub-01, 100-iteration timing audit.

    The Python rows time the model ``fit`` call after loading the shared
    PCA-projected array. The Fortran row times the external executable after
    its input files have been written. The records are therefore a controlled
    fixed-workload audit with an explicitly different Fortran timing boundary,
    not repeated process-level benchmark estimates.
    """
    cpu_root = MEMORY_JSON_ROOT / "cpu/ds004505_sub-01_mem"
    paths = {
        "amica JAX-GPU (chunked)": RUNTIME_GPU_ROOTS[100]
        / "amica_python_jax_chunked_sub-01_seed0_result.json",
        "amica JAX-CPU": cpu_root / "amica_python_jax_sub-01_seed0_result.json",
        "Scott–Huberty amica-python 0.1.1": cpu_root
        / "scott_huberty_torch_sub-01_seed0_result.json",
        "PyAMICA 0.3.0": cpu_root / "pyamica_torch_sub-01_seed0_result.json",
        "Fortran AMICA 1.7": cpu_root / "fortran_amica17_sub-01_seed0_result.json",
    }
    records = {label: json.loads(path.read_text(encoding="utf-8")) for label, path in paths.items()}
    reference = records["amica JAX-CPU"]
    rows = []
    for label, record in records.items():
        if int(record["n_components"]) != 64 or int(record["n_samples"]) != 785328:
            raise ValueError(f"Fixed-workload metadata mismatch for {label}")
        if int(record["n_iter"]) != 100 or int(record["max_iter"]) != 100:
            raise ValueError(f"Fixed-workload iteration mismatch for {label}")
        median_r, minimum_r = _matched_unmixing_summary(reference["W"], record["W"])
        rows.append(
            {
                "display": label,
                "implementation": record["implementation"],
                "device": "gpu" if record.get("device") in {"gpu", "cuda"} else "cpu",
                "n_components": int(record["n_components"]),
                "n_samples": int(record["n_samples"]),
                "n_iter": int(record["n_iter"]),
                "fit_time_s": float(record["fit_time_s"]),
                "ll_final": float(record["ll_final"]),
                "abs_ll_difference_vs_amica_jax_cpu": abs(
                    float(record["ll_final"]) - float(reference["ll_final"])
                ),
                "median_matched_row_correlation_vs_amica_jax_cpu": median_r,
                "minimum_matched_row_correlation_vs_amica_jax_cpu": minimum_r,
                "timing_boundary": (
                    "external executable; input serialisation excluded; initialisation and output writing included"
                    if record["implementation"] == "fortran_amica17"
                    else "model.fit after shared PCA-array loading; first-use compilation included"
                ),
                "n_process_repetitions": 1,
                "source_path": paths[label],
            }
        )
    audit = pd.DataFrame(rows)
    if write_output:
        exported = audit.drop(columns=["source_path"])
        exported.to_csv(FIG4_RUNTIME_AUDIT, index=False)
    return audit


def load_paired_chunking_memory() -> pd.DataFrame:
    """Load paired full/chunked CPU RSS for ds004505 sub-01--sub-06."""
    roots = [MEMORY_JSON_ROOT / "cpu/ds004505_sub-01_mem"]
    roots.extend(sorted(MEMORY_MULTISUBJECT_ROOT.glob("ds004505_sub-*_mem")))
    rows = []
    for root in roots:
        subject = root.name.split("_")[1]
        records = {}
        for path in root.glob("*_result.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            records[record.get("implementation")] = (record, path)
        required = {"amica_python_jax", "amica_python_jax_chunked"}
        if not required.issubset(records):
            raise ValueError(f"Missing paired full/chunked records for {subject}")
        full, full_path = records["amica_python_jax"]
        chunked, chunked_path = records["amica_python_jax_chunked"]
        for key in ("n_components", "n_samples", "n_iter", "max_iter"):
            if full[key] != chunked[key]:
                raise ValueError(f"Paired memory metadata mismatch for {subject}: {key}")
        if int(full["n_components"]) != 64:
            raise ValueError("Paired chunking audit must use 64 retained components")
        full_peak = float(full["peak_rss_gb"])
        chunked_peak = float(chunked["peak_rss_gb"])
        rows.append(
            {
                "subject": subject,
                "n_samples": int(full["n_samples"]),
                "n_iter": int(full["n_iter"]),
                "full_peak_rss_gib": full_peak,
                "chunked_peak_rss_gib": chunked_peak,
                "total_peak_reduction_pct": 100.0 * (1.0 - chunked_peak / full_peak),
                "full_source_path": full_path,
                "chunked_source_path": chunked_path,
            }
        )
    memory = pd.DataFrame(rows).sort_values("n_samples").reset_index(drop=True)
    if len(memory) != 6 or set(memory.subject) != {f"sub-{index:02d}" for index in range(1, 7)}:
        raise ValueError("Expected paired memory records for ds004505 sub-01--sub-06")
    return memory


def _figure4_integrity_stats(
    bench: pd.DataFrame,
    memory: pd.DataFrame,
    scaling: pd.DataFrame,
) -> dict:
    method_order = [
        "AMICA-Python (JAX-GPU)",
        "Picard",
        "Infomax",
        "FastICA",
        "AMICA-Python (JAX-CPU)",
        "AMICA-Python (NumPy-CPU)",
    ]
    if set(bench.method) != set(method_order):
        raise ValueError("Unexpected method set in the ds004505 runtime archive")
    subject_sets = {
        method: set(bench.loc[bench.method == method, "subject"])
        for method in method_order
    }
    if any(len(subjects) != 25 for subjects in subject_sets.values()):
        raise ValueError("Every runtime method must contain 25 unique subjects")
    if len({frozenset(subjects) for subjects in subject_sets.values()}) != 1:
        raise ValueError("Runtime subjects are not paired across all six methods")
    if bench.fit_runtime_s.isna().any():
        raise ValueError("Missing subject-level runtime values")
    if set(bench.n_components.astype(int)) != {64}:
        raise ValueError("Figure 3 runtime rows do not all use 64 retained components")

    runtime = {}
    for method in method_order:
        rows = bench[bench.method == method]
        q1, median, q3 = rows.fit_runtime_s.quantile([0.25, 0.5, 0.75])
        runtime[method] = {
            "n_subjects": int(rows.subject.nunique()),
            "median_s": float(median),
            "iqr_s": [float(q1), float(q3)],
            "requested_iteration_cap": int(rows.max_iter.iloc[0]),
            "actual_iteration_range": [
                int(rows.n_iter_actual.min()),
                int(rows.n_iter_actual.max()),
            ],
            "n_reaching_cap": int((rows.n_iter_actual == rows.max_iter).sum()),
            "hostnames": sorted(rows.hardware.dropna().astype(str).unique().tolist()),
        }

    required_memory = {
        ("amica_python_jax", "cpu"),
        ("amica_python_jax_chunked", "cpu"),
        ("scott_huberty_torch", "cpu"),
        ("pyamica_torch", "cpu"),
        ("fortran_amica17", "cpu"),
        ("amica_python_jax_chunked", "gpu"),
        ("scott_huberty_torch", "gpu"),
        ("pyamica_torch", "gpu"),
    }
    present_memory = set(zip(memory.implementation, memory.device))
    if present_memory != required_memory:
        raise ValueError("Unexpected implementation/device set in the memory fixture")
    if set(memory.n_components) != {64} or set(memory.n_samples) != {785328}:
        raise ValueError("Memory fixture metadata do not match ds004505 sub-01")
    if set(memory.n_iter) != {100} or set(memory.max_iter) != {100}:
        raise ValueError("Memory fixture must contain 100-iteration runs")

    full = scaling[scaling["mode"] == "full"].sort_values("n_samples")
    chunked = scaling[scaling["mode"] == "chunked"].sort_values("chunk")
    if len(full) != 5 or len(chunked) != 4:
        raise ValueError("Expected five full-batch and four chunk-size scaling runs")
    if set(scaling.n_components) != {64} or set(scaling.n_iter) != {60}:
        raise ValueError("Scaling sweeps must use C=64 and 60 iterations")
    if set(chunked.n_samples) != {785328}:
        raise ValueError("Chunk-size scaling sweep must hold T fixed at 785,328")

    def file_record(path: Path) -> dict:
        return {
            "path": path.relative_to(WORKSPACE).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    memory_values = {}
    for row in memory.itertuples(index=False):
        key = f"{row.implementation}/{row.device}"
        memory_values[key] = {
            "peak_process_rss_gib": float(row.peak_rss_gib),
            "baseline_process_rss_gib": float(row.baseline_rss_gib)
            if np.isfinite(row.baseline_rss_gib)
            else None,
            "incremental_process_rss_gib": float(row.delta_rss_gib)
            if np.isfinite(row.delta_rss_gib)
            else None,
            "peak_vram_gib": float(row.peak_vram_gib)
            if np.isfinite(row.peak_vram_gib)
            else None,
        }

    full_core = memory[
        (memory.implementation == "amica_python_jax") & (memory.device == "cpu")
    ].iloc[0]
    chunk_core = memory[
        (memory.implementation == "amica_python_jax_chunked")
        & (memory.device == "cpu")
    ].iloc[0]
    gpu_chunk = memory[
        (memory.implementation == "amica_python_jax_chunked")
        & (memory.device == "gpu")
    ].iloc[0]
    total_reduction_pct = 100.0 * (
        1.0 - chunk_core.peak_rss_gib / full_core.peak_rss_gib
    )
    incremental_reduction_pct = 100.0 * (
        1.0 - chunk_core.delta_rss_gib / full_core.delta_rss_gib
    )

    with REFERENCE_EVIDENCE.open(encoding="utf-8") as fh:
        chunking_agreement = json.load(fh)["chunking"]

    scaling_endpoint = full.iloc[-1]

    return {
        "input_files": {
            "runtime_csv": file_record(BENCH_505),
            "memory_summary_csv": file_record(MEMORY_CSV),
            "memory_jsons": [file_record(p) for p in memory.source_path],
            "cpu_scaling_jsons": [file_record(p) for p in scaling.source_path],
            "chunking_agreement": file_record(REFERENCE_EVIDENCE),
        },
        "producer_files": {
            "runtime_measurement": file_record(RUNTIME_RUNNER),
            "mne_fit_integration": file_record(MNE_INTEGRATION),
            "rss_measurement": file_record(MEMORY_MEASUREMENT),
            "jax_vram_measurement": file_record(MEMORY_AMICA_RUNNER),
            "scott_vram_measurement": file_record(MEMORY_SCOTT_RUNNER),
            "pyamica_vram_measurement": file_record(MEMORY_PYAMICA_RUNNER),
            "cpu_scaling_submission": file_record(SCALING_SUBMISSION),
            "cpu_scaling_measurement": file_record(SCALING_RUNNER),
        },
        "runtime": runtime,
        "runtime_subjects_paired": True,
        "runtime_definition": (
            "Wall time of fit_ica (amica) or MNE ICA.fit (comparators): includes "
            "data extraction, pre-whitening/PCA, optimisation, MNE object construction, "
            "and any first-use JAX compilation; excludes file loading, filtering, "
            "dipole fitting, ICLabel, and downstream benchmark metrics."
        ),
        "gpu_synchronisation": (
            "fit_ica materialises fitted JAX arrays with numpy.asarray before returning; "
            "the outer timer therefore ends after device work is complete."
        ),
        "runtime_configuration": {
            "dataset": "ds004505",
            "n_subjects": 25,
            "retained_components": 64,
            "amica_models_M": 1,
            "adaptive_density_components_K": 3,
            "gpu": "one NVIDIA H100",
        },
        "memory_fixture": {
            "dataset": "ds004505",
            "subject": "sub-01",
            "n_samples": 785328,
            "retained_components": 64,
            "iterations": 100,
            "protocol": (
                "AMICA core fit from an already PCA-projected 64 x T NumPy array; "
                "pre-fit RSS was recorded after data and framework loading."
            ),
            "values": memory_values,
            "host_total_reduction_pct": float(total_reduction_pct),
            "host_incremental_reduction_pct": float(incremental_reduction_pct),
            "chunked_gpu_allocator_peak_gib": float(gpu_chunk.peak_vram_gib),
        },
        "memory_units": (
            "GiB (binary). Legacy source columns end in _gb, but producer code divides "
            "bytes by 1024**3 and Linux ru_maxrss KiB by 1024**2."
        ),
        "cpu_memory_definition": "process RSS high-water mark from resource.getrusage",
        "gpu_memory_definition": (
            "allocator high-water: XLA peak_bytes_in_use or "
            "torch.cuda.max_memory_allocated"
        ),
        "scaling_fixture": {
            "dataset": "ds004505",
            "subject": "sub-01",
            "retained_components": 64,
            "iterations": 60,
            "protocol": (
                "MNE estimator pipeline from Raw input; the process RSS high-water "
                "therefore includes data extraction, pre-whitening/PCA, and fitting."
            ),
            "full_batch_n_samples": full.n_samples.astype(int).tolist(),
            "full_batch_peak_rss_gib": full.peak_rss_gib.astype(float).tolist(),
            "chunk_sizes": chunked.chunk.astype(int).tolist(),
            "chunked_peak_rss_gib": chunked.peak_rss_gib.astype(float).tolist(),
            "chunked_fixed_n_samples": 785328,
        },
        "memory_protocol_reconciliation": {
            "core_fit_full_batch_peak_rss_gib": float(full_core.peak_rss_gib),
            "mne_pipeline_full_batch_peak_rss_gib": float(scaling_endpoint.peak_rss_gib),
            "difference_gib": float(
                scaling_endpoint.peak_rss_gib - full_core.peak_rss_gib
            ),
            "explanation": (
                "The 11.4-GiB core-fit fixture and 14.7-GiB scaling endpoint are "
                "not replicate measurements: the former starts from a pre-projected "
                "array, whereas the latter includes the MNE Raw-to-PCA estimator path."
            ),
        },
        "full_chunked_agreement": {
            "fixture": chunking_agreement["fixture"],
            "absolute_final_ll_difference": float(
                chunking_agreement["absolute_final_ll_difference"]
            ),
            "unmixing_frobenius_relative_error": float(
                chunking_agreement["unmixing_frobenius_relative_error"]
            ),
            "note": "Independent numerical-agreement fixture; not the memory run.",
        },
        "main_figure_scope": (
            "Only amica runtime backends and full/chunked memory measurements are "
            "shown. Third-party implementation memory remains in Supplementary Table S5."
        ),
        "provenance_limitations": [
            "The aggregate runtime archive records cluster hostnames but not CPU model names.",
            "The memory archive does not record third-party package commit SHAs or a complete software lockfile.",
        ],
    }


def _make_figure4_runtime_legacy() -> dict:
    set_style()
    bench = pd.read_csv(BENCH_505)
    mem = load_memory_rows()
    scaling = load_scaling_rows()
    integrity = _figure4_integrity_stats(bench, mem, scaling)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.05))
    fig.subplots_adjust(
        left=0.175,
        right=0.985,
        bottom=0.09,
        top=0.885,
        wspace=0.47,
        hspace=0.58,
    )
    fig.suptitle(
        "Computational feasibility and memory control",
        fontsize=10.5,
        fontweight="bold",
        y=0.985,
    )

    method_order = [
        "AMICA-Python (JAX-GPU)",
        "Picard",
        "Infomax",
        "FastICA",
        "AMICA-Python (JAX-CPU)",
        "AMICA-Python (NumPy-CPU)",
    ]
    display = {
        "AMICA-Python (JAX-GPU)": "amica JAX-GPU",
        "AMICA-Python (JAX-CPU)": "amica JAX-CPU",
        "AMICA-Python (NumPy-CPU)": "amica NumPy-CPU",
        "Picard": "Picard",
        "Infomax": "Ext. Infomax",
        "FastICA": "FastICA",
    }
    colors = {
        "AMICA-Python (JAX-GPU)": BLUE,
        "AMICA-Python (JAX-CPU)": LIGHT_BLUE,
        "AMICA-Python (NumPy-CPU)": "#7BA6C2",
        **COMPARATOR_COLORS,
    }
    markers = {
        "AMICA-Python (JAX-GPU)": "o",
        "AMICA-Python (JAX-CPU)": "s",
        "AMICA-Python (NumPy-CPU)": "^",
        **COMPARATOR_MARKERS,
    }

    def runtime_panel(
        ax: mpl.axes.Axes,
        methods: list[str],
        label: str,
        title: str,
        subtitle: str,
        xlim: tuple[float, float],
    ) -> None:
        panel_title(ax, label, title, y=1.09)
        ax.text(
            0.0,
            1.02,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.8,
            color=GREY,
        )
        for y, method in enumerate(methods):
            vals = bench.loc[
                bench.method == method, "fit_runtime_s"
            ].dropna().to_numpy(float)
            rng = np.random.default_rng(1100 + method_order.index(method))
            ax.scatter(
                vals,
                y + rng.normal(0, 0.075, len(vals)),
                s=8,
                color=colors[method],
                alpha=0.34,
                linewidths=0,
            )
            q1, med, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
            ax.plot([q1, q3], [y, y], color="black", linewidth=1.45, zorder=7)
            ax.plot(
                med,
                y,
                markers[method],
                color=colors[method],
                markeredgecolor="black",
                markeredgewidth=0.65,
                markersize=6.2,
                zorder=8,
            )
            ax.annotate(
                f"{med:,.0f} s",
                (med, y),
                xytext=(5, -8),
                textcoords="offset points",
                ha="left",
                va="top",
                fontsize=6.8,
                color=INK,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.4},
                zorder=9,
            )
        ax.set_xscale("log")
        ax.set_xticks([1e2, 1e3] if xlim[1] < 1e4 else [1e2, 1e3, 1e4])
        ax.set_xlim(*xlim)
        ax.set_yticks(range(len(methods)), [display[m] for m in methods])
        ax.set_ylim(len(methods) - 0.52, -0.52)
        ax.set_xlabel("Fit time per subject (s)")
        finish_axes(ax)

    runtime_panel(
        axes[0, 0],
        [
            "AMICA-Python (JAX-GPU)",
            "Picard",
            "Infomax",
            "FastICA",
        ],
        "A",
        "Practical pipeline runtime",
        "H100 amica vs CPU MNE solvers; stopping rules differ",
        (40, 3.2e3),
    )
    runtime_panel(
        axes[0, 1],
        [
            "AMICA-Python (JAX-GPU)",
            "AMICA-Python (JAX-CPU)",
            "AMICA-Python (NumPy-CPU)",
        ],
        "B",
        "amica backend cost",
        "Same 3,000-iteration configuration; hardware-specific",
        (75, 5.2e4),
    )

    # Main-memory claim: only the reproducible amica full/chunked measurements.
    cpu = mem[mem.device == "cpu"].set_index("implementation")
    gpu = mem[mem.device == "gpu"].set_index("implementation")
    full = cpu.loc["amica_python_jax"]
    chunked = cpu.loc["amica_python_jax_chunked"]
    gpu_chunked = gpu.loc["amica_python_jax_chunked"]
    total_reduction = 100.0 * (1.0 - chunked.peak_rss_gib / full.peak_rss_gib)
    incremental_reduction = 100.0 * (
        1.0 - chunked.delta_rss_gib / full.delta_rss_gib
    )
    with REFERENCE_EVIDENCE.open(encoding="utf-8") as fh:
        agreement = json.load(fh)["chunking"]

    def math_sci(value: float) -> str:
        exponent = int(np.floor(np.log10(abs(value))))
        coefficient = value / (10.0**exponent)
        return rf"{coefficient:.2f}\!\times\!10^{{{exponent}}}"

    ll_agreement = math_sci(float(agreement["absolute_final_ll_difference"]))
    w_agreement = math_sci(float(agreement["unmixing_frobenius_relative_error"]))

    ax = axes[1, 0]
    panel_title(ax, "C", "Memory control", y=1.09)
    ax.text(
        0.0,
        1.02,
        "Core-fit fixture: pre-projected input, 100 iterations",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.8,
        color=GREY,
    )
    y = np.arange(3)
    baseline = np.array([full.baseline_rss_gib, chunked.baseline_rss_gib])
    increment = np.array([full.delta_rss_gib, chunked.delta_rss_gib])
    ax.barh(
        y[:2],
        baseline,
        color="#E5E7E9",
        edgecolor="black",
        linewidth=0.45,
        height=0.58,
        label="pre-fit host baseline",
    )
    ax.barh(
        y[:2],
        increment,
        left=baseline,
        color=[LIGHT_BLUE, BLUE],
        edgecolor="black",
        linewidth=0.45,
        height=0.58,
        label="fit increment",
    )
    ax.barh(
        y[2],
        gpu_chunked.peak_vram_gib,
        color=BLUE,
        alpha=0.75,
        hatch="///",
        edgecolor="black",
        linewidth=0.55,
        height=0.58,
        label="allocator VRAM",
    )
    totals = [full.peak_rss_gib, chunked.peak_rss_gib, gpu_chunked.peak_vram_gib]
    for yi, value in zip(y, totals):
        suffix = " total" if yi < 2 else " VRAM"
        ax.text(value + 0.22, yi, f"{value:.1f}{suffix}", va="center", fontsize=6.9)
    ax.set_yticks(
        y,
        ["full-batch host", "chunked host", "chunked GPU"],
    )
    ax.set_ylim(2.55, -0.55)
    ax.set_xlim(0, 12.8)
    ax.set_xlabel("Peak memory (GiB)")
    ax.text(
        12.45,
        1.46,
        f"Host peak: -{total_reduction:.0f}% total\n"
        f"Fit increment: -{incremental_reduction:.0f}%",
        ha="right",
        va="center",
        fontsize=6.35,
        color=INK,
        bbox={"facecolor": "white", "edgecolor": "#D9DCE0", "linewidth": 0.45, "pad": 1.5},
    )
    ax.text(
        12.45,
        2.13,
        "Separate agreement fixture\n"
        + rf"$|\Delta \ell|={ll_agreement}$"
        + "\n"
        + rf"$\|\Delta W\|_F/\|W\|_F={w_agreement}$",
        ha="right",
        va="center",
        fontsize=6.15,
        color=INK,
        bbox={"facecolor": "white", "edgecolor": "#D9DCE0", "linewidth": 0.45, "pad": 1.4},
    )
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.19),
        frameon=False,
        ncol=3,
        borderaxespad=0,
        handlelength=1.3,
        handletextpad=0.35,
        columnspacing=0.65,
        fontsize=6.2,
    )
    finish_axes(ax)

    # One engineering panel containing the two complementary scaling checks.
    host = axes[1, 1]
    host.axis("off")
    panel_title(host, "D", "Memory scaling", y=1.09)
    host.text(
        0.0,
        1.02,
        "Separate MNE-pipeline fixture: Raw input, 60 iterations",
        transform=host.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.8,
        color=GREY,
    )
    left = host.inset_axes([0.02, 0.08, 0.46, 0.78])
    right = host.inset_axes([0.56, 0.08, 0.42, 0.78])
    full = scaling[scaling["mode"] == "full"].sort_values("n_samples")
    chunk = scaling[scaling["mode"] == "chunked"].sort_values("chunk")
    ymax = max(full.peak_rss_gib.max(), chunk.peak_rss_gib.max()) * 1.08
    left.plot(full.n_samples / 1000, full.peak_rss_gib, "o-", color=LIGHT_BLUE, markeredgecolor="black", markeredgewidth=0.5, linewidth=1.8)
    left.set_xlabel(r"samples $T$ ($\times10^3$)", fontsize=7.2)
    left.set_ylabel("peak process RSS (GiB)", fontsize=7.2)
    left.set_title("Full batch", fontsize=6.7, fontweight="bold", pad=10)
    left.text(
        0.04,
        0.96,
        r"sample sweep; $C=64$",
        transform=left.transAxes,
        ha="left",
        va="top",
        fontsize=5.9,
        color=GREY,
    )
    left.set_ylim(0, ymax)
    finish_axes(left)
    right.semilogx(chunk.chunk, chunk.peak_rss_gib, "s-", color=BLUE, markeredgecolor="black", markeredgewidth=0.5, linewidth=1.8)
    right.set_xlabel(r"chunk size $B$", fontsize=7.2)
    right.set_title("Chunked", fontsize=6.7, fontweight="bold", pad=10)
    right.text(
        0.04,
        0.96,
        r"$B$ sweep; $C=64$; $T=785$k",
        transform=right.transAxes,
        ha="left",
        va="top",
        fontsize=5.7,
        color=GREY,
    )
    right.set_ylim(0, ymax)
    right.tick_params(labelleft=False)
    finish_axes(right)

    save_figure(fig, "fig4_runtime_memory", png_dpi=600)
    return integrity


# ---------------------------------------------------------------------------
# Figure 5: fixed-budget convergence
# ---------------------------------------------------------------------------
CONVERGENCE_ROLLING_WINDOW = 25
CONVERGENCE_TERMINAL_START = 2751
CONVERGENCE_TERMINAL_END = 3000


def _file_record(path: Path) -> dict:
    return {
        "path": str(path.relative_to(WORKSPACE)).replace("\\", "/"),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _amica_config_defaults() -> dict:
    """Read relevant AmicaConfig defaults without importing JAX."""
    wanted = {
        "lrate",
        "minlrate",
        "lratefact",
        "rholrate",
        "rholratefact",
        "do_newton",
        "newt_start",
        "newt_ramp",
        "newtrate",
        "min_dll",
        "use_min_dll",
        "max_decs",
        "max_incs",
    }
    tree = ast.parse(AMICA_CONFIG_SOURCE.read_text(encoding="utf-8"))
    defaults: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "AmicaConfig":
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    if item.target.id in wanted and item.value is not None:
                        defaults[item.target.id] = ast.literal_eval(item.value)
    missing = wanted.difference(defaults)
    if missing:
        raise ValueError(f"Could not read AmicaConfig defaults: {sorted(missing)}")
    return defaults


def _max_true_run(values: np.ndarray) -> int:
    longest = current = 0
    for value in values.astype(bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return int(longest)


def _first_gain_crossing(iterations: np.ndarray, gain: np.ndarray, fraction: float) -> int:
    target = fraction * float(gain[-1])
    indices = np.flatnonzero(gain >= target)
    return int(iterations[indices[0]]) if indices.size else -1


def _load_convergence_audit(*, write_outputs: bool) -> dict:
    config = _amica_config_defaults()
    tol = float(config["min_dll"])
    required_consecutive = int(config["max_incs"]) + 1

    trace_all = pd.read_csv(ITER_TRACE)
    trace = trace_all.loc[trace_all.method == AMICA_BENCHMARK_METHOD].copy()
    trace = trace.sort_values(["subject", "iteration"])
    summary = pd.read_csv(BENCH_505)
    summary = summary.loc[summary.method == AMICA_BENCHMARK_METHOD].copy()

    if trace.subject.nunique() != 25 or summary.subject.nunique() != 25:
        raise ValueError("Figure 4 requires all 25 ds004505 JAX-GPU subjects")
    if trace.duplicated(["subject", "iteration"]).any():
        raise ValueError("Duplicate subject/iteration records in convergence trace")
    if set(trace.subject) != set(summary.subject):
        raise ValueError("Trace and benchmark summary subject sets differ")
    if not np.isfinite(trace.log_likelihood.to_numpy(float)).all():
        raise ValueError("Non-finite likelihood values in Figure 4 trace")
    if trace[["step_size", "gradient_norm"]].notna().any().any():
        raise ValueError("Unexpected per-iteration optimiser fields; update the audit")

    counts = trace.groupby("subject").iteration.agg(["min", "max", "count"])
    if not ((counts["min"] == 1) & (counts["max"] == 3000) & (counts["count"] == 3000)).all():
        raise ValueError("Each JAX-GPU history must contain archived iterations 1-3000")

    pivot = trace.pivot(index="iteration", columns="subject", values="log_likelihood").sort_index()
    delta = pivot.diff().iloc[1:]
    cumulative = pivot - pivot.iloc[0]
    rolling = delta.rolling(
        window=CONVERGENCE_ROLLING_WINDOW,
        min_periods=CONVERGENCE_ROLLING_WINDOW,
    ).median()
    terminal_delta = delta.loc[CONVERGENCE_TERMINAL_START:CONVERGENCE_TERMINAL_END]
    if len(terminal_delta) != 250:
        raise ValueError("Terminal diagnostic must contain 250 increments")
    terminal = terminal_delta.median(axis=0)

    trace_hash = hashlib.sha256(ITER_TRACE.read_bytes()).hexdigest()
    summary_hash = hashlib.sha256(BENCH_505.read_bytes()).hexdigest()
    rows: list[dict] = []
    summary_by_subject = summary.set_index("subject")
    for subject in pivot.columns:
        ll = pivot[subject].to_numpy(float)
        iterations = pivot.index.to_numpy(int)
        dll = np.diff(ll)
        dll_iterations = iterations[1:]
        terminal_mask = (
            (dll_iterations >= CONVERGENCE_TERMINAL_START)
            & (dll_iterations <= CONVERGENCE_TERMINAL_END)
        )
        terminal_values = dll[terminal_mask]
        total_gain = float(ll[-1] - ll[0])
        gain = ll - ll[0]
        info = summary_by_subject.loc[subject]
        reached_cap = int(info.n_iter_actual) == int(info.max_iter)
        converged_before_cap = bool(info.converged_before_cap)
        if reached_cap and not converged_before_cap:
            stopping_reason = "max_iter"
        elif converged_before_cap:
            stopping_reason = "configured_stopping_rule"
        else:
            stopping_reason = "unknown"
        rows.append(
            {
                "subject": subject,
                "dataset": str(info.dataset),
                "method": str(info.method),
                "backend": str(info.backend),
                "device": str(info.device),
                "random_seed": int(info.random_seed),
                "n_components": int(info.n_components),
                "n_iterations": int(len(ll)),
                "max_iter": int(info.max_iter),
                "stopping_reason": stopping_reason,
                "stopping_reason_source": "derived from n_iter_actual, max_iter, and converged_before_cap",
                "first_archived_iteration": int(iterations[0]),
                "last_archived_iteration": int(iterations[-1]),
                "initial_likelihood": float(ll[0]),
                "final_likelihood": float(ll[-1]),
                "total_gain": total_gain,
                "median_delta_last_250": float(np.median(terminal_values)),
                "fraction_last_250_below_tol": float(np.mean(terminal_values < tol)),
                "min_delta": float(np.min(dll)),
                "max_delta": float(np.max(dll)),
                "number_negative_deltas": int(np.sum(dll < 0)),
                "number_zero_deltas": int(np.sum(dll == 0)),
                "number_nonfinite_values": int(np.sum(~np.isfinite(ll))),
                "iteration_95pct_gain": _first_gain_crossing(iterations, gain, 0.95),
                "iteration_99pct_gain": _first_gain_crossing(iterations, gain, 0.99),
                "final_250_gain": float(np.sum(terminal_values)),
                "final_250_gain_fraction": float(np.sum(terminal_values) / total_gain),
                "max_consecutive_below_tol": _max_true_run(dll < tol),
                "formal_consecutive_required": required_consecutive,
                "tolerance": tol,
                "newton_start_solver_iteration": int(config["newt_start"]),
                "source_trace_sha256": trace_hash,
                "source_summary_sha256": summary_hash,
            }
        )
    audit = pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)

    if set(audit.stopping_reason) != {"max_iter"}:
        raise ValueError("All main convergence fits should stop at max_iter")
    if int((audit.median_delta_last_250 > tol).sum()) != 22:
        raise ValueError("Terminal diagnostic count changed; review Figure 4")
    if int(audit.max_consecutive_below_tol.max()) >= required_consecutive:
        raise ValueError("A trace appears to meet the configured persistence rule")

    cohort = {
        "n_subjects": int(len(audit)),
        "n_reaching_max_iter": int((audit.stopping_reason == "max_iter").sum()),
        "n_terminal_medians_above_tolerance": int((audit.median_delta_last_250 > tol).sum()),
        "terminal_median_cohort": float(audit.median_delta_last_250.median()),
        "terminal_median_iqr": [
            float(audit.median_delta_last_250.quantile(0.25)),
            float(audit.median_delta_last_250.quantile(0.75)),
        ],
        "median_iteration_95pct_gain": float(audit.iteration_95pct_gain.median()),
        "iqr_iteration_95pct_gain": [
            float(audit.iteration_95pct_gain.quantile(0.25)),
            float(audit.iteration_95pct_gain.quantile(0.75)),
        ],
        "median_iteration_99pct_gain": float(audit.iteration_99pct_gain.median()),
        "iqr_iteration_99pct_gain": [
            float(audit.iteration_99pct_gain.quantile(0.25)),
            float(audit.iteration_99pct_gain.quantile(0.75)),
        ],
        "median_solver_iteration_99pct_gain": float(audit.iteration_99pct_gain.median() - 1),
        "iqr_solver_iteration_99pct_gain": [
            float(audit.iteration_99pct_gain.quantile(0.25) - 1),
            float(audit.iteration_99pct_gain.quantile(0.75) - 1),
        ],
        "n_subjects_with_negative_raw_increments": int((audit.number_negative_deltas > 0).sum()),
        "n_negative_raw_increments": int(audit.number_negative_deltas.sum()),
        "n_subjects_with_negative_rolling_median": int(((rolling < 0).sum(axis=0) > 0).sum()),
        "n_negative_rolling_medians": int((rolling < 0).sum().sum()),
        "maximum_consecutive_below_tolerance": int(audit.max_consecutive_below_tol.max()),
        "median_final_250_gain_fraction": float(audit.final_250_gain_fraction.median()),
        "iqr_final_250_gain_fraction": [
            float(audit.final_250_gain_fraction.quantile(0.25)),
            float(audit.final_250_gain_fraction.quantile(0.75)),
        ],
    }
    integrity = {
        "input_files": {
            "iteration_trace": _file_record(ITER_TRACE),
            "benchmark_summary": _file_record(BENCH_505),
        },
        "producer_files": {
            "configuration": _file_record(AMICA_CONFIG_SOURCE),
            "solver_and_stopping_rule": _file_record(AMICA_SOLVER_SOURCE),
            "likelihood_normalisation": _file_record(AMICA_LIKELIHOOD_SOURCE),
            "mne_fit_integration": _file_record(MNE_INTEGRATION),
            "gpu_submission": _file_record(AMICA_GPU_SUBMISSION),
        },
        "configuration": {
            **config,
            "benchmark_max_iter": int(summary.max_iter.iloc[0]),
            "retained_components": int(summary.n_components.iloc[0]),
            "random_seed": int(summary.random_seed.iloc[0]),
        },
        "likelihood_definition": {
            "normalisation": "mean log-likelihood divided by retained component count",
            "unit": "nats per retained component per sample",
            "iteration_numbering": "archive is 1-based; solver index is archive iteration minus one",
            "iteration_zero": "the first stored objective is solver iteration 0 after its update; no separate pre-update objective is archived",
            "delta": "raw signed difference between consecutive stored objectives",
        },
        "stopping_rule": {
            "primary": f"raw delta_ll < {tol:g} for {required_consecutive} consecutive stored increments",
            "implementation_detail": f"numincs increments when raw delta_ll < min_dll and stops when numincs > max_incs ({int(config['max_incs'])})",
            "reset": "numincs resets to zero when raw delta_ll >= min_dll",
            "negative_increments": "count as below tolerance; no absolute value or relative normalisation is used",
            "separate_learning_rate_stop": f"after a preceding likelihood decrease, stop if lrate <= minlrate ({float(config['minlrate']):g})",
            "newton_effect": "the configured Newton onset does not change the stopping criterion",
            "cap_order": "the stopping checks occur inside each loop iteration; max_iter is reported only after the loop ends without a stop",
        },
        "display_statistics": {
            "rolling_window": CONVERGENCE_ROLLING_WINDOW,
            "rolling_definition": "trailing subject-level median of signed raw increments; first 24 values omitted",
            "cohort_line": "median across subjects",
            "cohort_band": "interquartile range across subjects",
            "terminal_window": f"archived iterations {CONVERGENCE_TERMINAL_START}-{CONVERGENCE_TERMINAL_END} inclusive",
            "terminal_statistic": "within-subject median of signed raw increments",
            "nonpositive_display": "retained without clipping or absolute value on symmetric-log axes",
        },
        "cohort": cohort,
        "audit_csv": str(CONVERGENCE_AUDIT_CSV.relative_to(WORKSPACE)).replace("\\", "/"),
        "archive_limitations": [
            "The iteration archive contains likelihood and timing only; step size, gradient norm, Newton-use flags, learning rates, and the live convergence counter are not retained.",
            "The benchmark summary does not contain an explicit stopping_reason field; max_iter status is derived from n_iter_actual, max_iter, and converged_before_cap.",
            "The source capsule .git file points to an unavailable cluster worktree, so its exact Git commit cannot be resolved locally; SHA-256 hashes are recorded for all source and result files.",
        ],
    }

    if write_outputs:
        CONVERGENCE_AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(CONVERGENCE_AUDIT_CSV, index=False, float_format="%.17g")
        CONVERGENCE_AUDIT_JSON.write_text(
            json.dumps(integrity, indent=2) + "\n", encoding="utf-8"
        )
    return {
        "pivot": pivot,
        "cumulative": cumulative,
        "delta": delta,
        "rolling": rolling,
        "terminal": terminal,
        "audit": audit,
        "integrity": integrity,
        "tol": tol,
        "newton_start": int(config["newt_start"]),
    }


def _signed_sci_tick(value: float, _position: int | None = None) -> str:
    if value == 0:
        return "0"
    exponent = int(np.round(np.log10(abs(value))))
    sign = "-" if value < 0 else ""
    return rf"${sign}10^{{{exponent}}}$"


def _terminal_strip_offsets(values: np.ndarray) -> np.ndarray:
    """Deterministic one-category strip offsets; y has no quantitative meaning."""
    pattern = np.array([0.0, 0.075, -0.075, 0.15, -0.15, 0.225, -0.225])
    offsets = np.empty(values.size, dtype=float)
    order = np.argsort(values)
    for rank, index in enumerate(order):
        offsets[index] = pattern[rank % len(pattern)]
    return offsets


def make_figure5(
    *,
    layout: str = "equal",
    output_dir: Path | None = None,
    stem: str = "fig5_convergence",
    write_audit: bool = True,
) -> dict:
    set_style()
    data = _load_convergence_audit(write_outputs=write_audit)
    cumulative = data["cumulative"]
    rolling = data["rolling"]
    terminal = data["terminal"]
    integrity = data["integrity"]
    tol = data["tol"]
    newton_start = data["newton_start"]

    if layout == "equal":
        width_ratios = [1.0, 1.0, 1.0]
        wspace = 0.49
    elif layout == "wide_trajectory":
        width_ratios = [1.45, 0.95, 0.95]
        wspace = 0.45
    else:
        raise ValueError("layout must be 'equal' or 'wide_trajectory'")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.1, 2.75),
        gridspec_kw={"width_ratios": width_ratios},
    )
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.20, top=0.78, wspace=wspace)
    fig.suptitle(
        "Fixed-budget convergence diagnostics on Table tennis",
        fontsize=10.5,
        fontweight="bold",
        y=0.98,
    )

    # The aggregate archive is one-based. Plot solver indices 0-2999 so the
    # configured newt_start=50 aligns with the solver's actual iteration index.
    solver_iteration = cumulative.index.to_numpy(float) - 1
    subject_color = "#78909C"

    ax = axes[0]
    panel_title(ax, "A", "Cumulative likelihood gain", y=1.09)
    for subject in cumulative.columns:
        ax.plot(
            solver_iteration,
            cumulative[subject],
            color=subject_color,
            linewidth=0.55,
            alpha=0.22,
        )
    med = cumulative.median(axis=1)
    q1 = cumulative.quantile(0.25, axis=1)
    q3 = cumulative.quantile(0.75, axis=1)
    ax.fill_between(solver_iteration, q1, q3, color=LIGHT_BLUE, alpha=0.25, linewidth=0)
    ax.plot(solver_iteration, med, color=BLUE, linewidth=2.1)
    ax.axvline(newton_start, color=GREY, linestyle=":", linewidth=1.0)
    ax.text(
        newton_start + 25,
        0.97,
        "Newton phase\nbegins",
        transform=ax.get_xaxis_transform(),
        fontsize=6.8,
        color=GREY,
        ha="left",
        va="top",
    )
    iter99 = int(integrity["cohort"]["median_solver_iteration_99pct_gain"])
    ax.text(
        0.94,
        0.06,
        f"99% of the iteration-3,000 gain\nmedian first crossing: {iter99}",
        transform=ax.transAxes,
        fontsize=6.4,
        color=INK,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    ax.set_xlim(0, 3000)
    ax.set_xticks([0, 1000, 2000, 3000])
    ax.set_xlabel("iteration")
    ax.set_ylabel(
        "Likelihood gain from iteration 0\n(nats component$^{-1}$ sample$^{-1}$)",
        fontsize=7.7,
    )
    finish_axes(ax, "both")

    ax = axes[1]
    panel_title(ax, "B", "Per-iteration improvement", y=1.09)
    rolling_iteration = rolling.index.to_numpy(float) - 1
    for subject in rolling.columns:
        ax.plot(
            rolling_iteration,
            rolling[subject],
            color=subject_color,
            linewidth=0.5,
            alpha=0.11,
        )
    med = rolling.median(axis=1)
    q1 = rolling.quantile(0.25, axis=1)
    q3 = rolling.quantile(0.75, axis=1)
    ax.fill_between(rolling_iteration, q1, q3, color=LIGHT_BLUE, alpha=0.18, linewidth=0)
    ax.plot(rolling_iteration, med, color=BLUE, linewidth=2.1)
    ax.axhline(0, color="#B8BDC3", linewidth=0.75)
    ax.axhline(tol, color=ORANGE, linestyle="--", linewidth=1.1)
    ax.axvline(newton_start, color=GREY, linestyle=":", linewidth=1.0)
    ax.text(
        2970,
        tol,
        " per-iteration tolerance",
        fontsize=6.6,
        color=ORANGE,
        ha="right",
        va="bottom",
    )
    ax.set_yscale("symlog", linthresh=1e-10, linscale=0.8, base=10)
    ax.set_yticks([-1e-5, -1e-7, -1e-9, 0, 1e-9, 1e-7, 1e-5, 1e-3])
    ax.yaxis.set_major_formatter(FuncFormatter(_signed_sci_tick))
    ax.set_xlim(0, 3000)
    ax.set_xticks([0, 1000, 2000, 3000])
    ax.set_xlabel("iteration")
    ax.set_ylabel(
        "25-iteration rolling median $\\Delta\\ell$\n(nats component$^{-1}$ sample$^{-1}$)",
        fontsize=7.7,
    )
    finish_axes(ax, "both")

    ax = axes[2]
    panel_title(ax, "C", "Terminal improvement", y=1.09)
    terminal_values = terminal.to_numpy(float)
    offsets = _terminal_strip_offsets(terminal_values)
    above = terminal_values > tol
    ax.scatter(
        terminal_values[above],
        offsets[above],
        s=19,
        color=BLUE,
        edgecolor="white",
        linewidth=0.45,
        alpha=0.85,
        marker="o",
        zorder=4,
    )
    ax.scatter(
        terminal_values[~above],
        offsets[~above],
        s=22,
        color=ORANGE,
        edgecolor="black",
        linewidth=0.45,
        alpha=0.9,
        marker="D",
        zorder=5,
    )
    tq1, tmed, tq3 = np.quantile(terminal_values, [0.25, 0.5, 0.75])
    summary_y = -0.31
    ax.plot([tq1, tq3], [summary_y, summary_y], color="black", linewidth=1.9, zorder=6)
    ax.plot(
        tmed,
        summary_y,
        "s",
        color=BLUE,
        markeredgecolor="black",
        markeredgewidth=0.65,
        markersize=6.0,
        zorder=7,
    )
    ax.axvline(tol, color=ORANGE, linestyle="--", linewidth=1.1)
    ax.set_xscale("symlog", linthresh=1e-10, linscale=0.75, base=10)
    ax.set_xlim(-2e-8, 1e-4)
    ax.set_xticks([-1e-8, 0, 1e-8, 1e-6, 1e-4])
    ax.xaxis.set_major_formatter(FuncFormatter(_signed_sci_tick))
    ax.set_ylim(-0.43, 0.58)
    ax.set_yticks([0])
    ax.set_yticklabels(["subjects"])
    ax.tick_params(axis="y", length=0, pad=4)
    ax.spines["left"].set_visible(False)
    count_above = int(integrity["cohort"]["n_terminal_medians_above_tolerance"])
    count_cap = int(integrity["cohort"]["n_reaching_max_iter"])
    ax.text(
        0.02,
        0.96,
        f"{count_cap}/25 reached the iteration cap\n{count_above}/25 medians above tolerance",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        color=INK,
        fontweight="bold",
    )
    ax.set_xlabel(
        "Median $\\Delta\\ell$, iterations 2751-3000\n"
        "(nats component$^{-1}$ sample$^{-1}$)"
    )
    finish_axes(ax, "x")

    target_dir = FIG_DIR if output_dir is None else output_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(target_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(target_dir / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    integrity["selected_layout"] = layout
    integrity["output_files"] = {
        "pdf": str((target_dir / f"{stem}.pdf").resolve()),
        "png": str((target_dir / f"{stem}.png").resolve()),
        "png_dpi": 600,
    }
    if write_audit:
        CONVERGENCE_AUDIT_JSON.write_text(
            json.dumps(integrity, indent=2) + "\n", encoding="utf-8"
        )
    return integrity


def make_figure4() -> dict:
    """Focused computation figure: deployment time, implementation audit, memory."""
    set_style()
    bench = pd.read_csv(BENCH_505)
    mem = load_memory_rows()
    scaling = load_scaling_rows()
    runtime_integrity = _figure4_integrity_stats(bench, mem, scaling)
    fixed_workload = load_fixed_workload_runtime_audit(write_output=True)
    paired_memory = load_paired_chunking_memory()

    fig = plt.figure(figsize=(7.2, 3.25))
    gs = fig.add_gridspec(
        1,
        3,
        left=0.10,
        right=0.985,
        bottom=0.20,
        top=0.80,
        wspace=0.66,
        width_ratios=[0.95, 1.22, 1.08],
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    fig.suptitle(
        "Computational performance and memory-bounded fitting",
        fontsize=10.3,
        fontweight="bold",
        y=0.985,
    )

    display = {
        "AMICA-Python (JAX-GPU)": "amica JAX-GPU",
        "Picard": "Picard",
        "Infomax": "Ext. Infomax",
        "FastICA": "FastICA",
    }
    colors = {
        "AMICA-Python (JAX-GPU)": BLUE,
        **COMPARATOR_COLORS,
    }
    markers = {
        "AMICA-Python (JAX-GPU)": "o",
        **COMPARATOR_MARKERS,
    }

    # Panel A: deployment-level estimator-call runtime.
    panel_title(ax_a, "A", "End-to-end runtime", y=1.08)
    ax_a.text(
        0,
        1.015,
        "H100 amica vs CPU solvers; stopping rules differ",
        transform=ax_a.transAxes,
        fontsize=5.7,
        color=GREY,
        va="bottom",
    )
    methods_a = ["AMICA-Python (JAX-GPU)", "Picard", "Infomax", "FastICA"]
    for y, method in enumerate(methods_a):
        values = bench.loc[bench.method == method, "fit_runtime_s"].dropna().to_numpy(float)
        rng = np.random.default_rng(4400 + y)
        ax_a.scatter(
            values,
            y + rng.normal(0, 0.065, len(values)),
            s=7.0,
            color=colors[method],
            alpha=0.28,
            linewidths=0,
            rasterized=True,
        )
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        ax_a.plot([q1, q3], [y, y], color=INK, linewidth=1.35)
        ax_a.plot(
            median,
            y,
            marker=markers[method],
            color=colors[method],
            markeredgecolor=INK,
            markeredgewidth=0.55,
            markersize=5.4,
        )
        ax_a.annotate(
            f"{median:,.0f}",
            (median, y),
            xytext=(4, -7),
            textcoords="offset points",
            fontsize=5.9,
            ha="left",
            va="top",
        )
    ax_a.set_xscale("log")
    ax_a.set_xlim(40, 3.2e3)
    ax_a.set_yticks(range(len(methods_a)), [display[m] for m in methods_a], fontsize=6.2)
    ax_a.set_ylim(len(methods_a) - 0.55, -0.55)
    ax_a.set_xlabel("Estimator-call time per subject (s)", fontsize=7.1)
    finish_axes(ax_a)

    # Panel B: one fixed-workload audit. Do not imply repeated-run uncertainty.
    panel_title(ax_b, "B", "Implementation audit", y=1.08)
    ax_b.text(
        0,
        1.015,
        r"sub-01; $64\times785{,}328$; 100 iterations; one run",
        transform=ax_b.transAxes,
        fontsize=5.7,
        color=GREY,
        va="bottom",
    )
    order_b = [
        "amica JAX-GPU (chunked)",
        "amica JAX-CPU",
        "Scott–Huberty amica-python 0.1.1",
        "Fortran AMICA 1.7",
        "PyAMICA 0.3.0",
    ]
    label_b = {
        "amica JAX-GPU (chunked)": "amica JAX-GPU\n(chunked)",
        "amica JAX-CPU": "amica JAX-CPU",
        "Scott–Huberty amica-python 0.1.1": "Scott–Huberty\n0.1.1 CPU",
        "Fortran AMICA 1.7": "Fortran AMICA\n1.7 CPU",
        "PyAMICA 0.3.0": "PyAMICA\n0.3.0 CPU",
    }
    color_b = {
        "amica JAX-GPU (chunked)": BLUE,
        "amica JAX-CPU": LIGHT_BLUE,
        "Scott–Huberty amica-python 0.1.1": "#8C8C8C",
        "Fortran AMICA 1.7": "#525252",
        "PyAMICA 0.3.0": "#B0B0B0",
    }
    marker_b = {
        "amica JAX-GPU (chunked)": "o",
        "amica JAX-CPU": "s",
        "Scott–Huberty amica-python 0.1.1": "D",
        "Fortran AMICA 1.7": "^",
        "PyAMICA 0.3.0": "v",
    }
    fixed = fixed_workload.set_index("display")
    for y, label in enumerate(order_b):
        value = float(fixed.loc[label, "fit_time_s"])
        face = "white" if label == "Fortran AMICA 1.7" else color_b[label]
        ax_b.plot(
            value,
            y,
            marker=marker_b[label],
            markersize=6.0,
            markerfacecolor=face,
            markeredgecolor=INK,
            markeredgewidth=0.7,
            linestyle="none",
            zorder=4,
        )
        ax_b.annotate(
            f"{value:,.1f}",
            (value, y),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=5.8,
            va="center",
        )
    ax_b.axhline(1.5, color="#D5D8DB", linewidth=0.7)
    ax_b.text(0.98, 0.02, "lower is faster", transform=ax_b.transAxes, fontsize=5.7, color=GREY, ha="right")
    ax_b.set_xscale("log")
    ax_b.set_xlim(20, 1.35e3)
    ax_b.set_yticks(range(len(order_b)), [label_b[label] for label in order_b], fontsize=5.9)
    ax_b.set_ylim(len(order_b) - 0.55, -0.55)
    ax_b.set_xlabel("Fixed-workload time (s)", fontsize=7.1)
    finish_axes(ax_b)

    # Panel C: directly paired full/chunked process peaks across six recordings.
    panel_title(ax_c, "C", "Memory scaling", y=1.08)
    ax_c.text(
        0,
        1.015,
        "six paired recordings; 64 PCs; fresh CPU processes",
        transform=ax_c.transAxes,
        fontsize=5.7,
        color=GREY,
        va="bottom",
    )
    x = paired_memory.n_samples.to_numpy(float) / 1e6
    ax_c.plot(
        x,
        paired_memory.full_peak_rss_gib,
        "o-",
        color=LIGHT_BLUE,
        markeredgecolor=INK,
        markeredgewidth=0.45,
        linewidth=1.8,
        label="full batch",
    )
    ax_c.plot(
        x,
        paired_memory.chunked_peak_rss_gib,
        "s-",
        color=BLUE,
        markeredgecolor=INK,
        markeredgewidth=0.45,
        linewidth=1.8,
        label="automatic chunking",
    )
    median_reduction = float(np.median(paired_memory.total_peak_reduction_pct))
    ax_c.text(
        0.04,
        0.19,
        f"median paired reduction: {median_reduction:.0f}%",
        transform=ax_c.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.0,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 0.8},
    )
    ax_c.set_xlim(0.74, 1.41)
    ax_c.set_ylim(5.5, 20.8)
    ax_c.set_xlabel(r"Samples, $T$ ($\times10^6$)", fontsize=7.1)
    ax_c.set_ylabel("Peak process RSS (GiB)", fontsize=7.1)
    ax_c.legend(frameon=False, fontsize=6.2, loc="upper left", handlelength=1.6)
    finish_axes(ax_c, "both")

    save_figure(fig, "fig4_computational_performance", png_dpi=600)

    fixed_export = fixed_workload.drop(columns=["source_path"])
    paired_export = paired_memory.drop(
        columns=["full_source_path", "chunked_source_path"]
    )
    return {
        "deployment_runtime": runtime_integrity["runtime"],
        "deployment_runtime_definition": runtime_integrity["runtime_definition"],
        "fixed_workload_audit": {
            "records": fixed_export.to_dict(orient="records"),
            "scope": (
                "one archived run per implementation on Table tennis sub-01; Python model.fit timing; "
                "Fortran external-executable timing; descriptive, not a repeated implementation ranking"
            ),
            "missing_controlled_path": (
                "No matched NumPy-CPU record or repeated Fortran process runs were archived for this fixture."
            ),
        },
        "paired_chunking_memory": {
            "records": paired_export.to_dict(orient="records"),
            "median_total_peak_reduction_pct": median_reduction,
            "range_total_peak_reduction_pct": [
                float(paired_memory.total_peak_reduction_pct.min()),
                float(paired_memory.total_peak_reduction_pct.max()),
            ],
            "memory_unit": "GiB (binary process RSS high-water mark)",
        },
    }


# ---------------------------------------------------------------------------
# Figure 5: native six-panel multi-model validation figure
# ---------------------------------------------------------------------------
MULTIMODEL_META = {
    "ds004505 task (120 ch)": {
        "root": MM_ROOTS["ds004505 task (120 ch)"],
        "cohort_id": "ds004505_task_120ch",
        "dataset": "ds004505",
        "display": "Table tennis — 120 ch",
        "source_type": "real_eeg",
        "surrogate_type": "none",
        "recorded_channel_count": 120,
    },
    "ds004505 task (19 ch)": {
        "root": MM_ROOTS["ds004505 task (19 ch)"],
        "cohort_id": "ds004505_task_19ch",
        "dataset": "ds004505",
        "display": "Table tennis — 19 ch",
        "source_type": "real_eeg",
        "surrogate_type": "none",
        "recorded_channel_count": 120,
    },
    "ds004504 rest (19 ch)": {
        "root": MM_ROOTS["ds004504 rest (19 ch)"],
        "cohort_id": "ds004504_rest_19ch",
        "dataset": "ds004504",
        "display": "Eyes-closed rest — 19 ch",
        "source_type": "real_eeg",
        "surrogate_type": "none",
        "recorded_channel_count": 19,
    },
    "ds004621 rest (127 ch)": {
        "root": MM_ROOTS["ds004621 rest (127 ch)"],
        "cohort_id": "ds004621_rest_127ch",
        "dataset": "ds004621",
        "display": "Eyes-open rest — 127 ch",
        "source_type": "real_eeg",
        "surrogate_type": "none",
        "recorded_channel_count": 128,
    },
    "ds004505 phase surrogate": {
        "root": SURR_ROOTS["ds004505 phase surrogate"],
        "cohort_id": "ds004505_phase_surrogate",
        "dataset": "ds004505",
        "display": "Table tennis phase surrogate",
        "source_type": "phase_surrogate",
        "surrogate_type": "common_phase_fourier",
        "recorded_channel_count": 120,
    },
    "ds004504 phase surrogate": {
        "root": SURR_ROOTS["ds004504 phase surrogate"],
        "cohort_id": "ds004504_phase_surrogate",
        "dataset": "ds004504",
        "display": "Eyes-closed rest phase surrogate",
        "source_type": "phase_surrogate",
        "surrogate_type": "common_phase_fourier",
        "recorded_channel_count": 19,
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _scalar(z: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in z.files:
        return default
    value = np.asarray(z[key])
    return value.item() if value.size == 1 else value


def _posterior_switch_rate(post: np.ndarray, step: int, sfreq: float) -> float:
    if post.shape[0] == 1 or post.shape[1] < 2:
        return 0.0
    hard = np.argmax(post, axis=0)
    duration = (hard.size - 1) * float(step) / float(sfreq)
    return float(np.count_nonzero(np.diff(hard)) / duration) if duration > 0 else np.nan


def _load_multimodel_audit() -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    source_hashes: dict[str, str] = {}

    for meta in MULTIMODEL_META.values():
        root = meta["root"]
        actual_subjects: set[int] = set()
        actual_orders: dict[int, set[int]] = {}
        for path in sorted(root.glob("*.npz")):
            with np.load(path, allow_pickle=True) as z:
                subject = int(_scalar(z, "subject"))
                model_order = int(_scalar(z, "num_models"))
                actual_subjects.add(subject)
                actual_orders.setdefault(subject, set()).add(model_order)
                skipped = bool(_scalar(z, "skipped_underpowered", False))
                gm = np.atleast_1d(np.asarray(z["gm"], dtype=float)) if "gm" in z.files else np.array([])
                if gm.size:
                    gm = gm / gm.sum()
                    neff = float(np.exp(-np.sum(gm * np.log(np.clip(gm, 1e-30, None)))))
                else:
                    neff = np.nan
                post = (
                    np.asarray(z["model_posteriors_ds"], dtype=float)
                    if "model_posteriors_ds" in z.files
                    else np.ones((model_order, 1), dtype=float) / model_order
                )
                sfreq = float(_scalar(z, "sfreq", np.nan))
                post_step = int(_scalar(z, "post_downsample_step", 1))
                final_ll = float(_scalar(z, "ll_final", np.nan))
                n_iter = int(_scalar(z, "n_iter", 0))
                max_iter = int(_scalar(z, "max_iter", 2000))
                finite = bool(np.isfinite(final_ll) and np.isfinite(gm).all())
                weight_collapse = bool(model_order > 1 and np.isfinite(neff) and neff < 1.05)
                path_key = str(path.resolve())
                source_hashes[path_key] = _sha256_file(path)
                rows.append(
                    {
                        "source_type": meta["source_type"],
                        "cohort_id": meta["cohort_id"],
                        "dataset": meta["dataset"],
                        "subject": f"sub-{subject:02d}",
                        "channel_count": int(len(z["ch_names"])) if "ch_names" in z.files else np.nan,
                        "recorded_channel_count": meta["recorded_channel_count"],
                        "retained_pca_rank": int(_scalar(z, "n_components", 16)),
                        "model_order": model_order,
                        "iteration_budget": max_iter,
                        "n_iterations": n_iter,
                        "stopping_status": (
                            "iteration_cap" if n_iter >= max_iter else "stopped_before_cap_reason_not_archived"
                        ),
                        "backend": f"jax_{str(_scalar(z, 'device', 'unknown')).lower()}",
                        "seed": 0,
                        "seed_source": "runner default; submission did not override",
                        "n_samples": int(_scalar(z, "n_samples", 0)),
                        "sfreq_hz": sfreq,
                        "final_ll_per_sample": final_ll,
                        "likelihood_normalisation": "nats_per_retained_component_per_sample",
                        "delta_ll_vs_m1": np.nan,
                        "model_weights": json.dumps(gm.tolist()),
                        "neff": neff,
                        "switch_rate": _posterior_switch_rate(post, post_step, sfreq),
                        "switch_rate_definition": "hard-posterior transitions per second after 10-Hz block averaging",
                        "fit_success": bool(not skipped and finite),
                        "missing_or_degenerate_flag": bool(skipped or not finite or weight_collapse),
                        "flag_reason": (
                            "weight_collapse_neff_lt_1.05"
                            if weight_collapse
                            else ("skipped_underpowered" if skipped else ("nonfinite" if not finite else ""))
                        ),
                        "surrogate_type": meta["surrogate_type"],
                        "source_map_error": np.nan,
                        "SIR": np.nan,
                        "true_regime": "unknown_real_eeg" if meta["source_type"] == "real_eeg" else "stationary_reference",
                        "posterior_sum_max_abs_error": float(np.max(np.abs(post.sum(axis=0) - 1.0))),
                        "source_file": path_key,
                        "source_file_sha256": source_hashes[path_key],
                    }
                )

        # Missing archives are explicit audit rows rather than silently dropped.
        for subject in sorted(actual_subjects):
            for model_order in sorted(set(range(1, 11)) - actual_orders.get(subject, set())):
                rows.append(
                    {
                        "source_type": meta["source_type"],
                        "cohort_id": meta["cohort_id"],
                        "dataset": meta["dataset"],
                        "subject": f"sub-{subject:02d}",
                        "channel_count": np.nan,
                        "recorded_channel_count": meta["recorded_channel_count"],
                        "retained_pca_rank": 16,
                        "model_order": model_order,
                        "iteration_budget": 2000,
                        "n_iterations": np.nan,
                        "stopping_status": "missing_archive",
                        "backend": "jax_gpu",
                        "seed": 0,
                        "seed_source": "runner default; submission did not override",
                        "n_samples": np.nan,
                        "sfreq_hz": 250.0,
                        "final_ll_per_sample": np.nan,
                        "likelihood_normalisation": "nats_per_retained_component_per_sample",
                        "delta_ll_vs_m1": np.nan,
                        "model_weights": "",
                        "neff": np.nan,
                        "switch_rate": np.nan,
                        "switch_rate_definition": "hard-posterior transitions per second after 10-Hz block averaging",
                        "fit_success": False,
                        "missing_or_degenerate_flag": True,
                        "flag_reason": "missing_archive",
                        "surrogate_type": meta["surrogate_type"],
                        "source_map_error": np.nan,
                        "SIR": np.nan,
                        "true_regime": "unknown_real_eeg" if meta["source_type"] == "real_eeg" else "stationary_reference",
                        "posterior_sum_max_abs_error": np.nan,
                        "source_file": "",
                        "source_file_sha256": "",
                    }
                )

    audit = pd.DataFrame(rows)
    for (cohort_id, subject), group in audit[audit["fit_success"]].groupby(["cohort_id", "subject"]):
        base = group.loc[group["model_order"] == 1, "final_ll_per_sample"]
        if base.empty:
            continue
        mask = (audit["cohort_id"] == cohort_id) & (audit["subject"] == subject) & audit["fit_success"]
        audit.loc[mask, "delta_ll_vs_m1"] = audit.loc[mask, "final_ll_per_sample"] - float(base.iloc[0])

    with SYNTHETIC_JSON.open(encoding="utf-8") as handle:
        synth = json.load(handle)
    stationary = pd.DataFrame(synth["stationary"]).rename(columns={"H": "M"})
    nonstationary = pd.DataFrame(synth["non_stationary"]).rename(columns={"H": "M"})
    synthetic_hash = _sha256_file(SYNTHETIC_JSON)
    for frame, source_type, dataset, true_regime in (
        (stationary, "synthetic_stationary", "synthetic_stationary", "stationary_single_mixture"),
        (nonstationary, "synthetic_nonstationary", "synthetic_nonstationary", "M_true=3"),
    ):
        frame["delta_ll"] = frame.ll - frame.loc[frame.M == 1, "ll"].iloc[0]
        for row in frame.itertuples():
            audit.loc[len(audit)] = {
                "source_type": source_type,
                "cohort_id": dataset,
                "dataset": dataset,
                "subject": "seed-0",
                "channel_count": 16,
                "recorded_channel_count": np.nan,
                "retained_pca_rank": 16,
                "model_order": int(row.M),
                "iteration_budget": int(synth["config"]["max_iter"]),
                "n_iterations": np.nan,
                "stopping_status": "not_archived",
                "backend": "not_archived",
                "seed": int(synth["config"]["seed"]),
                "seed_source": "synthetic JSON configuration",
                "n_samples": int(synth["config"]["tseg"] * synth["config"]["n_regimes"]),
                "sfreq_hz": float(synth["config"]["sfreq"]),
                "final_ll_per_sample": float(row.ll),
                "likelihood_normalisation": "nats_per_component_per_sample",
                "delta_ll_vs_m1": float(row.delta_ll),
                "model_weights": json.dumps(list(row.gm)),
                "neff": float(row.n_eff),
                "switch_rate": float(row.switching_rate_hz),
                "switch_rate_definition": "hard-posterior transitions per second",
                "fit_success": True,
                "missing_or_degenerate_flag": False,
                "flag_reason": "",
                "surrogate_type": "none",
                "source_map_error": float(row.model_error),
                "SIR": float(row.sir_db),
                "true_regime": true_regime,
                "posterior_sum_max_abs_error": np.nan,
                "source_file": str(SYNTHETIC_JSON.resolve()),
                "source_file_sha256": synthetic_hash,
            }

    # The displayed posterior is a predefined archived demo, not a data-driven selection.
    demo_m1 = DEMO_NPZ.with_name("mm_demo_sub-04_M1.npz")
    with np.load(DEMO_NPZ, allow_pickle=True) as z, np.load(demo_m1, allow_pickle=True) as z1:
        gm = np.asarray(z["gm"], dtype=float)
        gm = gm / gm.sum()
        post = np.asarray(z["model_posteriors"], dtype=float)
        sfreq = float(z["sfreq"])
        audit.loc[len(audit)] = {
            "source_type": "illustrative_demo",
            "cohort_id": "ds004505_demo_sub04",
            "dataset": "ds004505",
            "subject": "sub-04",
            "channel_count": 120,
            "recorded_channel_count": 120,
            "retained_pca_rank": int(z["n_components"]),
            "model_order": int(z["num_models"]),
            "iteration_budget": 2000,
            "n_iterations": int(z["n_iter"]),
            "stopping_status": "iteration_cap" if int(z["n_iter"]) >= 2000 else "stopped_before_cap_reason_not_archived",
            "backend": f"jax_{str(z['device'].item()).lower()}",
            "seed": 0,
            "seed_source": "demo runner default; submission did not override",
            "n_samples": int(z["n_samples"]),
            "sfreq_hz": sfreq,
            "final_ll_per_sample": float(z["ll_final"]),
            "likelihood_normalisation": "nats_per_retained_component_per_sample",
            "delta_ll_vs_m1": float(z["ll_final"] - z1["ll_final"]),
            "model_weights": json.dumps(gm.tolist()),
            "neff": float(np.exp(-np.sum(gm * np.log(np.clip(gm, 1e-30, None))))),
            "switch_rate": _posterior_switch_rate(post, 1, sfreq),
            "switch_rate_definition": "hard-posterior transitions per second on unsmoothed posterior",
            "fit_success": True,
            "missing_or_degenerate_flag": False,
            "flag_reason": "",
            "surrogate_type": "none",
            "source_map_error": np.nan,
            "SIR": np.nan,
            "true_regime": "unknown_real_eeg",
            "posterior_sum_max_abs_error": float(np.max(np.abs(post.sum(axis=0) - 1.0))),
            "source_file": str(DEMO_NPZ.resolve()),
            "source_file_sha256": _sha256_file(DEMO_NPZ),
        }

    valid = audit[audit["fit_success"]].copy()
    real = valid[valid["source_type"] == "real_eeg"]
    phase = valid[valid["source_type"] == "phase_surrogate"]
    subject_counts = {
        cohort: {str(int(m)): int(n) for m, n in group.groupby("model_order")["subject"].nunique().items()}
        for cohort, group in real.groupby("cohort_id")
    }
    m10_medians = {
        cohort: float(group.loc[group["model_order"] == 10, "delta_ll_vs_m1"].median())
        for cohort, group in real.groupby("cohort_id")
    }
    phase_m10_medians = {
        cohort: float(group.loc[group["model_order"] == 10, "delta_ll_vs_m1"].median())
        for cohort, group in phase.groupby("cohort_id")
    }
    early = real[real["n_iterations"] < real["iteration_budget"]]
    collapsed = real[real["flag_reason"] == "weight_collapse_neff_lt_1.05"]
    missing = audit[audit["flag_reason"] == "missing_archive"]
    archived_failed = audit[
        audit["source_type"].isin(["real_eeg", "phase_surrogate"])
        & (audit["flag_reason"] != "missing_archive")
        & ~audit["fit_success"]
    ]
    integrity = {
        "original_figure_audit": {
            "native_assembly": "The current draft is already native, but its predecessor and filenames still reflect two separately generated source figures.",
            "visual_hierarchy": "Synthetic and real rows are separated, but the real likelihood panel lacks variability and panel f is cramped at manuscript width.",
            "panel_labels": "Continuous a-f in the current draft; legacy source PDFs duplicate a-c and use H/K notation.",
            "aggregation": "Current real curves use cohort means without uncertainty; the revision uses medians and IQRs.",
            "posterior_context": "Current panel identifies sub-04 and task onset but omits the 64-PC configuration and calls a predefined example representative.",
            "scientific_scope": "The rejection ablation addresses a different intervention and remains supplementary.",
        },
        "notation_consistency": {
            "final_model_order": "M",
            "true_generating_order": "M_true",
            "posterior": "p(m | t)",
            "model_prior": "pi_m",
            "density_mixture_count": "K (only for adaptive source-density terms)",
            "archived_legacy_notation": "The source JSON and runner use H; the final figure and manuscript translate this to M without altering values.",
        },
        "dataset_channel_rank_audit": {
            "ds004505_task_120ch": "120 selected scalp-EEG channels, reduced to 16 PCs for the model-order sweep.",
            "ds004505_task_19ch": "19 name-matched 10-20 channels selected from ds004505, reduced to 16 PCs.",
            "ds004504_rest_19ch": "19 EEG channels, reduced to 16 PCs.",
            "ds004621_rest_127ch": "A nominal 128-channel system with 127 stored EEG channels because FCz was the online reference; reduced to 16 PCs.",
            "posterior_demo": "ds004505 sub-04, 120 selected scalp channels reduced to 64 PCs, first 600 s.",
        },
        "surrogate_method_audit": {
            "stage": "Applied after PCA and per-component variance normalisation.",
            "method": "One seed-0 Fourier surrogate per subject; a common random phase is applied at each positive frequency to every retained component, with DC and Nyquist unchanged and inverse rFFT preserving a real signal.",
            "preserved": "Each component amplitude spectrum and the full cross-spectral matrix are preserved exactly up to floating-point error.",
            "changed": "Fourier phase, and therefore the original temporal organisation, is randomised.",
            "realisations": "Five ds004505 subjects and five ds004504 subjects; one surrogate realisation per subject, reused across M through the same seed/preprocessed input.",
        },
        "aggregation": {
            "real_and_phase_curves": "median across subjects",
            "uncertainty": "interquartile range across subjects",
            "synthetic": "single seed-0 simulation; no uncertainty available",
            "likelihood_normalisation": "nats per retained component per sample",
            "model_order_range": [1, 10],
            "maximum_iterations": 2000,
        },
        "cohort_subject_counts_by_model_order": subject_counts,
        "m10_real_delta_ll_medians": m10_medians,
        "m10_phase_surrogate_delta_ll_medians": phase_m10_medians,
        "fit_integrity": {
            "real_fit_rows": int(len(real)),
            "phase_surrogate_fit_rows": int(len(phase)),
            "real_fits_stopping_before_2000": int(len(early)),
            "early_stop_records": early[["dataset", "subject", "model_order", "n_iterations"]].to_dict("records"),
            "missing_archives": int(len(missing)),
            "missing_records": missing[["dataset", "subject", "model_order"]].to_dict("records"),
            "weight_collapse_diagnostics": int(len(collapsed)),
            "weight_collapse_records": collapsed[["dataset", "subject", "model_order", "neff"]].to_dict("records"),
            "failed_or_nonfinite_archived_fits": int(len(archived_failed)),
            "posterior_normalisation_max_error": float(real["posterior_sum_max_abs_error"].max()),
        },
        "representative_subject_selection": {
            "label_used": "illustrative example",
            "rule": "sub-04 was one of the two subjects predefined in submit_multimodel_demo.sh (array 1,4), before this figure revision; it was not selected from the cohort sweep by effect size.",
            "smoothing": "2-s centred moving mean applied after posterior normalisation; the smoothed traces are renormalised to sum to one at each sample.",
        },
        "model_complexity_audit": "No archived held-out or complexity-corrected likelihood result corresponds to these real-EEG model-order sweeps; panels d-e therefore retain explicitly labelled in-sample quantities.",
        "task_decoding_audit": "Decoding utilities exist, but no committed cohort-level, leakage-audited decoding result with uncertainty was found for these fit archives; decoding is not promoted.",
        "layout_comparison": {
            "candidate_A_equal_2x3": "Balanced column alignment, readable axes, and a clear synthetic-versus-real row grammar.",
            "candidate_B_wide_panel_d": "Gives panel d more width but makes N_eff and the posterior context too narrow and breaks cross-row column alignment.",
            "candidate_C_direct_matrices": "Not prototyped because the two-regime matrix has no archived machine-readable result object; retaining it in Supplementary Table S3 avoids presenting copied values as a regenerated panel.",
            "selected": "candidate_A_equal_2x3",
        },
        "supplementary_decisions": {
            "fig_synthetic_stationarity_contrast.pdf": "Retained as an archived legacy output but superseded by panels a-c.",
            "fig6_multimodel.pdf": "Retained for before/after review but retired from LaTeX after the new native figure is selected.",
            "fig_multimodel_demo_sub04.pdf": "Retired from LaTeX because panel f promotes its interpretable content and the remaining single-subject likelihood panel duplicates panel d.",
            "fig_ablation_ds004505.pdf": "Retained supplementary as a sensitivity analysis; not promoted.",
            "Supplementary_Table_S2": "Retained for exact cohort/model-order mean values; Figure 5 uses medians and IQRs.",
            "Supplementary_Table_S3": "Retained as explicitly single-configuration evidence; not promoted because its result object is not archived.",
        },
        "provenance": {
            "multimodel_generator_commit": _git_head(WORKSPACE / "figdata/synth/amica-mm"),
            "demo_repository_commit": _git_head(WORKSPACE / "repos/amica-python"),
            # Was _git_head(FIG_DIR.parent), which resolved to the Overleaf
            # project root only because FIG_DIR happened to be figures/ there.
            # The producers live here now, so record this repository instead --
            # under a key that says which repository it actually is.
            "producer_repository_commit": _git_head(REPO_ROOT),
            "synthetic_json": {"path": str(SYNTHETIC_JSON.resolve()), "sha256": synthetic_hash},
            "runner": {"path": str(MULTIMODEL_RUNNER.resolve()), "sha256": _sha256_file(MULTIMODEL_RUNNER)},
            "submission": {"path": str(MULTIMODEL_SUBMISSION.resolve()), "sha256": _sha256_file(MULTIMODEL_SUBMISSION)},
            "synthetic_runner": {"path": str(MULTIMODEL_SYNTHETIC_RUNNER.resolve()), "sha256": _sha256_file(MULTIMODEL_SYNTHETIC_RUNNER)},
            "demo_submission": {"path": str(MULTIMODEL_DEMO_SUBMISSION.resolve()), "sha256": _sha256_file(MULTIMODEL_DEMO_SUBMISSION)},
            "per_fit_source_hashes": "Stored row-wise in fig6_multimodel_integrity.csv.",
        },
        "unresolved_issues": [
            "The two-model recovery values in Supplementary Table S3 are reproducible from a unit test but have no archived result object; they were not redrawn.",
            "The per-fit NPZ files do not store the random seed or exact early stopping reason; seed 0 is inferred from the unmodified runner/submission defaults, and early stops are labelled reason-not-archived.",
            "The ds004505 19-channel archive contains 14 subjects, with sub-14 missing M=8 and M=9; no subject is silently imputed.",
            "The synthetic sweep contains one seed, so its panels have no replication uncertainty.",
            "Model labels are not aligned across subjects; only label-invariant cohort quantities are aggregated, and panel f is one fit.",
        ],
    }
    return audit, integrity, stationary, nonstationary


def _summary_by_model(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        data.groupby("model_order", as_index=False)
        .agg(
            median=(metric, "median"),
            q1=(metric, lambda x: x.quantile(0.25)),
            q3=(metric, lambda x: x.quantile(0.75)),
            n=(metric, "count"),
        )
        .rename(columns={"model_order": "M"})
    )


def _draw_posterior_dynamics(ax, *, letter: str | None = None,
                             label_fontsize: float = 6.4) -> dict:
    """Draw the illustrative posterior model-probability time course.

    Shared by the main multi-model figure and the supplementary figure so the
    two renderings cannot drift apart. Returns the provenance of what it drew.
    """
    if letter:
        panel_title(ax, letter, "Posterior model dynamics", y=1.04)
    with np.load(DEMO_NPZ, allow_pickle=True) as z:
        post = np.asarray(z["model_posteriors"], dtype=float)
        sfreq = float(z["sfreq"])
        onsets = np.asarray(z["event_onsets"], dtype=float)
        event_types = np.asarray(z["event_types"], dtype=object)
        demo_rank = int(z["n_components"])
        demo_order = int(z["num_models"])
    win = max(1, int(round(2.0 * sfreq)))
    weights = np.ones(win, dtype=float)
    denominator = np.convolve(np.ones(post.shape[1], dtype=float), weights, mode="same")
    smooth = np.vstack([np.convolve(row, weights, mode="same") / denominator for row in post])
    smooth /= np.clip(smooth.sum(axis=0, keepdims=True), 1e-30, None)
    time_s = np.arange(post.shape[1]) / sfreq
    model_colors = ["#3B4CC0", "#B34F8C", "#D89000"]
    for model_idx, color in enumerate(model_colors):
        ax.plot(time_s, smooth[model_idx], color=color, linewidth=1.25, label=f"model {model_idx + 1}")
    task_events = onsets[event_types == "cooperative"]
    task_onset = float(task_events.min()) if task_events.size else np.nan
    if np.isfinite(task_onset):
        ax.axvspan(task_onset, time_s[-1], color="#E8EAEC", alpha=0.65, linewidth=0, zorder=0)
        ax.axvline(task_onset, color="#4D5257", linestyle="--", linewidth=0.9)
        epoch_bbox = {"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0}
        ax.text(task_onset - 8, 0.995, "status / baseline", ha="right", va="top", fontsize=label_fontsize, color=GREY, bbox=epoch_bbox)
        ax.text(task_onset + 8, 0.995, "cooperative task", ha="left", va="top", fontsize=label_fontsize, color=GREY, bbox=epoch_bbox)
    if task_events.size:
        idx = np.unique(np.linspace(0, task_events.size - 1, min(100, task_events.size)).astype(int))
        ax.plot(task_events[idx], np.full(idx.size, 0.015), "|", color="#555B61", markersize=3.0, alpha=0.42)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(time_s[0], time_s[-1])
    ax.set_xticks([0, 150, 300, 450, 600])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"Posterior probability, $p(m\mid t)$")
    ax.text(0.02, 0.06, f"Table tennis sub-04 | $M={demo_order}$ | {demo_rank} PCs\n2-s moving mean; illustrative", transform=ax.transAxes, fontsize=label_fontsize, color="#555B61", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 1.2})
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.02, 0.38), ncol=1, fontsize=6.2, handlelength=1.6, labelspacing=0.35)
    finish_axes(ax, "both")
    return {"subject": "Table tennis sub-04", "num_models": demo_order,
            "n_components": demo_rank, "smoothing_s": 2.0,
            "source_npz": str(DEMO_NPZ)}


def make_supplementary_figure4() -> dict:
    """Standalone posterior model-probability figure (moved out of Figure 5).

    It is an illustrative single-subject view from an analysis prespecified as
    exploratory; giving it a main-figure panel overstated its status.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    info = _draw_posterior_dynamics(ax, letter=None, label_fontsize=7.0)
    save_figure(fig, "figS4_posterior_dynamics", png_dpi=600)
    return info


def make_figure5_multimodel_published(
    *,
    layout: str = "equal",
    output_dir: Path | None = None,
    stem: str = "fig6_multimodel_stationarity",
    write_audit: bool = True,
) -> dict:
    """The six-panel multi-model figure that the manuscript actually prints.

    Despite the ``fig6_`` stem this is *Figure 5* in the paper -- the stem is
    historical and is kept because ``zenodo.tex`` includes it by name. This
    function was previously called ``_make_figure6_legacy`` and was not wired
    into ``main()``, so a full regeneration rebuilt the unused four-panel
    ``fig5_*`` instead and recorded its provenance as ``figure5``. See
    ``_make_figure5_four_panel_superseded`` for that other rendering.
    """
    set_style()
    audit, integrity, stationary, nonstationary = _load_multimodel_audit()
    valid = audit[audit["fit_success"]].copy()

    fig = plt.figure(figsize=(7.15, 5.72))
    if layout == "equal":
        gs = fig.add_gridspec(2, 3, left=0.075, right=0.985, bottom=0.175, top=0.83, wspace=0.43, hspace=0.82)
        axes = np.array([[fig.add_subplot(gs[0, j]) for j in range(3)], [fig.add_subplot(gs[1, j]) for j in range(3)]])
    elif layout == "emphasised_real":
        gs = fig.add_gridspec(2, 12, left=0.075, right=0.985, bottom=0.175, top=0.83, wspace=1.7, hspace=0.82)
        axes = np.array(
            [
                [fig.add_subplot(gs[0, 0:4]), fig.add_subplot(gs[0, 4:8]), fig.add_subplot(gs[0, 8:12])],
                [fig.add_subplot(gs[1, 0:5]), fig.add_subplot(gs[1, 5:8]), fig.add_subplot(gs[1, 8:12])],
            ]
        )
    elif layout == "five_panel":
        # Posterior dynamics moved to the supplement, so the bottom row carries
        # two panels instead of three. Two separate gridspecs rather than one
        # 12-column grid: with a spanned grid, wspace inserts a gap between
        # every column, so a block spanning four columns swallows the internal
        # gaps and the panels end up too close for their y-axis labels.
        # Row extents reproduce the "equal" layout (bottom=0.175, top=0.83,
        # hspace=0.82 over two rows).
        gs_top = fig.add_gridspec(1, 3, left=0.075, right=0.985,
                                  bottom=0.598, top=0.83, wspace=0.43)
        gs_bot = fig.add_gridspec(1, 2, left=0.075, right=0.985,
                                  bottom=0.175, top=0.407, wspace=0.30)
        axes = np.empty((2, 3), dtype=object)
        for j in range(3):
            axes[0, j] = fig.add_subplot(gs_top[0, j])
        for j in range(2):
            axes[1, j] = fig.add_subplot(gs_bot[0, j])
        axes[1, 2] = None
    else:
        raise ValueError(f"Unknown Figure 5 layout: {layout}")

    fig.suptitle("Multi-model recovery and stationarity signatures", fontsize=10.5, fontweight="bold", y=0.985)
    fig.text(0.075, 0.885, "Synthetic controls — known ground truth", fontsize=8.8, fontweight="bold", color="#4F555B")
    fig.text(0.075, 0.485, "Real EEG — exploratory stationarity signatures", fontsize=8.8, fontweight="bold", color="#4F555B")

    synthetic_specs = [
        ("delta_ll", r"$\Delta LL$ relative to $M=1$" "\n" r"(nats component$^{-1}$ sample$^{-1}$)", "Likelihood gain"),
        ("sir_db", "Signal-to-interference ratio (dB)", "Source recovery"),
        ("model_error", "Matched map error, $1-|r|$", "Ground-truth map recovery"),
    ]
    for idx, (key, ylabel, title) in enumerate(synthetic_specs):
        ax = axes[0, idx]
        # Uppercase to match the caption's (A--C)/(D--E)/(F) and Figures 2-4.
        panel_title(ax, chr(ord("A") + idx), title, y=1.04)
        ax.plot(
            nonstationary.M,
            nonstationary[key],
            "o-",
            color=BLUE,
            markeredgecolor=INK,
            markeredgewidth=0.45,
            linewidth=1.9,
            label=r"non-stationary, $M_{\mathrm{true}}=3$",
        )
        ax.plot(
            stationary.M,
            stationary[key],
            "s--",
            color=GREY,
            markeredgecolor=INK,
            markeredgewidth=0.4,
            linewidth=1.45,
            label="stationary control",
        )
        ax.axvline(3, color="#666B70", linestyle=":", linewidth=0.95)
        ax.set_xticks([1, 3, 5, 7, 10])
        ax.set_xlabel(r"Number of fitted models, $M$")
        ax.set_ylabel(ylabel)
        finish_axes(ax, "both")
    # The stationary control is flat at zero, so an inline label near y=0 lands on
    # its own curve; lift it into the empty band above and give both a white bbox.
    inline_bbox = {"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0}
    axes[0, 0].text(0.98, 0.84, r"non-stationary, $M_{\mathrm{true}}=3$", transform=axes[0, 0].transAxes, ha="right", va="top", color=BLUE, fontsize=6.6, bbox=inline_bbox)
    axes[0, 0].text(0.98, 0.28, "stationary control", transform=axes[0, 0].transAxes, ha="right", va="top", color=GREY, fontsize=6.6, bbox=inline_bbox)
    axes[0, 2].text(0.98, 0.97, "lower is better", transform=axes[0, 2].transAxes, ha="right", va="top", fontsize=6.7, color=GREY, bbox=inline_bbox)

    real_styles = {
        "ds004505_task_120ch": (BLUE, "o", "-", "Table tennis — 120 ch"),
        "ds004505_task_19ch": (LIGHT_BLUE, "s", "-", "Table tennis — 19 ch"),
        "ds004504_rest_19ch": ("#4C956C", "^", "-", "Eyes-closed rest — 19 ch"),
        "ds004621_rest_127ch": ("#7A5195", "D", "-", "Eyes-open rest — 127 ch"),
    }
    phase_styles = {
        "ds004505_phase_surrogate": ("#777D83", "--", "Table tennis phase surrogate"),
        "ds004504_phase_surrogate": ("#A4A9AE", "-.", "Eyes-closed rest phase surrogate"),
    }
    legend_handles: list[Line2D] = []
    for j, (metric, ylabel, title) in enumerate(
        [
            ("delta_ll_vs_m1", r"$\Delta LL$ relative to $M=1$" "\n" r"(nats component$^{-1}$ sample$^{-1}$)", "Likelihood gain"),
            ("neff", r"Effective model count, $N_{\mathrm{eff}}$", "Effective model count"),
        ]
    ):
        ax = axes[1, j]
        panel_title(ax, chr(ord("D") + j), title, y=1.04)
        for cohort_id, (color, marker, linestyle, label) in real_styles.items():
            cohort = valid[(valid["cohort_id"] == cohort_id) & (valid["source_type"] == "real_eeg")]
            summary = _summary_by_model(cohort, metric)
            ax.fill_between(summary.M, summary.q1, summary.q3, color=color, alpha=0.10, linewidth=0)
            ax.plot(summary.M, summary["median"], color=color, marker=marker, linestyle=linestyle, linewidth=1.75, markersize=3.9, markeredgecolor=INK, markeredgewidth=0.35)
            if j == 0:
                legend_handles.append(Line2D([0], [0], color=color, marker=marker, linewidth=1.75, label=label, markeredgecolor=INK, markeredgewidth=0.35))
        for cohort_id, (color, linestyle, label) in phase_styles.items():
            cohort = valid[(valid["cohort_id"] == cohort_id) & (valid["source_type"] == "phase_surrogate")]
            summary = _summary_by_model(cohort, metric)
            ax.fill_between(summary.M, summary.q1, summary.q3, color=color, alpha=0.12, linewidth=0)
            ax.plot(summary.M, summary["median"], color=color, linestyle=linestyle, linewidth=1.35)
            if j == 0:
                legend_handles.append(Line2D([0], [0], color=color, linestyle=linestyle, linewidth=1.35, label=label))
        stationary_metric = "delta_ll" if metric == "delta_ll_vs_m1" else "n_eff"
        ax.plot(stationary.M, stationary[stationary_metric], color="#42474C", linestyle=":", linewidth=1.35)
        ax.set_xticks([1, 3, 5, 7, 10])
        ax.set_xlabel(r"Number of fitted models, $M$")
        ax.set_ylabel(ylabel)
        finish_axes(ax, "both")
        if metric == "delta_ll_vs_m1":
            ax.axhline(0, color="#BFC3C7", linewidth=0.7, zorder=0)
        else:
            ax.plot([1, 10], [1, 10], color="#D4D7DA", linestyle=":", linewidth=0.9, zorder=0)
            ax.set_ylim(0.75, 10.35)
            ax.text(0.04, 0.95, "phase surrogates also increase", transform=ax.transAxes, ha="left", va="top", fontsize=6.7, color=GREY, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 0.8})
    legend_handles.append(Line2D([0], [0], color="#42474C", linestyle=":", linewidth=1.35, label="synthetic stationary"))

    # The posterior-dynamics panel now lives in the supplement (it is an
    # illustrative single-subject view from a prespecified-exploratory
    # analysis). Drawing it is shared with make_supplementary_figure4 so the
    # two can never diverge.
    if axes.shape[1] > 2 and axes[1, 2] is not None:
        _draw_posterior_dynamics(axes[1, 2], letter="F", label_fontsize=6.4)

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.53, 0.018),
        ncol=4,
        frameon=False,
        fontsize=6.4,
        handlelength=2.0,
        columnspacing=0.9,
        labelspacing=0.55,
    )

    target_dir = FIG_DIR if output_dir is None else output_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(target_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(target_dir / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    integrity["selected_layout"] = layout
    integrity["output_files"] = {
        "pdf": str((target_dir / f"{stem}.pdf").resolve()),
        "png": str((target_dir / f"{stem}.png").resolve()),
        "png_dpi": 600,
    }
    # Record the panels this figure actually contains, so the stats artifact
    # describes the rendering the manuscript prints rather than a superseded one.
    integrity["main_figure_panels"] = [
        "synthetic likelihood gain",
        "synthetic source recovery",
        "synthetic ground-truth map recovery",
        "real and stationary-reference likelihood gain",
        "real and surrogate effective model count",
    ] + ([] if layout == "five_panel" else ["illustrative posterior model dynamics"])
    if layout == "five_panel":
        integrity["posterior_dynamics_panel"] = (
            "moved to the supplement (figS4_posterior_dynamics); it is an "
            "illustrative single-subject view from a prespecified-exploratory analysis"
        )
    integrity["audit_csv"] = str(MULTIMODEL_AUDIT_CSV.resolve())
    integrity["audit_json"] = str(MULTIMODEL_AUDIT_JSON.resolve())
    if write_audit:
        MULTIMODEL_AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(MULTIMODEL_AUDIT_CSV, index=False)
        MULTIMODEL_AUDIT_JSON.write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    # Return the complete integrity record, not a summary: main() stores this as
    # stats["figure5"], and a thin dict is why the published figure's provenance
    # was previously missing from main_figure_stats.json.
    return integrity


def _make_figure5_four_panel_superseded() -> dict:
    """Four-panel multi-model rendering with two-regime alignment. NOT PUBLISHED.

    Writes ``fig5_multimodel_stationarity.*``, which ``zenodo.tex`` does not
    include. Kept for before/after comparison only. Do not wire this into
    ``main()``: it overwrites the shared multi-model audit JSON with a panel
    list describing a figure the manuscript does not print.
    """
    set_style()
    audit, integrity, stationary, nonstationary = _load_multimodel_audit()
    valid = audit[audit["fit_success"]].copy()
    alignment = pd.read_csv(FIG5_TWO_REGIME)

    fig = plt.figure(figsize=(7.2, 5.35))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.09,
        right=0.985,
        bottom=0.19,
        top=0.86,
        wspace=0.50,
        hspace=0.62,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    fig.suptitle("Multi-model regime recovery and stationarity signatures", fontsize=10.4, fontweight="bold", y=0.985)
    fig.text(0.09, 0.91, "SYNTHETIC CONTROLS: KNOWN GROUND TRUTH", fontsize=7.4, color=GREY, fontweight="bold")
    fig.text(0.55, 0.49, "REAL EEG: EXPLORATORY IN-SAMPLE SIGNATURES", fontsize=7.4, color=GREY, fontweight="bold")

    ax_a.axis("off")
    panel_title(ax_a, "A", "Regime-posterior alignment", y=1.05)
    matrices = []
    for metric in ["matched_source_correlation", "mean_posterior"]:
        matrices.append(
            alignment.pivot(index="learned_model", columns="true_regime", values=metric)
            .sort_index()
            .sort_index(axis=1)
            .to_numpy(float)
        )
    for index, (matrix, title) in enumerate(zip(matrices, ["matched source |r|", r"mean $p(m\mid t)$"])):
        inset = ax_a.inset_axes([0.03 + 0.50 * index, 0.08, 0.43, 0.75])
        inset.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="equal")
        for i in range(2):
            for j in range(2):
                inset.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=7.0, color="white" if matrix[i, j] > 0.65 else INK, fontweight="bold" if i == j else "normal")
        inset.add_patch(Rectangle((-0.48, -0.48), 0.96, 0.96, fill=False, edgecolor=ORANGE, linewidth=1.0))
        inset.add_patch(Rectangle((0.52, 0.52), 0.96, 0.96, fill=False, edgecolor=ORANGE, linewidth=1.0))
        inset.set_xticks([0, 1], ["regime 1", "regime 2"], fontsize=6.2)
        inset.set_yticks([0, 1], ["model 1", "model 2"] if index == 0 else ["", ""], fontsize=6.2)
        inset.set_title(title, fontsize=7.0, fontweight="bold", pad=4)
        inset.tick_params(length=0)
        for spine in inset.spines.values():
            spine.set_visible(False)

    panel_title(ax_b, "B", "Likelihood gain versus model order", y=1.05)
    ax_b.plot(nonstationary.M, nonstationary.delta_ll, "o-", color=BLUE, markeredgecolor=INK, markeredgewidth=0.4, linewidth=1.9, label=r"three regimes, $M_{\mathrm{true}}=3$")
    ax_b.plot(stationary.M, stationary.delta_ll, "s--", color=GREY, markeredgecolor=INK, markeredgewidth=0.35, linewidth=1.35, label="stationary control")
    ax_b.axvline(3, color="#666B70", linestyle=":", linewidth=0.9)
    ax_b.set_xticks([1, 3, 5, 7, 10])
    ax_b.set_xlabel(r"Number of fitted models, $M$")
    ax_b.set_ylabel(r"$\Delta LL$ relative to $M=1$" "\n" r"(nats component$^{-1}$ sample$^{-1}$)")
    ax_b.text(
        0.97,
        0.76,
        r"three regimes, $M_{\mathrm{true}}=3$",
        transform=ax_b.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        color=BLUE,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.45},
    )
    ax_b.text(
        0.97,
        0.12,
        "stationary control",
        transform=ax_b.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color=GREY,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.45},
    )
    finish_axes(ax_b, "both")

    ax_c.axis("off")
    panel_title(ax_c, "C", "Synthetic source and map recovery", y=1.05)
    sir_ax = ax_c.inset_axes([0.02, 0.57, 0.96, 0.33])
    sir_ax.plot(nonstationary.M, nonstationary.sir_db, "o-", color=BLUE, linewidth=1.7, markeredgecolor=INK, markeredgewidth=0.4)
    sir_ax.axvline(3, color="#666B70", linestyle=":", linewidth=0.85)
    sir_ax.set_ylabel("SIR (dB)", color=BLUE, fontsize=7.0)
    sir_ax.tick_params(axis="y", labelcolor=BLUE, labelsize=6.2)
    sir_ax.tick_params(axis="x", labelbottom=False)
    finish_axes(sir_ax, "both")
    map_ax = ax_c.inset_axes([0.02, 0.09, 0.96, 0.33])
    map_ax.plot(nonstationary.M, nonstationary.model_error, "s--", color=ORANGE, linewidth=1.45, markeredgecolor=INK, markeredgewidth=0.35)
    map_ax.axvline(3, color="#666B70", linestyle=":", linewidth=0.85)
    map_ax.set_ylabel(r"Map error, $1-|r|$", color=ORANGE, fontsize=7.0)
    map_ax.tick_params(axis="y", labelcolor=ORANGE, labelsize=6.2)
    map_ax.set_xticks([1, 3, 5, 7, 10])
    map_ax.set_xlabel(r"Number of fitted models, $M$", fontsize=7.2)
    finish_axes(map_ax, "both")

    panel_title(ax_d, "D", "Real EEG versus stationary references", y=1.05)
    real_styles = {
        "ds004505_task_120ch": (BLUE, "o", "-", "Table tennis, 120 ch"),
        "ds004505_task_19ch": (LIGHT_BLUE, "s", "-", "Table tennis, 19 ch"),
        "ds004504_rest_19ch": (GREEN, "^", "-", "Eyes-closed rest, 19 ch"),
        "ds004621_rest_127ch": ("#7A5195", "D", "-", "Eyes-open rest, 127 ch"),
    }
    phase_styles = {
        "ds004505_phase_surrogate": ("#777D83", "--", "Table tennis phase surrogate"),
        "ds004504_phase_surrogate": ("#A4A9AE", "-.", "Eyes-closed rest phase surrogate"),
    }
    legend_handles: list[Line2D] = []
    for cohort_id, (color, marker, linestyle, label) in real_styles.items():
        cohort = valid[(valid.cohort_id == cohort_id) & (valid.source_type == "real_eeg")]
        summary = _summary_by_model(cohort, "delta_ll_vs_m1")
        ax_d.fill_between(summary.M, summary.q1, summary.q3, color=color, alpha=0.10, linewidth=0)
        ax_d.plot(summary.M, summary["median"], color=color, marker=marker, linestyle=linestyle, linewidth=1.65, markersize=3.6, markeredgecolor=INK, markeredgewidth=0.3)
        legend_handles.append(Line2D([0], [0], color=color, marker=marker, linewidth=1.6, label=label, markersize=3.5))
    for cohort_id, (color, linestyle, label) in phase_styles.items():
        cohort = valid[(valid.cohort_id == cohort_id) & (valid.source_type == "phase_surrogate")]
        summary = _summary_by_model(cohort, "delta_ll_vs_m1")
        ax_d.fill_between(summary.M, summary.q1, summary.q3, color=color, alpha=0.10, linewidth=0)
        ax_d.plot(summary.M, summary["median"], color=color, linestyle=linestyle, linewidth=1.25)
        legend_handles.append(Line2D([0], [0], color=color, linestyle=linestyle, linewidth=1.25, label=label))
    ax_d.plot(stationary.M, stationary.delta_ll, color="#42474C", linestyle=":", linewidth=1.25)
    legend_handles.append(Line2D([0], [0], color="#42474C", linestyle=":", linewidth=1.25, label="synthetic stationary"))
    ax_d.axhline(0, color="#BFC3C7", linewidth=0.7)
    ax_d.set_xticks([1, 3, 5, 7, 10])
    ax_d.set_xlabel(r"Number of fitted models, $M$")
    ax_d.set_ylabel(r"In-sample $\Delta LL$ vs $M=1$" "\n" r"(nats component$^{-1}$ sample$^{-1}$)")
    finish_axes(ax_d, "both")

    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.52, 0.025), ncol=4, frameon=False, fontsize=6.2, handlelength=1.8, columnspacing=0.8)
    save_figure(fig, "fig5_multimodel_stationarity", png_dpi=600)
    integrity["two_regime_alignment"] = {
        "source": str(FIG5_TWO_REGIME.resolve()),
        "scope": "regime-averaged legacy numerical audit; full posterior time course was not archived",
    }
    integrity["main_figure_panels"] = [
        "two-regime source/posterior alignment",
        "synthetic likelihood gain",
        "synthetic source and map recovery",
        "real and stationary-reference likelihood gain",
    ]
    MULTIMODEL_AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(MULTIMODEL_AUDIT_CSV, index=False)
    MULTIMODEL_AUDIT_JSON.write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    return integrity


def make_supplementary_figure1() -> dict:
    """Dipolarity threshold sensitivity across all three real cohorts."""
    set_style()
    thresholds = np.linspace(0, 100, 201)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.22, top=0.79, wspace=0.24)
    fig.suptitle("Dipolarity threshold sensitivity", fontsize=10.2, fontweight="bold", y=0.97)
    report: dict[str, dict] = {}
    for index, (dataset, ax) in enumerate(zip(DATASET_ORDER, axes)):
        path = COMPONENT_METRICS[dataset]
        data = pd.read_csv(path)
        report[dataset] = {
            "source": _file_record(path),
            "n_subjects": int(data.subject.nunique()),
            "methods": sorted(data.method.unique().tolist()),
        }
        panel_title(ax, chr(ord("A") + index), DATASET_DISPLAY[dataset], y=1.08)
        for method in [AMICA_BENCHMARK_METHOD, "Picard", "Infomax", "FastICA"]:
            subset = data[data.method == method]
            subjects = sorted(subset.subject.unique())
            subject_curves = []
            for subject in subjects:
                rv = subset.loc[subset.subject == subject, "dipole_residual_variance_percent"].dropna().to_numpy(float)
                subject_curves.append(np.array([(rv <= threshold).mean() * 100.0 for threshold in thresholds]))
            curves = np.vstack(subject_curves)
            mean = curves.mean(axis=0)
            rng = np.random.default_rng(5100 + index * 10 + [AMICA_BENCHMARK_METHOD, "Picard", "Infomax", "FastICA"].index(method))
            samples = curves[rng.integers(0, len(curves), size=(2000, len(curves)))].mean(axis=1)
            lo, hi = np.percentile(samples, [2.5, 97.5], axis=0)
            label = "amica" if method == AMICA_BENCHMARK_METHOD else COMPARATOR_LABELS.get(method, method)
            color = BLUE if method == AMICA_BENCHMARK_METHOD else COMPARATOR_COLORS[method]
            linestyle = "-" if method != "Infomax" else "--"
            ax.fill_between(thresholds, lo, hi, color=color, alpha=0.07, linewidth=0)
            ax.plot(thresholds, mean, color=color, linestyle=linestyle, linewidth=1.45, label=label)
        ax.axvline(5, color="#BFC3C7", linestyle=":", linewidth=0.8)
        ax.axvline(10, color="#BFC3C7", linestyle=":", linewidth=0.8)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Residual-variance threshold (%)")
        if index == 0:
            ax.set_ylabel("Components below threshold (%)")
        ax.text(0.97, 0.05, rf"$n={len(subjects)}$", transform=ax.transAxes, ha="right", fontsize=6.5, color=GREY)
        finish_axes(ax, "both")
    axes[0].legend(frameon=False, fontsize=6.5, loc="upper left")
    save_figure(fig, "figS1_dipolarity_sensitivity", png_dpi=600)
    return report


def make_supplementary_figure2() -> dict:
    """Detailed controlled runtime and memory scaling fixtures."""
    set_style()
    perphase = pd.read_csv(PERPHASE_RUNTIME_CSV)
    scaling = load_scaling_rows()
    gpu_rows = []
    for path in sorted(GPU_SCALING_ROOT.glob("Tsec-*/*.json")):
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        gpu_rows.append(
            {
                "n_samples": int(record["_data"]["n_samples"]),
                "steady_iter_s": float(record["amica"]["steady_iter_s"]),
                "peak_vram_gib": float(record["amica"]["peak_vram_gb"]),
                "source": str(path.relative_to(WORKSPACE)).replace("\\", "/"),
            }
        )
    gpu_scaling = pd.DataFrame(gpu_rows).sort_values("n_samples")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.95))
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.20, top=0.78, wspace=0.42)
    fig.suptitle("Detailed computational scaling", fontsize=10.3, fontweight="bold", y=0.985)

    ax = axes[0]
    panel_title(ax, "A", "Per-iteration cost by phase", y=1.06)
    x = np.arange(len(perphase))
    width = 0.34
    ax.bar(x - width / 2, perphase.natgrad_iter_s, width, color=LIGHT_BLUE, edgecolor=INK, linewidth=0.45, label="natural gradient")
    ax.bar(x + width / 2, perphase.newton_iter_s, width, color=BLUE, edgecolor=INK, linewidth=0.45, label="Newton")
    ax.set_yscale("log")
    ax.set_xticks(x, ["JAX-GPU", "JAX-CPU", "NumPy-CPU"])
    ax.set_ylabel("Seconds per iteration")
    ax.legend(frameon=False, fontsize=6.6)
    finish_axes(ax, "y")

    full = scaling[scaling["mode"] == "full"].sort_values("n_samples")
    chunked = scaling[scaling["mode"] == "chunked"].sort_values("chunk")
    # Host-memory-versus-sample-count and versus-block-size panels were removed:
    # Figure 4D already plots both from the same fixtures, and carrying two
    # uncited displays of the same result reads as padding.

    ax = axes[1]
    panel_title(ax, "B", "GPU sample-count sweep", y=1.06)
    line_time = ax.plot(gpu_scaling.n_samples / 1000, gpu_scaling.steady_iter_s * 1000, "o-", color=BLUE, linewidth=1.8, markeredgecolor=INK, markeredgewidth=0.45, label="steady iteration")[0]
    ax.set_xlabel(r"Samples, $T$ ($\times10^3$)")
    ax.set_ylabel("Steady iteration time (ms)", color=BLUE)
    ax.tick_params(axis="y", labelcolor=BLUE)
    finish_axes(ax, "both")
    twin = ax.twinx()
    line_mem = twin.plot(gpu_scaling.n_samples / 1000, gpu_scaling.peak_vram_gib, "s--", color=ORANGE, linewidth=1.45, markeredgecolor=INK, markeredgewidth=0.4, label="peak VRAM")[0]
    twin.set_ylabel("Peak allocator VRAM (GiB)", color=ORANGE)
    twin.tick_params(axis="y", labelcolor=ORANGE)
    twin.spines["top"].set_visible(False)
    ax.legend([line_time, line_mem], ["steady iteration", "peak VRAM"], frameon=False, fontsize=6.5, loc="upper left")
    ax.text(0.97, 0.06, "one H100; 64 PCs; 100 iterations", transform=ax.transAxes, ha="right", fontsize=6.2, color=GREY)

    save_figure(fig, "figS2_computational_scaling", png_dpi=600)
    return {
        "per_phase_source": _file_record(PERPHASE_RUNTIME_CSV),
        "cpu_scaling": scaling.drop(columns=["source_path"], errors="ignore").to_dict(orient="records"),
        "gpu_scaling": gpu_rows,
        "scope": "controlled single-subject engineering fixtures; descriptive, not formal complexity estimates",
    }


def main() -> None:
    required = [
        REFERENCE_EVIDENCE,
        REFERENCE_DENSITY,
        REFERENCE_BACKEND,
        FIG1_EMPIRICAL_DENSITIES,
        FIG1_EMPIRICAL_AUDIT,
        FIG5_TWO_REGIME,
        BENCH_505,
        BENCH_504,
        BENCH_621,
        ITER_TRACE,
        AMICA_CONFIG_SOURCE,
        AMICA_SOLVER_SOURCE,
        AMICA_LIKELIHOOD_SOURCE,
        AMICA_GPU_SUBMISSION,
        MEMORY_CSV,
        MEMORY_JSON_ROOT,
        MEMORY_MULTISUBJECT_ROOT,
        *RUNTIME_GPU_ROOTS.values(),
        RUNTIME_RUNNER,
        MNE_INTEGRATION,
        MEMORY_MEASUREMENT,
        MEMORY_AMICA_RUNNER,
        MEMORY_SCOTT_RUNNER,
        MEMORY_PYAMICA_RUNNER,
        SCALING_SUBMISSION,
        SCALING_RUNNER,
        SYNTHETIC_JSON,
        MULTIMODEL_RUNNER,
        MULTIMODEL_SUBMISSION,
        MULTIMODEL_SYNTHETIC_RUNNER,
        MULTIMODEL_DEMO_SUBMISSION,
        SEED_ROBUSTNESS_CSV,
        PERPHASE_RUNTIME_CSV,
        *COMPONENT_METRICS.values(),
        *SINGLE_MODEL_SYNTHETIC_ROOTS.values(),
        *MM_ROOTS.values(),
        *SURR_ROOTS.values(),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required figure inputs:\n" + "\n".join(missing))
    make_figure1()
    stats = {"figure2": make_figure2_reference()}
    stats["figure3"] = make_figure3()
    stats["figure4"] = make_figure4()
    stats["figure5"] = make_figure5_multimodel_published()
    stats["supplementary_figure1"] = make_supplementary_figure1()
    stats["supplementary_figure2"] = make_supplementary_figure2()
    stats["supplementary_figure3"] = make_figure5(
        layout="equal",
        stem="figS3_convergence",
        write_audit=True,
    )
    # Emit beside the figures, NOT over HERE/main_figure_stats.json. That file
    # is the frozen artifact the manuscript was written from, and three table
    # producers read it as input -- make_tab_correctness names it as such. A
    # regeneration here does not reproduce it key-for-key (backend_worst_row_
    # agreement is one it drops), so overwriting it silently breaks the tables
    # that depend on it. Compare the two deliberately instead.
    (FIG_DIR / "main_figure_stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print("Wrote coordinated main figures to", FIG_DIR)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
