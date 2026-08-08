"""Cross-implementation cost table: CPU and GPU, matched fixture.

Restores a comparison that existed in the manuscript source but sat inside an
\\iffalse block and therefore never rendered. It is the only evidence in this
work that speaks to how the JAX implementation compares with the other Python
AMICA implementations on the same data and the same hardware.

Two things about it need stating rather than smoothing over:

  * On a 100-iteration run the JAX implementation is the SLOWEST on GPU,
    because just-in-time compilation dominates a short fit. Its 600-iteration
    run reuses the XLA cache and finishes in less wall time than its own
    100-iteration run.
  * Consequently the usual two-point estimator (T600-T100)/500 returns a
    negative number for it. The per-iteration figure reported for the JAX
    implementation is therefore T600/600, while the PyTorch implementations use
    the two-point form. These are different estimators and the table says so.

Run from figures/src/:  python make_tab_cross_implementation.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from _paths import DATA_ROOT as _WS, out

_HERE = Path(__file__).resolve().parent
DEST = out("tab_cross_implementation.tex")

MEM_CSV = _WS / "results/mem_compare/mem_comparison_table.csv"
CPU_JSON_DIR = _WS / "results/mem_compare/cpu/ds004505_sub-01_mem"
CPU600 = _WS / "results/rt_cpu_600"
GPU100 = _WS / "results/rt_gpu_100"
GPU600 = _WS / "results/rt_gpu_600"


def per_iteration_ms(t100: float, t600: float) -> tuple[float, str]:
    """Steady-state per-iteration cost, and which estimator produced it.

    The two-point form (T600-T100)/500 cancels fixed start-up cost -- import,
    allocation, first-touch, compilation -- and is what a long fit actually
    pays per iteration. It is undefined when a 600-iteration run finishes in
    less wall time than its own 100-iteration run, which happens whenever a
    compilation cache is reused; T600/600 is reported then instead.

    Chosen from the measurements rather than by implementation name: the
    condition is a property of the run, and hard-coding which backend is
    expected to compile is how the table would keep reporting a cached estimator
    for a backend that had stopped caching, or miss one that started.
    """
    if t600 > t100:
        return (t600 - t100) / 500 * 1000, r"$^{\ddagger}$"
    return t600 / 600 * 1000, r"$^{\dagger}$"
# Numerical agreement on the same fixture. This used to be a separate table
# (tab_fixed_workload) whose every fit time was already reported here; only the
# two agreement columns were unique to it, so they are folded in and the
# duplicate table retired.
AUDIT = _HERE / "main_figure_stats.json"

DISPLAY = {
    "amica_python_jax": r"\texttt{amica} (JAX, full batch)",
    "amica_python_jax_chunked": r"\texttt{amica} (JAX, chunked)",
    "scott_huberty_torch": r"AMICA-Python (PyTorch)",
    "pyamica_torch": r"\texttt{pyamica} (PyTorch)",
    "pamica_torch": r"\texttt{pAMICA} (PyTorch)",
    "fortran_amica17": r"Fortran AMICA~1.7",
}
CPU_ORDER = ["amica_python_jax", "amica_python_jax_chunked",
             "scott_huberty_torch", "pyamica_torch", "pamica_torch",
             "fortran_amica17"]
GPU_ORDER = ["amica_python_jax_chunked", "pyamica_torch", "scott_huberty_torch",
             "pamica_torch"]


def load_gpu(d: Path) -> dict[str, dict]:
    return {p.name.split("_sub-01")[0]: json.loads(p.read_text())
            for p in d.glob("*_result.json")}


def _matched_unmixing_summary(reference, candidate) -> tuple[float, float]:
    """Median and minimum unsigned row correlation after Hungarian matching.

    Same definition as the fixed-workload audit in make_main_figures.py; kept
    identical on purpose so the two cannot report different numbers for the
    same quantity.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    reference = reference / np.linalg.norm(reference, axis=1, keepdims=True)
    candidate = candidate / np.linalg.norm(candidate, axis=1, keepdims=True)
    correlation = np.abs(reference @ candidate.T)
    row, col = linear_sum_assignment(-correlation)
    matched = correlation[row, col]
    return float(np.median(matched)), float(np.min(matched))


def load_agreement() -> dict[tuple[str, str], dict]:
    """Agreement against the amica JAX-CPU run, keyed by (implementation, device).

    Computed here from the archived per-run JSONs rather than read out of
    main_figure_stats.json. That file is the frozen artifact the manuscript was
    written from, so it cannot contain a record for an implementation added
    afterwards -- pamica would have shown a blank agreement column while its
    unmixing sat in the same directory as everybody else's. Reading the runs
    directly also keeps every cell in this table sourced from one campaign.

    Rows whose run is absent are simply missing from the mapping; callers
    tolerate a miss.
    """
    ref_path = CPU_JSON_DIR / "amica_python_jax_sub-01_seed0_result.json"
    if not ref_path.exists():
        return {}
    ref = json.loads(ref_path.read_text(encoding="utf-8"))

    out: dict[tuple[str, str], dict] = {}
    for device, d in (("cpu", CPU_JSON_DIR), ("gpu", GPU100)):
        for p in sorted(d.glob("*_result.json")):
            rec = json.loads(p.read_text(encoding="utf-8"))
            if "error" in rec or "W" not in rec:
                continue
            _, minimum_r = _matched_unmixing_summary(ref["W"], rec["W"])
            out[(rec["implementation"], device)] = {
                "abs_ll_difference_vs_amica_jax_cpu":
                    abs(float(rec["ll_final"]) - float(ref["ll_final"])),
                "minimum_matched_row_correlation_vs_amica_jax_cpu": minimum_r,
            }
    return out


def sci(value: float, digits: int = 1) -> str:
    if value == 0:
        return "$0$"
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return rf"${mantissa}{{\times}}10^{{{int(exponent)}}}$"


def agree_cells(rec: dict | None) -> str:
    if rec is None:
        return "-- & --"
    return (f"{sci(rec['abs_ll_difference_vs_amica_jax_cpu'])} & "
            f"${rec['minimum_matched_row_correlation_vs_amica_jax_cpu']:.4f}$")


def main() -> None:
    cpu = {}
    for r in csv.DictReader(MEM_CSV.open(encoding="utf-8")):
        if r["device"] == "cpu":
            cpu[r["implementation"]] = r
    g100, g600 = load_gpu(GPU100), load_gpu(GPU600)
    agree = load_agreement()

    out: list[str] = []
    add = out.append
    add(r"% Generated by figures/src/make_tab_cross_implementation.py -- do not hand-edit.")
    add(r"\begin{table}[htbp]")
    add(r"\centering")
    add(r"\caption{")
    add(r"\textbf{Cross-implementation cost and agreement on a matched fixture.}")
    add(r"One archived")
    add(r"run per implementation on the PCA-projected Table tennis sub-01 array")
    add(r"($64\times785{,}328$) at 100 iterations. CPU rows ran on eight cores of")
    add(r"a dual-socket AMD EPYC~9655 node (96 cores per socket) with thread")
    add(r"counts bound to the allocation; GPU rows on one H100. All rows of a")
    add(r"given device come from a single campaign on one machine.")
    add(r"Within each device, rows are ordered by steady-state per-iteration")
    add(r"cost, and \texttt{amica} is set in bold wherever that ordering places")
    add(r"it. Rows whose 600-iteration companion run is unavailable have no")
    add(r"per-iteration figure and are listed last.")
    add(r"The two right-hand columns compare each decomposition with the")
    add(r"\texttt{amica} JAX-CPU run, Hungarian-matched and sign-aligned.")
    add(r"\textbf{These are single runs, not repeated measurements}, and")
    add(r"the timing boundaries are not identical: the Python rows time the")
    add(r"model-fit call including first-use compilation, whereas the Fortran row")
    add(r"times the external executable including initialisation and output")
    add(r"writing but excluding input serialisation.")
    add(r"\textbf{Per-iteration cost is not the same estimator in every")
    add(r"row}: the JAX implementation's 600-iteration run reuses the XLA")
    add(r"compilation cache and finishes in less wall time than its own")
    add(r"100-iteration run, so the two-point form $(T_{600}-T_{100})/500$ is")
    add(r"undefined for it and $T_{600}/600$ is reported instead; the PyTorch")
    add(r"implementations use the two-point form. The two runs entering that")
    add(r"form were separate jobs and, on CPU, were scheduled onto different")
    add(r"nodes of the same model; for the one implementation whose wall time")
    add(r"varied appreciably between such nodes this shifts its per-iteration")
    add(r"figure by around $10\%$. On the 100-iteration run the")
    add(r"JAX implementation is slower on GPU than the two PyTorch")
    add(r"implementations it is closest to, because compilation dominates a short")
    add(r"fit; that ordering reverses once the iteration budget is large enough")
    add(r"for compilation to amortise, which is the regime AMICA fits actually")
    add(r"occupy. \texttt{pAMICA} is run with its own algorithm constants and the")
    add(r"shared protocol (iteration budget, mixture count, learning rate, Newton")
    add(r"enabled); its cost is not sensitive to its block size, which was swept")
    add(r"from $1{,}533$ to $11$ blocks per iteration for a fit-time change under")
    add(r"$7\%$. Third-party revisions were not")
    add(r"archived and these measurements predate the Anderson-accelerated")
    add(r"variant of AMICA-Python, so this is a descriptive comparison of one")
    add(r"configuration rather than a durable ranking of implementations.")
    add(r"}")
    add(r"\label{tab:cross-implementation}")
    add(r"\footnotesize")
    add(r"\setlength{\tabcolsep}{3.5pt}")
    add(r"\begin{tabular}{lcccccc}")
    add(r"\toprule")
    add(r"Implementation & \shortstack{Fit time\\(s)} & "
        r"\shortstack{Per iteration\\(ms)} & \shortstack{Peak host\\RSS (GiB)} & "
        r"\shortstack{Peak\\VRAM (GiB)} & "
        r"\shortstack{$|\Delta$ final $\ell|$\\vs JAX-CPU} & "
        r"\shortstack{Worst matched\\row $|r|$} \\")

    add(r"\midrule")
    add(r"\multicolumn{7}{l}{\emph{CPU, 100 iterations}}\\")
    add(r"\addlinespace[1pt]")
    # Implementations added after the archived campaigns (pamica) are absent from
    # older result trees. Skip a row whose measurement does not exist rather than
    # dying, so this producer still regenerates the published table from the
    # archive it was written against, and picks the new row up once its run
    # lands. What is skipped is reported at the end, never silently dropped.
    skipped: list[str] = []
    c600 = load_gpu(CPU600)  # same shape: one *_result.json per implementation

    def build_rows(order, at100, at600, device, vram_cell):
        """(sort key, label, cells) per implementation, ordered by per-iteration cost.

        Sorting by the steady-state figure rather than by a fixed list makes the
        column readable, and puts this package wherever the measurement puts it
        instead of first by construction. Rows whose 600-iteration companion is
        missing have no per-iteration figure and sort last: they cannot be
        placed, and guessing a position for them would be the one thing a sorted
        table must not do.
        """
        built = []
        for k in order:
            r = at100.get(k)
            if r is None:
                skipped.append(f"{device}100:{k}")
                continue
            t1 = float(r["fit_time_s"])
            if k in at600:
                per, mark = per_iteration_ms(t1, float(at600[k]["fit_time_s"]))
                per_cell, key = f"${per:.1f}${mark}", per
            else:
                skipped.append(f"{device}600:{k}")
                per_cell, key = "--", float("inf")
            # This package is bolded so it stays findable once the order is the
            # measurement's rather than the author's. The label only: bolding a
            # row's numbers would put a thumb on the comparison the table exists
            # to make.
            label = DISPLAY[k]
            if k.startswith("amica_python_"):
                label = rf"\textbf{{{label}}}"
            built.append((key, k, f"{label} & ${t1:.1f}$ & {per_cell} & "
                                  f"${float(r['peak_rss_gb']):.2f}$ & {vram_cell(k)} & "
                                  f"{agree_cells(agree.get((k, device)))} \\\\"))
        built.sort(key=lambda b: b[0])
        return built

    for _, _, line in build_rows(CPU_ORDER, cpu, c600, "cpu", lambda k: "--"):
        add(line)

    add(r"\midrule")
    add(r"\multicolumn{7}{l}{\emph{GPU (one H100), 100-iteration run}}\\")
    add(r"\addlinespace[1pt]")
    for _, _, line in build_rows(
            GPU_ORDER, g100, g600, "gpu",
            lambda k: f"${g100[k]['peak_vram_gb']:.2f}$"):
        add(line)

    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"")
    add(r"\vspace{2pt}")
    add(r"{\footnotesize $^{\dagger}$ $T_{600}/600$ (compilation cache reused). "
        r"$^{\ddagger}$ $(T_{600}-T_{100})/500$.}")
    add(r"\end{table}")

    DEST.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {DEST}")
    for k in GPU_ORDER:
        if k not in g100:
            continue
        t1 = g100[k]["fit_time_s"]
        if k in g600:
            per, _ = per_iteration_ms(t1, float(g600[k]["fit_time_s"]))
            per_s = f"{per:6.2f}ms"
        else:
            per_s = "     n/a"
        print(f"  {DISPLAY[k]:34} T100={t1:6.2f}s  per-iter={per_s}  "
              f"VRAM={g100[k]['peak_vram_gb']:.2f}GiB")
    if skipped:
        print("  measurements absent, rows/cells omitted: " + ", ".join(skipped))


if __name__ == "__main__":
    main()
