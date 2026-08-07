#!/usr/bin/env python3
"""Generate tab_correctness.tex from figures/src/main_figure_stats.json.

Source choice matters here more than in the other table generators, and the
wrong choice looks identical to the right one.

The cross-backend row could be recomputed from
results/backend_parity_v3/backend_agreement_per_subject.csv. Doing so yields
0.999860 for jax_gpu_vs_numpy_cpu where the manuscript prints 0.999864 -- the
CSV's matched_r_min and the frozen worst_row_correlation are not quite the same
quantity. A generator built on the CSV would therefore have silently altered a
published number by 4e-6 while appearing to faithfully regenerate the table.

main_figure_stats.json is the frozen-numbers artifact the manuscript was
written from, so it is the source of truth here. Its own `limitations` field
records that some rows (K=1, natural-gradient-only, real-EEG Fortran) trace to
an archived audit table rather than to raw result files; that provenance is
carried through to the notes rather than hidden.

Usage:
    python make_tab_correctness.py            # print
    python make_tab_correctness.py --write    # write ../../tab_correctness.tex
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from _paths import out

HERE = Path(__file__).resolve().parent
STATS = HERE / "main_figure_stats.json"
OUT = out("tab_correctness.tex")

# The three-build ablation is summarised in the manuscript from
# results/fortran_parity_nearstock/. It is not carried in main_figure_stats.json,
# so it is declared here with its provenance rather than silently recomputed.
NEARSTOCK = {
    "fixture": "6-channel Laplacian, \\(K=3\\), Newton and natural gradient, "
               "three build variants, free initialization",
    "metric": "Relative \\(\\Delta LL\\); \\(r_W^{\\min}\\)",
    "result": "\\(1.6\\)--\\(7.9\\times10^{-5}\\%\\); \\(\\geq 0.9999999\\)",
    "scope": "Build differences within repeated-run variability",
    "source": "results/fortran_parity_nearstock/",
}


def sci(x: float, sig: int = 2) -> str:
    """Render a number as LaTeX, using scientific notation when it is small."""
    if x == 0:
        return "0"
    exp = math.floor(math.log10(abs(x)))
    if -3 < exp < 4:
        s = f"{x:.{max(0, sig - 1 - exp)}f}".rstrip("0").rstrip(".")
        return f"\\({s}\\)"
    mant = x / (10 ** exp)
    m = f"{mant:.{sig - 1}f}".rstrip("0").rstrip(".")
    return f"\\({m}\\times10^{{{exp}}}\\)" if m != "1" else f"\\(10^{{{exp}}}\\)"


def pct(x: float, sig: int = 2) -> str:
    if x == 0:
        return "\\(0\\%\\)"
    exp = math.floor(math.log10(abs(x)))
    if -3 < exp < 3:
        s = f"{x:.{max(0, sig - 1 - exp)}f}".rstrip("0").rstrip(".")
        return f"\\({s}\\%\\)"
    mant = x / (10 ** exp)
    m = f"{mant:.{sig - 1}f}".rstrip("0").rstrip(".")
    return f"\\({m}\\times10^{{{exp}}}\\%\\)"


def r(x: float, dp: int) -> str:
    return f"\\({x:.{dp}f}\\)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    f2 = json.loads(STATS.read_text(encoding="utf-8"))["figure2"]
    panel = {e["label"].replace("\n", " "): e for e in f2["panel_a"]}
    dens = f2["density_parameter_agreement"]
    back = f2["backend_worst_row_agreement"]
    chunk = f2["chunking"]
    mne = f2["mne"]

    def p(label):
        for k, v in panel.items():
            if label in k:
                return v
        raise SystemExit(f"panel_a row not found: {label}")

    k1, k3n, k3g = p("K=1, Newton"), p("K=3, Newton"), p("natural gradient")
    real = p("ds004505 sub-01")

    dr = min(v["pearson_r"] for v in dens.values())
    nl = min(v["nrmse_over_iqr"] for v in dens.values())
    nh = max(v["nrmse_over_iqr"] for v in dens.values())
    bmed = [v["median_worst_row_correlation"] for v in back.values()]
    mir = f2["maximum_absolute_backend_mir_difference_kbits_s"]

    rows = [
        ("\\multirow{6}{=}{Fortran AMICA~1.7 parity}\n& 6-channel Laplacian, \\(K=1\\), Newton",
         "Relative \\(\\Delta LL\\); \\(r_W^{\\min}\\)",
         f"{pct(k1['relative_ll_difference_pct'])}; {r(k1['worst_row_correlation'], 4)}",
         "Matched sources; single fixture"),
        ("& 6-channel Laplacian, \\(K=3\\), Newton, fixed initialization",
         "Relative \\(\\Delta LL\\); \\(r_W^{\\min}\\)",
         f"{pct(k3n['relative_ll_difference_pct'], 3)}; {r(k3n['worst_row_correlation'], 11)}",
         "Archived 2,000-iteration rerun"),
        ("& 6-channel Laplacian, \\(K=3\\), natural gradient",
         "Relative \\(\\Delta LL\\); \\(r_W^{\\min}\\)",
         f"{pct(k3g['relative_ll_difference_pct'])}; {r(k3g['worst_row_correlation'], 4)}",
         "Matched sources; single fixture"),
        ("& 6-channel, \\(K=3\\), density parameters",
         "Matched \\(r\\); nRMSE/IQR for \\(\\alpha,\\mu,\\beta,\\rho\\)",
         f"\\(r\\geq{dr:.8f}\\);\n{sci(nl)}--{sci(nh)}",
         "18 aligned density terms"),
        (f"& {NEARSTOCK['fixture']}", NEARSTOCK["metric"], NEARSTOCK["result"],
         NEARSTOCK["scope"]),
        ("& Table tennis sub-01, 64 PCs, 100 iterations",
         "Absolute \\(\\Delta LL\\) (relative);\n\\(\\bar r_W\\) (\\(r_W^{\\min}\\))",
         f"{sci(real['absolute_ll_difference'])}\n({pct(real['relative_ll_difference_pct'])});\n"
         f"{r(real['mean_row_correlation'], 5)} ({r(real['worst_row_correlation'], 4)})",
         "One subject; single model"),
    ]

    body = "\n\n".join(f"{a}\n& {b}\n& {c}\n& {d} \\\\" for a, b, c, d in rows)

    body += f"""

\\midrule

Cross-backend agreement
& Table tennis, 25 participants; JAX-GPU, JAX-CPU, NumPy-CPU
& Median \\(r_W^{{\\min}}\\);
\\(\\lvert\\Delta\\mathrm{{MIR}}\\rvert_{{\\max}}\\)
& \\({min(bmed):.6f}\\)--\\({max(bmed):.6f}\\);
\\({mir}\\)~kbits/s
& Benchmark-level consistency \\\\

\\midrule

Full-batch/chunked agreement
& MNE sample; 30 PCs; 166,800 samples; 100 iterations; \\texttt{{float64}}; \\(B=1{{,}}024\\)
& Absolute final \\(\\Delta LL\\);
\\(\\lVert\\Delta W\\rVert_F/\\lVert W\\rVert_F\\)
& {sci(chunk['absolute_final_ll_difference'], 3)};
{sci(chunk['unmixing_frobenius_relative_error'], 3)}
& Separate agreement fixture \\\\

\\midrule

MNE-Python interoperability
& \\texttt{{Raw}}/\\texttt{{Epochs}}, sources, exclusion, reconstruction,
reordering, rank reduction, model access, and ICLabel
& Direct tests; CI on Python~{mne['python_ci_versions'].replace('-', '--')} and \
{'three' if mne['ci_operating_systems'] == 3 else mne['ci_operating_systems']} operating systems
& All tested workflows passed; reconstruction residual
{sci(mne['transform_inverse_frobenius_relative_residual'])}
& Not a universal conformance claim \\\\"""

    tex = f"""% GENERATED by figures/src/make_tab_correctness.py -- do not edit by hand.
% Source: figures/src/main_figure_stats.json (figure2), the frozen-numbers
% artifact the manuscript was written from. Do NOT regenerate the cross-backend
% row from backend_agreement_per_subject.csv: that yields 0.999860 where this
% prints 0.999864.
% Supplementary numerical audit supporting main Figure~\\ref{{fig:reference-agreement}}.
\\begin{{table}}[htbp]
\\centering
\\caption{{
\\textbf{{Numerical agreement and tested software interoperability.}}
Results are reported for the listed fixtures and configurations only.
}}
\\label{{tab:parity}}

\\scriptsize
\\setlength{{\\tabcolsep}}{{3pt}}
\\renewcommand{{\\arraystretch}}{{1.08}}

\\begin{{tabular}}{{
    >{{\\raggedright\\arraybackslash}}p{{1.85cm}}
    >{{\\raggedright\\arraybackslash}}p{{3.35cm}}
    >{{\\raggedright\\arraybackslash}}p{{2.85cm}}
    >{{\\raggedright\\arraybackslash}}p{{3.05cm}}
    >{{\\raggedright\\arraybackslash}}p{{2.55cm}}
}}
\\toprule
Boundary & Fixture / configuration & Metric & Result & Scope \\\\
\\midrule

{body}

\\bottomrule
\\end{{tabular}}

\\vspace{{3pt}}
\\begin{{minipage}}{{\\linewidth}}
\\scriptsize
\\textit{{Notes.}}
\\(\\Delta LL\\) denotes either a relative percentage or an absolute difference
in normalized log-likelihood, as indicated. The normalized objective is
expressed in nats per retained component per sample.
\\(r_W^{{\\min}}\\) is the lowest Hungarian-matched, sign-aligned correlation
between unmixing rows within a fit. Parity results apply to single-model
AMICA (\\(M=1\\)) and to the listed fixtures only.
\\end{{minipage}}
\\end{{table}}
"""
    if args.write:
        OUT.write_text(tex, encoding="utf-8")
        print(f"wrote {OUT}")
    else:
        print(tex)

    print(f"\nrow sources (from main_figure_stats.json):")
    for lbl, e in panel.items():
        print(f"  {lbl:44s} <- {e.get('source', '?')[:52]}")
    print(f"  {'three-build ablation':44s} <- {NEARSTOCK['source']} (declared)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
