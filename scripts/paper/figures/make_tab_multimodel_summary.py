#!/usr/bin/env python3
"""Generate tab_multimodel_summary.tex from the multi-model benchmark archives.

Definitions were recovered by reproducing the published table rather than
assumed, because two plausible ones disagree:

    N_eff(10), Table tennis   published            8.761
               perplexity  exp(-sum g log g)       8.761   <- this one
               inverse-Simpson  1 / sum(g^2)       8.072

All of dLL(2), dLL(5), dLL(10) and N_eff(10) reproduce to the published number
of digits for the Table tennis cohort under the perplexity definition.

dLL(M) is ll_final(M) - ll_final(1) per participant, then the cohort median.
The "missing or flagged" column is derived, not transcribed: participants whose
M-series has gaps are detected by scanning the archives.

Usage:
    python make_tab_multimodel_summary.py           # print
    python make_tab_multimodel_summary.py --write   # write ../../tab_multimodel_summary.tex
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import statistics as st
from pathlib import Path

import numpy as np

from _paths import DATA_ROOT as WORKSPACE, out

HERE = Path(__file__).resolve().parent
OUT = out("tab_multimodel_summary.tex")

M_COLUMNS = (2, 5, 10)
M_EXPECTED = tuple(range(1, 11))

COHORTS = [
    {"label": "Table tennis, 120 channels", "dir": "figdata/mmbench_ds004505",
     "group": "real"},
    {"label": "Table tennis, 19 channels", "dir": "figdata/mmbench_ds004505_ch19",
     "group": "real"},
    {"label": "Eyes-closed rest, 19 channels", "dir": "figdata/mmbench_ds004504",
     "group": "real"},
    {"label": "Eyes-open rest, 127 channels", "dir": "figdata/mmbench_ds004621",
     "group": "real"},
    {"label": "Table tennis phase surrogate", "dir": "figdata/mmbench_ds004505_surr",
     "group": "surrogate"},
    {"label": "Eyes-closed rest phase surrogate", "dir": "figdata/mmbench_ds004504_surr",
     "group": "surrogate"},
]


def series(d: Path) -> dict[str, dict[int, str]]:
    """subject -> {M: path} from mmbench_*_sub-XX_N##_M#.npz"""
    out: dict[str, dict[int, str]] = {}
    for f in glob.glob(str(d / "*.npz")):
        # Variant cohorts carry a suffix after the M index: the 19-channel
        # subset uses _tentwenty and the phase surrogates use _surrphase.
        m = re.search(r"sub-(\w+)_N(\d+)_M(\d+)(?:_\w+)?\.npz$", os.path.basename(f))
        if m:
            out.setdefault(m.group(1), {})[int(m.group(3))] = f
    return out


def n_eff(path: str) -> float:
    """Effective model count as the perplexity of the model weights."""
    g = np.asarray(np.load(path, allow_pickle=True)["gm"], dtype=float).ravel()
    g = g / g.sum()
    return float(np.exp(-(g * np.log(g + 1e-300)).sum()))


def ll(path: str) -> float:
    return float(np.load(path, allow_pickle=True)["ll_final"])


# A fit whose model weights collapse onto a single model has N_eff == 1 exactly.
# Verified against the published flag: eyes-closed sub-38 gives N_eff 1.000 at
# M=8 and M=10 (flagged) but 1.200 at M=7 and 2.550 at M=9 (not flagged).
COLLAPSE_NEFF = 1.01


def math_m(values) -> str:
    r"""Render an M list as inline math, e.g. \(M=8,10\).

    Built by concatenation rather than in an f-string: the manuscript uses
    \(...\) delimiters, and those braces and backslashes are painful to escape
    inside a format string.
    """
    return "\\(M=" + ",".join(str(v) for v in values) + "\\)"


def summarise(d: Path) -> dict | None:
    subs = series(d)
    if not subs:
        return None
    dll: dict[int, list[float]] = {m: [] for m in M_COLUMNS}
    neffs: list[float] = []
    gaps: list[str] = []
    for s, by_m in sorted(subs.items()):
        missing = [m for m in M_EXPECTED if m not in by_m]
        if missing and missing != list(M_EXPECTED):
            gaps.append("sub-%s absent at %s" % (s, math_m(missing)))
        collapsed = [m for m in sorted(by_m) if n_eff(by_m[m]) < COLLAPSE_NEFF and m > 1]
        if collapsed:
            gaps.append("sub-%s weight collapse at %s" % (s, math_m(collapsed)))
        if 1 not in by_m:
            continue
        base = ll(by_m[1])
        for m in M_COLUMNS:
            if m in by_m:
                dll[m].append(ll(by_m[m]) - base)
        if 10 in by_m:
            neffs.append(n_eff(by_m[10]))
    return {
        "n": len(subs),
        "dll": {m: (st.median(v) if v else float("nan")) for m, v in dll.items()},
        "neff": st.median(neffs) if neffs else float("nan"),
        "gaps": gaps,
    }


def synthetic_row(ws: Path) -> dict | None:
    """The stationary control: one archived seed, a different archive shape."""
    import json
    f = ws / "figdata/multimodel_synthetic_2000/synthetic_summary.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    dll = {m: d["delta_ll_stationary"][str(m)]["delta"] for m in M_COLUMNS
           if str(m) in d.get("delta_ll_stationary", {})}
    ten = [e for e in d.get("stationary", []) if e.get("H") == 10]
    return {"n": "1 seed", "dll": dll,
            "neff": ten[0]["n_eff"] if ten else float("nan"), "gaps": []}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--workspace", default=str(WORKSPACE))
    ap.add_argument("--allow-missing", action="store_true",
                    help="emit the table even if archive inputs are absent; "
                         "each missing input silently drops a row")
    args = ap.parse_args()
    ws = Path(args.workspace)

    rows, missing_dirs = {"real": [], "surrogate": []}, []

    def emit(label, s, group, dp):
        flag = "; ".join(s["gaps"]) if s["gaps"] else "None"
        rows[group].append(
            f"{label}\n& {s['n']}\n"
            + "".join(f"& {s['dll'][m]:.{dp}f}\n" for m in M_COLUMNS)
            + f"& {s['neff']:.3f}\n& {flag} \\\\"
        )
        print(f"  {label:34s} n={str(s['n']):>6s}  "
              + "  ".join(f"dLL({m})={s['dll'][m]:.{dp}f}" for m in M_COLUMNS)
              + f"  Neff={s['neff']:.3f}  flags={flag}")

    for c in COHORTS:
        s = summarise(ws / c["dir"])
        if s is None:
            missing_dirs.append(c["dir"])
            continue
        # Published precision differs by magnitude: real cohorts print 5 dp,
        # surrogate and synthetic rows print 6 (0.000019 would render 0.00002).
        emit(c["label"], s, c["group"], 5 if c["group"] == "real" else 6)

    syn = synthetic_row(ws)
    if syn:
        emit("Synthetic stationary control", syn, "surrogate", 6)
    else:
        missing_dirs.append("figdata/multimodel_synthetic_2000/synthetic_summary.json")

    if missing_dirs:
        # Omitting a row and carrying on produces a table that is not short --
        # it is wrong, and it exits 0. The multi-model .npz fits are ~485 MB and
        # deliberately not in git, so a bare clone hits this by default. Say so
        # and stop, unless the caller has explicitly accepted a partial table.
        detail = "\n".join(f"  ! {d}" for d in missing_dirs)
        if not args.allow_missing:
            raise SystemExit(
                f"refusing to write {OUT.name}: {len(missing_dirs)} archive "
                f"input(s) missing, and each one silently drops a row.\n"
                f"{detail}\n"
                "Point AMICA_BENCH_DATA at a tree that has them, or pass "
                "--allow-missing to accept a partial table."
            )
        print("\nMISSING archive directories (row omitted):")
        print(detail)

    body = "\n\n".join(rows["real"]) + "\n\n\\midrule\n\n" + "\n\n".join(rows["surrogate"])
    tex = f"""% GENERATED by figures/src/make_tab_multimodel_summary.py -- do not edit by hand.
\\begin{{table}}[htbp]
\\centering
\\caption{{
\\textbf{{Multi-model numerical summary.}}
Cohort medians of in-sample likelihood gain relative to \\(M=1\\) and effective
model count at \\(M=10\\).
}}
\\label{{tab:multimodel-summary}}

\\scriptsize
\\setlength{{\\tabcolsep}}{{3.5pt}}
\\renewcommand{{\\arraystretch}}{{1.08}}

\\begin{{tabular}}{{
    >{{\\raggedright\\arraybackslash}}p{{4.0cm}}
    c
    ccc
    c
    >{{\\raggedright\\arraybackslash}}p{{3.0cm}}
}}
\\toprule
Cohort / reference &
\\(n\\) &
\\(\\Delta LL(2)\\) &
\\(\\Delta LL(5)\\) &
\\(\\Delta LL(10)\\) &
\\(N_{{\\mathrm{{eff}}}}(10)\\) &
Missing or flagged fits \\\\
\\midrule

{body}

\\bottomrule
\\end{{tabular}}

\\vspace{{3pt}}
\\begin{{minipage}}{{\\linewidth}}
\\scriptsize
\\textit{{Notes.}}
\\(\\Delta LL(M)\\) is the in-sample likelihood gain relative to \\(M=1\\), in
nats per retained component per sample. \\(N_{{\\mathrm{{eff}}}}\\) is the
perplexity of the estimated model weights, \\(\\exp(-\\sum_m g_m \\log g_m)\\).
Real and phase-surrogate fits used 16 retained PCs and a 2,000-iteration
maximum. One phase-randomized surrogate was generated per listed
participant; the synthetic stationary control used one archived seed.
\\(N_{{\\mathrm{{eff}}}}\\) is not a specific stationarity index because it also
increased in the surrogate cohorts.
\\end{{minipage}}
\\end{{table}}
"""
    if args.write:
        OUT.write_text(tex, encoding="utf-8")
        print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
