"""Rebuild results/mem_compare/mem_comparison_table.csv from the archived run JSONs.

The CSV is what make_tab_cross_implementation reads for its CPU rows, and it had
no generator in the repository -- it was written once and then hand-carried,
which is how a table and the records behind it drift apart. This regenerates it
from the per-implementation result JSONs so the two cannot disagree.

    python benchmark/comparator/build_mem_table.py            # report only
    python benchmark/comparator/build_mem_table.py --write
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CPU_DIR = REPO / "results/mem_compare/cpu/ds004505_sub-01_mem"
GPU_DIR = REPO / "results/rt_gpu_100"
DEST = REPO / "results/mem_compare/mem_comparison_table.csv"

COLUMNS = ["implementation", "device", "peak_rss_gb", "delta_rss_gb",
           "peak_vram_gb", "fit_time_s", "n_iter"]

# Row order as the manuscript table presents them: amica first, then the other
# implementations, then the Fortran reference.
ORDER = ["amica_python_jax", "amica_python_jax_chunked", "scott_huberty_torch",
         "pamica_torch", "pyamica_torch", "fortran_amica17"]


def load(d: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(d.glob("*_result.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if "error" in rec:
            print(f"  skipping {p.name}: {rec['error']}")
            continue
        out[rec["implementation"]] = rec
    return out


def fmt(v, digits: int) -> str:
    return "" if v is None else f"{float(v):.{digits}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    cpu, gpu = load(CPU_DIR), load(GPU_DIR)
    rows = []
    for key in ORDER:
        if key in cpu:
            r = cpu[key]
            rows.append({
                "implementation": key, "device": "cpu",
                "peak_rss_gb": fmt(r.get("peak_rss_gb"), 2),
                "delta_rss_gb": fmt(r.get("delta_rss_gb"), 2),
                "peak_vram_gb": "",
                "fit_time_s": fmt(r.get("fit_time_s"), 1),
                "n_iter": r.get("n_iter", ""),
            })
    for key in ORDER:
        if key in gpu:
            r = gpu[key]
            rows.append({
                "implementation": key, "device": "gpu",
                "peak_rss_gb": fmt(r.get("peak_rss_gb"), 2),
                "delta_rss_gb": "",
                "peak_vram_gb": fmt(r.get("peak_vram_gb"), 3),
                "fit_time_s": fmt(r.get("fit_time_s"), 1),
                "n_iter": r.get("n_iter", ""),
            })

    missing = [k for k in ORDER if k not in cpu]
    if missing:
        print(f"  CPU rows absent: {', '.join(missing)}")

    for r in rows:
        print("  " + "  ".join(f"{r[c]}" for c in COLUMNS))

    if args.write:
        with DEST.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {DEST}  ({len(rows)} rows)")
    else:
        print("(dry run; pass --write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
