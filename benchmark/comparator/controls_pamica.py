"""Fairness controls for the pAMICA row of the cross-implementation table.

The table imposes one experimental protocol on every implementation. Two things
about pamica could make that protocol, rather than pamica, the thing being
measured, so both were checked before its numbers were reported. Keeping the
controls next to the comparison means the claim "this was controlled" is
inspectable rather than asserted.

``--control config``
    Runs pamica under three configurations on one input: the shared protocol
    with its own tuning constants (what the runner does), the same protocol
    carrying another implementation's constants, and pamica's full library
    defaults. The middle one was the original runner's behaviour and cost most
    of pamica's apparent accuracy.

``--control blocks``
    Sweeps ``block_size``. pamica's default of 512 is 1,533 sequential blocks
    per iteration on a 785k-sample recording, while the amica row it is compared
    against runs auto-tuned chunking, so a runtime gap could have been a
    comparison artefact.

Both need a run directory that already holds the orchestrator's
``input_*.npz`` and the amica reference result, i.e. one produced by
``implementation_perf.py``. Run inside an allocation::

    source benchmark/cc_benchmark/pamica_env.sh
    srun --account=def-kjerbi_gpu --gres=gpu:h100:1 --time=1:00:00 --mem=32G \
        python benchmark/comparator/controls_pamica.py \
            --run-dir /scratch/$USER/.../comparator/gpu/ds004505_sub-01 \
            --control blocks
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "runners" / "run_pamica.py"

# The shared protocol. Identical for every implementation in the table, and the
# only thing a control is allowed to hold fixed across configurations.
PROTOCOL = {"max_iter": 100, "n_mix": 3, "lrate": 0.1, "do_newton": True, "seed": 0}

CONFIGS = {
    # What the runner does: shared protocol, pamica's own tuning constants.
    "own-constants": {},
    # What the runner used to do: pyamica's constants carried across.
    "borrowed-constants": {"newt_start": 50, "invsigmin": 1e-8, "invsigmax": 100.0},
    # pamica as shipped. Not a fair setting for a fixed 100-iteration budget --
    # do_newton=False targets the long parity runs its documentation describes.
    "library-defaults": {"lrate": 0.05, "do_newton": False, "newt_start": 20,
                         "invsigmin": 1e-4, "invsigmax": 1000.0},
}
BLOCK_SIZES = [512, 4096, 16384, 65536]


def worst_matched_r(a, b) -> float | None:
    """Worst Hungarian-matched, sign-aligned correlation between unmixing rows."""
    A, B = np.asarray(a, float), np.asarray(b, float)
    if A.shape != B.shape:
        return None
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    C = np.abs(A @ B.T)
    r, c = linear_sum_assignment(-C)
    return float(np.min(C[r, c]))


def run(cfg: dict, out_path: Path, input_path: Path, device: str) -> dict | None:
    if out_path.exists():
        print(f"  reusing {out_path.name}")
    else:
        cp = subprocess.run(
            [sys.executable, str(RUNNER), "--input", str(input_path),
             "--output", str(out_path), "--config", json.dumps(cfg)],
            capture_output=True, text=True, timeout=7200,
            env={**os.environ, "TORCH_DEVICE": device})
        if cp.returncode != 0:
            print(f"  FAILED: {cp.stderr.strip()[-400:]}")
            return None
        print("  " + (cp.stdout.strip().splitlines() or ["(no stdout)"])[-1])
    d = json.loads(out_path.read_text())
    if "error" in d:
        print(f"  ERROR: {d['error']}")
        return None
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path,
                    help="orchestrator run dir holding input_*.npz and the amica reference")
    ap.add_argument("--control", choices=["config", "blocks"], required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--device", default=os.environ.get("TORCH_DEVICE", "cuda"))
    ap.add_argument("--reference", default=None,
                    help="reference result filename; defaults to the amica chunked run")
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    inputs = sorted(run_dir.glob("input_*.npz"))
    if not inputs:
        return print(f"no input_*.npz under {run_dir}") or 1
    input_path = inputs[0]

    ref_name = args.reference or "amica_python_jax_chunked_sub-01_seed0_result.json"
    ref_path = run_dir / ref_name
    if not ref_path.exists():
        cands = sorted(p.name for p in run_dir.glob("amica_python_jax*_result.json"))
        return print(f"reference {ref_name} not found; candidates: {cands}") or 1
    ref = json.loads(ref_path.read_text())

    out_dir = args.out_dir or (run_dir / f"controls_{args.control}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    if args.control == "config":
        for name, extra in CONFIGS.items():
            print(f"[{name}]")
            d = run({**PROTOCOL, **extra}, out_dir / f"pamica_{name}.json",
                    input_path, args.device)
            if d:
                rows.append((name, d))
        label = "config"
    else:
        n_samples = int(np.load(input_path)["X"].shape[1])
        for bs in BLOCK_SIZES:
            print(f"[block_size={bs}]  {n_samples // bs} blocks/iter")
            d = run({**PROTOCOL, "block_size": bs}, out_dir / f"pamica_bs{bs}.json",
                    input_path, args.device)
            if d:
                rows.append((str(bs), d))
        label = "block_size"

    print()
    print(f"reference: {ref_path.name}  ll_final={ref['ll_final']:.10f}  "
          f"fit={ref['fit_time_s']:.1f}s")
    print()
    print(f"{label:20} {'fit_s':>9} {'VRAM_GiB':>9} {'ll_final':>16} "
          f"{'|dll|':>11} {'worst|r|':>9}")
    print("-" * 80)
    for name, d in rows:
        vram = d.get("peak_vram_gb")
        r = worst_matched_r(ref["W"], d["W"])
        print(f"{name:20} {d['fit_time_s']:9.1f} "
              f"{(f'{vram:9.3f}' if vram is not None else '        -')} "
              f"{d['ll_final']:16.10f} "
              f"{abs(d['ll_final'] - ref['ll_final']):11.3e} "
              f"{(f'{r:9.4f}' if r is not None else '        -')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
