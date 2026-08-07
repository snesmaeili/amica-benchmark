"""Smoke-test run_pamica.py end to end on a small synthetic fixture.

Cheap gate before committing an array job to the queue, and the thing to re-run
after bumping the pinned pamica version. Builds a pre-whitened
``(n_components, n_samples)`` input like the orchestrator produces, invokes the
runner as a subprocess exactly as implementation_perf.py does, and checks the
result JSON carries every RESULT_KEYS field with a usable value.

Run inside an allocation, never on a login node::

    source benchmark/cc_benchmark/pamica_env.sh
    srun --account=def-kjerbi_cpu --time=0:20:00 --cpus-per-task=4 --mem=16G \
        python benchmark/comparator/smoke_pamica.py

    # GPU: add --gres=gpu:h100:1 and TORCH_DEVICE=cuda
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "runners" / "run_pamica.py"
COMMON = HERE / "runners" / "_common.py"

n_comp = int(os.environ.get("SMOKE_NCOMP", 16))
n_samples = int(os.environ.get("SMOKE_NSAMP", 20000))
max_iter = int(os.environ.get("SMOKE_ITER", 30))
device = os.environ.get("TORCH_DEVICE", "cpu")


def result_keys() -> tuple[str, ...]:
    """RESULT_KEYS as _common.py declares it, without importing psutil."""
    src = COMMON.read_text(encoding="utf-8")
    literal = src.split("RESULT_KEYS = ")[1].split(")")[0] + ")"
    return tuple(eval(literal))  # noqa: S307 - a tuple literal from our own file


def main() -> int:
    # Super-Gaussian sources through a random mixing, then whitened -- the shape
    # the orchestrator hands runners after its PCA projection, so do_sphere and
    # do_mean stay off exactly as in a real run.
    rng = np.random.default_rng(0)
    S = rng.laplace(size=(n_comp, n_samples))
    A = rng.normal(size=(n_comp, n_comp))
    X = A @ S
    X -= X.mean(axis=1, keepdims=True)
    d, E = np.linalg.eigh(np.cov(X))
    X = (E @ np.diag(d ** -0.5) @ E.T) @ X

    tmp = Path(tempfile.mkdtemp(prefix="pamica_smoke_"))
    inp, outp = tmp / "input.npz", tmp / "result.json"
    np.savez(inp, X=X)

    cfg = {"max_iter": max_iter, "n_mix": 3, "lrate": 0.1, "do_newton": True, "seed": 0}
    cmd = [sys.executable, str(RUNNER), "--input", str(inp),
           "--output", str(outp), "--config", json.dumps(cfg)]
    print(f"[smoke] device={device} {n_comp}x{n_samples} max_iter={max_iter}")

    cp = subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                        env={**os.environ, "TORCH_DEVICE": device})
    print("--- runner stdout ---")
    print(cp.stdout.strip() or "(empty)")
    if cp.returncode != 0:
        print("--- runner stderr ---")
        print(cp.stderr.strip()[-3000:])
        print(f"FAIL: runner exited {cp.returncode}")
        return 1

    res = json.loads(outp.read_text())
    if "error" in res:
        print(f"FAIL: runner reported error: {res['error']}")
        return 1

    W = np.asarray(res["W"], dtype=float)
    print("\n--- result ---")
    for k in ("implementation", "pamica_version", "device", "dtype", "n_iter",
              "fit_time_s", "ll_final", "ll_last_iterate", "stop_reason",
              "converged_before_cap", "peak_rss_gb", "delta_rss_gb", "peak_vram_gb"):
        print(f"  {k:22} {res.get(k)}")
    print(f"  {'W.shape':22} {W.shape}")

    keys = result_keys()
    problems = []
    missing = [k for k in keys if k not in res]
    # VRAM keys are legitimately None on CPU and without the NVML cross-check.
    null = [k for k in keys
            if res.get(k) is None and k not in ("peak_vram_gb", "nvml_peak_vram_gb")]
    if missing:
        problems.append(f"missing RESULT_KEYS: {missing}")
    if null:
        problems.append(f"null RESULT_KEYS: {null}")
    if W.shape != (n_comp, n_comp):
        problems.append(f"W shape {W.shape}, expected {(n_comp, n_comp)}")
    if not np.isfinite(W).all():
        problems.append("W has non-finite entries")
    if not np.isfinite(res["ll_final"]):
        problems.append("ll_final is not finite")
    if len(res["ll_history"]) != res["n_iter"]:
        problems.append("n_iter disagrees with len(ll_history)")
    if device == "cuda" and res.get("peak_vram_gb") in (None, 0):
        problems.append("device=cuda but no VRAM peak recorded -- did CUDA init?")

    print()
    for p in problems:
        print("  FAIL " + p)
    if problems:
        return 1
    print("SMOKE PASS — every RESULT_KEYS field present and usable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
