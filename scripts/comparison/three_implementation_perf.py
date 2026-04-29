"""Three-implementation perf comparison on MNE sample (60-ch EEG, ~1 min).

Runs four configurations side-by-side:
  - amica-python (Sina, JAX)
  - amica-python (Sina, NumPy fallback via AMICA_NO_JAX=1)
  - pyamica (DerAndereJohannes, PyTorch)
  - amica-python (Scott Huberty, PyTorch sklearn-style)

Each runs in a SEPARATE subprocess with its own venv to avoid the
JAX/Torch import-order conflict and to keep peak-RSS measurements clean.

Outputs:
  results/comparison/three_implementation_perf.json     (aggregated)
  results/comparison/<impl>_result.json                 (per-runner)

Usage:
  python scripts/comparison/three_implementation_perf.py [--max-iter 100]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "comparison"
RUNNERS_DIR = ROOT / "scripts" / "comparison" / "runners"

# Venv pythons — adjust if your local layout differs.
VENV_AMICA = Path(r"D:\mne-amica\.venv311\Scripts\python.exe")
VENV_COMPETITORS = ROOT / ".venv-competitors" / "Scripts" / "python.exe"


def preprocess_mne_sample(n_components: int = 30, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Pick EEG, 1 Hz HP, average ref, then sklearn PCA → (n_components, n_samples)."""
    import mne
    from sklearn.decomposition import PCA

    sample_path = mne.datasets.sample.data_path()
    raw_fname = sample_path / "MEG" / "sample" / "sample_audvis_raw.fif"
    raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
    raw.pick_types(eeg=True, exclude="bads")
    raw.filter(1.0, None, verbose=False)
    raw.set_eeg_reference("average", verbose=False)

    data = raw.get_data().astype(np.float64)  # (n_ch, n_samples)
    n_ch, n_samples = data.shape
    n_comp = min(n_components, n_ch)

    pca = PCA(n_components=n_comp, whiten=False, random_state=seed)
    projected = pca.fit_transform(data.T).T  # (n_comp, n_samples)

    # Per-component variance normalisation (matches Sina's mne_integration path)
    stds = np.std(projected, axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    projected = projected / stds

    meta = {
        "dataset": "mne_sample_audvis",
        "n_channels": int(n_ch),
        "n_samples": int(n_samples),
        "n_components": int(n_comp),
        "sfreq": float(raw.info["sfreq"]),
        "filter_l_freq": 1.0,
        "reference": "average",
    }
    return projected, meta


def run_subprocess(python_exe: Path, runner: Path, input_path: Path, output_path: Path,
                   config: dict, env_extra: dict | None = None) -> dict:
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    if env_extra:
        env.update(env_extra)

    cmd = [
        str(python_exe), str(runner),
        "--input", str(input_path),
        "--output", str(output_path),
        "--config", json.dumps(config),
    ]
    print(f"[orchestrator] {python_exe.name} {runner.name} {env_extra or ''}")
    t0 = time.perf_counter()
    try:
        cp = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=3600)
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "wall_s": 3600.0}
    wall = time.perf_counter() - t0

    if cp.returncode != 0:
        return {
            "error": "nonzero_exit",
            "returncode": cp.returncode,
            "stdout": cp.stdout[-2000:],
            "stderr": cp.stderr[-2000:],
            "wall_s": wall,
        }
    if not output_path.exists():
        return {
            "error": "no_output_file",
            "stdout": cp.stdout[-2000:],
            "stderr": cp.stderr[-2000:],
            "wall_s": wall,
        }
    with open(output_path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--n-components", type=int, default=30)
    parser.add_argument("--n-mix", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lrate", type=float, default=0.1)
    parser.add_argument("--skip", nargs="*", default=[],
                        help="implementations to skip (e.g. --skip pyamica scott_huberty)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Build the shared input
    print(f"[orchestrator] preprocessing MNE sample ...")
    X, meta = preprocess_mne_sample(n_components=args.n_components, seed=args.seed)
    input_path = RESULTS_DIR / "input.npz"
    np.savez(input_path, X=X, **{k: v for k, v in meta.items() if isinstance(v, (int, float))})
    print(f"[orchestrator]   X={X.shape}, sfreq={meta['sfreq']} Hz")

    cfg = dict(
        max_iter=args.max_iter, n_mix=args.n_mix, lrate=args.lrate,
        seed=args.seed, do_newton=True,
    )

    # 2. Define the 4 runs
    runs = [
        ("amica_python_jax",    VENV_AMICA,        RUNNERS_DIR / "run_amica_python.py", None),
        ("amica_python_numpy",  VENV_AMICA,        RUNNERS_DIR / "run_amica_python.py", {"AMICA_NO_JAX": "1"}),
        ("pyamica_torch",       VENV_COMPETITORS,  RUNNERS_DIR / "run_pyamica.py",      None),
        ("scott_huberty_torch", VENV_COMPETITORS,  RUNNERS_DIR / "run_scott_huberty.py",None),
    ]

    # 3. Run each
    summary = {"meta": meta, "config": cfg, "results": {}}
    for name, py, runner, env_extra in runs:
        if name in args.skip:
            print(f"[orchestrator] SKIPPING {name}")
            continue
        if not py.exists():
            summary["results"][name] = {"error": f"venv python not found at {py}"}
            print(f"[orchestrator] FAIL {name}: venv python missing at {py}")
            continue
        out_json = RESULTS_DIR / f"{name}_result.json"
        if out_json.exists():
            out_json.unlink()
        result = run_subprocess(py, runner, input_path, out_json, cfg, env_extra)
        summary["results"][name] = result

    # 4. Aggregate the table
    rows = []
    for name, r in summary["results"].items():
        if r.get("error"):
            rows.append((name, "error", "—", "—", "—", r.get("error")))
        else:
            rows.append((
                name,
                f"{r['fit_time_s']:.2f}",
                f"{r['peak_rss_gb']:.2f}",
                f"{r['ll_final']:.4f}",
                f"{r['n_iter']}",
                "ok",
            ))

    print()
    print(f"{'impl':<24} {'time(s)':>10} {'rss(GB)':>10} {'ll_final':>14} {'iters':>8} {'status':>8}")
    print("-" * 80)
    for row in rows:
        print(f"{row[0]:<24} {row[1]:>10} {row[2]:>10} {row[3]:>14} {row[4]:>8} {row[5]:>8}")

    # 5. Pairwise W correlations with Hungarian-matched component permutation
    from scipy.optimize import linear_sum_assignment

    def matched_mean_corr(Wa: np.ndarray, Wb: np.ndarray) -> float:
        """Mean unsigned correlation after Hungarian permutation matching."""
        if Wa.shape != Wb.shape:
            return float("nan")
        # Cost matrix: cost[i, j] = 1 - |corr(Wa[i], Wb[j])|
        n = Wa.shape[0]
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                v = np.corrcoef(Wa[i], Wb[j])[0, 1]
                C[i, j] = 1.0 - (abs(v) if np.isfinite(v) else 0.0)
        row_ind, col_ind = linear_sum_assignment(C)
        return float(np.mean(1.0 - C[row_ind, col_ind]))

    ok = {n: r for n, r in summary["results"].items() if "W" in r and r["W"] is not None}
    pairs = sorted(ok.keys())
    summary["pairwise_W_correlation"] = {}
    if len(pairs) >= 2:
        print()
        print("Pairwise W correlation (Hungarian-matched, unsigned):")
        for i, a in enumerate(pairs):
            for b in pairs[i + 1:]:
                Wa = np.asarray(ok[a]["W"])
                Wb = np.asarray(ok[b]["W"])
                mean_corr = matched_mean_corr(Wa, Wb)
                summary["pairwise_W_correlation"][f"{a}__vs__{b}"] = mean_corr
                print(f"  {a:<24} vs {b:<24}  mean|r| = {mean_corr:.3f}")

    # 6. Write the aggregated JSON
    out = RESULTS_DIR / "three_implementation_perf.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[orchestrator] aggregated -> {out}")


if __name__ == "__main__":
    main()
