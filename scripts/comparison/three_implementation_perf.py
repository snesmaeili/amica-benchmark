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

# Venv pythons. Override either via env vars for portability between machines:
#   AMICA_PYTHON_VENV       — path to amica-python's venv python
#   COMPETITORS_VENV        — path to competitors venv python (pyamica + scott)
# Default: assume a sibling-repo layout where amica-python and amica-python-benchmark
# live next to each other (the workspace convention under repos/).
_default_amica_venv = ROOT.parent / "amica-python" / ".venv311" / "Scripts" / "python.exe"
VENV_AMICA = Path(os.environ.get("AMICA_PYTHON_VENV", str(_default_amica_venv)))
VENV_COMPETITORS = Path(os.environ.get("COMPETITORS_VENV",
                                        str(ROOT / ".venv-competitors" / "Scripts" / "python.exe")))


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


def _aggregate_per_impl(seed_results: list[dict]) -> dict:
    """Compute mean/std across seed-results for a single implementation."""
    valid = [r for r in seed_results if "error" not in r]
    out: dict = {"n_seeds_ok": len(valid), "n_seeds_total": len(seed_results)}
    if not valid:
        out["error"] = "all seeds failed"
        return out
    for key in ("fit_time_s", "peak_rss_gb", "ll_final", "n_iter"):
        vals = [r[key] for r in valid if key in r and r[key] is not None]
        if vals:
            out[f"{key}_mean"] = float(np.mean(vals))
            out[f"{key}_std"] = float(np.std(vals))
    return out


def _matched_mean_corr(Wa: np.ndarray, Wb: np.ndarray) -> float:
    """Mean unsigned correlation after Hungarian permutation matching."""
    from scipy.optimize import linear_sum_assignment
    if Wa.shape != Wb.shape:
        return float("nan")
    n = Wa.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            v = np.corrcoef(Wa[i], Wb[j])[0, 1]
            C[i, j] = 1.0 - (abs(v) if np.isfinite(v) else 0.0)
    row_ind, col_ind = linear_sum_assignment(C)
    return float(np.mean(1.0 - C[row_ind, col_ind]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--n-components", type=int, default=30)
    parser.add_argument("--n-mix", type=int, default=3)
    parser.add_argument("--seeds", default="0",
                        help="Comma-separated seed list (e.g. '0,1,2'). Default '0' for single-seed.")
    parser.add_argument("--lrate", type=float, default=0.1)
    parser.add_argument("--skip", nargs="*", default=[],
                        help="implementations to skip (e.g. --skip pyamica_torch scott_huberty_torch)")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Build the shared input (PCA seed = first seed for determinism)
    print(f"[orchestrator] preprocessing MNE sample (PCA seed={seeds[0]})...")
    X, meta = preprocess_mne_sample(n_components=args.n_components, seed=seeds[0])
    input_path = RESULTS_DIR / "input.npz"
    np.savez(input_path, X=X, **{k: v for k, v in meta.items() if isinstance(v, (int, float))})
    print(f"[orchestrator]   X={X.shape}, sfreq={meta['sfreq']} Hz")

    base_cfg = dict(
        max_iter=args.max_iter, n_mix=args.n_mix, lrate=args.lrate,
        do_newton=True,
    )

    # 2. Define the 4 runs (name, venv, runner script, env_extra)
    runs = [
        ("amica_python_jax",    VENV_AMICA,        RUNNERS_DIR / "run_amica_python.py", None),
        ("amica_python_numpy",  VENV_AMICA,        RUNNERS_DIR / "run_amica_python.py", {"AMICA_NO_JAX": "1"}),
        ("pyamica_torch",       VENV_COMPETITORS,  RUNNERS_DIR / "run_pyamica.py",      None),
        ("scott_huberty_torch", VENV_COMPETITORS,  RUNNERS_DIR / "run_scott_huberty.py",None),
    ]

    # 3. Run each (impl × seed)
    summary: dict = {
        "meta": meta,
        "config": base_cfg,
        "seeds": seeds,
        "results": {},               # impl -> list[result_dict] (one per seed)
        "aggregated": {},            # impl -> mean/std summary
        "pairwise_W_correlation": {},  # "a__vs__b" -> {seed_n: corr, ..., "mean": ...}
    }

    for name, py, runner, env_extra in runs:
        if name in args.skip:
            print(f"[orchestrator] SKIPPING {name}")
            continue
        if not py.exists():
            summary["results"][name] = [{"error": f"venv python not found at {py}"}]
            print(f"[orchestrator] FAIL {name}: venv python missing at {py}")
            continue
        per_seed: list = []
        for seed in seeds:
            cfg_seeded = dict(base_cfg, seed=seed)
            out_json = RESULTS_DIR / f"{name}_seed{seed}_result.json"
            if out_json.exists():
                out_json.unlink()
            print(f"[orchestrator] {name} seed={seed} ...")
            result = run_subprocess(py, runner, input_path, out_json, cfg_seeded, env_extra)
            result["seed_used"] = seed
            per_seed.append(result)
        summary["results"][name] = per_seed
        summary["aggregated"][name] = _aggregate_per_impl(per_seed)

    # 4. Per-impl aggregated table
    print()
    print(f"{'impl':<24} {'time mean±std':>20} {'rss mean±std':>20} {'ll mean±std':>22} {'iters':>10}")
    print("-" * 100)
    for name, agg in summary["aggregated"].items():
        if agg.get("error"):
            print(f"{name:<24} {'(all seeds failed)':>20}")
            continue
        t = f"{agg.get('fit_time_s_mean', 0):.2f}±{agg.get('fit_time_s_std', 0):.2f}"
        r = f"{agg.get('peak_rss_gb_mean', 0):.2f}±{agg.get('peak_rss_gb_std', 0):.2f}"
        ll = f"{agg.get('ll_final_mean', 0):.4f}±{agg.get('ll_final_std', 0):.4e}"
        it = f"{agg.get('n_iter_mean', 0):.0f}±{agg.get('n_iter_std', 0):.0f}"
        print(f"{name:<24} {t:>20} {r:>20} {ll:>22} {it:>10}")

    # 5. Pairwise W correlations per seed, plus mean across seeds
    impl_names = sorted(summary["results"].keys())
    if len(impl_names) >= 2:
        print()
        print("Pairwise W correlation (Hungarian-matched, unsigned), per seed:")
        for i, a in enumerate(impl_names):
            for b in impl_names[i + 1:]:
                pair_key = f"{a}__vs__{b}"
                per_seed_corr: dict = {}
                a_results = summary["results"].get(a) or []
                b_results = summary["results"].get(b) or []
                for ar, br in zip(a_results, b_results):
                    if ar.get("error") or br.get("error"):
                        continue
                    if "W" not in ar or "W" not in br:
                        continue
                    if ar["W"] is None or br["W"] is None:
                        continue
                    seed = ar.get("seed_used", "?")
                    mc = _matched_mean_corr(np.asarray(ar["W"]), np.asarray(br["W"]))
                    per_seed_corr[f"seed_{seed}"] = mc
                if per_seed_corr:
                    vals = list(per_seed_corr.values())
                    per_seed_corr["mean"] = float(np.mean(vals))
                    per_seed_corr["std"] = float(np.std(vals))
                summary["pairwise_W_correlation"][pair_key] = per_seed_corr
                if per_seed_corr:
                    print(f"  {a:<24} vs {b:<24}  mean|r| = "
                          f"{per_seed_corr.get('mean', float('nan')):.3f} "
                          f"(±{per_seed_corr.get('std', 0):.3f}, n={len(per_seed_corr)-2})")

    # 6. Write the aggregated JSON
    out = RESULTS_DIR / "three_implementation_perf.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[orchestrator] aggregated -> {out}")


if __name__ == "__main__":
    main()
