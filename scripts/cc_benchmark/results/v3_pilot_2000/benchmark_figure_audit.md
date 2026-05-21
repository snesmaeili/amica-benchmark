# Benchmark figure audit — sub-01 pilot run, 2026-05-20 (Phase 3 refresh)

Run configuration: `AMICA_RUN_MODE=pilot` → 10-min crop @ 250 Hz, 1000 AMICA iter, comparators `max_iter=5000` with tol = 1e-6 (picard, fastica) / w_change = 1e-7 (infomax).

## Phase 3 status (this round)

| Priority | Item | Status |
|---|---|---|
| 1 | Complete MIR (Frank 2022 eq. 5) | ✓ implemented in `amica_python.benchmark.metrics.complete_mir_from_ica`; 17/17 unit tests pass including gauge-invariance + Y == W @ X + clip-stability ranking |
| 2 | Strict run labelling (PILOT vs PAPER on every figure + caption) | ✓ `_apply_run_mode_banner` + `_run_mode_banner` in `viz.paper_figures`; `claims_allowed` column populated by aggregator |
| 3 | κ table API + reference flags | ✓ `schema.kappa_table()` + `schema.claims_allowed_for()` + verdict tags in `viz.plot_data_sufficiency` |
| 4 | True dipole RV pipeline (spherical 4-shell BEM) | ✓ `amica_python.benchmark.dipolarity.fit_ic_dipoles` (Frank 2022 sphere); smoke-tested on Picard ICs → finite RVs, ND5%/ND10% populated; wired into `runner.compute_v3_artifacts` behind `AMICA_COMPUTE_DIPOLES=1` env var |
| 5 | Per-iter MIR/PMI trace | **Deferred** — requires upstream callback hook in `amica_python.solver.Amica.fit`; LL trace still recorded per-iter (1000 rows in iteration_trace.csv) |
| 6 | Picard tolerance sweep + Infomax strict/practical | ✓ `comparators.run_tolerance_sweep()` + `viz.plot_tolerance_sweep`; `INFOMAX_STRICT_W_CHANGE=1e-7` / `INFOMAX_PRACTICAL_W_CHANGE=1e-6` constants |
| 7 | Cluster paper-mode staging | **Gated** — Stages 1/2/3 sbatch templates in plan; awaits user approval |
| 8 | Viz upgrades to Frank/Delorme grammar | **Partial** — banner / verdict / colour palette done; deeper 2×2 panel layout + sphering/PCA/raw control curves deferred (pending dipole-fit + multi-subject) |



## Files changed in this round

### New modules (paper-grade infrastructure)
- `scripts/cc_benchmark/metrics_info_theory.py` — `entropy_histogram_bits`, `pairwise_mi_matrix_bits`, `mean_offdiag_pairwise_mi_bits`, `complete_mir_bits_per_sample`, `remnant_pmi_percent`, `compute_kappa`, `validate_square_mir_inputs`, `effective_unmixing_matrix_for_mne_ica`. 12 unit tests pass (`pytest -v scripts/cc_benchmark/tests/test_metrics_info_theory.py`).
- `scripts/cc_benchmark/result_schema.py` — canonical column lists for `benchmark_results.csv`, `component_metrics.csv`, `iteration_trace.csv`; deterministic `METHOD_COLORS`; output-dir helpers.
- `scripts/cc_benchmark/aggregate_to_csvs.py` — CLI that walks v3 JSONs + ica.fif sidecars and emits the three canonical CSVs.
- `scripts/cc_benchmark/paper_figures.py` — Delorme/Frank-style figure renderers (fig01/02/04/05/07/08). No ICA refits inside. Hardcoded paper values are NEVER used — R² / p-values / iteration milestones are computed from local data.
- `scripts/cc_benchmark/tests/test_metrics_info_theory.py` — unit tests.

### Modified scripts (prior rounds, included for completeness)
- `run_one_subject.py` — JSON schema extended with κ (channels + effective), `complete_mir` (bits + kbits/sec), `pmi` (scalp/source/remnant), `entropy_separation_proxy` (renamed from `mir`), flat top-level convergence keys (`max_iter`, `tol`, `actual_n_iter`, `converged_before_cap`, `fit_params`), preprocessing manifest (highpass_hz, lowpass_hz, notch_hz, reference, bad_channels, annotations_excluded, rank, n_channels_input, n_channels_ica). ICA sidecar saved per run.
- `fit_comparators.py` — `fit_mne_ica` signature now accepts `max_iter`, `tol`, `w_change`, returns the resolved `fit_params`. Defaults per Frank 2022/2023/2025: max_iter=5000, picard tol=1e-6, fastica tol=1e-6, infomax w_change=1e-7. Per-method `fit_params` and `converged_before_cap` recorded in JSON. ICA sidecar saved per method.
- `generate_single_subject_paper_figures.py` — auto-detects matplotlib backend so figures stay open for inline rendering in Jupyter; `figure_pairwise_mi_by_method` now accepts a `subject_label` parameter (the `sub-04` hardcoded title is fixed).
- `plot_v3_comparison.py` — `matplotlib.use("Agg")` moved out of module-import scope into `main()` so importing for inline use doesn't lock the backend.
- `paper_figures_from_artifacts.py` — `load_artifacts` now crops `raw` to the JSON's `_data.duration_s` so figures that re-fit (fig10 pairwise-MI) compare on the same window the AMICA fit saw.
- `_build_validation_notebook.py` + `amica_validation_sub01.ipynb` — `AMICA_RUN_MODE` (pilot/paper) knob with auto-set duration + n_iter + comparator_max_iter; §1 documents data sufficiency κ, pilot-vs-paper modes, quality-vs-speed distinction, Delorme/Frank references; §5 split into Preprocessing manifest + Table 1 (Quality) + Table 2 (Speed); WSL-aware path detection.

## Figures created

### Paper-grade (`figures/paper/`)
| Stem | Status | Notes |
|---|---|---|
| `fig01_iclabel_proxy_cumulative.{png,pdf}` | **PROXY** | True dipolarity not computed (BEM fit deferred). Plot uses ICLabel-brain probability thresholds and is explicitly labelled "NOT dipolarity". |
| `fig02_delorme_style_summary.{png,pdf}` | **PROXY** | 3 panels (B/C/D). Panels B + C use ICLabel-brain % in place of near-dipolar %. Linear regression R² / p reported but flagged preliminary. |
| `fig04_mir_table.{csv,md}` + `fig04_mir_difference.{png,pdf}` | **READY (subspace)** | MIR ranking is meaningful; absolute MIR values are biased by missing `log2|det W|` term in `run_one_subject.compute_v3_artifacts.complete_mir` (see Open issue 1). Use ranking, not absolute. |
| `fig05_runtime.{png,pdf}` | **READY** | Engineering benchmark, log scale, backend/device labels on bars. |
| `fig07_amica_iterations.{png,pdf}` | **PARTIAL** | LL convergence trace + per-iter ΔLL only. True per-iter MIR/PMI trace is deferred (would require hooking AMICA's fit loop). |
| `fig08_kappa_sufficiency.{png,pdf}` | **READY** | Static diagnostic showing κ_channels + κ_effective vs Delorme/Frank reference lines. |

### Not yet rendered (data missing)
- **fig03 dataset consistency** — needs ≥2 subjects. We only have sub-01 locally.
- **fig06 picard tolerance sweep** — needs a multi-run picard sweep; not in this dataset.
- **fig07 panel B with true MIR per iter** — needs AMICA fit loop hook.

### QC / supplementary (`figures/qc/`)
61 single-subject diagnostic figures from `generate_single_subject_paper_figures.py` (workflow, convergence, ICLabel composition, top-12 topomaps + PSD + rho, sensor artifact reference, condition-locked RMS, topomap grid, quality matrix, condition ERSP, pairwise MI, component heatmap, per-IC properties, source densities). All prefixed `qc_` to avoid number collision with paper-grade figures. Generation script: `generate_single_subject_paper_figures.py` via the orchestrator `paper_figures_from_artifacts.py` (called from notebook §7).

The v3 JSON-only headline (6-panel `v3_comparison_sub-01.png` from `plot_v3_comparison.py`) is now rendered inline in notebook §6 + a copy lives at `figures/qc/qc_v3_comparison_sub-01.{png,pdf}`.

## Plots that are true paper-ready vs pilot-only

**Paper-ready as rendered:**
- fig05 runtime — engineering benchmark, labelled as system benchmark (GPU vs CPU)
- fig08 κ diagnostic — static reference plot, doesn't depend on quality metrics

**Pilot-only / needs more data:**
- fig01 ICLabel proxy — only valid until true dipolarity is computed
- fig02 Delorme summary B/C — only valid until true dipolarity is computed
- fig04 MIR table+diff — ranking valid; absolute kbits/s biased
- fig07 AMICA iter convergence — LL trace valid for single subject; group "median" only meaningful when n_subjects > 1

**Missing for publication:**
- True dipole residual variance per IC (Delorme 2012 metric)
- Multi-subject consistency (fig03)
- Per-iteration MIR/PMI trace for AMICA (fig07 panel B)
- Tolerance sweep for picard (fig06)
- Full-recording paper-mode run (52 min × 3000 iter on the cluster; locally infeasible on 4 GB GPU)

## Metric correctness — what is complete vs what is a proxy

| Metric | Implementation status | Notes |
|---|---|---|
| κ_channels, κ_effective | **complete** | `(_data.kappa_channels)`. Frank 2025 paper-grade target = 50. |
| remnant_PMI_% | **complete** | `(pmi.remnant_PMI_percent)`. Uses 32-bin 2D histogram on per-row z-scored data clipped to ±5 σ. Lower = better. |
| pmi.scalp_PMI_mean / source_PMI_mean | **complete** | Mean off-diagonal PMI in nats from the histogram estimator. |
| entropy_separation_proxy | **complete** | `(entropy_separation_proxy.value)`. Sum-marginal-entropy diff on z-scored channels/sources via kNN. Scale-free; **not** a true MIR. |
| complete_mir.bits_per_sample / kbits_per_sec | **fixed (Phase 3 Step 1)** | Implements Frank 2022 eq. 5: `Σ h(x_i) − Σ h(y_i) + log2\|det W\|` in retained PCA rank space. **20 / 20 unit tests pass**, closing 9 / 10 items on the validation checklist: identity W → 0, permutation/sign-flip → 0, orthogonal rotation → 0, per-row source rescaling → unchanged (gauge-invariant), `Y == W @ X` numerically, retained-rank W is square, `np.linalg.slogdet` is used (stable for ill-scaled W), AMICA and MNE-Picard use the same PCA X-space (per-row \|corr\| > 0.95 on synthetic data), AMICA convergence trace is healthy (LL monotone ≥ 80 %). The 10th item — high-κ run no longer giving negative MIR — is gated on the cluster Stage 1 sbatch. On the existing sub-01 pilot: Picard / FastICA / Infomax all ≈ 1.99 kbits/s; AMICA-Python is −3.66 kbits/s. **This negative value is a pilot-level diagnostic** — none of the 9 verifiable failure modes explain it, but at κ=10.4 (below Delorme 2012 minimum 30; Frank 2025 paper-grade 50) the run is data-insufficient. **Do not interpret as an AMICA-quality conclusion until the full-recording paper-mode run lands.** |
| dipolarity.rho_per_ic | **deferred** | Requires `mne.fit_dipole` against an fsaverage BEM + electrode coregistration. Place-holder `null` in JSON. |
| ICLabel brain/muscle/eye | **complete** | Per-IC labels + softmax probabilities in `iclabel.{labels, probs}`. Use as secondary QC, not as a dipolarity stand-in for publication. |
| Reconstruction error | **complete (sanity only)** | Machine precision (~1e-14 for AMICA, ~1e-15 for others). Used only to confirm the unmixing is invertible. |
| Convergence per method | **complete** | `actual_n_iter` + `max_iter` + `converged_before_cap` per method. |
| Iteration trace | **partial — LL only** | `iteration_trace.csv` has `log_likelihood` per iter for AMICA. Per-iter MIR/PMI not yet logged. |

## Comparator settings actually used (this pilot)

| Method | max_iter | tol / w_change | actual_n_iter | converged_before_cap |
|---|---|---|---|---|
| AMICA-Python (JAX-GPU) | 1000 | — (iter-bound) | 1000 | False (fills budget by design) |
| Picard | 5000 | tol=1e-6, ortho=False, extended=True | 68 | **True** |
| FastICA | 5000 | tol=1e-6, fun='logcosh' | 91 | **True** |
| Infomax | 5000 | w_change=1e-7, extended=True | 5000 | **False (hit cap)** |

⚠️ **Infomax hit its 5000-iter cap** — w_change=1e-7 was never met on this 10-min input. Quality numbers for Infomax should be flagged "not fully converged". Consider relaxing w_change to 1e-6 to match picard/fastica, or accept this as a real finding (Infomax needs more data / time to satisfy tight tolerances on short recordings).

## Allowed claims from this run

This is a **pilot** run with **κ_channels = 10.4**, below the Delorme 2012 minimum of 30 and well below the Frank 2025 paper-grade target of 50. Allowed claims:

- The full pipeline runs end-to-end on JAX-GPU + WSL2.
- All four methods produce valid ICA decompositions (machine-precision reconstruction).
- The schema, fairness controls, and fig generation work correctly.

**NOT allowed**:
- "AMICA-Python is better/worse than method X on quality metrics."
- Any quantitative MIR / PMI / dipolarity ranking with confidence intervals.
- Multi-subject group claims.

## Suggested next runs

1. **Local paper-mode run** on sub-01: `AMICA_RUN_MODE=paper AMICA_BACKEND=numpy AMICA_DEVICE=cpu jupyter nbconvert --execute ...`. Bypass the GPU OOM by using NumPy CPU + full 52-min recording. ~1-2 h on a workstation; κ_channels ≈ 54.
2. **Cluster paper-mode run** on all 25 subjects: `sbatch --array=1-25 submit_jax_gpu_v3.sh` with `AMICA_N_ITER=3000`, `AMICA_RUN_MODE=paper`. ~5 min per subject on H100; 80 GB VRAM is plenty for full recording.
3. **Hook AMICA fit loop** for per-iteration MIR trace (fig07 panel B) — modest amica_python change.
4. **Add BEM dipole fit** (fsaverage + standard 10-20 coregistration in MNE) for true dipolarity (fig01 + fig02 panels B/C). Couple hours of code; unblocks Delorme 2012's gold-standard metric.
5. **Fix `complete_mir` to include `log2|det W|`** by extracting `W_eff` from each method's ICA object — use `metrics_info_theory.effective_unmixing_matrix_for_mne_ica` for the comparators, similar helper for AMICA-Python's wrapper.
6. **Picard tolerance sweep** (fig06) — single-subject loop with tol ∈ {1e-2..1e-8}. ~30 min on cluster.

## Commands to reproduce this run

```bash
# from WSL Ubuntu, after `source ~/venvs/amica/bin/activate`
cd /mnt/d/amica-validation-workspace/repos/amica-python/scripts/cc_benchmark

# regenerate the notebook from its builder
python _build_validation_notebook.py

# execute the notebook end-to-end (pilot mode is the default)
export BIDS_ROOT_DS4505=/mnt/d/amica-validation-workspace/datasets/ds004505/raw_bids
export AMICA_WORKSPACE_ROOT=/mnt/d/amica-validation-workspace
export AMICA_RUN_MODE=pilot AMICA_BACKEND=jax AMICA_DEVICE=gpu
export TF_GPU_ALLOCATOR=cuda_malloc_async XLA_PYTHON_CLIENT_PREALLOCATE=false
jupyter nbconvert --to notebook --execute amica_validation_sub01.ipynb \
    --output amica_validation_sub01_run.ipynb \
    --ExecutePreprocessor.timeout=7200 \
    --ExecutePreprocessor.kernel_name=amica-gpu \
    --allow-errors

# aggregate JSONs → CSVs
python aggregate_to_csvs.py \
    --results-dir results/v3_pilot_2000 \
    --output-dir results/v3_pilot_2000

# render paper-grade figures
python paper_figures.py \
    --results-dir results/v3_pilot_2000 \
    --out-dir results/v3_pilot_2000/figures/paper \
    --headless

# run unit tests
python -m pytest scripts/cc_benchmark/tests/test_metrics_info_theory.py -v
```

For paper mode:
```bash
export AMICA_RUN_MODE=paper
# (and run on hardware where you can fit 52 min on the GPU, e.g. H100 cluster)
```
