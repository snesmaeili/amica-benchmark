# Stage I: i.i.d. multi-model AMICA validation

This directory contains the predeclared validation layer for the existing
multi-model solver. It does not implement a temporal state model.

For fitted model order \(M\), the current solver is an i.i.d. finite mixture
of complete ICA emissions:

\[
p(x_t)=\sum_{m=1}^{M}\pi_m p_m(x_t).
\]

Consequently, its total likelihood is invariant to a permutation of sample
order. A likelihood improvement can support distributional heterogeneity, but
cannot by itself establish temporal non-stationarity, persistence, or
physiological brain states.

## Evidence gates

1. **Implementation gate**
   - Atomic rollback of every coupled multi-model parameter.
   - Documented density-update flags are honoured.
   - Symmetric fixed initialisations are rejected.
   - The objective is recomputed from the returned final state.
   - Terminal posteriors respect the configured time chunk.
   - Versioned multi-model save/load round-trips all shapes and optional
     posterior state.
   - Current-package path, commit, manifest, command, hardware and environment
     provenance are complete.

2. **One-fit cluster smoke**
   - One current-package \(M=2\) fit.
   - Strict JSON and readable NPZ output.
   - Finite returned-state likelihood and normalised posteriors.
   - Exact manifest and package commit match.
   - No Slurm campaign proceeds until this record passes.

3. **Replicated synthetic core**
   - Seven mechanisms, \(C=16\), 250 Hz, \(T=150{,}000\), \(K=3\).
   - Fitted \(M=1,\ldots,8\).
   - Thirty independent generating seeds crossed with three fit seeds.
   - Every fit uses the frozen current-code configuration: JAX-GPU float64,
     2,000 iterations, a 65,536-sample chunk, and sample rejection disabled.
     This is a fixed-budget result, not a declaration of formal convergence.
   - Transition-boundary F1 uses a predeclared 0.5-s (125-sample) tolerance.
   - The manifest records and checks the synthetic-generator schema version.
   - The stationary control has \(M_{\mathrm{true}}=1\).
   - The five discrete non-stationary mechanisms use
     \(M_{\mathrm{true}}=3\).
   - Continuous drift is deliberately misspecified and has no finite discrete
     \(M_{\mathrm{true}}\).

4. **Synthetic stress campaign**
   - Run only after the core has passed and been reviewed.
   - One factor at a time: true order, samples per model and \(C^2\),
     occupancy, dwell time, SNR, shared sources and temporal form.
   - The complete declared core contains 5,040 fits; the complete stress
     manifest contains 16,560 result cells. Five stress reference levels
     explicitly reuse the matching core reference through
     `reference_condition_id`, so only 12,960 stress rows require new fitting.
     A submission runner must not execute rows with `requires_fit=False`.
     These are separate gated campaigns, not one array to submit unchanged.
   - The `density_only` mechanism changes the AMICA density weight and shape
     parameters \(\alpha\) and \(\rho\) while holding \(\mu\) and \(\beta\)
     fixed. It is not a moment-matched shape-only control: realised marginal
     variances may change as a consequence of the altered density.

5. **Real-EEG held-out validation**
   - All preprocessing parameters are learned from training samples only.
   - Prepared fold arrays are already centred, projected and scaled; runners
     must disable AMICA's internal mean/sphere/PCA estimation and must score
     these exact arrays without refitting a transform.
   - Five contiguous test folds use 5-s guards.
   - Every fitted \(M\) is scored on identical held-out samples.
   - The subject is the inferential unit.
   - The primary endpoint is block-held-out log predictive-density gain over
     \(M=1\), not in-sample likelihood or effective model count.

6. **Null, stability and nuisance gates**
   - Common-phase, multivariate IAAFT-type and stationary parametric nulls are
     separately documented by the properties they preserve.
   - Circular shifts and block permutations are used only for posterior
     temporal statistics.
   - Returned fitted priors \(\pi_m<0.02\) and Kish
     \(N_{\mathrm{eff}}/C^2<25\) are flagged; thresholds 10, 25 and 50 are
     reported as sensitivity analyses. Fitted priors, posterior occupancy and
     hard occupancy are stored separately rather than treated as synonyms.
   - Seed-aligned posterior, source and model stability are reported.
   - EOG/EMG, movement, amplitude, discontinuities, rejection and
     preprocessing-boundary associations are audited.

## Stop conditions

Stop before further fitting when a current-code smoke fails, Fortran parity
fails, a non-finite state appears, provenance is incomplete, or a result
contradicts a manuscript claim. Do not change the solver to improve a
benchmark outcome. Figure 5 and its associated manuscript text remain frozen
until these gates pass and a separate figure revision is approved.

An HMM emission wrapper is a separate Stage II project. It must not be used to
reinterpret the present i.i.d. likelihood as temporal evidence.

## Manifest and provenance contract

The synthetic CSV is a typed, versioned manifest. It must be loaded with
`read_manifest_csv`, which parses optional nulls and exact booleans, recomputes
row hashes, and validates the complete seed/order crossing. Raw
`csv.DictReader` strings are not admissible run inputs.

Completed-run provenance records include checksummed inputs and outputs,
resolved software and OS versions, clean-worktree flags, physical and logical
CPU information, GPU identity, thread settings, JAX/XLA settings, and the
requested Slurm allocation. Stage I JAX-GPU records must be float64 with
`jax_enable_x64=True` and an explicitly resolved GPU platform.
