# Figure 2 clean known-topography audit

These compact files are the complete plotting inputs for Figure 2C-D. They were
copied without modification from the completed Compute Canada Slurm run
48954350 in /scratch/sesma/amica_figure2_topography.

- Benchmark repository commit: 90f63b38503b8af461b23becf85b6d1cd9f19b95
- Full fit archive SHA-256:
  3F7C22C84CF5542A7EA207D8BA59BB0BF0240A54CB33DB04D645C1CCEF4DA25A
- Full fit archive location in the validation workspace:
  repos/amica-capsule/results/figure2_topography/figure2_topography_fit_outputs.npz
- Job resources: one NVIDIA H100, 4 CPUs, 32 GiB, 1 hour
- Job outcome: completed in 3 min 16 s; all five primary configurations and
  three stricter comparator sensitivity fits finished.

figure2_topography_maps.npz contains only the planted and aligned fitted maps
needed to regenerate the figure. The 42 MB full fit archive remains in the
benchmark-results workspace rather than the Overleaf repository. The manifest,
configuration table, source-level matches, sensitivity results, summary table,
and selected-source table retain the data and protocol provenance.

The Slurm log recorded a JAX warning about the installed ptxas 12.6.77
toolchain with CUDA 12.6.2. The job completed and produced finite outputs, but
this environment-specific warning is retained as an unresolved provenance
qualification; the figure is therefore described as a one-fixture audit rather
than universal backend evidence.
