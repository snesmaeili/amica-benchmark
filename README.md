# amica-benchmark

Validation, benchmarking, and manuscript-reproduction materials for
[`amica`](https://github.com/snesmaeili/amica) — a Python/JAX implementation of
AMICA for MNE-Python.

This repository holds the scripts, Slurm jobs, and analysis pipelines used to
validate `amica` against the reference Fortran AMICA 1.7, to compare it with
other AMICA implementations and with Picard, extended Infomax, and FastICA on
real EEG, and to generate every figure and table in the preprint.

## Start here

| If you want to… | Go to |
|---|---|
| Compare `amica` against **other AMICA implementations** | `scripts/comparison/runners/` — one runner each for pyAMICA, neuromechanist/pyAMICA, Scott Huberty's amica-python, and `amica`; `benchmark/comparator/runners/` adds the Fortran runner |
| Compare against **Picard / Infomax / FastICA** | `benchmark/cc_benchmark/submit_{picard,infomax,fastica}_cpu_v3.sh` |
| Check **Fortran AMICA 1.7 parity** | `scripts/parity/` — adapters for Fortran, pyAMICA and `amica`, plus metrics and the run manifest. The patched reference source and its Docker build are in `fortran/` |
| **Regenerate a manuscript figure or table** | `scripts/paper/figures/` — see its README |
| Run the **multi-model** benchmark | `scripts/multimodel/` |
| Reproduce a **cluster campaign** | `benchmark/cc_benchmark/` and `slurm/` |
| Get the **data** | `benchmark/cc_benchmark/download_ds00450{4,5}.sh`, `download_ds004621.sh` |

## Install

```bash
pip install amica            # the package being benchmarked
pip install -e ".[jax-cpu]"  # this repository's helpers
```

On an Alliance/Compute Canada cluster, `source conf/narval.env` first, then
`make check-env`.

> **Version note.** The preprint's results were produced with `amica` 0.0.1.
> Version 0.1.0 changed chunked multi-model fitting and added rank estimation;
> see the manuscript's availability statement for what that does and does not
> affect. Reproduction should use the dependency versions recorded in the
> archived release.

## Datasets

Three openly available OpenNeuro datasets, downloaded by the scripts above:

| Accession | Recording | Reference |
|---|---|---|
| `ds004505` | Table tennis, 120 channels | Studnicki et al. 2022 |
| `ds004504` | Eyes-closed rest, 19 channels | Miltiadous et al. 2023 |
| `ds004621` | Eyes-open rest, 127 channels | Dzianok et al. 2022 |

## What is here, and what is not

Committed here: every producer script, the Slurm jobs, the run manifests, and
the **aggregated** results those scripts consume — roughly 20 MB, so a clone
stays usable.

Not committed: the bulk fitted objects (`.npy`/`.npz`), the raw per-run trees,
and the large multi-model fits (~485 MB alone). Those live in the Zenodo record
that accompanies the paper. Producers that need them **refuse to run and name
the missing tree** rather than emit a partial table; point `AMICA_BENCH_DATA` at
an extracted archive to supply them. Five of the eight manuscript tables
regenerate byte-identically from a bare clone with no extra data.

## Layout

```
amica_python/    vendored package snapshot used by some benchmark entry points
benchmark/       cluster campaigns (cc_benchmark), cross-implementation
                 comparator, MNE synthetic fixtures, figure renderers
conf/            HPC config (narval.env, datasets.yaml, paths.py)
docs/            audit reports and validation guides
figdata/         small figure inputs
fortran/         patched AMICA 1.7 source, Dockerfile, build script
manifests/       parity and multi-model run manifests
phaseB_figures/  phase-B runtime and robustness inputs
results/         aggregated results and emitted figures
scripts/         comparison, parity, performance, multimodel, paper figures
slurm/           job submission scripts by campaign
tests/           unit tests for the benchmark tooling
```

## License

BSD-3-Clause, matching [`amica`](https://github.com/snesmaeili/amica) — see
`LICENSE`. The vendored Fortran in `fortran/` remains under its upstream terms;
see `fortran/LICENSE.upstream`.

## Related

- [`amica`](https://github.com/snesmaeili/amica) — the package being benchmarked
- [scott-huberty/amica-benchmark](https://github.com/scott-huberty/amica-benchmark) — an independent benchmark repository
