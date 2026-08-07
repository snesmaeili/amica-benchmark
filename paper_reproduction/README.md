# paper_reproduction — point-in-time snapshot

> **The repository root is canonical.** This directory is a snapshot of the
> benchmark suite as it stood for an earlier paper-reproduction run, kept for
> history. New work, and anything you intend to run, belongs at the root — see
> [`../README.md`](../README.md). The two will drift; where they disagree, the
> root is correct.

Validation and benchmarking suite for [`amica`](https://github.com/snesmaeili/amica).

Scripts, Slurm job templates, and analysis pipelines used to validate `amica`
against the reference Fortran AMICA 1.7 and to benchmark it against Picard,
extended Infomax, and FastICA on real EEG data.

## Setup

### On Narval (Alliance HPC)

```bash
git clone git@github.com:snesmaeili/amica-benchmark.git
cd amica-benchmark

# Use existing virtual environment
source conf/narval.env

# Install the package being benchmarked
pip install amica

# Verify
make check-env
```

### Local

```bash
pip install -e ".[jax-cpu]"
pip install amica
```

## Benchmarking goals

| Goal | Scripts | Slurm |
|------|---------|-------|
| **Fortran parity** | `scripts/parity/` | `slurm/parity/` |
| **Algorithm comparison** | `scripts/comparison/` | `slurm/comparison/` |
| **CPU/GPU performance** | `scripts/performance/` | `slurm/performance/` |
| **Parameter sensitivity** | `scripts/sensitivity/` | — |
| **Real EEG validation** | `scripts/real_eeg/` | `slurm/real_eeg/` |
| **Paper figures** | `scripts/paper/` | `slurm/paper/` |

## Quick start

```bash
# Run Fortran parity checks locally
make parity

# Run quick real-EEG validation (needs ds004505)
make real-eeg

# Submit 25-subject GPU comparison on Narval
make slurm-comparison

# Generate paper figures (after results are ready)
make paper-all
```

## Directory structure

```
conf/           # HPC config (narval.env, datasets.yaml, paths.py)
docs/           # Research docs (audit reports, validation guides)
scripts/        # Benchmark scripts organized by goal
slurm/          # Slurm job submission scripts
results/        # Output directory (.gitignored except README)
```

## Datasets

- **ds004505**: MoBI dual-layer 120-channel EEG (Studnicki et al. 2022) — `/home/sesma/scratch/ds004505`
- **MNE sample**: Built-in MNE dataset (auto-downloaded)

## Related

- [`amica`](https://github.com/snesmaeili/amica) — the package being benchmarked
- [scott-huberty/amica-benchmark](https://github.com/scott-huberty/amica-benchmark) — reference benchmark repo
