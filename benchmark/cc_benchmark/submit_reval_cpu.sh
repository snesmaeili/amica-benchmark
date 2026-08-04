#!/bin/bash
# S6 re-validation, CPU half: full-batch vs chunked agreement and the MNE
# interoperability tests, run against the release candidate.
#
# Purpose is narrow and worth stating: this is NOT a re-benchmark. It exists to
# show that the v0.1.0 release did not move the published numbers. Nothing here
# asserts agreement -- every value is written to JSON and diffed numerically
# against figures/src/main_figure_stats.json on the workstation afterwards.
#
# Submit from the repository root on fir:
#   sbatch benchmark/cc_benchmark/submit_reval_cpu.sh
#SBATCH --job-name=amica_reval_cpu
#SBATCH --account=def-kjerbi
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/home/sesma/scratch/amica_reval_cpu_%j.out
#SBATCH --error=/home/sesma/scratch/amica_reval_cpu_%j.err

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="/scratch/${USER}/amica_reval/cpu_${SLURM_JOB_ID:-manual}"
mkdir -p "$OUT"

source "${REPO}/benchmark/cc_benchmark/fir_env.sh"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

echo "=== provenance ==="
hostname
python -c "import sys; print('python', sys.version.split()[0])"
python -c "import numpy, scipy; print('numpy', numpy.__version__, '| scipy', scipy.__version__)"
python -c "import mne; print('mne', mne.__version__)" || echo "mne unavailable"
git -C "$REPO" rev-parse HEAD
echo

# --- 1. full-batch vs chunked -------------------------------------------------
# Published pair (main_figure_stats.json -> figure2.chunking):
#   absolute_final_ll_difference        5.91657431492365e-07
#   unmixing_frobenius_relative_error   1.448769887359409e-04
echo "=== full-batch vs chunked (MNE sample, 30 PCs, 100 iterations, float64) ==="
for CHUNK in none 1024; do
  ARG=()
  [ "$CHUNK" != "none" ] && ARG=(--chunk-size "$CHUNK")
  python -m amica_python.benchmark.runner \
      --dataset mne --subject 1 --device cpu --backend numpy \
      --n-components 30 --n-iter 100 --dtype float64 \
      "${ARG[@]}" \
      > "${OUT}/chunk_${CHUNK}.json" 2> "${OUT}/chunk_${CHUNK}.log" \
    || echo "chunk=${CHUNK} FAILED (see log)"
done

# --- 2. MNE interoperability --------------------------------------------------
echo "=== MNE interoperability tests ==="
python -m pytest -q "${REPO}/tests" -k "mne or interop" \
    --junitxml="${OUT}/mne_interop.xml" \
    > "${OUT}/mne_interop.log" 2>&1 || echo "pytest reported failures (see log)"

echo
echo "=== outputs ==="
ls -la "$OUT"
echo "DONE"
