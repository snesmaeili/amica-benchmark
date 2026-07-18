#!/bin/bash
#SBATCH --job-name=amica-parity
#SBATCH --account=def-kjerbi_cpu
#SBATCH --partition=cpubase_bycore_b2
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=parity_%A_%a.out
#SBATCH --error=parity_%A_%a.err

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 scipy-stack
module load gcc/12.3 openmpi/4.1.5 flexiblas/3.3.1

: "${AMICA_BENCHMARK_ROOT:?set AMICA_BENCHMARK_ROOT}"
: "${AMICA_PACKAGE_REPO:?set AMICA_PACKAGE_REPO}"
: "${AMICA_ENV:?set AMICA_ENV to the paper virtual environment}"
: "${PARITY_MANIFEST:?set PARITY_MANIFEST}"
: "${PARITY_OUTPUT_DIR:?set PARITY_OUTPUT_DIR}"
: "${AMICA_FORTRAN_BIN:=/scratch/sesma/fortran_parity/amica17_build/amica17}"

source "$AMICA_ENV/bin/activate"
cd "$AMICA_BENCHMARK_ROOT"
export PYTHONPATH="$AMICA_PACKAGE_REPO:$AMICA_BENCHMARK_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export JAX_PLATFORMS=cpu
export JAX_PLATFORM_NAME=cpu
export JAX_ENABLE_X64=1
export JAX_COMPILATION_CACHE_DIR="$SLURM_TMPDIR/amica_jax_cache"
unset XLA_FLAGS

python scripts/parity/run_manifest_cell.py \
  --manifest "$PARITY_MANIFEST" \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --fortran-binary "$AMICA_FORTRAN_BIN" \
  --output-dir "$PARITY_OUTPUT_DIR"
