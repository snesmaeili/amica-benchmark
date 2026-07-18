#!/bin/bash
#SBATCH --job-name=amica-parity-diag
#SBATCH --account=def-kjerbi_cpu
#SBATCH --partition=cpubase_bycore_b2
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=parity_diag_%j.out
#SBATCH --error=parity_diag_%j.err

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 scipy-stack
module load gcc/12.3 openmpi/4.1.5 flexiblas/3.3.1

: "${AMICA_BENCHMARK_ROOT:?set AMICA_BENCHMARK_ROOT}"
: "${AMICA_PACKAGE_REPO:?set AMICA_PACKAGE_REPO}"
: "${AMICA_ENV:?set AMICA_ENV to the paper virtual environment}"
: "${PARITY_MANIFEST:?set PARITY_MANIFEST}"
: "${PARITY_DIAGNOSTIC_OUTPUT:?set PARITY_DIAGNOSTIC_OUTPUT}"
: "${AMICA_FORTRAN_BIN:=/scratch/sesma/fortran_parity/amica17_build/amica17}"
: "${PARITY_TASK_INDEX:=1}"
: "${PARITY_CHECKPOINTS:=1,2,3,10,20,49,50,51,60,100,200}"

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

python scripts/parity/diagnose_manifest_cell.py \
  --manifest "$PARITY_MANIFEST" \
  --task-index "$PARITY_TASK_INDEX" \
  --fortran-binary "$AMICA_FORTRAN_BIN" \
  --checkpoints "$PARITY_CHECKPOINTS" \
  --output "$PARITY_DIAGNOSTIC_OUTPUT"
