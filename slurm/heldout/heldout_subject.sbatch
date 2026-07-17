#!/bin/bash
# Resources, account, partition, and array are supplied explicitly at sbatch time.
#SBATCH --job-name=heldout-mir
#SBATCH --output=heldout_%A_%a.out
#SBATCH --error=heldout_%A_%a.err

set -euo pipefail
module purge
module load StdEnv/2023 python/3.11 scipy-stack cuda/12.6 cudnn

: "${AMICA_BENCHMARK_ROOT:?set AMICA_BENCHMARK_ROOT}"
: "${AMICA_PACKAGE_REPO:?set AMICA_PACKAGE_REPO}"
: "${AMICA_ENV:?set AMICA_ENV to the paper virtual environment}"
: "${HELDOUT_DATASET:?set HELDOUT_DATASET}"
: "${HELDOUT_N_COMPONENTS:?set HELDOUT_N_COMPONENTS}"
: "${HELDOUT_OUTPUT_DIR:?set HELDOUT_OUTPUT_DIR}"

source "$AMICA_ENV/bin/activate"
cd "$AMICA_BENCHMARK_ROOT"
mkdir -p "$HELDOUT_OUTPUT_DIR/$HELDOUT_DATASET"
export PYTHONPATH="$AMICA_PACKAGE_REPO:$AMICA_BENCHMARK_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export JAX_PLATFORM_NAME=gpu
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME:-$CUDA_ROOT}"

python scripts/heldout/run_heldout_cv.py \
  --dataset "$HELDOUT_DATASET" \
  --subject "$SLURM_ARRAY_TASK_ID" \
  --n-components "$HELDOUT_N_COMPONENTS" \
  --folds 5 \
  --guard-seconds 5 \
  --histogram-bins 50 100 200 \
  --amica-max-iter 3000 \
  --comparator-max-iter 5000 \
  --chunk-size 65536 \
  --device gpu \
  --output "$HELDOUT_OUTPUT_DIR/$HELDOUT_DATASET/sub-${SLURM_ARRAY_TASK_ID}.json"
