#!/bin/bash
# S6 re-validation, GPU half: representative ds004505 fits at the published
# configuration, run against the release candidate.
#
# Three subjects, not twenty-five. Enough to detect a shift in the published
# numbers, deliberately not a re-benchmark -- the full cohort is already
# archived and re-running it would neither add evidence nor be a good use of
# the allocation.
#
# Results are written to JSON and diffed numerically against the published
# per-subject values on the workstation afterwards. Nothing is asserted here.
#
# Submit from the repository root on fir:
#   sbatch benchmark/cc_benchmark/submit_reval_gpu.sh
#SBATCH --job-name=amica_reval_gpu
#SBATCH --account=def-kjerbi_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/home/sesma/scratch/amica_reval_gpu_%j.out
#SBATCH --error=/home/sesma/scratch/amica_reval_gpu_%j.err

set -euo pipefail

# Slurm copies the submitted script into a spool directory, so BASH_SOURCE
# points at /localscratch/spool/... and not at the repository. Use the
# directory the job was submitted from instead.
REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
OUT="/scratch/${USER}/amica_reval/gpu_${SLURM_JOB_ID:-manual}"
mkdir -p "$OUT"

# reval_env.sh, not fir_env.sh -- see submit_reval_cpu.sh for why.
export AMICA_REVAL_GPU=1
source "${REPO}/benchmark/cc_benchmark/reval_env.sh"

echo "=== provenance ==="
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import sys; print('python', sys.version.split()[0])"
python -c "import jax; print('jax', jax.__version__, '| devices', jax.devices())"
# Which algorithm actually ran -- see submit_reval_cpu.sh.
python -c "import amica; print('amica  ->', amica.__file__)"
python -c "import amica_python.benchmark.runner as r; print('harness->', r.__file__)"
echo "harness commit: $(git -C "$REPO" rev-parse HEAD)"
echo "release commit: $(git -C "${AMICA_RELEASE:-/scratch/$USER/amica_release}" rev-parse HEAD)"
echo

# Published configuration for the ds004505 single-model benchmark:
#   64 retained PCs, 3,000-iteration cap, seed 42, M=1, K=3, float64.
# Subjects chosen as the first three of the 25-participant cohort.
for SUBJ in 1 2 3; do
  echo "=== ds004505 sub-$(printf '%02d' "$SUBJ"), 64 PCs, 3000 iterations, JAX-GPU ==="
  python -m amica_python.benchmark.runner \
      --dataset ds004505 --subject "$SUBJ" \
      --device gpu --backend jax \
      --n-components 64 --n-iter 3000 --dtype float64 \
      > "${OUT}/ds004505_sub-$(printf '%02d' "$SUBJ").json" \
      2> "${OUT}/ds004505_sub-$(printf '%02d' "$SUBJ").log" \
    || echo "subject ${SUBJ} FAILED (see log)"
done

echo
echo "=== outputs ==="
ls -la "$OUT"
echo "DONE"
