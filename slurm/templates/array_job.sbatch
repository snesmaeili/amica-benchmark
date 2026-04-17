#!/bin/bash
#SBATCH --account=def-kjerbi_gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=1-25
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source conf/narval.env
export JAX_PLATFORMS=cuda

mkdir -p logs

SUBJECT=$(printf "sub-%02d" "$SLURM_ARRAY_TASK_ID")
echo "=== $(date) === Job $SLURM_JOB_ID, task $SLURM_ARRAY_TASK_ID ($SUBJECT) ==="
echo "Script: $SCRIPT"

python -u "$SCRIPT" --subject "$SUBJECT" ${ARGS:-}

echo "=== $(date) === Done ==="
