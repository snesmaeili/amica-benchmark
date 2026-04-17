#!/bin/bash
#SBATCH --job-name=3way-parity
#SBATCH --account=def-kjerbi
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/3way-parity-%j.out
#SBATCH --error=logs/3way-parity-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source conf/narval.env
export JAX_PLATFORMS=cpu

mkdir -p logs results/parity

echo "=== $(date) === Job $SLURM_JOB_ID on $(hostname) ==="
python -u scripts/parity/three_way_parity.py --levels 1,2,3,4,5 --l5-iters 500
echo "=== $(date) === Done ==="
