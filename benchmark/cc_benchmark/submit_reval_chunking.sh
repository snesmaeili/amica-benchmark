#!/bin/bash
# S6 re-validation, chunking half: reproduce the PUBLISHED chunking parity pair
# against the released amica.
#
# Why this exists separately from submit_reval_cpu.sh: that job drives
# runner.py, which preprocesses differently from the script that produced the
# published numbers -- 1-100 Hz plus a 60 Hz notch and no average reference,
# against this script's 1-40 Hz with average reference -- and compares two
# separate runs rather than two fits in one process. Those are different data
# and a different metric, so the published pair is not reachable that way.
# Running the original script is the only like-for-like comparison.
#
# Published values (figures/src/main_figure_stats.json -> figure2.chunking,
# from results/mne_chunking/summary.json):
#     absolute_final_ll_difference          5.91657431492365e-07
#     unmixing_frobenius_relative_error     1.448769887359409e-04
#     sensor_unmixing_frobenius_relative_error  1.6841895262134987e-04
# and for reference, published wall times: full 107.6 s, chunked 90.4 s
# (chunked was FASTER, ratio 0.84).
#
# This job asserts nothing. It writes summary.json and the comparison happens
# off-cluster, so the thing being tested does not get to judge itself.
#
# Submit from the repository root on fir:
#   sbatch benchmark/cc_benchmark/submit_reval_chunking.sh
#SBATCH --job-name=amica_reval_chunk
#SBATCH --account=def-kjerbi
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/sesma/scratch/amica_reval_chunk_%j.out
#SBATCH --error=/home/sesma/scratch/amica_reval_chunk_%j.err

set -eo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
OUT="/scratch/${USER}/amica_reval/chunking_${SLURM_JOB_ID:-manual}"
mkdir -p "$OUT"

source "${REPO}/benchmark/cc_benchmark/reval_env.sh"

# Run the released package, and keep the output away from the published
# archive at results/mne_chunking (whose sha256 is recorded in the manuscript
# stats file). Both are belt and braces: the script already defaults a release
# run to a different directory.
export AMICA_USE_RELEASE=1
export AMICA_CHUNKING_OUT="$OUT"

echo "=== provenance ==="
hostname
python -c "import sys; print('python', sys.version.split()[0])"
python -c "import numpy, scipy, mne; print('numpy', numpy.__version__, '| scipy', scipy.__version__, '| mne', mne.__version__)"
python -c "import amica; print('amica  ->', amica.__file__)"
echo "harness commit: $(git -C "$REPO" rev-parse HEAD)"
echo "release commit: $(git -C "${AMICA_RELEASE:-/scratch/$USER/amica_release}" rev-parse HEAD)"
echo

echo "=== full-batch vs chunked, published fixture (1-40 Hz, average reference, 30 PCs, 100 iterations) ==="
cd "$REPO"
python scripts/performance/validate_chunking_mne.py

echo
echo "=== summary.json ==="
cat "${OUT}/summary.json" 2>/dev/null || echo "WARNING: no summary.json produced"

echo
echo "=== outputs ==="
ls -la "$OUT"
echo "DONE"
