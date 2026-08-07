#!/bin/bash
# Build .venv_competitors with the 3 external AMICA reimplementations used by the
# cross-implementation MEMORY comparison (benchmark/comparator). Run ONCE per site.
#
#   pyamica  (DerAndereJohannes)  PyTorch     -> import pyamica
#   amica    (scott-huberty)      PyTorch     -> import amica
#   pyAMICA  (sccn/pAMICA @526aa32, the March-2025 pre-rename snapshot
#             formerly at neuromechanist/pyAMICA)  pure NumPy  -> import pyAMICA
#
# Current pAMICA (v0.3.1, package `pamica`) is NOT here: it needs Python >= 3.12.
# It gets its own venv -> see setup_pamica.sh.
#
# git+pip needs internet, so run this on the LOGIN node — it is a one-time ENV BUILD
# (not compute), which is the supported use of pip on a login node. On Alliance the
# `--no-index torch` wheel is CUDA-capable and serves both the CPU and GPU comparisons.
#
#   bash setup_competitors.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # benchmark/cc_benchmark/
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
VENV="${COMPETITORS_VENV_DIR:-$REPO_ROOT/.venv_competitors}"

# Caches off $HOME (Alliance quota), mirroring fir_env.sh.
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/scratch/$USER/.cache/pip}"
mkdir -p "$PIP_CACHE_DIR"

module purge 2>/dev/null || true
module load StdEnv/2023 python/3.11 2>/dev/null || true

if [ ! -d "$VENV" ]; then
    echo "Creating venv at $VENV ..."
    virtualenv --no-download "$VENV" 2>/dev/null || python -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --upgrade pip

# torch: Alliance wheelhouse wheel (--no-index) is CUDA-enabled and works for BOTH the CPU
# and GPU comparisons; fall back to PyPI off-Alliance.
echo "Installing torch ..."
pip install --no-index torch 2>/dev/null || pip install torch

# The competitor AMICA implementations, pinned to commits.
#
# The pins are not cosmetic. github.com/neuromechanist/pyAMICA was renamed and
# transferred to github.com/sccn/pAMICA (same GitHub repository id, 301 redirect),
# so the unpinned URL below now resolves to that project's current state: package
# `pamica`, which provides no `import pyAMICA` and needs Python >= 3.12. Left
# floating, this line silently stopped installing what the paper measured and
# would fail the verify step at the bottom of this script.
#
# 526aa32 is the March-2025 pure-NumPy state actually benchmarked. Current pAMICA
# is a separate environment with its own interpreter; see setup_pamica.sh.
echo "Installing the competitor implementations (pinned) ..."
pip install "git+https://github.com/DerAndereJohannes/pyamica.git"
pip install "git+https://github.com/scott-huberty/amica-python.git"
pip install "git+https://github.com/sccn/pAMICA.git@526aa32"

# Optional: NVML neutral cross-check for the GPU comparison (enable with AMICA_MEM_NVML=1).
pip install nvidia-ml-py 2>/dev/null || echo "(nvidia-ml-py not installed; NVML cross-check stays off)"

echo "=== verify imports ==="
python -c "import torch, pyamica, amica, pyAMICA; print('OK — torch', torch.__version__, '| pyamica + amica(scott) + pyAMICA(neuromechanist) all import')"
echo "competitors venv ready: $VENV"
