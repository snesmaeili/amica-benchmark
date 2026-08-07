#!/bin/bash
# Environment for the pAMICA (sccn/pAMICA) comparator on fir.
#
# Companion to fir_env.sh, which serves the JAX/amica jobs on python/3.11.
# pamica needs python >= 3.12, so it gets its own interpreter and its own venv
# (.venv_pamica, built by setup_pamica.sh) and cannot share that one.
#
# Source this, do not execute it:
#     source pamica_env.sh
#
# Loading cuda/12.6 is NOT optional for GPU runs and the failure is misleading
# without it: torch reports cuda.is_available() == True and nvidia-smi shows the
# H100 with 80 GiB free, but CUDA context creation then fails and every
# allocation -- including a 1 MiB one, and torch.cuda.mem_get_info() itself --
# raises "CUDA error: out of memory". That reads as a memory problem and is not
# one.

_pamica_had_u=0
case $- in *u*) _pamica_had_u=1 ;; esac
set +u

# sbatch and srun give a NON-login shell, where `module` is not defined at all.
# Without this the loads below silently no-op and the job runs against the
# system interpreter.
[ -f /cvmfs/soft.computecanada.ca/config/profile/bash.sh ] && \
  source /cvmfs/soft.computecanada.ca/config/profile/bash.sh

# Caches off $HOME (Alliance quota).
export XDG_CACHE_HOME="/scratch/$USER/.cache"
export XDG_DATA_HOME="/scratch/$USER/.local/share"
export PIP_CACHE_DIR="/scratch/$USER/.cache/pip"
mkdir -p "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$PIP_CACHE_DIR"

_PAMICA_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$_PAMICA_ENV_DIR/env.local" ] && source "$_PAMICA_ENV_DIR/env.local"

module purge
module load StdEnv/2023 || true
module load python/3.12
module load cuda/12.6
module load cudnn

# Dataset paths, shared with fir_env.sh so both read the same recordings.
export BIDS_ROOT_DS4505="${BIDS_ROOT_DS4505:-/project/rrg-kjerbi/datasets/openneuro/ds004505/raw_bids}"

# Thread tuning. pamica's own backend guide measures ~4 intra-op threads as the
# sweet spot on CPU, with 8+ regressing, so honour the allocation rather than
# letting torch grab every core on the node.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

REPO_ROOT="$(cd "$_PAMICA_ENV_DIR/../.." && pwd)"
VENV_PATH="${PAMICA_VENV_DIR:-$REPO_ROOT/.venv_pamica}"
export PAMICA_VENV="$VENV_PATH/bin/python"

# Completeness test, not a directory test: a build that died part-way leaves a
# stub that later jobs would otherwise "reuse". Probe what the runner imports.
if [ -x "$VENV_PATH/bin/python" ] && \
   "$VENV_PATH/bin/python" -c "import torch, numpy, psutil, pamica" >/dev/null 2>&1; then
    source "$VENV_PATH/bin/activate"
else
    echo "ERROR: $VENV_PATH is missing or incomplete." >&2
    echo "       Build it on the LOGIN node: bash $_PAMICA_ENV_DIR/setup_pamica.sh" >&2
    if [ "$_pamica_had_u" = 1 ]; then set -u; fi
    return 1 2>/dev/null || exit 1
fi

# `if`, not `[ ... ] && set -u`: as the last command in a sourced file, that
# form returns 1 whenever the caller did not have -u set, so
# `source pamica_env.sh && python ...` short-circuits and the job silently does
# nothing while looking like a clean module load.
if [ "$_pamica_had_u" = 1 ]; then set -u; fi
