#!/bin/bash
# Environment for the S6 re-validation jobs.
#
# Deliberately NOT fir_env.sh. That script is sourced by ~28 other submit
# scripts, installs `-e "$REPO_ROOT[all,gpu]"` -- extras this repository does
# not define -- and provisions a venv shared with the historical runs. S6 needs
# the opposite: a self-contained environment with no editable installs, in
# which the algorithm under test is unambiguous.
#
# Why that matters here specifically: this repository vendors its own copy of
# the algorithm under amica_python/, and that copy predates PR #17 (it has no
# rank guard and no _blocked_posteriors). The published numbers came from it,
# so it is kept as the historical record and left alone. The re-validation must
# exercise the *release* instead, so the harness imports `amica` and the two
# packages are separated by path:
#
#   amica                     -> $AMICA_RELEASE   the release under test
#   amica_python.benchmark.*  -> $REPO            the harness
#
# Nothing is pip-installed editable, so PYTHONPATH is authoritative and no
# meta-path finder can silently redirect an import. (A shared venv's editable
# finder is exactly how /scratch/$USER/amica-python/.venv_fir ended up
# resolving amica_python to an unrelated feature-branch checkout.)
#
# Verified on compute node fc30669, job 53256186: numpy 2.4.2, scipy 1.17.0,
# mne 1.12.1, jax 0.9.1 -- matching the published stack -- with amica resolving
# to the release checkout and the rank/chunking fixes both present.
#
# Requires $REPO to be set by the caller (the submit scripts derive it from
# SLURM_SUBMIT_DIR). Set AMICA_REVAL_GPU=1 before sourcing for a GPU job.

# `set -u` must be OFF across the module block: the Compute Canada profile
# reads SKIP_CC_CVMFS while unset (bash.sh:2), and sbatch/srun give a
# non-login shell in which the `module` function is not defined at all.
# Both facts cost a failed job apiece before they were pinned down.
_amica_had_u=0
case $- in *u*) _amica_had_u=1 ;; esac
set +u

[ -f /cvmfs/soft.computecanada.ca/config/profile/bash.sh ] && \
  source /cvmfs/soft.computecanada.ca/config/profile/bash.sh

# Caches off $HOME: the Alliance quota there is small and jobs hit I/O errors.
export XDG_CACHE_HOME="/scratch/$USER/.cache"
export PIP_CACHE_DIR="/scratch/$USER/.cache/pip"
mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR"

module purge
module load StdEnv/2023 python/3.11 scipy-stack
if [ "${AMICA_REVAL_GPU:-0}" = "1" ]; then
  module load cuda/12.6 cudnn
fi

AMICA_REVAL_BASE="${AMICA_REVAL_BASE:-/scratch/$USER/amica_reval}"
AMICA_RELEASE="${AMICA_RELEASE:-/scratch/$USER/amica_release}"
VENV="$AMICA_REVAL_BASE/.venv_reval"
mkdir -p "$AMICA_REVAL_BASE"

# Completeness test, not a directory test. A venv whose build died part-way
# still has bin/activate, so `[ -d "$VENV" ]` reports it as usable and every
# later job "reuses" a stub -- which is precisely how job 53097938 came to fail
# on `import jax` twelve minutes after job 53097937 died mid-bootstrap.
if [ -x "$VENV/bin/python" ] && \
   "$VENV/bin/python" -c "import numpy, scipy, mne, jax, pytest" >/dev/null 2>&1; then
  source "$VENV/bin/activate"
  echo "reval env: reusing complete venv at $VENV"
else
  echo "reval env: building $VENV (incomplete or absent)"
  rm -rf "$VENV"
  virtualenv --no-download "$VENV"
  source "$VENV/bin/activate"
  pip install --no-index --upgrade pip
  # --ignore-installed: scipy-stack supplies numpy/scipy via PYTHONPATH, so pip
  # would call them satisfied and skip them, leaving the venv without its own
  # copies -- which then vanish the moment PYTHONPATH is rewritten.
  pip install --no-index --ignore-installed numpy scipy pandas jax jaxlib
  # Not mirrored in the Compute Canada wheelhouse. Compute nodes on fir do
  # reach PyPI (verified: HTTP/2 200 to pypi.org from fc30669).
  pip install mne mne-icalabel seaborn tabulate pyyaml matplotlib scikit-learn
  # pytest runs the release's own MNE interop suite in the CPU job.
  pip install pytest
fi

if [ ! -d "$AMICA_RELEASE/amica" ]; then
  echo "reval env: FATAL -- no release checkout at $AMICA_RELEASE" >&2
  echo "  git clone https://github.com/snesmaeili/amica.git $AMICA_RELEASE" >&2
  return 1 2>/dev/null || exit 1
fi

# APPEND, never replace: scipy-stack puts entries on PYTHONPATH and clobbering
# it silently removes numpy and scipy from the environment.
export PYTHONPATH="${REPO:?REPO must be set by the caller}:$AMICA_RELEASE:${PYTHONPATH:-}"

export BIDS_ROOT_DS4505="${BIDS_ROOT_DS4505:-/project/rrg-kjerbi/datasets/openneuro/ds004505/raw_bids}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

[ "$_amica_had_u" = 1 ] && set -u
unset _amica_had_u
