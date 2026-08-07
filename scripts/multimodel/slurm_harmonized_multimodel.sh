#!/bin/bash
# One manifest row per H100 Slurm array task. Submit explicitly with sbatch.
set -euo pipefail

REPO_ROOT="${AMICA_MM_REPO_ROOT:-/scratch/${USER}/amica-mm}"
RUNNER_DIR="${CMIR_RUNNER_DIR:-/scratch/${USER}/amica-cmir-workflow}"
MANIFEST="${MANIFEST:?set MANIFEST to the absolute CSV path}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR to the absolute result directory}"
EXPECTED_PACKAGE_COMMIT="${EXPECTED_PACKAGE_COMMIT:?pin the deployed amica Git commit}"
N_COMPONENTS="${N_COMPONENTS:-15}"
NUM_MIX="${NUM_MIX:-3}"
DURATION_SEC="${DURATION_SEC:-600}"
SFREQ="${SFREQ:-250}"
MAX_ITER="${MAX_ITER:-2000}"
CHUNK_SIZE="${CHUNK_SIZE:-65536}"

cd "${REPO_ROOT}/scripts/cc_benchmark"
source fir_env.sh

task_id="${SLURM_ARRAY_TASK_ID:?this script must run as a Slurm array}"
row_number=$((task_id + 2))
row="$(sed -n "${row_number}p" "${MANIFEST}" | tr -d '\r')"
if [[ -z "${row}" ]]; then
    echo "No manifest row for zero-based task ${task_id}" >&2
    exit 2
fi
IFS=',' read -r dataset subject num_models fit_seed surrogate surrogate_seed <<< "${row}"

mkdir -p "${OUTPUT_DIR}"
echo "manifest=${MANIFEST} task=${task_id} dataset=${dataset} subject=${subject} M=${num_models} fit_seed=${fit_seed} surrogate=${surrogate} surrogate_seed=${surrogate_seed}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "n_components=${N_COMPONENTS} K=${NUM_MIX} duration=${DURATION_SEC}s sfreq=${SFREQ} max_iter=${MAX_ITER} chunk_size=${CHUNK_SIZE}"

export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
python "${RUNNER_DIR}/run_harmonized_multimodel.py" \
    --manifest-path "${MANIFEST}" \
    --manifest-row-index "${task_id}" \
    --expected-package-commit "${EXPECTED_PACKAGE_COMMIT}" \
    --dataset "${dataset}" \
    --subject "${subject}" \
    --num-models "${num_models}" \
    --fit-seed "${fit_seed}" \
    --surrogate "${surrogate}" \
    --surrogate-seed "${surrogate_seed}" \
    --n-components "${N_COMPONENTS}" \
    --num-mix "${NUM_MIX}" \
    --duration-sec "${DURATION_SEC}" \
    --sfreq "${SFREQ}" \
    --max-iter "${MAX_ITER}" \
    --chunk-size "${CHUNK_SIZE}" \
    --min-effective-n 2000 \
    --min-posterior-mass 2000 \
    --posterior-window-sec 5 \
    --transition-buffer-sec 30 \
    --output-dir "${OUTPUT_DIR}"
