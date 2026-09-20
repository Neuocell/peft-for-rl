#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/runs}"
export EXP_NAME="${EXP_NAME:-oracle_justrl_stablerank_ceil_rmean15p87_rmax44_a2x_b64m16n8_270_4gpu_boxed_v1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export DRY_RUN="${DRY_RUN:-1}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-peft-for-rl}"
LOG_DIR="${LOG_DIR:-${RUNTIME_ROOT}/logs/verl}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${EXP_NAME}.log}"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not available on PATH" >&2
    exit 2
fi

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Repository: ${REPO_ROOT}"
echo "Runtime:    ${RUNTIME_ROOT}"
echo "Conda env:  ${CONDA_ENV_NAME}"
echo "Log:        ${LOG_FILE}"
echo "DRY_RUN:    ${DRY_RUN}"

exec conda run --no-capture-output -n "${CONDA_ENV_NAME}" \
    bash "${REPO_ROOT}/examples/verl_train/run_dapo_math_boxed_oracle_rank_lora_1p5b_4gpu_8k.sh"
