#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/runs}"
export EXP_NAME="${EXP_NAME:-rl_gradient_probe_stable_snr_w5x3_c2_v2_pbudget16_v3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-peft-for-rl}"
LOG_DIR="${LOG_DIR:-${RUNTIME_ROOT}/logs/verl}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${EXP_NAME}.log}"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}"
exec >>"${LOG_FILE}" 2>&1

echo "Repository: ${REPO_ROOT}"
echo "Runtime:    ${RUNTIME_ROOT}"
echo "Conda env:  ${CONDA_ENV_NAME}"
echo "Log:        ${LOG_FILE}"

exec conda run --no-capture-output -n "${CONDA_ENV_NAME}" \
    bash "${REPO_ROOT}/examples/verl_train/run_dapo_math_boxed_gradient_probe_stable_snr_1p5b_4gpu_8k.sh"
