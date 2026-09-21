#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/runs}"
export EXP_NAME="${EXP_NAME:-spar_v0_positive_probe_uniform_r8_50_seed42}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-peft-for-rl}"
LOG_DIR="${LOG_DIR:-${RUNTIME_ROOT}/logs/verl}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${EXP_NAME}.log}"
mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}"
exec >>"${LOG_FILE}" 2>&1
exec conda run --no-capture-output -n "${CONDA_ENV_NAME}" \
    bash "${REPO_ROOT}/examples/verl_train/run_dapo_math_boxed_spar_uniform_r8_1p5b_4gpu_8k.sh"
