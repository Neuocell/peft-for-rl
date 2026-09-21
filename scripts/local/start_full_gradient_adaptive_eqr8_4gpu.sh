#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/runs}"
CANDIDATE_METHOD="${CANDIDATE_METHOD:-mean}"
ALLOCATION_DIR="${ALLOCATION_DIR:-${RUNTIME_ROOT}/analysis/full_gradient_signed_grpo_probe_v1_seed42/${CANDIDATE_METHOD}_adaptive_eqr8}"
export EXP_NAME="${EXP_NAME:-full_gradient_${CANDIDATE_METHOD}_adaptive_eqr8_50_seed42}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PEFT_TYPE=grad_subspace
# vLLM uses the global rank as its capacity ceiling; PEFT modules use rank_pattern.
export LORA_RANK=32
export LORA_ALPHA=64
export LORA_DROPOUT=0.0
export LORA_FREEZE_A=False
export GRADIENT_SUBSPACE_RANK_MAP_PATH="${GRADIENT_SUBSPACE_RANK_MAP_PATH:-${ALLOCATION_DIR}/rank_map.json}"
export GRADIENT_SUBSPACE_PATH="${GRADIENT_SUBSPACE_PATH:-${ALLOCATION_DIR}/subspaces.safetensors}"
export GRADIENT_SUBSPACE_SCALING=2.0
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-50}"
export SAVE_FREQ="${SAVE_FREQ:-25}"
export RESUME_MODE="${RESUME_MODE:-disable}"
export DATA_SEED="${DATA_SEED:-42}"
export PPO_DATA_LOADER_SEED="${PPO_DATA_LOADER_SEED:-42}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-peft-for-rl}"
export LOG_DIR="${LOG_DIR:-${RUNTIME_ROOT}/logs/verl}"
export LOG_FILE="${LOG_FILE:-${LOG_DIR}/${EXP_NAME}.log}"

if [[ ! -s "${GRADIENT_SUBSPACE_RANK_MAP_PATH}" ]] || [[ ! -s "${GRADIENT_SUBSPACE_PATH}" ]]; then
    echo "Missing full-gradient adaptive allocation under ${ALLOCATION_DIR}" >&2
    exit 2
fi
mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}"
exec >>"${LOG_FILE}" 2>&1
exec conda run --no-capture-output -n "${CONDA_ENV_NAME}" \
    bash "${REPO_ROOT}/examples/verl_train/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh"
