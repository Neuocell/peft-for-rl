#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/runs/windowed-adam-consensus-v1}"
CANDIDATE_METHOD="${CANDIDATE_METHOD:-consensus_hybrid}"
PROBE_DIR="${PROBE_DIR:-${RUNTIME_ROOT}/analysis/windowed_adam_consensus_probe_b64n8_seed42}"
ALLOCATION_DIR="${ALLOCATION_DIR:-${PROBE_DIR}/${CANDIDATE_METHOD}_uniform_r8_future_lcb}"
export EXP_NAME="${EXP_NAME:-windowed_${CANDIDATE_METHOD}_uniform_r8_b64_50_seed42}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PEFT_TYPE=grad_subspace
export LORA_RANK=8
export LORA_ALPHA=16
export LORA_DROPOUT=0.0
export LORA_FREEZE_A=False
export GRADIENT_SUBSPACE_RANK_MAP_PATH="${ALLOCATION_DIR}/rank_map.json"
export GRADIENT_SUBSPACE_PATH="${ALLOCATION_DIR}/subspaces.safetensors"
export GRADIENT_SUBSPACE_SCALING=2.0
export TRAIN_PROMPT_BSZ=64
export TRAIN_PROMPT_MINI_BSZ=16
export N_RESP_PER_PROMPT=8
export ACTOR_PPO_MAX_TOKEN_LEN=12288
export INFER_PPO_MAX_TOKEN_LEN=12288
export ROLLOUT_MAX_NUM_SEQS=256
export TOTAL_TRAINING_STEPS=50
export SAVE_FREQ=25
export RESUME_MODE=disable
export DATA_SEED=42
export PPO_DATA_LOADER_SEED=42
export ACTOR_SHUFFLE=False
export SAVE_CONTENTS="['model','optimizer','extra']"
export MODEL_PATH="${MODEL_PATH:-/data/peft-for-rl-runtime/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base}"
export TRAIN_FILE="${TRAIN_FILE:-/data/peft-for-rl-runtime/datasets/verl_dapo/data/dapo-math-17k-boxed.parquet}"
export TEST_FILE="${TEST_FILE:-${TRAIN_FILE}}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray-wac-v1}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-peft-for-rl}"
export LOG_DIR="${LOG_DIR:-${RUNTIME_ROOT}/logs/verl}"
export LOG_FILE="${LOG_FILE:-${LOG_DIR}/${EXP_NAME}.log}"

if [[ ! -s "${GRADIENT_SUBSPACE_RANK_MAP_PATH}" ]] || [[ ! -s "${GRADIENT_SUBSPACE_PATH}" ]]; then
    echo "Missing windowed uniform-r8 allocation under ${ALLOCATION_DIR}" >&2
    exit 2
fi
mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}"
exec >>"${LOG_FILE}" 2>&1
exec conda run --no-capture-output -n "${CONDA_ENV_NAME}" \
    bash "${REPO_ROOT}/examples/verl_train/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh"
