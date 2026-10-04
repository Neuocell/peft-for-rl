#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/runs/full-gradient-v1}"
SOURCE_EXP="full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_to100_seed42"
TARGET_EXP="full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step100_to200_warm_seed42"
CKPT_ROOT="${RUNTIME_ROOT}/ckpts/verl/DAPO-Math-17k"
SOURCE_CKPT="${CKPT_ROOT}/${SOURCE_EXP}/global_step_100"
TARGET_ROOT="${CKPT_ROOT}/${TARGET_EXP}"
TARGET_CKPT="${TARGET_ROOT}/global_step_100"
LATEST_CKPT_FILE="${TARGET_ROOT}/latest_checkpointed_iteration.txt"

required=(
    "actor/peft_adapter/adapter_model.safetensors"
    "actor/extra_state_world_size_4_rank_0.pt"
    "actor/extra_state_world_size_4_rank_1.pt"
    "actor/extra_state_world_size_4_rank_2.pt"
    "actor/extra_state_world_size_4_rank_3.pt"
    "data.pt"
)
for relative_path in "${required[@]}"; do
    if [[ ! -s "${SOURCE_CKPT}/${relative_path}" ]]; then
        echo "Incomplete source checkpoint: ${SOURCE_CKPT}/${relative_path}" >&2
        exit 2
    fi
done
if [[ ! -d "${TARGET_CKPT}" ]]; then
    mkdir -p "${TARGET_ROOT}"
    cp -a "${SOURCE_CKPT}" "${TARGET_CKPT}"
fi
for relative_path in "${required[@]}"; do
    if [[ ! -s "${TARGET_CKPT}/${relative_path}" ]]; then
        echo "Incomplete staged checkpoint: ${TARGET_CKPT}/${relative_path}" >&2
        exit 2
    fi
done
if [[ ! -s "${LATEST_CKPT_FILE}" ]]; then
    printf '100\n' >"${LATEST_CKPT_FILE}"
fi

export RUNTIME_ROOT
export EXP_NAME="${TARGET_EXP}"
export ALLOCATION_DIR="${RUNTIME_ROOT}/analysis/full_gradient_signed_grpo_probe_r32_d16c16a8_seed42/mean_uniform_r8"
export MODEL_PATH="/data/peft-for-rl-runtime/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base"
export TRAIN_FILE="/data/peft-for-rl-runtime/datasets/verl_dapo/data/dapo-math-17k-boxed.parquet"
export TEST_FILE="${TRAIN_FILE}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_fg_r8_100_200_warm}"
export LOG_DIR="${RUNTIME_ROOT}/logs/verl"
export LOG_FILE="${LOG_DIR}/${TARGET_EXP}.log"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TRAIN_PROMPT_BSZ=64
export TRAIN_PROMPT_MINI_BSZ=16
export N_RESP_PER_PROMPT=8
export ACTOR_PPO_MAX_TOKEN_LEN=12288
export INFER_PPO_MAX_TOKEN_LEN=12288
export ROLLOUT_MAX_NUM_SEQS=256
export LORA_RANK=32
export LORA_ALPHA=64
export TOTAL_TRAINING_STEPS=200
export SAVE_FREQ=25
export MAX_ACTOR_CKPT_TO_KEEP=5
export LORA_ADAPTER_PATH="${TARGET_CKPT}/actor/peft_adapter"
export RESUME_MODE=disable
export WARM_START_GLOBAL_STEP=100
export WARM_START_DATA_PATH="${TARGET_CKPT}/data.pt"
export DATA_SEED=42
export PPO_DATA_LOADER_SEED=42

exec bash "${SCRIPT_DIR}/start_full_gradient_uniform_r8_4gpu.sh"
