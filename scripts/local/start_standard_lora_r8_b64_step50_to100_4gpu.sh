#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/runs/standard-lora-rank8-v1}"
SOURCE_EXP="standard_lora_r8_b64m16n8_50_seed42_v1"
TARGET_EXP="standard_lora_r8_b64m16n8_step50_to100_seed42_v1"
CKPT_ROOT="${RUNTIME_ROOT}/ckpts/verl/DAPO-Math-17k"
SOURCE_CKPT="${CKPT_ROOT}/${SOURCE_EXP}/global_step_50"
TARGET_ROOT="${CKPT_ROOT}/${TARGET_EXP}"
TARGET_CKPT="${TARGET_ROOT}/global_step_50"
LATEST_CKPT_FILE="${TARGET_ROOT}/latest_checkpointed_iteration.txt"

required=(
    "actor/model_world_size_4_rank_0.pt"
    "actor/model_world_size_4_rank_1.pt"
    "actor/model_world_size_4_rank_2.pt"
    "actor/model_world_size_4_rank_3.pt"
    "actor/extra_state_world_size_4_rank_0.pt"
    "actor/extra_state_world_size_4_rank_1.pt"
    "actor/extra_state_world_size_4_rank_2.pt"
    "actor/extra_state_world_size_4_rank_3.pt"
    "actor/peft_adapter/adapter_model.safetensors"
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
    cp -al "${SOURCE_CKPT}" "${TARGET_CKPT}"
fi
for relative_path in "${required[@]}"; do
    if [[ ! -s "${TARGET_CKPT}/${relative_path}" ]]; then
        echo "Incomplete staged checkpoint: ${TARGET_CKPT}/${relative_path}" >&2
        exit 2
    fi
done
if [[ ! -s "${LATEST_CKPT_FILE}" ]]; then
    printf '50\n' >"${LATEST_CKPT_FILE}"
fi

export RUNTIME_ROOT
export EXP_NAME="${TARGET_EXP}"
export CUDA_VISIBLE_DEVICES=0,1,2,3

export PEFT_TYPE=lora
export LORA_RANK=8
export LORA_ALPHA=16
export LORA_DROPOUT=0.05
export LORA_FREEZE_A=False
export LORA_RANK_PATTERN_PATH=null
export LORA_ADAPTER_PATH=null

export TRAIN_PROMPT_BSZ=64
export TRAIN_PROMPT_MINI_BSZ=16
export N_RESP_PER_PROMPT=8
export ACTOR_PPO_MAX_TOKEN_LEN=12288
export INFER_PPO_MAX_TOKEN_LEN=12288
export ROLLOUT_MAX_NUM_SEQS=256
export TOTAL_TRAINING_STEPS=100
export SAVE_FREQ=25
export MAX_ACTOR_CKPT_TO_KEEP=3
export SAVE_CONTENTS="['model','extra']"
export RESUME_MODE=auto
export DATA_SEED=42
export PPO_DATA_LOADER_SEED=42
export ACTOR_SHUFFLE=False

export MODEL_PATH="/data/peft-for-rl-runtime/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base"
export TRAIN_FILE="/data/peft-for-rl-runtime/datasets/verl_dapo/data/dapo-math-17k-boxed.parquet"
export TEST_FILE="${TRAIN_FILE}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_std_r8_50_100}"
export RAY_local_fs_capacity_threshold="${RAY_LOCAL_FS_CAPACITY_THRESHOLD:-0.99}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-peft-for-rl}"
export LOG_DIR="${RUNTIME_ROOT}/logs/verl"
export LOG_FILE="${LOG_DIR}/${TARGET_EXP}.log"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}"
exec >>"${LOG_FILE}" 2>&1
exec conda run --no-capture-output -n "${CONDA_ENV_NAME}" \
    bash "${REPO_ROOT}/examples/verl_train/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh"
