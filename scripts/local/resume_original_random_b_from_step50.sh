#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/runs/random-b-original-repro-v1}"
PROBE_EXP_NAME="rl_gradient_probe_b16n8_w8_cap64_e95_s6to12_seed42_original_repro_v1"
TRAIN_EXP_NAME="gradtop_probe12_r8to32_original_random_b_a2_b64m16n8_270_seed42_repro_v1"
CKPT_ROOT="${RUNTIME_ROOT}/ckpts/verl/DAPO-Math-17k/${TRAIN_EXP_NAME}"
STEP50="${CKPT_ROOT}/global_step_50"

required=(
    "actor/model_world_size_4_rank_0.pt"
    "actor/model_world_size_4_rank_1.pt"
    "actor/model_world_size_4_rank_2.pt"
    "actor/model_world_size_4_rank_3.pt"
    "actor/optim_world_size_4_rank_0.pt"
    "actor/optim_world_size_4_rank_1.pt"
    "actor/optim_world_size_4_rank_2.pt"
    "actor/optim_world_size_4_rank_3.pt"
    "actor/extra_state_world_size_4_rank_0.pt"
    "actor/extra_state_world_size_4_rank_1.pt"
    "actor/extra_state_world_size_4_rank_2.pt"
    "actor/extra_state_world_size_4_rank_3.pt"
    "data.pt"
)
for relative_path in "${required[@]}"; do
    if [[ ! -s "${STEP50}/${relative_path}" ]]; then
        echo "Incomplete step-50 checkpoint: ${STEP50}/${relative_path}" >&2
        exit 2
    fi
done
if [[ "$(tr -d '[:space:]' <"${CKPT_ROOT}/latest_checkpointed_iteration.txt")" != "50" ]]; then
    echo "Refusing to resume: latest checkpoint marker is not 50" >&2
    exit 2
fi

RUNTIME_ROOT="${RUNTIME_ROOT}" \
EXP_NAME="${TRAIN_EXP_NAME}" \
MODEL_PATH="/data/peft-for-rl-runtime/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base" \
TRAIN_FILE="/data/peft-for-rl-runtime/datasets/verl_dapo/data/dapo-math-17k-boxed.parquet" \
TEST_FILE="/data/peft-for-rl-runtime/datasets/verl_dapo/data/dapo-math-17k-boxed.parquet" \
CKPTS_DIR="${CKPT_ROOT}" \
GRADIENT_PROBE_ARTIFACT_DIR="${RUNTIME_ROOT}/analysis/${PROBE_EXP_NAME}" \
RAY_TEMP_DIR="${TRAIN_RAY_TEMP_DIR:-/root/rbrt}" \
TRAIN_PROMPT_BSZ=64 \
TRAIN_PROMPT_MINI_BSZ=16 \
N_RESP_PER_PROMPT=8 \
TOTAL_TRAINING_STEPS=270 \
SAVE_FREQ=50 \
MAX_ACTOR_CKPT_TO_KEEP=2 \
SAVE_CONTENTS="['model','optimizer','extra']" \
RESUME_MODE=auto \
DATA_SEED=42 \
PPO_DATA_LOADER_SEED=42 \
bash "${SCRIPT_DIR}/start_gradient_subspace_init_4gpu.sh"
