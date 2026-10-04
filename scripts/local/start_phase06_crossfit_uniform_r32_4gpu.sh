#!/usr/bin/env bash
set -euo pipefail

if (( $# > 1 )); then
    echo "Usage: $0 [P0|P2|C0|C2]" >&2
    exit 2
fi
METHOD="${1:-${METHOD:-}}"
case "${METHOD}" in
    P0|P2|C0|C2) ;;
    *) echo "A Phase-0.6 method is required: P0, P2, C0 or C2" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/node/anaconda3/envs/peft-for-rl/bin/python}"
SOURCE_ARTIFACT_DIR="${SOURCE_ARTIFACT_DIR:-${REPO_ROOT}/runs/phase0-gradient-diagnostics-v1/analysis/phase05_hybrid_shared_rollout_d64c32a16_r32_seed42_v3}"
REPLAY_ARTIFACT_DIR="${REPLAY_ARTIFACT_DIR:-${REPO_ROOT}/runs/phase0-gradient-diagnostics-v1/analysis/phase06_crossfit_replay_d64c32a16_r32_seed42_v1}"
ALLOCATION_DIR="${REPLAY_ARTIFACT_DIR}/training_allocations/${METHOD}_uniform_r32"
TRAIN_SEED="${TRAIN_SEED:-42}"
METHOD_LOWER="${METHOD,,}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}"
"${PYTHON_BIN}" scripts/analysis/prepare_phase06_training_artifacts.py \
    --source-artifact "${SOURCE_ARTIFACT_DIR}" \
    --replay-artifact "${REPLAY_ARTIFACT_DIR}" \
    --verify-method "${METHOD}"

export RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/runs/phase06-crossfit-training-v1}"
export EXP_NAME="${EXP_NAME:-phase06_${METHOD_LOWER}_uniform_r32_b64m16n8_step50_seed${TRAIN_SEED}_v1}"
export MODEL_PATH="${MODEL_PATH:-/data/peft-for-rl-runtime/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base}"
export TRAIN_FILE="${TRAIN_FILE:-/data/peft-for-rl-runtime/datasets/verl_dapo/data/dapo-math-17k-boxed.parquet}"
export TEST_FILE="${TEST_FILE:-${TRAIN_FILE}}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray-phase06-${METHOD_LOWER}-seed${TRAIN_SEED}-v1}"
export LOG_DIR="${LOG_DIR:-${RUNTIME_ROOT}/logs/verl}"
export LOG_FILE="${LOG_FILE:-${LOG_DIR}/${EXP_NAME}.log}"
export CKPTS_DIR="${CKPTS_DIR:-${RUNTIME_ROOT}/ckpts/verl/DAPO-Math-17k/${EXP_NAME}}"
if [[ -d "${CKPTS_DIR}" ]] && [[ -n "$(find "${CKPTS_DIR}" -mindepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite an existing Phase-0.6 training directory: ${CKPTS_DIR}" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export ALLOCATION_DIR
export PEFT_TYPE=grad_subspace
export LORA_RANK=32
export LORA_ALPHA=64
export LORA_DROPOUT=0.0
export LORA_FREEZE_A=False
export GRADIENT_SUBSPACE_RANK_MAP_PATH="${ALLOCATION_DIR}/rank_map.json"
export GRADIENT_SUBSPACE_PATH="${ALLOCATION_DIR}/subspaces.safetensors"
export GRADIENT_SUBSPACE_SCALING=2.0
export LR=1e-6
export LR_WARMUP_STEPS=0
export LR_SCHEDULER_TYPE=constant
export WEIGHT_DECAY=0
export PPO_EPOCHS=1
export CLIP_RATIO_LOW=0.2
export CLIP_RATIO_HIGH=0.28
export USE_KL_LOSS=False
export KL_LOSS_COEF=0.0
export TEMPERATURE=1.0
export TOP_P=1.0
export TOP_K=-1
export TRAIN_PROMPT_BSZ=64
export TRAIN_PROMPT_MINI_BSZ=16
export N_RESP_PER_PROMPT=8
export ACTOR_PPO_MAX_TOKEN_LEN=12288
export INFER_PPO_MAX_TOKEN_LEN=12288
export ROLLOUT_MAX_NUM_SEQS=256
export TOTAL_TRAINING_STEPS=50
export SAVE_FREQ=25
export MAX_ACTOR_CKPT_TO_KEEP=2
export SAVE_CONTENTS="['model','optimizer','extra']"
export RESUME_MODE=disable
export TRAIN_SHUFFLE=True
export ACTOR_SHUFFLE=False
export DATA_SEED="${TRAIN_SEED}"
export PPO_DATA_LOADER_SEED="${TRAIN_SEED}"
export ROLLOUT_SEED="${TRAIN_SEED}"

exec bash "${SCRIPT_DIR}/start_full_gradient_uniform_r8_4gpu.sh"
