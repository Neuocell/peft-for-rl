#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-${SCRIPT_DIR}/../../runs}"

export PEFT_TYPE=tinylora
export EXP_NAME="${EXP_NAME:-tinylora_r2u1_tiled16_13p_lr2e4_b64m16n8_50_4gpu_boxed}"
export CKPTS_DIR="${CKPTS_DIR:-${RUNTIME_ROOT}/ckpts/verl/DAPO-Math-17k/${EXP_NAME}}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-${HOME}/rtiny}"
export RAY_local_fs_capacity_threshold="${RAY_LOCAL_FS_CAPACITY_THRESHOLD:-0.995}"

export TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-64}"
export TRAIN_PROMPT_MINI_BSZ="${TRAIN_PROMPT_MINI_BSZ:-16}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-8}"
export PPO_EPOCHS="${PPO_EPOCHS:-1}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-50}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-10}"
export RESUME_MODE="${RESUME_MODE:-disable}"

export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
export ACTOR_PPO_MAX_TOKEN_LEN="${ACTOR_PPO_MAX_TOKEN_LEN:-12288}"
export INFER_PPO_MAX_TOKEN_LEN="${INFER_PPO_MAX_TOKEN_LEN:-12288}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-256}"

export LORA_RANK=0
export TARGET_MODULES="${TARGET_MODULES:-all-linear}"
export USE_ORIG_PARAMS=True
export TINY_LORA_RANK="${TINY_LORA_RANK:-2}"
export TINY_LORA_PROJECTION_DIM="${TINY_LORA_PROJECTION_DIM:-1}"
export TINY_LORA_TIE_FACTOR="${TINY_LORA_TIE_FACTOR:-16}"
export TINY_LORA_TIE_STRATEGY="${TINY_LORA_TIE_STRATEGY:-tiled}"
export TINY_LORA_SEED="${TINY_LORA_SEED:-42}"
export TINY_LORA_SVD_DEVICE="${TINY_LORA_SVD_DEVICE:-auto}"
export TINY_LORA_SVD_METHOD="${TINY_LORA_SVD_METHOD:-lowrank}"
export TINY_LORA_SVD_OVERSAMPLE="${TINY_LORA_SVD_OVERSAMPLE:-4}"
export TINY_LORA_SVD_NITER="${TINY_LORA_SVD_NITER:-2}"
export TINY_LORA_PROJECTION_STD="${TINY_LORA_PROJECTION_STD:-1.0}"

# The paper does not publish the winning LR per parameter budget. Use the
# highest candidate in its reported sweep for this 13-parameter experiment.
export LR="${LR:-2e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
export LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"
export PPO_DATA_LOADER_SEED="${PPO_DATA_LOADER_SEED:-42}"
export DATA_SEED="${DATA_SEED:-42}"
export TRAIN_SHUFFLE="${TRAIN_SHUFFLE:-True}"
export ACTOR_SHUFFLE="${ACTOR_SHUFFLE:-False}"

exec bash "${SCRIPT_DIR}/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh"
