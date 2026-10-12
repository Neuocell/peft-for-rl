#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-${SCRIPT_DIR}/../../runs/nora-lr1e5-probe-v1}"

export PEFT_TYPE=lora
export EXP_NAME="${EXP_NAME:-nora_init_r8a8_lr1e5_b64m16n8_10_4gpu_boxed}"
export CKPTS_DIR="${CKPTS_DIR:-${RUNTIME_ROOT}/ckpts/verl/DAPO-Math-17k/${EXP_NAME}}"

export LORA_RANK="${LORA_RANK:-8}"
export LORA_ALPHA="${LORA_ALPHA:-8}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
export LORA_FREEZE_A="${LORA_FREEZE_A:-False}"
export LORA_INIT_MODE="${LORA_INIT_MODE:-nora_init}"
export LORA_INIT_SEED="${LORA_INIT_SEED:-42}"
export LORA_INIT_AUDIT_PATH="${LORA_INIT_AUDIT_PATH:-${RUNTIME_ROOT}/analysis/initialization/${EXP_NAME}.json}"
export LORA_INIT_PEER_AUDIT_PATH=null

export LR="${LR:-1e-5}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
export LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"

export TRAIN_PROMPT_BSZ=64
export TRAIN_PROMPT_MINI_BSZ=16
export N_RESP_PER_PROMPT=8
export PPO_EPOCHS=1
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-10}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-5}"
export SAVE_CONTENTS="${SAVE_CONTENTS:-['model','optimizer','extra']}"
export RESUME_MODE="${RESUME_MODE:-disable}"

export MAX_PROMPT_LENGTH=1024
export MAX_RESPONSE_LENGTH=8192
export ACTOR_PPO_MAX_TOKEN_LEN=12288
export INFER_PPO_MAX_TOKEN_LEN=12288
export ROLLOUT_MAX_NUM_SEQS=256

export DATA_SEED=42
export PPO_DATA_LOADER_SEED=42
export ROLLOUT_SEED=42
export TRAIN_SHUFFLE=True
export ACTOR_SHUFFLE=False

exec bash "${SCRIPT_DIR}/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh"
