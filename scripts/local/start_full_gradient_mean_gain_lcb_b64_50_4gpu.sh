#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/runs/full-gradient-v1}"
export EXP_NAME="${EXP_NAME:-full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_50_seed42}"
export ALLOCATION_DIR="${ALLOCATION_DIR:-${RUNTIME_ROOT}/analysis/full_gradient_signed_grpo_probe_r32_d16c16a8_seed42/mean_uniform_r8}"
export MODEL_PATH="${MODEL_PATH:-/data/peft-for-rl-runtime/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base}"
export TRAIN_FILE="${TRAIN_FILE:-/data/peft-for-rl-runtime/datasets/verl_dapo/data/dapo-math-17k-boxed.parquet}"
export TEST_FILE="${TEST_FILE:-${TRAIN_FILE}}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-${RUNTIME_ROOT}/ray}"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export TRAIN_PROMPT_BSZ=64
export TRAIN_PROMPT_MINI_BSZ=16
export N_RESP_PER_PROMPT=8
export ACTOR_PPO_MAX_TOKEN_LEN=12288
export INFER_PPO_MAX_TOKEN_LEN=12288
export ROLLOUT_MAX_NUM_SEQS=256
export LORA_RANK=32
export LORA_ALPHA=64
export TOTAL_TRAINING_STEPS=50
export SAVE_FREQ=50
export RESUME_MODE=disable
export DATA_SEED=42
export PPO_DATA_LOADER_SEED=42

exec bash "${SCRIPT_DIR}/start_full_gradient_uniform_r8_4gpu.sh"
