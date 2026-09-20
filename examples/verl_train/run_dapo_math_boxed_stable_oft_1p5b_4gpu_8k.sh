#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/runs}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export EXP_NAME=${EXP_NAME:-dapo_math_boxed_stable_oft_1p5b_4gpu_8k_v1_20260724}
export CKPTS_DIR=${CKPTS_DIR:-${RUNTIME_ROOT}/ckpts/verl/DAPO-Math-17k/${EXP_NAME}}
export RAY_TEMP_DIR=${RAY_TEMP_DIR:-${RUNTIME_ROOT}/ray_oft_boxed_stable_8k}

export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
export SAVE_FREQ=${SAVE_FREQ:-20}
export MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-2}
export RESUME_MODE=${RESUME_MODE:-disable}

export PEFT_TYPE=oft
export LORA_RANK=0
export OFT_BLOCK_SIZE=${OFT_BLOCK_SIZE:-32}
export OFT_RANK=${OFT_RANK:-0}
export OFT_DROPOUT=${OFT_DROPOUT:-0.0}

bash "$BASE_SCRIPT"
