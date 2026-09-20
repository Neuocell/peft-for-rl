#!/usr/bin/env bash
set -xeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export EXP_NAME=${EXP_NAME:-dapo_math_boxed_iso_oft_1p5b_4gpu_8k}
export CKPTS_DIR=${CKPTS_DIR:-${RUNTIME_ROOT:-${SCRIPT_DIR}/../../runs}/ckpts/verl/DAPO-Math-17k/${EXP_NAME}}
export RAY_TEMP_DIR=${RAY_TEMP_DIR:-${RUNTIME_ROOT:-${SCRIPT_DIR}/../../runs}/ray_iso_oft_boxed_8k}

export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
export SAVE_FREQ=${SAVE_FREQ:-20}
export MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-2}
export RESUME_MODE=${RESUME_MODE:-disable}

# ISO-style PEFT: PEFT/OFT implements a fixed-spectrum input-frame adapter.
# Effective linear weight W' = W R with orthogonal R, so singular values are
# inherited from the frozen base W up to numerical Cayley approximation error.
export PEFT_TYPE=oft
export LORA_RANK=0
export OFT_RANK=${OFT_RANK:-0}
export OFT_BLOCK_SIZE=${OFT_BLOCK_SIZE:-64}
export OFT_DROPOUT=${OFT_DROPOUT:-0.0}
export OFT_COFT=${OFT_COFT:-True}
export OFT_EPS=${OFT_EPS:-0.0001}
export OFT_BLOCK_SHARE=${OFT_BLOCK_SHARE:-False}
export OFT_USE_CAYLEY_NEUMANN=${OFT_USE_CAYLEY_NEUMANN:-True}
export OFT_NUM_CAYLEY_NEUMANN_TERMS=${OFT_NUM_CAYLEY_NEUMANN_TERMS:-8}

bash "$BASE_SCRIPT"
