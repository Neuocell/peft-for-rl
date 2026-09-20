#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-${SCRIPT_DIR}/../../runs}"

export PEFT_TYPE=rlpo
export EXP_NAME="${EXP_NAME:-chr_rlpo_v0_d1act95_r8to28_mean16p78_a2_b64m16n8_270_4gpu_boxed_v1}"
export CKPTS_DIR="${CKPTS_DIR:-${RUNTIME_ROOT}/ckpts/verl/DAPO-Math-17k/${EXP_NAME}}"
export LORA_RANK_PATTERN_PATH="${LORA_RANK_PATTERN_PATH:-${RUNTIME_ROOT}/analysis/d1_rlpo_r32_step270_activation_rank/rank_map_activation_prefix.json}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-${HOME}/rchr1}"
export RAY_local_fs_capacity_threshold="${RAY_LOCAL_FS_CAPACITY_THRESHOLD:-0.98}"

export TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-64}"
export TRAIN_PROMPT_MINI_BSZ="${TRAIN_PROMPT_MINI_BSZ:-16}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-8}"
export PPO_EPOCHS="${PPO_EPOCHS:-1}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-270}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-10}"
export RESUME_MODE="${RESUME_MODE:-disable}"

export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
export ACTOR_PPO_MAX_TOKEN_LEN="${ACTOR_PPO_MAX_TOKEN_LEN:-12288}"
export INFER_PPO_MAX_TOKEN_LEN="${INFER_PPO_MAX_TOKEN_LEN:-12288}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-256}"

# Global rank/alpha define maximum capacity and rollout scaling. The D1 map
# supplies each module's actual r_m and alpha_m=2*r_m.
export LORA_RANK="${LORA_RANK:-32}"
export LORA_ALPHA="${LORA_ALPHA:-64}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
export TARGET_MODULES="${TARGET_MODULES:-all-linear}"
export RLPO_SVD_DEVICE="${RLPO_SVD_DEVICE:-auto}"

export LR="${LR:-1e-6}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
export LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"
export PPO_DATA_LOADER_SEED="${PPO_DATA_LOADER_SEED:-42}"
export DATA_SEED="${DATA_SEED:-42}"
export TRAIN_SHUFFLE="${TRAIN_SHUFFLE:-True}"
export ACTOR_SHUFFLE="${ACTOR_SHUFFLE:-False}"

python3 -m verl.utils.peft_oracle_lora \
    "${LORA_RANK_PATTERN_PATH}" \
    --base-rank "${LORA_RANK}" \
    --base-alpha "${LORA_ALPHA}"

exec bash "${SCRIPT_DIR}/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh"
