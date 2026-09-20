#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/runs}"

export PEFT_TYPE=lora
export EXP_NAME="${EXP_NAME:-oracle_justrl_stablerank_ceil_rmean15p87_rmax44_a2x_b64m16n8_270_4gpu_boxed_v1}"
export CKPTS_DIR="${CKPTS_DIR:-${RUNTIME_ROOT}/ckpts/verl/DAPO-Math-17k/${EXP_NAME}}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-${HOME}/roracle}"
export RAY_local_fs_capacity_threshold="${RAY_LOCAL_FS_CAPACITY_THRESHOLD:-0.98}"

export TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-64}"
export TRAIN_PROMPT_MINI_BSZ="${TRAIN_PROMPT_MINI_BSZ:-16}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-8}"
export PPO_EPOCHS="${PPO_EPOCHS:-1}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-270}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export RESUME_MODE="${RESUME_MODE:-disable}"

export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
export ACTOR_PPO_MAX_TOKEN_LEN="${ACTOR_PPO_MAX_TOKEN_LEN:-12288}"
export INFER_PPO_MAX_TOKEN_LEN="${INFER_PPO_MAX_TOKEN_LEN:-12288}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-256}"

# The global values are the oracle maximum and matching alpha. PEFT applies
# the per-module patterns, while vLLM uses the same global alpha/r=2 scaling.
export LORA_RANK="${LORA_RANK:-44}"
export LORA_ALPHA="${LORA_ALPHA:-88}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
export LORA_RANK_PATTERN_PATH="${LORA_RANK_PATTERN_PATH:-${SCRIPT_DIR}/config/oracle_justrl_stable_rank_ceil_1p5b.json}"
export TARGET_MODULES="${TARGET_MODULES:-all-linear}"

export LR="${LR:-1e-6}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
export LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"
export PPO_DATA_LOADER_SEED="${PPO_DATA_LOADER_SEED:-42}"
export DATA_SEED="${DATA_SEED:-42}"
export TRAIN_SHUFFLE="${TRAIN_SHUFFLE:-True}"
export ACTOR_SHUFFLE="${ACTOR_SHUFFLE:-False}"

# Stay in review mode unless explicitly enabled after config inspection.
export DRY_RUN="${DRY_RUN:-1}"

python3 "${REPO_ROOT}/verl/utils/peft_oracle_lora.py" \
    "${LORA_RANK_PATTERN_PATH}" \
    --base-rank "${LORA_RANK}" \
    --base-alpha "${LORA_ALPHA}"

exec bash "${SCRIPT_DIR}/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh"
