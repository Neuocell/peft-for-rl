#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-${SCRIPT_DIR}/../../runs}"

export PEFT_TYPE=adalora
export EXP_NAME="${EXP_NAME:-adalora_r32_to_r8_a64_b64m16n8_270_4gpu_boxed_official}"
export CKPTS_DIR="${CKPTS_DIR:-${RUNTIME_ROOT}/ckpts/verl/DAPO-Math-17k/${EXP_NAME}}"
# Keep this path short: Ray appends long session and socket suffixes.
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-${RUNTIME_ROOT}/ray}"
# /home is a large shared volume that is already above Ray's default 95%
# cutoff despite having hundreds of GiB free. Keep a 2% reserve while allowing
# object spilling for this run.
export RAY_local_fs_capacity_threshold="${RAY_LOCAL_FS_CAPACITY_THRESHOLD:-0.98}"

export TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-64}"
export TRAIN_PROMPT_MINI_BSZ="${TRAIN_PROMPT_MINI_BSZ:-16}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-8}"
export PPO_EPOCHS="${PPO_EPOCHS:-1}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-270}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export RESUME_MODE="${RESUME_MODE:-disable}"

# Match the token and vLLM scheduling limits of the previously stable 4-GPU
# run. Scheduling all 512 responses concurrently raises memory to ~44 GiB.
export ACTOR_PPO_MAX_TOKEN_LEN="${ACTOR_PPO_MAX_TOKEN_LEN:-12288}"
export INFER_PPO_MAX_TOKEN_LEN="${INFER_PPO_MAX_TOKEN_LEN:-12288}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-256}"

export LORA_RANK="${LORA_RANK:-32}"
export LORA_ALPHA="${LORA_ALPHA:-64}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
export ADALORA_INIT_R="${ADALORA_INIT_R:-32}"
export ADALORA_TARGET_R="${ADALORA_TARGET_R:-8}"
export ADALORA_TINIT="${ADALORA_TINIT:-100}"
export ADALORA_TFINAL="${ADALORA_TFINAL:-200}"
export ADALORA_DELTA_T="${ADALORA_DELTA_T:-20}"
export ADALORA_BETA1="${ADALORA_BETA1:-0.85}"
export ADALORA_BETA2="${ADALORA_BETA2:-0.85}"
export ADALORA_ORTH_REG_WEIGHT="${ADALORA_ORTH_REG_WEIGHT:-1e-3}"

export LR="${LR:-1e-6}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
export LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"
export PPO_DATA_LOADER_SEED="${PPO_DATA_LOADER_SEED:-42}"
export DATA_SEED="${DATA_SEED:-42}"
export TRAIN_SHUFFLE="${TRAIN_SHUFFLE:-True}"
export ACTOR_SHUFFLE="${ACTOR_SHUFFLE:-False}"

# Official RankAllocator needs complete A/B/E parameters and gradients after
# backward. Keep original parameters and do not reshard after forward.
export USE_ORIG_PARAMS=True
export FSDP_RESHARD_AFTER_FORWARD=False

exec bash "${SCRIPT_DIR}/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh"
