#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-${SCRIPT_DIR}/../../runs}"

export PEFT_TYPE=lora
export EXP_NAME="${EXP_NAME:-gradtop_step100_act90_compact_r8to28_mean18p29_a2_b64m16n8_rem170_v1}"
export CKPTS_DIR="${CKPTS_DIR:-${RUNTIME_ROOT}/ckpts/verl/DAPO-Math-17k/${EXP_NAME}}"
export LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-${RUNTIME_ROOT}/initializers/gradtop_step100_act90_compact}"
export LORA_RANK_PATTERN_PATH="${LORA_RANK_PATTERN_PATH:-${LORA_ADAPTER_PATH}/rank_pattern.json}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-${HOME}/rca1}"
export RAY_local_fs_capacity_threshold="${RAY_LOCAL_FS_CAPACITY_THRESHOLD:-0.98}"

if [[ ! -s "${LORA_ADAPTER_PATH}/adapter_config.json" ]] || \
   [[ ! -s "${LORA_ADAPTER_PATH}/adapter_model.safetensors" ]]; then
    echo "Missing compact initialization adapter: ${LORA_ADAPTER_PATH}" >&2
    exit 2
fi

export TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-64}"
export TRAIN_PROMPT_MINI_BSZ="${TRAIN_PROMPT_MINI_BSZ:-16}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-8}"
export PPO_EPOCHS=1
# The policy and dataloader continue from source step 100, while the changed
# tensor shapes require a fresh optimizer. Checkpoints retain the 150/200/250/270
# numbering so analysis remains aligned with the original one-epoch schedule.
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-270}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-10}"
export RESUME_MODE=disable
export WARM_START_GLOBAL_STEP="${WARM_START_GLOBAL_STEP:-100}"
export WARM_START_DATA_PATH="${WARM_START_DATA_PATH:-${RUNTIME_ROOT}/ckpts/verl/DAPO-Math-17k/gradtop_probe12_r8to32_mean31p63_a2_b64m16n8_270_v1/global_step_100/data.pt}"
if [[ ! -s "${WARM_START_DATA_PATH}" ]]; then
    echo "Missing source dataloader state: ${WARM_START_DATA_PATH}" >&2
    exit 2
fi

export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
export ACTOR_PPO_MAX_TOKEN_LEN="${ACTOR_PPO_MAX_TOKEN_LEN:-12288}"
export INFER_PPO_MAX_TOKEN_LEN="${INFER_PPO_MAX_TOKEN_LEN:-12288}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-256}"

export LORA_RANK=28
export LORA_ALPHA=56
export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
export TARGET_MODULES="${TARGET_MODULES:-all-linear}"

export LR="${LR:-1e-6}"
export WEIGHT_DECAY=0
export LR_WARMUP_STEPS=0
export LR_SCHEDULER_TYPE=constant
export DATA_SEED="${DATA_SEED:-42}"
export PPO_DATA_LOADER_SEED="${PPO_DATA_LOADER_SEED:-42}"
export TRAIN_SHUFFLE=True
export ACTOR_SHUFFLE=False

python3 -m verl.utils.peft_oracle_lora \
    "${LORA_RANK_PATTERN_PATH}" \
    --base-rank "${LORA_RANK}" \
    --base-alpha "${LORA_ALPHA}"

exec bash "${SCRIPT_DIR}/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh"
