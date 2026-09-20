#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-${SCRIPT_DIR}/../../runs}"

export PEFT_TYPE=grad_subspace
export LORA_FREEZE_A=True
export EXP_NAME="${EXP_NAME:-stable_snr_fixedA_bonly_pbudget16_a2_b64m16n8_270_v1}"
export CKPTS_DIR="${CKPTS_DIR:-${RUNTIME_ROOT}/ckpts/verl/DAPO-Math-17k/${EXP_NAME}}"
export GRADIENT_PROBE_ARTIFACT_DIR="${GRADIENT_PROBE_ARTIFACT_DIR:-${RUNTIME_ROOT}/analysis/rl_gradient_probe_stable_snr_w5x3_pbudget16_v2}"
export GRADIENT_SUBSPACE_RANK_MAP_PATH="${GRADIENT_SUBSPACE_RANK_MAP_PATH:-${GRADIENT_PROBE_ARTIFACT_DIR}/rank_map.json}"
export GRADIENT_SUBSPACE_PATH="${GRADIENT_SUBSPACE_PATH:-${GRADIENT_PROBE_ARTIFACT_DIR}/subspaces.safetensors}"
export GRADIENT_SUBSPACE_SCALING="${GRADIENT_SUBSPACE_SCALING:-2.0}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-${HOME}/gsf1}"
export RAY_local_fs_capacity_threshold="${RAY_LOCAL_FS_CAPACITY_THRESHOLD:-0.98}"

if [[ ! -s "${GRADIENT_SUBSPACE_RANK_MAP_PATH}" ]]; then
    echo "Missing stable-SNR rank map: ${GRADIENT_SUBSPACE_RANK_MAP_PATH}" >&2
    exit 2
fi
if [[ ! -s "${GRADIENT_SUBSPACE_PATH}" ]]; then
    echo "Missing stable-SNR subspace tensors: ${GRADIENT_SUBSPACE_PATH}" >&2
    exit 2
fi

export TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-64}"
export TRAIN_PROMPT_MINI_BSZ="${TRAIN_PROMPT_MINI_BSZ:-16}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-8}"
export PPO_EPOCHS=1
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-270}"
export SAVE_FREQ="${SAVE_FREQ:-50}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-10}"
export RESUME_MODE=disable

export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
export ACTOR_PPO_MAX_TOKEN_LEN="${ACTOR_PPO_MAX_TOKEN_LEN:-12288}"
export INFER_PPO_MAX_TOKEN_LEN="${INFER_PPO_MAX_TOKEN_LEN:-12288}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-256}"

export LORA_RANK="${LORA_RANK:-32}"
export LORA_ALPHA="${LORA_ALPHA:-64}"
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
    "${GRADIENT_SUBSPACE_RANK_MAP_PATH}" \
    --base-rank "${LORA_RANK}" \
    --base-alpha "${LORA_ALPHA}"

exec bash "${SCRIPT_DIR}/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh"
