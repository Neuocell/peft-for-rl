#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )) || [[ "$1" != "42" && "$1" != "43" ]]; then
    echo "Usage: $0 <42|43>" >&2
    exit 2
fi
train_seed="$1"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd -P)"
run_revision="${MATCHED_LORA_RUN_REVISION:-2}"
[[ "${run_revision}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid MATCHED_LORA_RUN_REVISION: ${run_revision}" >&2; exit 2; }
runtime_root="${RUNTIME_ROOT:-${repo_root}/runs/matched-standard-lora-r32-v${run_revision}}"
exp_name="${EXP_NAME:-matched_standard_lora_r32a64_b64m16n8_step50_seed${train_seed}_v${run_revision}}"
ckpt_dir="${runtime_root}/ckpts/verl/DAPO-Math-17k/${exp_name}"

if [[ -d "${ckpt_dir}" ]] && [[ -n "$(find "${ckpt_dir}" -mindepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite matched standard-LoRA training directory: ${ckpt_dir}" >&2
    exit 2
fi

export RUNTIME_ROOT="${runtime_root}"
export EXP_NAME="${exp_name}"
export PROJECT_NAME=DAPO-Math-17k
export MODEL_PATH="${MODEL_PATH:-${repo_root}/runs/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base}"
export TRAIN_FILE="${TRAIN_FILE:-${repo_root}/runs/phase1-signal-random-v1/imported-runtime/datasets/verl_dapo/data/dapo-math-17k-boxed.parquet}"
export TEST_FILE="${TRAIN_FILE}"
export CKPTS_DIR="${ckpt_dir}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-peft-for-rl}"
export PEFT_TYPE=lora
export LORA_RANK=32
export LORA_ALPHA=64
export LORA_DROPOUT=0.0
export LORA_FREEZE_A=False
export LORA_INIT_SEED=42
export LR=1e-6
export LR_WARMUP_STEPS=0
export LR_SCHEDULER_TYPE=constant
export WEIGHT_DECAY=0
export PPO_EPOCHS=1
export CLIP_RATIO_LOW=0.2
export CLIP_RATIO_HIGH=0.28
export USE_KL_LOSS=False
export KL_LOSS_COEF=0.0
export TEMPERATURE=1.0
export TOP_P=1.0
export TOP_K=-1
export TRAIN_PROMPT_BSZ=64
export TRAIN_PROMPT_MINI_BSZ=16
export N_RESP_PER_PROMPT=8
export ACTOR_PPO_MAX_TOKEN_LEN=12288
export INFER_PPO_MAX_TOKEN_LEN=12288
export ROLLOUT_MAX_NUM_SEQS=256
export TOTAL_TRAINING_STEPS=50
export SAVE_FREQ=25
export MAX_ACTOR_CKPT_TO_KEEP=2
export SAVE_CONTENTS="['model','optimizer','extra']"
export RESUME_MODE=disable
export TRAIN_SHUFFLE=True
export ACTOR_SHUFFLE=False
export DATA_SEED="${train_seed}"
export PPO_DATA_LOADER_SEED="${train_seed}"
export ROLLOUT_SEED="${train_seed}"
export RAY_ADDRESS=local
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/dev/shm/ray-matched-lora-s${train_seed}-v${run_revision}}"
export TMPDIR="${TMPDIR:-/dev/shm/tmp-matched-lora-s${train_seed}-v${run_revision}}"
export LOG_DIR="${runtime_root}/logs/verl"
export LOG_FILE="${LOG_FILE:-${LOG_DIR}/${exp_name}.log}"
export LORA_INIT_AUDIT_PATH="${LORA_INIT_AUDIT_PATH:-${runtime_root}/analysis/initialization/${exp_name}.json}"
peer_seed=43
if [[ "${train_seed}" == "43" ]]; then peer_seed=42; fi
export LORA_INIT_PEER_AUDIT_PATH="${LORA_INIT_PEER_AUDIT_PATH:-${runtime_root}/analysis/initialization/matched_standard_lora_r32a64_b64m16n8_step50_seed${peer_seed}_v${run_revision}.json}"

mkdir -p "${LOG_DIR}" "${TMPDIR}"
cd "${repo_root}"
export PYTHONPATH="${repo_root}"
export PATH="/home/wangls/miniconda3/bin:${PATH}"
exec >>"${LOG_FILE}" 2>&1
exec conda run --no-capture-output -n "${CONDA_ENV_NAME}" \
    bash "${repo_root}/examples/verl_train/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh"
