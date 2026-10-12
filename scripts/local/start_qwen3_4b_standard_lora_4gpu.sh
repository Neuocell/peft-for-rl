#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd -P)"
train_seed="${TRAIN_SEED:-42}"
[[ "${train_seed}" == "42" ]] || { echo "Qwen3-4B baseline is frozen to training seed 42" >&2; exit 2; }

run_revision="${QWEN_LORA_RUN_REVISION:-1}"
artifact_root="${ARTIFACT_ROOT:-${repo_root}/runs/qwen3-4b-standard-lora-v${run_revision}}"
storage_root="${STORAGE_ROOT:-/media/nfs_share/wangls/peft-for-rl-external-validity-v1/qwen3-4b-standard-lora-v${run_revision}}"
exp_name="${EXP_NAME:-qwen3_4b_standard_lora_r32a64_b64m16n8_step50_seed42_v${run_revision}}"
ckpt_dir="${CKPTS_DIR:-${storage_root}/ckpts/verl/DAPO-Math-17k/${exp_name}}"
resume_mode="${RESUME_MODE:-disable}"

if [[ "${resume_mode}" == "disable" && -d "${ckpt_dir}" && -n "$(find "${ckpt_dir}" -mindepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite Qwen3-4B training directory: ${ckpt_dir}" >&2
    exit 2
fi
if [[ "${resume_mode}" != "disable" && ! -s "${ckpt_dir}/latest_checkpointed_iteration.txt" ]]; then
    echo "Cannot resume without a checkpoint tracker: ${ckpt_dir}" >&2
    exit 2
fi

export ARTIFACT_ROOT="${artifact_root}"
export STORAGE_ROOT="${storage_root}"
export EXP_NAME="${exp_name}"
export PROJECT_NAME="DAPO-Math-17k-Qwen3-4B"
export MODEL_PATH="${MODEL_PATH:-/home/wangls/models/Qwen3-4B}"
export TRAIN_FILE="${TRAIN_FILE:-${repo_root}/runs/phase1-signal-random-v1/imported-runtime/datasets/verl_dapo/data/dapo-math-17k-boxed.parquet}"
export TEST_FILE="${TRAIN_FILE}"
export CKPTS_DIR="${ckpt_dir}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-peft-for-rl}"
export PEFT_TYPE=lora
export TARGET_MODULES=all-linear
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
export MAX_PROMPT_LENGTH=1024
export MAX_RESPONSE_LENGTH=8192
export TRAIN_PROMPT_BSZ=64
export TRAIN_PROMPT_MINI_BSZ=16
export N_RESP_PER_PROMPT=8
export ACTOR_PPO_MAX_TOKEN_LEN=12288
export INFER_PPO_MAX_TOKEN_LEN=12288
export ROLLOUT_MAX_NUM_SEQS=256
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-50}"
export SAVE_FREQ="${SAVE_FREQ:-25}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-2}"
export SAVE_CONTENTS="${SAVE_CONTENTS:-['model','optimizer','extra']}"
export RESUME_MODE="${resume_mode}"
export TRAIN_SHUFFLE=True
export ACTOR_SHUFFLE=False
export DATA_SEED=42
export PPO_DATA_LOADER_SEED=42
export ROLLOUT_SEED=42
export RAY_ADDRESS=local
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/dev/shm/ray-qwen3-lora-${exp_name}}"
export TMPDIR="${TMPDIR:-/dev/shm/tmp-qwen3-lora-${exp_name}}"
export LOG_DIR="${LOG_DIR:-${artifact_root}/logs/verl}"
export LOG_FILE="${LOG_FILE:-${LOG_DIR}/${exp_name}.log}"
export LORA_INIT_AUDIT_PATH="${LORA_INIT_AUDIT_PATH:-${artifact_root}/analysis/initialization/${exp_name}.json}"
unset LORA_INIT_PEER_AUDIT_PATH

mkdir -p "${LOG_DIR}" "${TMPDIR}" "$(dirname -- "${LORA_INIT_AUDIT_PATH}")" "$(dirname -- "${ckpt_dir}")"
cd "${repo_root}"
export PYTHONPATH="${repo_root}"
export PATH="/home/wangls/miniconda3/bin:${PATH}"
exec >>"${LOG_FILE}" 2>&1
exec conda run --no-capture-output -n "${CONDA_ENV_NAME}" \
    bash "${repo_root}/examples/verl_train/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh"
