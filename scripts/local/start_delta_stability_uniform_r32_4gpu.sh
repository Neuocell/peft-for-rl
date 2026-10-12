#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
    echo "Usage: $0 <s0|o0>" >&2
    exit 2
fi
method="${1,,}"
case "${method}" in
    s0)
        default_name="delta_stability_s0_standard_r32_seed42_v1"
        init_mode="standard"
        ;;
    o0)
        default_name="delta_stability_o0_random_orthogonal_r32_seed42_v1"
        init_mode="random_orthogonal"
        ;;
    *)
        echo "Unknown method: ${method}; expected s0 or o0" >&2
        exit 2
        ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd -P)"
artifact_root="${ARTIFACT_ROOT:-${repo_root}/runs/delta-stability-v1}"
storage_root="${STORAGE_ROOT:-/media/nfs_share/wangls/peft-for-rl-delta-stability-v1}"
exp_name="${EXP_NAME:-${default_name}}"
ckpt_dir="${CKPTS_DIR:-${storage_root}/ckpts/verl/DAPO-Math-17k/${exp_name}}"
snapshot_dir="${LORA_SNAPSHOT_DIR:-${storage_root}/snapshots/${exp_name}}"
base_model="${MODEL_PATH:-${repo_root}/runs/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base}"
train_file="${TRAIN_FILE:-${repo_root}/runs/phase1-signal-random-v1/imported-runtime/datasets/verl_dapo/data/dapo-math-17k-boxed.parquet}"
lock_path="${artifact_root}/locks/${exp_name}.lock"
mkdir -p "$(dirname -- "${lock_path}")"
exec 9>"${lock_path}"
echo "[$(date --iso-8601=seconds)] waiting for exclusive training lock: ${lock_path}"
flock 9

latest_file="${ckpt_dir}/latest_checkpointed_iteration.txt"
resume_mode="${RESUME_MODE:-auto}"
total_training_steps="${TOTAL_TRAINING_STEPS:-100}"
if [[ -s "${latest_file}" ]]; then
    latest_step="$(tr -d '[:space:]' <"${latest_file}")"
    if [[ ! "${latest_step}" =~ ^[0-9]+$ ]] || (( latest_step > total_training_steps )); then
        echo "Invalid checkpoint marker ${latest_step@Q} in ${latest_file}" >&2
        exit 2
    fi
    if (( latest_step >= total_training_steps )); then
        echo "[$(date --iso-8601=seconds)] ${exp_name} already reached step ${latest_step}; training skipped"
        exit 0
    fi
    if [[ "${resume_mode}" == "disable" ]]; then
        echo "Cannot use RESUME_MODE=disable with completed checkpoint step ${latest_step}" >&2
        exit 2
    fi
    echo "[$(date --iso-8601=seconds)] resuming ${exp_name} from completed checkpoint step ${latest_step}"
elif [[ "${resume_mode}" != "auto" && "${resume_mode}" != "disable" ]]; then
    echo "Cannot use RESUME_MODE=${resume_mode} without ${latest_file}" >&2
    exit 2
fi

export ARTIFACT_ROOT="${artifact_root}"
export STORAGE_ROOT="${storage_root}"
export EXP_NAME="${exp_name}"
export PROJECT_NAME=DAPO-Math-17k
export MODEL_PATH="${base_model}"
export TRAIN_FILE="${train_file}"
export TEST_FILE="${train_file}"
export CKPTS_DIR="${ckpt_dir}"
export LORA_SNAPSHOT_DIR="${snapshot_dir}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-peft-for-rl}"
export PEFT_TYPE=lora
export TARGET_MODULES=all-linear
export LORA_RANK=32
export LORA_ALPHA=64
export LORA_DROPOUT=0.0
export LORA_FREEZE_A=False
export LORA_INIT_SEED=42
export LORA_INIT_MODE="${init_mode}"
export LORA_INIT_AUDIT_PATH="${LORA_INIT_AUDIT_PATH:-${artifact_root}/analysis/initialization/${exp_name}.json}"
unset LORA_INIT_PEER_AUDIT_PATH
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
export LOSS_AGG_MODE=token-mean
export TRAIN_PROMPT_BSZ=64
export TRAIN_PROMPT_MINI_BSZ=16
export N_RESP_PER_PROMPT=8
export ACTOR_PPO_MAX_TOKEN_LEN=12288
export INFER_PPO_MAX_TOKEN_LEN=12288
export ROLLOUT_MAX_NUM_SEQS=256
export TOTAL_TRAINING_STEPS="${total_training_steps}"
export SAVE_FREQ="${SAVE_FREQ:-0}"
export SAVE_STEPS="${SAVE_STEPS:-[0,25,50,100]}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-4}"
export SAVE_CONTENTS="${SAVE_CONTENTS:-['model','optimizer','extra']}"
export LORA_SNAPSHOT_EVERY="${LORA_SNAPSHOT_EVERY:-1}"
export RESUME_MODE="${resume_mode}"
export TRAIN_SHUFFLE=True
export ACTOR_SHUFFLE=False
export DATA_SEED=42
export PPO_DATA_LOADER_SEED=42
export ROLLOUT_SEED=42
export RAY_ADDRESS=local
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.70}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/dev/shm/ray-delta-${method}-seed42-v1}"
export TMPDIR="${TMPDIR:-/dev/shm/tmp-delta-${method}-seed42-v1}"
export LOG_DIR="${LOG_DIR:-${artifact_root}/logs/verl}"
export LOG_FILE="${LOG_FILE:-${LOG_DIR}/${exp_name}.log}"

mkdir -p \
    "${LOG_DIR}" "${TMPDIR}" "${RAY_TEMP_DIR}" \
    "$(dirname -- "${LORA_INIT_AUDIT_PATH}")" \
    "$(dirname -- "${ckpt_dir}")" "$(dirname -- "${snapshot_dir}")"
cd "${repo_root}"
export PYTHONPATH="${repo_root}"
export PATH="/home/wangls/miniconda3/bin:${PATH}"
exec >>"${LOG_FILE}" 2>&1
exec conda run --no-capture-output -n "${CONDA_ENV_NAME}" \
    bash "${repo_root}/examples/verl_train/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh"
