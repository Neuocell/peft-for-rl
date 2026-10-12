#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd -P)"
run_root="${NORA_DORA_RUN_ROOT:-${repo_root}/runs/nora-dora-v1}"
exp_name="${EXP_NAME:-nora_dora_r8a8_mag_init_lr1e6_b64m16n8_50_4gpu_boxed}"
checkpoint_dir="${CKPTS_DIR:-${run_root}/ckpts/verl/DAPO-Math-17k/${exp_name}}"
gpu_ids="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
log_dir="${run_root}/logs"
log_file="${LOG_FILE:-${log_dir}/nora_dora_training.log}"
lock_dir="${run_root}/.start.lock"

mkdir -p "${run_root}" "${log_dir}"
if ! mkdir "${lock_dir}" 2>/dev/null; then
    echo "Another NoRA-DoRA launcher holds ${lock_dir}" >&2
    exit 2
fi
trap 'rmdir "${lock_dir}" 2>/dev/null || true' EXIT

if compgen -G "${checkpoint_dir}/global_step_*" >/dev/null; then
    echo "Checkpoint steps already exist under ${checkpoint_dir}; refusing a duplicate run" >&2
    exit 2
fi

IFS=',' read -r -a gpu_array <<< "${gpu_ids}"
if (( ${#gpu_array[@]} != 4 )); then
    echo "Expected four CUDA devices, got ${gpu_ids}" >&2
    exit 2
fi
for gpu in "${gpu_array[@]}"; do
    pids="$(nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
    if [[ -n "${pids//[[:space:]]/}" ]]; then
        echo "GPU ${gpu} is busy with PIDs: ${pids//$'\n'/,}" >&2
        exit 2
    fi
done

export RUNTIME_ROOT="${run_root}"
export EXP_NAME="${exp_name}"
export CKPTS_DIR="${checkpoint_dir}"
export MODEL_PATH="${MODEL_PATH:-${repo_root}/runs/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base}"
export TRAIN_FILE="${TRAIN_FILE:-${repo_root}/runs/datasets/verl_dapo/data/dapo-math-17k-boxed.parquet}"
export TEST_FILE="${TEST_FILE:-${TRAIN_FILE}}"
export CUDA_VISIBLE_DEVICES="${gpu_ids}"
export RAY_ADDRESS=local
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/dev/shm/ray-nora-dora-v1}"
export TMPDIR="${TMPDIR:-/dev/shm/tmp-nora-dora-v1}"
export SAVE_CONTENTS="${SAVE_CONTENTS:-['model','optimizer','extra']}"
export RESUME_MODE="${RESUME_MODE:-disable}"

mkdir -p "${TMPDIR}"
cd "${repo_root}"
export PYTHONPATH="${repo_root}"
export PATH="/home/wangls/miniconda3/bin:${PATH}"
exec >>"${log_file}" 2>&1
echo "[$(date --iso-8601=seconds)] starting NoRA-DoRA on GPUs ${gpu_ids}"
exec conda run --no-capture-output -n peft-for-rl \
    bash "${repo_root}/examples/verl_train/run_dapo_math_boxed_nora_dora_1p5b_4gpu_8k.sh"
