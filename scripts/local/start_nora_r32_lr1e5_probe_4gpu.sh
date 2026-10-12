#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd -P)"
run_root="${NORA_RUN_ROOT:-${repo_root}/runs/nora-r32-lr1e5-probe-v1}"
exp_name="${EXP_NAME:-nora_init_r32a64_lr1e5_b64m16n8_10_seed42_v1}"
checkpoint_dir="${CKPTS_DIR:-${run_root}/ckpts/verl/DAPO-Math-17k/${exp_name}}"
gpu_ids="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
resume_mode="${RESUME_MODE:-disable}"
log_dir="${run_root}/logs"
log_file="${LOG_FILE:-${log_dir}/nora_r32_lr1e5_probe_training.log}"
lock_dir="${run_root}/.start.lock"

mkdir -p "${run_root}" "${log_dir}"
if ! mkdir "${lock_dir}" 2>/dev/null; then
    echo "Another NoRA r32 LR probe launcher holds ${lock_dir}" >&2
    exit 2
fi
trap 'rmdir "${lock_dir}" 2>/dev/null || true' EXIT

if compgen -G "${checkpoint_dir}/global_step_*" >/dev/null && [[ "${resume_mode}" != "auto" ]]; then
    echo "Checkpoint steps already exist under ${checkpoint_dir}; refusing a duplicate run" >&2
    echo "Set RESUME_MODE=auto to resume the latest checkpoint explicitly" >&2
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
export RESUME_MODE="${resume_mode}"
export MODEL_PATH="${MODEL_PATH:-${repo_root}/runs/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base}"
export TRAIN_FILE="${TRAIN_FILE:-${repo_root}/runs/phase1-signal-random-v1/imported-runtime/datasets/verl_dapo/data/dapo-math-17k-boxed.parquet}"
export TEST_FILE="${TEST_FILE:-${TRAIN_FILE}}"
export CUDA_VISIBLE_DEVICES="${gpu_ids}"
export LORA_RANK=32
export LORA_ALPHA=64
export LR=1e-5
export RAY_ADDRESS=local
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/dev/shm/ray-nora-r32-lr1e5-probe-v1}"
export TMPDIR="${TMPDIR:-/dev/shm/tmp-nora-r32-lr1e5-probe-v1}"

mkdir -p "${TMPDIR}"
cd "${repo_root}"
export PYTHONPATH="${repo_root}"
export PATH="/home/wangls/miniconda3/bin:${PATH}"
exec >>"${log_file}" 2>&1
echo "[$(date --iso-8601=seconds)] starting NoRA-init r32a64 LR=1e-5 probe on GPUs ${gpu_ids}"
conda run --no-capture-output -n peft-for-rl \
    bash "${repo_root}/examples/verl_train/run_dapo_math_boxed_nora_init_1p5b_4gpu_8k.sh"
