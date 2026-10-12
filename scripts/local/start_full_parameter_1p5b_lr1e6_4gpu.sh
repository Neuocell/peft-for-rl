#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd -P)"
run_root="${FULL_PARAM_RUN_ROOT:-${repo_root}/runs/full-parameter-1p5b-unified-lr1e6-v2}"
exp_name="${EXP_NAME:-full_parameter_1p5b_lr1e6_b64m16n8_step50_seed42_v2}"
checkpoint_dir="${CKPTS_DIR:-${run_root}/ckpts/verl/DAPO-Math-17k/${exp_name}}"
gpu_ids="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
resume_mode="${RESUME_MODE:-disable}"
log_dir="${run_root}/logs"
log_file="${LOG_FILE:-${log_dir}/full_parameter_unified_training.log}"
lock_dir="${run_root}/.start.lock"

mkdir -p "${run_root}" "${log_dir}"
if ! mkdir "${lock_dir}" 2>/dev/null; then
    echo "Another full-parameter launcher holds ${lock_dir}" >&2
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

export EXP_NAME="${exp_name}"
export CKPTS_DIR="${checkpoint_dir}"
export RESUME_MODE="${resume_mode}"
export MODEL_PATH="${MODEL_PATH:-${repo_root}/runs/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base}"
export TRAIN_FILE="${TRAIN_FILE:-${repo_root}/runs/phase1-signal-random-v1/imported-runtime/datasets/verl_dapo/data/dapo-math-17k-boxed.parquet}"
export TEST_FILE="${TEST_FILE:-${TRAIN_FILE}}"
export CUDA_VISIBLE_DEVICES="${gpu_ids}"

# lora_rank=0 is verl's full-parameter path: no PEFT adapter is constructed.
export PEFT_TYPE="${PEFT_TYPE:-lora}"
export LORA_RANK="${LORA_RANK:-0}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
export LR="${LR:-1e-6}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
export LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0}"
export PPO_EPOCHS="${PPO_EPOCHS:-1}"
export TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-64}"
export TRAIN_PROMPT_MINI_BSZ="${TRAIN_PROMPT_MINI_BSZ:-16}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-8}"
export ACTOR_PPO_MAX_TOKEN_LEN="${ACTOR_PPO_MAX_TOKEN_LEN:-12288}"
export INFER_PPO_MAX_TOKEN_LEN="${INFER_PPO_MAX_TOKEN_LEN:-12288}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-256}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-50}"
export SAVE_FREQ="${SAVE_FREQ:-25}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-2}"
export SAVE_CONTENTS="${SAVE_CONTENTS:-['model','optimizer','extra']}"
export ENABLE_OVERLONG_BUFFER="${ENABLE_OVERLONG_BUFFER:-False}"
export TRAIN_SHUFFLE="${TRAIN_SHUFFLE:-True}"
export ACTOR_SHUFFLE="${ACTOR_SHUFFLE:-False}"
export DATA_SEED="${DATA_SEED:-42}"
export PPO_DATA_LOADER_SEED="${PPO_DATA_LOADER_SEED:-42}"
export ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
export DRY_RUN="${DRY_RUN:-0}"

export RAY_ADDRESS=local
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/dev/shm/ray-fp15b-u2}"
export TMPDIR="${TMPDIR:-/dev/shm/tmp-fp15b-u2}"

mkdir -p "${TMPDIR}"
cd "${repo_root}"
export PYTHONPATH="${repo_root}"
export PATH="/home/wangls/miniconda3/bin:${PATH}"
exec >>"${log_file}" 2>&1
echo "[$(date --iso-8601=seconds)] starting unified full-parameter 1.5B LR=1e-6 training on GPUs ${gpu_ids}"
conda run --no-capture-output -n peft-for-rl \
    bash "${repo_root}/examples/verl_train/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh"
