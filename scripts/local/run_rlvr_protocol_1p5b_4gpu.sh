#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd -P)"
runtime_root="${repo_root}/runs"
protocol="${repo_root}/configs/experiments/rlvr_dapo_math_1p5b_protocol_v1.json"
if [[ -n "${PYTHON_BIN:-}" ]]; then
    python_bin="${PYTHON_BIN}"
elif [[ -x "${repo_root}/.venv/bin/python" ]]; then
    python_bin="${repo_root}/.venv/bin/python"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/envs/peft-for-rl/bin/python" ]]; then
    python_bin="${CONDA_PREFIX}/envs/peft-for-rl/bin/python"
else
    python_bin="python3"
fi
if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "Protocol Python is not available: ${python_bin}" >&2
    exit 2
fi

method="${METHOD:-lora}"
lr="${LR:-1e-5}"
rank="${LORA_RANK:-32}"
target_step="${TARGET_STEP:-${TOTAL_TRAINING_STEPS:-270}}"
resume_mode="${RESUME_MODE:-disable}"

case "${lr}" in
    1e-6|5e-6|1e-5) ;;
    *) echo "LR must be exactly one of 1e-6, 5e-6, 1e-5; got ${lr}" >&2; exit 2 ;;
esac

case "${resume_mode}" in
    disable|auto) ;;
    *) echo "RESUME_MODE must be disable or auto, got ${resume_mode}" >&2; exit 2 ;;
esac

gpu_ids="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a gpu_array <<<"${gpu_ids}"
if (( ${#gpu_array[@]} != 4 )); then
    echo "The frozen protocol requires exactly four CUDA devices, got ${gpu_ids}" >&2
    exit 2
fi

train_file="${runtime_root}/datasets/verl_dapo/data/dapo-math-17k-boxed.parquet"
model_path="${repo_root}/runs/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base"
safe_lr="${lr//-/m}"
default_name="rlvr_v1_${method}_r${rank}_a64_lr${safe_lr}_b64m16n8_seed42"
exp_name="${EXP_NAME:-${default_name}}"
contract="${runtime_root}/protocol_contracts/${exp_name}.json"
ray_temp_dir="${RAY_TEMP_DIR:-${RAY_TEMP_BASE:-${runtime_root}/ray}/${method}_r${rank}_${safe_lr}}"
local_checkpoint_dir="${runtime_root}/ckpts/verl/peft-for-rl/${exp_name}"
local_snapshot_dir="${runtime_root}/snapshots/rlvr_protocol/${exp_name}"

mkdir -p "$(dirname -- "${contract}")"
pending_contract="$(mktemp "${contract}.check.XXXXXX")"
trap 'rm -f -- "${pending_contract}"' EXIT
"${python_bin}" "${repo_root}/scripts/analysis/verify_rlvr_protocol.py" \
    --protocol "${protocol}" \
    --learning-rate "${lr}" \
    --method "${method}" \
    --rank "${rank}" \
    --target-step "${target_step}" \
    --output "${pending_contract}" >/dev/null

lock_value() {
    local name="$1" expected="$2"
    if [[ -v "${name}" && "${!name}" != "${expected}" ]]; then
        echo "Frozen RLVR protocol rejects ${name}=${!name}; required ${expected}" >&2
        exit 2
    fi
    printf -v "${name}" '%s' "${expected}"
    export "${name}"
}

lock_value RUNTIME_ROOT "${runtime_root}"
lock_value DATASET dapo_full_boxed
lock_value TRAIN_FILE "${train_file}"
lock_value TEST_FILE "${train_file}"
lock_value MODEL_PATH "${model_path}"
lock_value CUSTOM_REWARD_PATH "${repo_root}/verl/utils/reward_score/boxed_math_accuracy.py"
lock_value PROJECT_NAME peft-for-rl
lock_value TRAIN_PROMPT_BSZ 64
lock_value TRAIN_PROMPT_MINI_BSZ 16
lock_value N_RESP_PER_PROMPT 8
lock_value MAX_PROMPT_LENGTH 1024
lock_value MAX_RESPONSE_LENGTH 8192
lock_value LOSS_AGG_MODE token-mean
lock_value LR_WARMUP_STEPS 0
lock_value WEIGHT_DECAY 0
lock_value TEMPERATURE 1.0
lock_value TOP_P 1.0
lock_value TOP_K -1
lock_value VAL_TOP_P 0.7
lock_value CLIP_RATIO_LOW 0.2
lock_value CLIP_RATIO_HIGH 0.28
lock_value CLIP_RATIO_C 10.0
lock_value DATA_SEED 42
lock_value PPO_DATA_LOADER_SEED 42
lock_value ROLLOUT_SEED 42
lock_value TRAIN_SHUFFLE True
lock_value ACTOR_SHUFFLE False
lock_value TARGET_MODULES all-linear
lock_value OFFLOAD True
lock_value FSDP_SIZE -1
lock_value FSDP_SEED 42
lock_value GEN_TP 1
lock_value GPU_MEMORY_UTILIZATION 0.70
lock_value ACTOR_PPO_MAX_TOKEN_LEN 12288
lock_value INFER_PPO_MAX_TOKEN_LEN 12288
lock_value ROLLOUT_MAX_NUM_BATCHED_TOKENS 9216
lock_value ROLLOUT_MAX_NUM_SEQS 256
lock_value RAY_ADDRESS local
lock_value RAY_TEMP_DIR "${ray_temp_dir}"
lock_value RLPO_SVD_DEVICE auto
lock_value LORA_ALPHA 64
lock_value LORA_DROPOUT 0.0
lock_value LORA_FREEZE_A False
lock_value LORA_INIT_SEED 42
lock_value NORA_DOFT_RANK "${rank}"
lock_value NORA_DOFT_BLOCK_SIZE 32
lock_value NORA_DOFT_ALPHA 64
lock_value NORA_DOFT_NORMALIZATION init
lock_value NORA_DOFT_USE_MAGNITUDE True
lock_value NORA_DOFT_EPS 1e-6
lock_value NORA_DOFT_SEED 42
lock_value NORA_DORA_RANK "${rank}"
lock_value NORA_DORA_ALPHA 64
if [[ "${method}" == "dora" ]]; then
    lock_value NORA_DORA_NORMALIZATION standard
elif [[ "${method}" == "nora_dora_continuous" ]]; then
    lock_value NORA_DORA_NORMALIZATION continuous
else
    lock_value NORA_DORA_NORMALIZATION init
fi
lock_value NORA_DORA_EPS 1e-6
lock_value NORA_DORA_SEED 42
lock_value NORA_DORA_INITIALIZATION_VERSION peft_nora_reference_v2
lock_value NGPUS_PER_NODE 4
lock_value NNODES 1
lock_value ALLOW_EXISTING_CKPT_DIR 0
lock_value CKPTS_DIR "${local_checkpoint_dir}"
lock_value LORA_SNAPSHOT_DIR "${local_snapshot_dir}"
lock_value LORA_SNAPSHOT_EVERY 10
lock_value SAVE_FREQ 50
lock_value MAX_ACTOR_CKPT_TO_KEEP 2
lock_value SAVE_CONTENTS "['model','optimizer','extra']"

export METHOD="${method}"
export LR="${lr}"
export LORA_RANK="${rank}"
export PYTHON_BIN="${python_bin}"
export TOTAL_TRAINING_STEPS="${target_step}"
export RESUME_MODE="${resume_mode}"
export EXP_NAME="${exp_name}"
export CUDA_VISIBLE_DEVICES="${gpu_ids}"
export DRY_RUN="${DRY_RUN:-1}"
export PROTOCOL_ID=rlvr_dapo_math_1p5b_protocol_v1
export PROTOCOL_CONTRACT="${contract}"

case "${method}" in
    nora_init)
        lock_value LORA_INIT_MODE nora_init
        lock_value USE_ORIG_PARAMS False
        lock_value USE_ORIG_PARAMS_OVERRIDE False
        ;;
    dora|nora_doft|nora_dora|nora_dora_continuous)
        lock_value LORA_INIT_MODE standard
        lock_value USE_ORIG_PARAMS True
        lock_value USE_ORIG_PARAMS_OVERRIDE True
        ;;
    *)
        lock_value LORA_INIT_MODE standard
        lock_value USE_ORIG_PARAMS False
        lock_value USE_ORIG_PARAMS_OVERRIDE False
        ;;
esac

runtime_args=(
    --contract "${pending_contract}"
    --manifest "${runtime_root}/protocol_contracts/${exp_name}.runtime.json"
    --checkpoints "${local_checkpoint_dir}"
    --snapshots "${local_snapshot_dir}"
    --resume-mode "${resume_mode}"
)
if [[ "${DRY_RUN}" != "1" ]]; then
    runtime_args+=(--record)
fi
"${python_bin}" "${repo_root}/scripts/analysis/record_rlvr_runtime.py" "${runtime_args[@]}"
# Keep the previous contract intact when configuration/provenance validation fails.
mv -- "${pending_contract}" "${contract}"
trap - EXIT

echo "Verified ${PROTOCOL_ID}: only LR=${lr}, METHOD=${method}, LORA_RANK=${rank} vary"
echo "Contract: ${contract}"
exec bash "${repo_root}/scripts/local/run_dapo_boxed_paper_clean_1p5b_4gpu.sh"
