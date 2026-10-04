#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-/data/peft-for-rl-runtime}"
TRAIN_SERVICE="${TRAIN_SERVICE:-spar-lora-tokenmask-270.service}"
TRAIN_NAME="tokenmask_cov_uncentered_adaptive_eqr28_rmin16_b64m16n8_270_seed42_v1"
CKPT_ROOT="${RUNTIME_ROOT}/ckpts/verl/DAPO-Math-17k/${TRAIN_NAME}"
TARGET_CKPT="${CKPT_ROOT}/global_step_100"
LATEST_FILE="${CKPT_ROOT}/latest_checkpointed_iteration.txt"

required=(
    "actor/peft_adapter/adapter_config.json"
    "actor/peft_adapter/adapter_model.safetensors"
    "actor/model_world_size_4_rank_0.pt"
    "actor/model_world_size_4_rank_1.pt"
    "actor/model_world_size_4_rank_2.pt"
    "actor/model_world_size_4_rank_3.pt"
    "actor/optim_world_size_4_rank_0.pt"
    "actor/optim_world_size_4_rank_1.pt"
    "actor/optim_world_size_4_rank_2.pt"
    "actor/optim_world_size_4_rank_3.pt"
    "actor/extra_state_world_size_4_rank_0.pt"
    "actor/extra_state_world_size_4_rank_1.pt"
    "actor/extra_state_world_size_4_rank_2.pt"
    "actor/extra_state_world_size_4_rank_3.pt"
    "data.pt"
)

echo "[$(date --iso-8601=seconds)] watching ${TRAIN_SERVICE} for a complete step-100 checkpoint"
while true; do
    if ! systemctl is-active --quiet "${TRAIN_SERVICE}"; then
        echo "Training service exited before the step-100 checkpoint was complete" >&2
        exit 3
    fi

    complete=1
    for relative_path in "${required[@]}"; do
        if [[ ! -s "${TARGET_CKPT}/${relative_path}" ]]; then
            complete=0
            break
        fi
    done
    checkpoint_marker="$(tr -d '[:space:]' <"${LATEST_FILE}" 2>/dev/null || true)"
    if (( complete == 1 )) && [[ "${checkpoint_marker}" == "100" ]]; then
        break
    fi
    sleep 10
done

echo "[$(date --iso-8601=seconds)] step-100 checkpoint complete; stopping ${TRAIN_SERVICE}"
systemctl stop "${TRAIN_SERVICE}"

for _ in $(seq 1 60); do
    gpu_processes="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
    if [[ -z "${gpu_processes}" ]]; then
        break
    fi
    sleep 5
done
gpu_processes="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
if [[ -n "${gpu_processes}" ]]; then
    echo "GPU processes remain after stopping training: ${gpu_processes}" >&2
    exit 4
fi

echo "[$(date --iso-8601=seconds)] GPUs released; starting step-100 evaluation"
exec bash "${SCRIPT_DIR}/run_token_mask_uncentered_step100_eval.sh"
