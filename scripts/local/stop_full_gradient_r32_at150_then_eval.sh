#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
TRAIN_PGID="${TRAIN_PGID:?TRAIN_PGID is required}"
TRAIN_NAME="full_gradient_mean_gain_lcb_uniform_r32_b64m16n8_270_seed42_v1"
TRAIN_LOG="${REPO_ROOT}/runs/full-gradient-rank32-v1/logs/verl/${TRAIN_NAME}.log"
CKPT_ROOT="${REPO_ROOT}/runs/full-gradient-rank32-v1/ckpts/verl/DAPO-Math-17k/${TRAIN_NAME}"
TARGET_CKPT="${CKPT_ROOT}/global_step_150"
LATEST_FILE="${CKPT_ROOT}/latest_checkpointed_iteration.txt"

required=(
    "actor/peft_adapter/adapter_config.json"
    "actor/peft_adapter/adapter_model.safetensors"
    "actor/model_world_size_4_rank_0.pt"
    "actor/model_world_size_4_rank_1.pt"
    "actor/model_world_size_4_rank_2.pt"
    "actor/model_world_size_4_rank_3.pt"
    "actor/extra_state_world_size_4_rank_0.pt"
    "actor/extra_state_world_size_4_rank_1.pt"
    "actor/extra_state_world_size_4_rank_2.pt"
    "actor/extra_state_world_size_4_rank_3.pt"
    "data.pt"
)

echo "[$(date --iso-8601=seconds)] watching training PGID ${TRAIN_PGID} for step 150"
while true; do
    if ! kill -0 -- "-${TRAIN_PGID}" 2>/dev/null; then
        echo "Training process group ${TRAIN_PGID} exited before step 150" >&2
        exit 3
    fi
    latest_step="$(sed -n 's/.*training\/global_step:\([0-9][0-9]*\).*/\1/p' "${TRAIN_LOG}" | tail -1)"
    if [[ -n "${latest_step}" ]] && (( latest_step >= 150 )); then
        complete=1
        for relative_path in "${required[@]}"; do
            if [[ ! -s "${TARGET_CKPT}/${relative_path}" ]]; then
                complete=0
                break
            fi
        done
        checkpoint_marker="$(tr -d '[:space:]' <"${LATEST_FILE}" 2>/dev/null || true)"
        if (( complete == 1 )) && [[ "${checkpoint_marker}" == "150" ]]; then
            break
        fi
    fi
    sleep 20
done

echo "[$(date --iso-8601=seconds)] step 150 checkpoint complete; stopping training PGID ${TRAIN_PGID}"
kill -TERM -- "-${TRAIN_PGID}"
for _ in $(seq 1 24); do
    if ! kill -0 -- "-${TRAIN_PGID}" 2>/dev/null; then
        break
    fi
    sleep 5
done
if kill -0 -- "-${TRAIN_PGID}" 2>/dev/null; then
    echo "[$(date --iso-8601=seconds)] training did not exit after 120 seconds; sending SIGKILL"
    kill -KILL -- "-${TRAIN_PGID}"
fi

for _ in $(seq 1 36); do
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

echo "[$(date --iso-8601=seconds)] GPUs released; starting evaluation"
exec bash "${SCRIPT_DIR}/run_full_gradient_r32_step150_eval.sh"
