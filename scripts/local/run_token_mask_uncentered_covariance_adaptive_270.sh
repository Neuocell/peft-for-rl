#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-/data/peft-for-rl-runtime}"
PYTHON_BIN="${PYTHON_BIN:-/home/node/anaconda3/envs/peft-for-rl/bin/python}"

PROBE_NAME="full_gradient_tokenmask_top50_cov_uncentered_r32_d64c32a16_seed42_v1"
ARTIFACT_DIR="${RUNTIME_ROOT}/analysis/${PROBE_NAME}"
ALLOCATION_DIR="${ARTIFACT_DIR}/covariance_adaptive_eqr28_rmin16_stable_energy"
TRAIN_NAME="tokenmask_cov_uncentered_adaptive_eqr28_rmin16_b64m16n8_270_seed42_v1"
STEP_270_ADAPTER="${RUNTIME_ROOT}/ckpts/verl/DAPO-Math-17k/${TRAIN_NAME}/global_step_270/actor/peft_adapter/adapter_model.safetensors"

mkdir -p "${RUNTIME_ROOT}/logs/verl"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}"

if [[ ! -s "${ARTIFACT_DIR}/probe_summary.json" ]]; then
    echo "[$(date --iso-8601=seconds)] starting masked uncentered-second-moment probe"
    RUNTIME_ROOT="${RUNTIME_ROOT}" \
    EXP_NAME="${PROBE_NAME}" \
    GRADIENT_PROBE_OUTPUT_DIR="${ARTIFACT_DIR}" \
    FULL_GRADIENT_PROBE_RANK=32 \
    FULL_GRADIENT_PROBE_SKETCH_WIDTH=40 \
    FULL_GRADIENT_PROBE_DISCOVERY_PROMPTS=64 \
    FULL_GRADIENT_PROBE_CALIBRATION_PROMPTS=32 \
    FULL_GRADIENT_PROBE_AUDIT_PROMPTS=16 \
    FULL_GRADIENT_PROBE_COVARIANCE_ESTIMATOR=uncentered_second_moment \
    FULL_GRADIENT_PROBE_TOKEN_MASK_MODE=top_surprisal \
    FULL_GRADIENT_PROBE_TOKEN_KEEP_RATIO=0.5 \
    FULL_GRADIENT_PROBE_TOKEN_MIN_KEEP=128 \
    FULL_GRADIENT_PROBE_TOKEN_KEEP_FINAL=128 \
    FULL_GRADIENT_PROBE_TOKEN_MASK_DISCOVERY_ONLY=False \
    FULL_GRADIENT_PROBE_TOKEN_MASK_SCOPE=discovery_calibration \
    TRAIN_PROMPT_BSZ=16 \
    TRAIN_PROMPT_MINI_BSZ=16 \
    N_RESP_PER_PROMPT=8 \
    TOTAL_TRAINING_STEPS=30 \
    MAX_RESPONSE_LENGTH=8192 \
    ACTOR_PPO_MAX_TOKEN_LEN=12288 \
    INFER_PPO_MAX_TOKEN_LEN=12288 \
    DATA_SEED=42 \
    PPO_DATA_LOADER_SEED=42 \
    bash "${SCRIPT_DIR}/start_full_gradient_rl_probe_4gpu.sh"
fi

if [[ ! -s "${ARTIFACT_DIR}/probe_summary.json" ]]; then
    echo "Probe exited without a complete artifact: ${ARTIFACT_DIR}" >&2
    exit 1
fi
if ! jq -e '
    .covariance_estimator == "uncentered_second_moment" and
    .token_mask.mode == "top_surprisal" and
    .token_mask.scope == "discovery_calibration"
' "${ARTIFACT_DIR}/probe_summary.json" >/dev/null; then
    echo "Probe artifact does not match the uncentered masked-selection protocol" >&2
    exit 2
fi

if [[ ! -s "${ALLOCATION_DIR}/rank_map.json" ]] || [[ ! -s "${ALLOCATION_DIR}/subspaces.safetensors" ]]; then
    echo "[$(date --iso-8601=seconds)] building adaptive covariance allocation"
    "${PYTHON_BIN}" scripts/analysis/build_full_gradient_uniform_allocation.py \
        --artifact-dir "${ARTIFACT_DIR}" \
        --output-dir "${ALLOCATION_DIR}" \
        --candidate-method covariance \
        --allocation-mode adaptive \
        --uniform-rank 28 \
        --r-min 16 \
        --adaptive-utility stable_energy
fi

if [[ ! -s "${STEP_270_ADAPTER}" ]]; then
    echo "[$(date --iso-8601=seconds)] starting/resuming one 270-step optimizer-continuous run"
    RUNTIME_ROOT="${RUNTIME_ROOT}" \
    ALLOCATION_DIR="${ALLOCATION_DIR}" \
    CANDIDATE_METHOD=covariance \
    EXP_NAME="${TRAIN_NAME}" \
    TRAIN_PROMPT_BSZ=64 \
    TRAIN_PROMPT_MINI_BSZ=16 \
    N_RESP_PER_PROMPT=8 \
    TOTAL_TRAINING_STEPS=270 \
    SAVE_FREQ=50 \
    MAX_ACTOR_CKPT_TO_KEEP=1 \
    SAVE_CONTENTS="['model','optimizer','extra']" \
    RESUME_MODE=auto \
    LORA_DROPOUT=0.05 \
    LR=1e-6 \
    DATA_SEED=42 \
    PPO_DATA_LOADER_SEED=42 \
    RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray_tokenmask_cov_uncentered_270}" \
    bash "${SCRIPT_DIR}/start_full_gradient_adaptive_eqr8_4gpu.sh"
fi

if [[ ! -s "${STEP_270_ADAPTER}" ]]; then
    echo "Training exited without the step-270 adapter: ${STEP_270_ADAPTER}" >&2
    exit 1
fi

echo "[$(date --iso-8601=seconds)] 270-step training complete: ${STEP_270_ADAPTER}"
