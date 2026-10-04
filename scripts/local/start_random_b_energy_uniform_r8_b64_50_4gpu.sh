#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/runs/random-b-rank8-v1}"
PROBE_DIR="${PROBE_DIR:-${RUNTIME_ROOT}/analysis/random_b_energy_probe_b16n8_w8_s6to12_seed42_v2}"
ALLOCATION_DIR="${ALLOCATION_DIR:-${PROBE_DIR}/uniform_r8}"
export EXP_NAME="${EXP_NAME:-random_b_energy_uniform_r8_b64m16n8_50_seed42_v1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

export PEFT_TYPE=grad_subspace
# Keep the global capacity identical to the historical strong baseline. The
# rank/alpha patterns below override every actual module to rank 8 / alpha 16.
export LORA_RANK=32
export LORA_ALPHA=64
export LORA_DROPOUT=0.05
export LORA_FREEZE_A=False
export GRADIENT_SUBSPACE_RANK_MAP_PATH="${ALLOCATION_DIR}/rank_map.json"
export GRADIENT_SUBSPACE_PATH="${ALLOCATION_DIR}/subspaces.safetensors"
export GRADIENT_SUBSPACE_SCALING=2.0

export TRAIN_PROMPT_BSZ=64
export TRAIN_PROMPT_MINI_BSZ=16
export N_RESP_PER_PROMPT=8
export ACTOR_PPO_MAX_TOKEN_LEN=12288
export INFER_PPO_MAX_TOKEN_LEN=12288
export ROLLOUT_MAX_NUM_SEQS=256
export TOTAL_TRAINING_STEPS=50
export SAVE_FREQ=50
export MAX_ACTOR_CKPT_TO_KEEP=1
export SAVE_CONTENTS="['model','extra']"
export RESUME_MODE=disable
export DATA_SEED=42
export PPO_DATA_LOADER_SEED=42
export ACTOR_SHUFFLE=False

export MODEL_PATH="${MODEL_PATH:-/data/peft-for-rl-runtime/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base}"
export TRAIN_FILE="${TRAIN_FILE:-/data/peft-for-rl-runtime/datasets/verl_dapo/data/dapo-math-17k-boxed.parquet}"
export TEST_FILE="${TEST_FILE:-${TRAIN_FILE}}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray-random-b-r8-train-v1}"
export RAY_local_fs_capacity_threshold="${RAY_LOCAL_FS_CAPACITY_THRESHOLD:-0.99}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-peft-for-rl}"
export LOG_DIR="${LOG_DIR:-${RUNTIME_ROOT}/logs/verl}"
export LOG_FILE="${LOG_FILE:-${LOG_DIR}/${EXP_NAME}.log}"

if [[ ! -s "${GRADIENT_SUBSPACE_RANK_MAP_PATH}" ]] || [[ ! -s "${GRADIENT_SUBSPACE_PATH}" ]]; then
    echo "Missing random-B uniform-r8 allocation under ${ALLOCATION_DIR}" >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}"
exec >>"${LOG_FILE}" 2>&1
exec conda run --no-capture-output -n "${CONDA_ENV_NAME}" \
    bash "${REPO_ROOT}/examples/verl_train/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh"
