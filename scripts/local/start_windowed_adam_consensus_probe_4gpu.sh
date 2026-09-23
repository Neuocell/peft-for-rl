#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/runs/windowed-adam-consensus-v1}"
export EXP_NAME="${EXP_NAME:-windowed_adam_consensus_probe_b64n8_seed42}"
export GRADIENT_PROBE_OUTPUT_DIR="${GRADIENT_PROBE_OUTPUT_DIR:-${RUNTIME_ROOT}/analysis/${EXP_NAME}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PEFT_TYPE=full_gradient_probe
export FULL_GRADIENT_PROBE_MODE=windowed_adam_consensus
export FULL_GRADIENT_PROBE_RANK=32
export FULL_GRADIENT_PROBE_WINDOW_PROMPTS=16
export FULL_GRADIENT_PROBE_DISCOVERY_WINDOWS=8
export FULL_GRADIENT_PROBE_CALIBRATION_WINDOWS=2
export FULL_GRADIENT_PROBE_AUDIT_WINDOWS=2
export FULL_GRADIENT_PROBE_LOCAL_ATOMS=4
export FULL_GRADIENT_PROBE_ADAM_BETA1=0.9
export FULL_GRADIENT_PROBE_ADAM_BETA2=0.999
export FULL_GRADIENT_PROBE_ADAM_EPS=1e-8
export FULL_GRADIENT_PROBE_FUTURE_LCB_Z=1.0
export FULL_GRADIENT_PROBE_SUPPORT_FLOOR=1e-4
export FULL_GRADIENT_PROBE_SVD_DEVICE=auto
export LORA_RANK=0
export LORA_ALPHA=0
export LORA_DROPOUT=0.0
export LORA_FREEZE_A=False
export TARGET_MODULES="${TARGET_MODULES:-all-linear}"
export MODEL_PATH="${MODEL_PATH:-/data/peft-for-rl-runtime/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base}"
export TRAIN_FILE="${TRAIN_FILE:-/data/peft-for-rl-runtime/datasets/verl_dapo/data/dapo-math-17k-boxed.parquet}"
export TEST_FILE="${TEST_FILE:-${TRAIN_FILE}}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-${RUNTIME_ROOT}/ray}"

# These are the historical training settings. Do not lower the global batch
# for probe convenience: each probe window is formed inside a batch64 rollout.
export TRAIN_PROMPT_BSZ=64
export TRAIN_PROMPT_MINI_BSZ=16
export N_RESP_PER_PROMPT=8
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-8}"
export SAVE_FREQ=0
export RESUME_MODE=disable
export DATA_SEED=42
export PPO_DATA_LOADER_SEED=42
export ACTOR_SHUFFLE=False
export USE_ORIG_PARAMS=True
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-peft-for-rl}"
export LOG_DIR="${LOG_DIR:-${RUNTIME_ROOT}/logs/verl}"
export LOG_FILE="${LOG_FILE:-${LOG_DIR}/${EXP_NAME}.log}"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}"
exec >>"${LOG_FILE}" 2>&1
exec conda run --no-capture-output -n "${CONDA_ENV_NAME}" \
    bash "${REPO_ROOT}/examples/verl_train/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh"
