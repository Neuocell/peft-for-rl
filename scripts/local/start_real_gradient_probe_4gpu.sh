#!/usr/bin/env bash
# REAL observation/debugging entry point; no optimizer updates are performed.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/runs}"
export REAL_PROBE_RANK="${REAL_PROBE_RANK:-32}"
export REAL_PROBE_OUTPUT_WIDTH="${REAL_PROBE_OUTPUT_WIDTH:-$((4 * REAL_PROBE_RANK))}"
export REAL_PROBE_INPUT_WIDTH="${REAL_PROBE_INPUT_WIDTH:-$((4 * REAL_PROBE_RANK))}"
export REAL_PROBE_STEPS="${REAL_PROBE_STEPS:-64}"
export REAL_PROBE_PROMPTS="${REAL_PROBE_PROMPTS:-0}"
export REAL_PROBE_SEED="${REAL_PROBE_SEED:-42}"
export EXP_NAME="${EXP_NAME:-real_probe_r${REAL_PROBE_RANK}_s${REAL_PROBE_STEPS}_p${REAL_PROBE_PROMPTS}_seed${REAL_PROBE_SEED}}"
export GRADIENT_PROBE_OUTPUT_DIR="${GRADIENT_PROBE_OUTPUT_DIR:-${RUNTIME_ROOT}/analysis/${EXP_NAME}}"
export PEFT_TYPE=real_probe
# The sketch adapter has identically zero output; rollout uses the base model.
export LORA_RANK=0
export LORA_ALPHA=0
export LORA_DROPOUT=0.0
export LORA_FREEZE_A=False
export USE_ORIG_PARAMS=True
export SP_SIZE=1
export TARGET_MODULES=all-linear
export TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-64}"
export TRAIN_PROMPT_MINI_BSZ="${TRAIN_PROMPT_MINI_BSZ:-16}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-8}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8192}"
export ACTOR_PPO_MAX_TOKEN_LEN="${ACTOR_PPO_MAX_TOKEN_LEN:-12288}"
export INFER_PPO_MAX_TOKEN_LEN="${INFER_PPO_MAX_TOKEN_LEN:-12288}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-256}"
export TOTAL_TRAINING_STEPS="${REAL_PROBE_STEPS}"
export GRADIENT_SUBSPACE_SCALING="${GRADIENT_SUBSPACE_SCALING:-2.0}"
export DATA_SEED="${DATA_SEED:-42}"
export PPO_DATA_LOADER_SEED="${PPO_DATA_LOADER_SEED:-42}"
export ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
export ACTOR_SHUFFLE=False
export PPO_EPOCHS=1
export USE_KL_LOSS=False
export KL_LOSS_COEF=0.0
export ENABLE_OVERLONG_BUFFER=False
export SAVE_FREQ=0
export SAVE_STEPS=null
export LORA_SNAPSHOT_EVERY=0
export LORA_SNAPSHOT_DIR=null
export TEST_FREQ=0
export RESUME_MODE=disable
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-${RUNTIME_ROOT}/ray/real_probe}"
export DRY_RUN="${DRY_RUN:-1}"

echo "REAL observation: rank=${REAL_PROBE_RANK}, sketches=${REAL_PROBE_OUTPUT_WIDTH}/${REAL_PROBE_INPUT_WIDTH}, steps=${REAL_PROBE_STEPS}, prompt cap=${REAL_PROBE_PROMPTS} (0=uncapped)"
echo "Artifact: ${GRADIENT_PROBE_OUTPUT_DIR}"
exec bash "${REPO_ROOT}/examples/verl_train/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh"
