#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-${SCRIPT_DIR}/../../runs}"

export PEFT_TYPE=grad_subspace
export LORA_FREEZE_A=False
export EXP_NAME="${EXP_NAME:-stable_snr_v3_trainableA_pbudget16_a2_b64m16n8_50_v1}"
export CKPTS_DIR="${CKPTS_DIR:-${RUNTIME_ROOT}/ckpts/verl/DAPO-Math-17k/${EXP_NAME}}"
export GRADIENT_PROBE_ARTIFACT_DIR="${GRADIENT_PROBE_ARTIFACT_DIR:-${RUNTIME_ROOT}/analysis/rl_gradient_probe_stable_snr_w5x3_c2_v2_pbudget16_v3}"
export GRADIENT_SUBSPACE_RANK_MAP_PATH="${GRADIENT_SUBSPACE_RANK_MAP_PATH:-${GRADIENT_PROBE_ARTIFACT_DIR}/rank_map.json}"
export GRADIENT_SUBSPACE_PATH="${GRADIENT_SUBSPACE_PATH:-${GRADIENT_PROBE_ARTIFACT_DIR}/subspaces.safetensors}"
export GRADIENT_SUBSPACE_SCALING="${GRADIENT_SUBSPACE_SCALING:-2.0}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-${HOME}/gsv3t1}"
export RAY_local_fs_capacity_threshold="${RAY_LOCAL_FS_CAPACITY_THRESHOLD:-0.98}"

export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-50}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-10}"
export RESUME_MODE=disable

exec bash "${SCRIPT_DIR}/run_dapo_math_boxed_gradient_subspace_init_1p5b_4gpu_8k.sh"
