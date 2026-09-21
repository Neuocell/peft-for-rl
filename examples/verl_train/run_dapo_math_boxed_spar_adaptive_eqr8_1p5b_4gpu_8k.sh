#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-${SCRIPT_DIR}/../../runs}"
ALLOCATION_ROOT="${SPAR_ALLOCATION_ROOT:-${RUNTIME_ROOT}/analysis/spar_v0_positive_probe_seed42/allocations}"

export PEFT_TYPE=grad_subspace
export LORA_FREEZE_A=False
export EXP_NAME="${EXP_NAME:-spar_v0_positive_probe_adaptive_eqr8_50_seed42}"
export CKPTS_DIR="${CKPTS_DIR:-${RUNTIME_ROOT}/ckpts/verl/DAPO-Math-17k/${EXP_NAME}}"
export GRADIENT_PROBE_ARTIFACT_DIR="${ALLOCATION_ROOT}/adaptive_eqr8"
export GRADIENT_SUBSPACE_RANK_MAP_PATH="${GRADIENT_PROBE_ARTIFACT_DIR}/rank_map.json"
export GRADIENT_SUBSPACE_PATH="${GRADIENT_PROBE_ARTIFACT_DIR}/subspaces.safetensors"
export GRADIENT_SUBSPACE_SCALING=2.0
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-${HOME}/rsa0}"
export RAY_local_fs_capacity_threshold="${RAY_LOCAL_FS_CAPACITY_THRESHOLD:-0.98}"
# The PEFT/vLLM global rank is the heterogeneous capacity ceiling.  Every
# module is still physically sized by rank_pattern, with alpha/r fixed at 2.
export LORA_RANK=32
export LORA_ALPHA=64
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-50}"
export SAVE_FREQ="${SAVE_FREQ:-25}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-2}"
export RESUME_MODE="${RESUME_MODE:-disable}"
export DATA_SEED=42
export PPO_DATA_LOADER_SEED=42

exec bash "${SCRIPT_DIR}/run_dapo_math_boxed_gradient_subspace_init_1p5b_4gpu_8k.sh"
