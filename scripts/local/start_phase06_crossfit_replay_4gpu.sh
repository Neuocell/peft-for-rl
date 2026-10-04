#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/node/anaconda3/envs/peft-for-rl/bin/python}"
SOURCE_ARTIFACT_DIR="${SOURCE_ARTIFACT_DIR:-${REPO_ROOT}/runs/phase0-gradient-diagnostics-v1/analysis/phase05_hybrid_shared_rollout_d64c32a16_r32_seed42_v3}"

export RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/runs/phase0-gradient-diagnostics-v1}"
export EXP_NAME="${EXP_NAME:-phase06_crossfit_replay_d64c32a16_r32_seed42_v1}"
export GRADIENT_PROBE_OUTPUT_DIR="${GRADIENT_PROBE_OUTPUT_DIR:-${RUNTIME_ROOT}/analysis/${EXP_NAME}}"
export LOG_DIR="${LOG_DIR:-${RUNTIME_ROOT}/logs/verl}"
export LOG_FILE="${LOG_FILE:-${LOG_DIR}/${EXP_NAME}.log}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}"

"${PYTHON_BIN}" - "${SOURCE_ARTIFACT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1]).expanduser().resolve()
summary = json.loads((source / "probe_summary.json").read_text(encoding="utf-8"))
audit = json.loads((source / "artifact_validation.json").read_text(encoding="utf-8"))
gate = json.loads((source / "phase05_gate_decision.json").read_text(encoding="utf-8"))
preparation = json.loads(
    (source / "phase05_training_preparation.json").read_text(encoding="utf-8")
)
if audit.get("status") != "valid":
    raise SystemExit("Phase 0.5 source has not passed strict artifact validation")
if gate.get("decision") != "no-go":
    raise SystemExit("Phase 0.6 may run only after the preregistered Phase-0.5 no-go")
if preparation.get("status") != "skipped_no_go":
    raise SystemExit("Phase 0.5 no-go training-preparation record is missing")
if int(summary.get("schema_version", -1)) != 4:
    raise SystemExit("Phase 0.6 requires a schema-4 Phase-0.5 source")
if tuple(summary.get(key, -1) for key in (
    "discovery_prompts", "calibration_prompts", "audit_prompts"
)) != (64, 32, 16):
    raise SystemExit("Phase 0.6 requires the formal 64/32/16 source split")
if int(summary.get("r_max", -1)) != 32 or int(summary.get("sketch_width", -1)) != 40:
    raise SystemExit("Phase 0.6 requires the formal rank-32, sketch-width-40 source")
if int(summary.get("crossfit_splits", -1)) != 3:
    raise SystemExit("Phase 0.6 requires the formal three cross-fit splits")
PY

if [[ -e "${GRADIENT_PROBE_OUTPUT_DIR}" ]] && \
   [[ -n "$(find "${GRADIENT_PROBE_OUTPUT_DIR}" -mindepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite a non-empty Phase-0.6 output: ${GRADIENT_PROBE_OUTPUT_DIR}" >&2
    exit 2
fi
if [[ -e "${LOG_FILE}" ]]; then
    echo "Refusing to overwrite an existing Phase-0.6 log: ${LOG_FILE}" >&2
    exit 2
fi

MIN_AVAILABLE_MEMORY_GIB="${PHASE06_MIN_AVAILABLE_MEMORY_GIB:-190}"
MEM_AVAILABLE_KIB="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
if [[ -z "${MEM_AVAILABLE_KIB}" ]]; then
    echo "Unable to read MemAvailable from /proc/meminfo; refusing to start Phase 0.6." >&2
    exit 1
fi
MEM_AVAILABLE_GIB="$((MEM_AVAILABLE_KIB / 1024 / 1024))"
echo "Phase 0.6 memory preflight: available=${MEM_AVAILABLE_GIB} GiB, required=${MIN_AVAILABLE_MEMORY_GIB} GiB."
if (( MEM_AVAILABLE_GIB < MIN_AVAILABLE_MEMORY_GIB )); then
    echo "Insufficient host memory for one sequential cross-fit accumulator." >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PEFT_TYPE=full_gradient_probe
export FULL_GRADIENT_PROBE_MODE=phase06_crossfit_replay
export FULL_GRADIENT_PROBE_REPLAY_SOURCE_DIR="${SOURCE_ARTIFACT_DIR}"
export FULL_GRADIENT_PROBE_RANK=32
export FULL_GRADIENT_PROBE_SKETCH_WIDTH=40
export FULL_GRADIENT_PROBE_DISCOVERY_PROMPTS=64
export FULL_GRADIENT_PROBE_CALIBRATION_PROMPTS=32
export FULL_GRADIENT_PROBE_AUDIT_PROMPTS=16
export FULL_GRADIENT_PROBE_CROSSFIT_SPLITS=3
export FULL_GRADIENT_PROBE_TOKEN_KEEP_RATIO=0.5
export FULL_GRADIENT_PROBE_TOKEN_MIN_KEEP=128
export FULL_GRADIENT_PROBE_TOKEN_KEEP_FINAL=128
export FULL_GRADIENT_PROBE_STABLE_SURPRISAL_QUANTILE=0.95
export FULL_GRADIENT_PROBE_SVD_DEVICE=auto
export LORA_RANK=0
export LORA_ALPHA=0
export LORA_DROPOUT=0.0
export LORA_FREEZE_A=False
export TARGET_MODULES=all-linear
export MODEL_PATH="${MODEL_PATH:-/data/peft-for-rl-runtime/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base}"
export TRAIN_FILE="${TRAIN_FILE:-/data/peft-for-rl-runtime/datasets/verl_dapo/data/dapo-math-17k-boxed.parquet}"
export TEST_FILE="${TEST_FILE:-${TRAIN_FILE}}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray-phase06-crossfit-replay-v1}"
export TRAIN_PROMPT_BSZ=1
export TRAIN_PROMPT_MINI_BSZ=1
export N_RESP_PER_PROMPT=8
export ACTOR_PPO_MAX_TOKEN_LEN=12288
export INFER_PPO_MAX_TOKEN_LEN=12288
export ROLLOUT_MAX_NUM_SEQS=8
export TOTAL_TRAINING_STEPS=1
export SAVE_FREQ=0
export RESUME_MODE=disable
export DATA_SEED=42
export PPO_DATA_LOADER_SEED=42
export ROLLOUT_SEED=42
export ACTOR_SHUFFLE=False
export USE_ORIG_PARAMS=True
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-peft-for-rl}"
export CONDA_BIN="${CONDA_BIN:-/home/node/anaconda3/bin/conda}"

mkdir -p "${LOG_DIR}"
exec >>"${LOG_FILE}" 2>&1
echo "Phase 0.6 source: ${SOURCE_ARTIFACT_DIR}"
echo "Phase 0.6 output: ${GRADIENT_PROBE_OUTPUT_DIR}"
exec "${CONDA_BIN}" run --no-capture-output -n "${CONDA_ENV_NAME}" \
    bash "${REPO_ROOT}/examples/verl_train/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh"
