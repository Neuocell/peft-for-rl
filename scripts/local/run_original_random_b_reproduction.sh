#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/runs/random-b-original-repro-v1}"
PROBE_EXP_NAME="${PROBE_EXP_NAME:-rl_gradient_probe_b16n8_w8_cap64_e95_s6to12_seed42_original_repro_v1}"
TRAIN_EXP_NAME="${TRAIN_EXP_NAME:-gradtop_probe12_r8to32_original_random_b_a2_b64m16n8_270_seed42_repro_v1}"
MODEL_PATH="${MODEL_PATH:-/data/peft-for-rl-runtime/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base}"
TRAIN_FILE="${TRAIN_FILE:-/data/peft-for-rl-runtime/datasets/verl_dapo/data/dapo-math-17k-boxed.parquet}"

PROBE_ARTIFACT_DIR="${RUNTIME_ROOT}/analysis/${PROBE_EXP_NAME}"
TRAIN_CKPT_DIR="${RUNTIME_ROOT}/ckpts/verl/DAPO-Math-17k/${TRAIN_EXP_NAME}"
CONTROLLER_LOG_DIR="${RUNTIME_ROOT}/logs"

mkdir -p "${CONTROLLER_LOG_DIR}"
cd "${REPO_ROOT}"

if [[ -e "${PROBE_ARTIFACT_DIR}/rank_map.json" || -d "${TRAIN_CKPT_DIR}/global_step_1" ]]; then
    echo "Refusing to overwrite an existing reproduction under ${RUNTIME_ROOT}" >&2
    exit 2
fi

echo "[$(date --iso-8601=seconds)] Starting original random-B energy probe"
RUNTIME_ROOT="${RUNTIME_ROOT}" \
EXP_NAME="${PROBE_EXP_NAME}" \
MODEL_PATH="${MODEL_PATH}" \
TRAIN_FILE="${TRAIN_FILE}" \
TEST_FILE="${TRAIN_FILE}" \
CKPTS_DIR="${RUNTIME_ROOT}/probe-no-checkpoints/${PROBE_EXP_NAME}" \
GRADIENT_PROBE_OUTPUT_DIR="${PROBE_ARTIFACT_DIR}" \
RAY_TEMP_DIR="${PROBE_RAY_TEMP_DIR:-/root/rbrp}" \
GRADIENT_PROBE_METHOD=energy \
GRADIENT_PROBE_WIDTH=8 \
GRADIENT_PROBE_CAPACITY=64 \
GRADIENT_PROBE_TARGET_ENERGY=0.95 \
GRADIENT_PROBE_RANK_BINS='[8,12,16,20,24,28,32]' \
GRADIENT_PROBE_SEED=42 \
GRADIENT_PROBE_MIN_STEPS=6 \
GRADIENT_PROBE_MAX_STEPS=12 \
GRADIENT_PROBE_STABILITY_PATIENCE=3 \
GRADIENT_PROBE_OVERLAP_THRESHOLD=0.98 \
GRADIENT_PROBE_RANK_TOLERANCE=1.0 \
TRAIN_PROMPT_BSZ=16 \
TRAIN_PROMPT_MINI_BSZ=16 \
N_RESP_PER_PROMPT=8 \
TOTAL_TRAINING_STEPS=12 \
DATA_SEED=42 \
PPO_DATA_LOADER_SEED=42 \
bash "${SCRIPT_DIR}/start_gradient_probe_4gpu.sh"

python3 - "${PROBE_ARTIFACT_DIR}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

artifact = Path(sys.argv[1])
required = [artifact / "rank_map.json", artifact / "summary.json", artifact / "subspaces.safetensors"]
missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
if missing:
    raise SystemExit(f"Original probe did not publish a complete artifact: {missing}")

rank_map = json.loads((artifact / "rank_map.json").read_text(encoding="utf-8"))
ranks = list(rank_map["rank_pattern"].values())
if rank_map.get("method") != "rl_policy_gradient_subspace":
    raise SystemExit(f"Unexpected probe method: {rank_map.get('method')!r}")
if len(ranks) != 196:
    raise SystemExit(f"Expected 196 target modules, got {len(ranks)}")
if not ranks or min(ranks) < 8 or max(ranks) > 32:
    raise SystemExit(f"Rank map is outside the original [8, 32] bins: {min(ranks)}..{max(ranks)}")
print(
    "Original probe artifact validated: "
    f"modules={len(ranks)}, rank_sum={sum(ranks)}, rank_mean={sum(ranks) / len(ranks):.4f}, "
    f"rank_range=[{min(ranks)}, {max(ranks)}], counts={dict(sorted(Counter(ranks).items()))}"
)
PY

echo "[$(date --iso-8601=seconds)] Probe complete; starting one continuous 270-step training run"
RUNTIME_ROOT="${RUNTIME_ROOT}" \
EXP_NAME="${TRAIN_EXP_NAME}" \
MODEL_PATH="${MODEL_PATH}" \
TRAIN_FILE="${TRAIN_FILE}" \
TEST_FILE="${TRAIN_FILE}" \
CKPTS_DIR="${TRAIN_CKPT_DIR}" \
GRADIENT_PROBE_ARTIFACT_DIR="${PROBE_ARTIFACT_DIR}" \
RAY_TEMP_DIR="${TRAIN_RAY_TEMP_DIR:-/root/rbrt}" \
TRAIN_PROMPT_BSZ=64 \
TRAIN_PROMPT_MINI_BSZ=16 \
N_RESP_PER_PROMPT=8 \
TOTAL_TRAINING_STEPS=270 \
SAVE_FREQ=50 \
MAX_ACTOR_CKPT_TO_KEEP=2 \
SAVE_CONTENTS="['model','optimizer','extra']" \
RESUME_MODE=disable \
DATA_SEED=42 \
PPO_DATA_LOADER_SEED=42 \
bash "${SCRIPT_DIR}/start_gradient_subspace_init_4gpu.sh"

echo "[$(date --iso-8601=seconds)] Original random-B reproduction finished"
