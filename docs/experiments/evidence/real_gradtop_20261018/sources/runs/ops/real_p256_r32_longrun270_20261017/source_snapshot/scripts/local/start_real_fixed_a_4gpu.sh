#!/usr/bin/env bash
# Implementation/debugging entry point. Formal experiments require a registered
# extension of RLVR_PROTOCOL.md for the REAL initialization and fixed-A method.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/runs}"
export LORA_RANK="${LORA_RANK:-32}"
export LORA_ALPHA="${LORA_ALPHA:-$((2 * LORA_RANK))}"
export GRADIENT_PROBE_ARTIFACT_DIR="${GRADIENT_PROBE_ARTIFACT_DIR:-${RUNTIME_ROOT}/analysis/real_probe_r${LORA_RANK}_s64_p0_seed42}"
export GRADIENT_SUBSPACE_RANK_MAP_PATH="${GRADIENT_PROBE_ARTIFACT_DIR}/rank_map.json"
export GRADIENT_SUBSPACE_PATH="${GRADIENT_PROBE_ARTIFACT_DIR}/subspaces.safetensors"
export LORA_FREEZE_A=True
export LORA_DROPOUT=0.0
export USE_ORIG_PARAMS=True
export ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
export EXP_NAME="${EXP_NAME:-real_fixed_a_r${LORA_RANK}_seed42}"
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-${RUNTIME_ROOT}/ray/real_fixed_a}"
export DRY_RUN="${DRY_RUN:-1}"

python3 - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["GRADIENT_SUBSPACE_RANK_MAP_PATH"])
if not path.is_file():
    raise SystemExit(f"REAL rank map is missing: {path}; run the observation phase first")
allocation = json.loads(path.read_text())
if allocation.get("method") != "real_reward_gradient_aligned_lora" or allocation.get("freeze_a") is not True:
    raise SystemExit("Expected a REAL fixed-A artifact; the supplied rank map describes another method")
ranks = allocation.get("rank_pattern", {})
alphas = allocation.get("alpha_pattern", {})
if not ranks or set(alphas) != set(ranks):
    raise SystemExit("REAL artifact must contain matching, non-empty rank and alpha patterns")
if set(ranks.values()) != {int(os.environ["LORA_RANK"])}:
    raise SystemExit("LORA_RANK must equal the uniform rank exported by the REAL probe")
if set(alphas.values()) != {int(os.environ["LORA_ALPHA"])}:
    raise SystemExit("LORA_ALPHA must equal the alpha exported by the REAL probe")
PY

echo "REAL fixed-A implementation run; see docs/real_reward_gradient_probe_zh.md for protocol differences."
exec bash "${REPO_ROOT}/examples/verl_train/run_dapo_math_boxed_gradient_subspace_init_1p5b_4gpu_8k.sh"
