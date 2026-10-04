#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
    echo "Usage: $0 <H0|H2|H4|H8|H16>" >&2
    exit 2
fi
METHOD="$1"
case "${METHOD}" in H0|H2|H4|H8|H16) ;; *) echo "Invalid Phase-0.5 method: ${METHOD}" >&2; exit 2 ;; esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/node/anaconda3/envs/peft-for-rl/bin/python}"
ARTIFACT_DIR="${ARTIFACT_DIR:-${REPO_ROOT}/runs/phase0-gradient-diagnostics-v1/analysis/phase05_hybrid_shared_rollout_d64c32a16_r32_seed42_v3}"
RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/runs/phase05-hybrid-training-v1}"
TRAIN_SEED="${TRAIN_SEED:-42}"
METHOD_LOWER="${METHOD,,}"
TRAIN_NAME="phase05_${METHOD_LOWER}_uniform_r32_b64m16n8_step50_seed${TRAIN_SEED}_v1"
ADAPTER_DIR="${RUNTIME_ROOT}/ckpts/verl/DAPO-Math-17k/${TRAIN_NAME}/global_step_50/actor/peft_adapter"
EVAL_NAME="${TRAIN_NAME}_fullbench_32768_seed42"
EVAL_DIR="${RUNTIME_ROOT}/outputs/full_bench_eval/${EVAL_NAME}"
EVAL_SUMMARY="${EVAL_DIR}/summary/${EVAL_NAME}.json"
BENCHMARK_SNAPSHOT_RECORDS="${BENCHMARK_SNAPSHOT_RECORDS:-/data/peft-for-rl-runtime/outputs/full_bench_eval/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42/records/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42.jsonl}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}"
if [[ ! -s "${ADAPTER_DIR}/adapter_model.safetensors" ]]; then
    echo "[$(date --iso-8601=seconds)] starting Phase-0.5 ${METHOD} step-50 training"
    ARTIFACT_DIR="${ARTIFACT_DIR}" \
    RUNTIME_ROOT="${RUNTIME_ROOT}" \
    TRAIN_SEED="${TRAIN_SEED}" \
    EXP_NAME="${TRAIN_NAME}" \
    bash "${SCRIPT_DIR}/start_phase05_hybrid_uniform_r32_4gpu.sh" "${METHOD}"
fi
if [[ ! -s "${ADAPTER_DIR}/adapter_model.safetensors" ]]; then
    echo "Training exited without the step-50 adapter: ${ADAPTER_DIR}" >&2
    exit 1
fi
if [[ ! -s "${BENCHMARK_SNAPSHOT_RECORDS}" ]]; then
    echo "Missing benchmark snapshot: ${BENCHMARK_SNAPSHOT_RECORDS}" >&2
    exit 2
fi

if [[ ! -s "${EVAL_SUMMARY}" ]]; then
    echo "[$(date --iso-8601=seconds)] starting Phase-0.5 ${METHOD} six-benchmark evaluation"
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    PROJECT_ROOT="${REPO_ROOT}/tina_run" \
    PYTHON_BIN="${PYTHON_BIN}" \
    BASE_MODEL="/data/peft-for-rl-runtime/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base" \
    LORA_ADAPTER="${ADAPTER_DIR}" \
    DIRECT_OPD_EVAL_ROOT="/data/peft-for-rl-runtime/datasets/Direct-OPD/datasets/eval" \
    CHECKPOINT_NAME="${EVAL_NAME}" \
    OUTPUT_DIR="${EVAL_DIR}" \
    BENCHMARK_SNAPSHOT_RECORDS="${BENCHMARK_SNAPSHOT_RECORDS}" \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    bash "${REPO_ROOT}/tina_run/scripts/local/eval/run_full_bench_vllm_4gpu.sh"
fi

"${PYTHON_BIN}" - "${EVAL_SUMMARY}" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
expected = {"aime24", "aime25", "amc23", "hmmt_feb", "math500", "minerva"}
actual = {row["benchmark"] for row in summary.get("benchmarks", [])}
if summary.get("num_samples") != 7248 or actual != expected:
    raise SystemExit(f"Incomplete full-benchmark summary: samples={summary.get('num_samples')} benchmarks={sorted(actual)}")
PY
echo "[$(date --iso-8601=seconds)] Phase-0.5 ${METHOD} pipeline complete: ${EVAL_SUMMARY}"
