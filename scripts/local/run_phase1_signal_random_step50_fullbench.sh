#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
    echo "Usage: $0 <I0|I8|I16|I32>" >&2
    exit 2
fi
METHOD="$1"
case "${METHOD}" in
    I0|I8|I16|I32) ;;
    *) echo "Invalid Phase-1 method: ${METHOD}" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/node/anaconda3/envs/peft-for-rl/bin/python}"
PHASE1_ARTIFACT_ROOT="${PHASE1_ARTIFACT_ROOT:-${REPO_ROOT}/runs/phase1-signal-random-v1/analysis/training_allocations}"
RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/runs/phase1-signal-random-v1}"
TRAIN_SEED="${TRAIN_SEED:-42}"
METHOD_LOWER="${METHOD,,}"
PHASE1_RUN_REVISION="${PHASE1_RUN_REVISION:-2}"
if [[ ! "${PHASE1_RUN_REVISION}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PHASE1_RUN_REVISION must be a positive integer" >&2
    exit 2
fi
TRAIN_NAME="phase1_${METHOD_LOWER}_uniform_r32_b64m16n8_step50_seed${TRAIN_SEED}_v${PHASE1_RUN_REVISION}"
ADAPTER_DIR="${RUNTIME_ROOT}/ckpts/verl/DAPO-Math-17k/${TRAIN_NAME}/global_step_50/actor/peft_adapter"
EVAL_NAME="${TRAIN_NAME}_fullbench_32768_seed42"
EVAL_DIR="${RUNTIME_ROOT}/outputs/full_bench_eval/${EVAL_NAME}"
EVAL_SUMMARY="${EVAL_DIR}/summary/${EVAL_NAME}.json"
EVAL_RECORDS="${EVAL_DIR}/records/${EVAL_NAME}.jsonl"
BENCHMARK_SNAPSHOT_RECORDS="${BENCHMARK_SNAPSHOT_RECORDS:-/data/peft-for-rl-runtime/outputs/full_bench_eval/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42/records/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42.jsonl}"
BENCHMARK_SNAPSHOT_SHA256="${BENCHMARK_SNAPSHOT_SHA256:-3416ba93286b324a9663777fa472d0b568af5446319f27f876a63b59b3e863da}"
PREPARATION="${PHASE1_ARTIFACT_ROOT}/phase1_training_preparation.json"
TRAINING_CONTRACT="${RUNTIME_ROOT}/analysis/training_manifests/${TRAIN_NAME}.json"
EVAL_BASE_MODEL="/data/peft-for-rl-runtime/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}"
"${PYTHON_BIN}" scripts/analysis/prepare_phase1_signal_random_artifacts.py \
    --output-root "${PHASE1_ARTIFACT_ROOT}" \
    --verify-method "${METHOD}" >/dev/null
if [[ ! -s "${ADAPTER_DIR}/adapter_model.safetensors" ]]; then
    "${PYTHON_BIN}" scripts/analysis/phase1_training_contract.py create \
        --preparation-kind phase1 \
        --preparation "${PREPARATION}" \
        --method "${METHOD}" \
        --training-seed "${TRAIN_SEED}" \
        --experiment-name "${TRAIN_NAME}" \
        --contract "${TRAINING_CONTRACT}" >/dev/null
    echo "[$(date --iso-8601=seconds)] starting Phase-1 ${METHOD} step-50 training"
    PHASE1_ARTIFACT_ROOT="${PHASE1_ARTIFACT_ROOT}" \
    RUNTIME_ROOT="${RUNTIME_ROOT}" \
    TRAIN_SEED="${TRAIN_SEED}" \
    PHASE1_RUN_REVISION="${PHASE1_RUN_REVISION}" \
    EXP_NAME="${TRAIN_NAME}" \
    bash "${SCRIPT_DIR}/start_phase1_signal_random_uniform_r32_4gpu.sh" "${METHOD}"
fi
if [[ ! -s "${ADAPTER_DIR}/adapter_model.safetensors" ]]; then
    echo "Training exited without the step-50 adapter: ${ADAPTER_DIR}" >&2
    exit 1
fi
CONTRACT_STATUS="$(${PYTHON_BIN} - "${TRAINING_CONTRACT}" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8")).get("status", ""))
PY
)"
if [[ "${CONTRACT_STATUS}" == "prepared" ]]; then
    "${PYTHON_BIN}" scripts/analysis/phase1_training_contract.py finalize \
        --contract "${TRAINING_CONTRACT}" \
        --adapter "${ADAPTER_DIR}/adapter_model.safetensors" >/dev/null
fi
"${PYTHON_BIN}" scripts/analysis/phase1_training_contract.py verify \
    --preparation-kind phase1 \
    --preparation "${PREPARATION}" \
    --method "${METHOD}" \
    --training-seed "${TRAIN_SEED}" \
    --experiment-name "${TRAIN_NAME}" \
    --contract "${TRAINING_CONTRACT}" \
    --adapter "${ADAPTER_DIR}/adapter_model.safetensors" >/dev/null
if [[ ! -s "${BENCHMARK_SNAPSHOT_RECORDS}" ]]; then
    echo "Missing benchmark snapshot: ${BENCHMARK_SNAPSHOT_RECORDS}" >&2
    exit 2
fi
ACTUAL_SNAPSHOT_SHA256="$(sha256sum "${BENCHMARK_SNAPSHOT_RECORDS}" | awk '{print $1}')"
if [[ "${ACTUAL_SNAPSHOT_SHA256}" != "${BENCHMARK_SNAPSHOT_SHA256}" ]]; then
    echo "Benchmark snapshot SHA-256 changed: expected=${BENCHMARK_SNAPSHOT_SHA256} actual=${ACTUAL_SNAPSHOT_SHA256}" >&2
    exit 2
fi

if [[ ! -s "${EVAL_SUMMARY}" ]]; then
    echo "[$(date --iso-8601=seconds)] starting Phase-1 ${METHOD} six-benchmark evaluation"
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    PROJECT_ROOT="${REPO_ROOT}/tina_run" \
    PYTHON_BIN="${PYTHON_BIN}" \
    BASE_MODEL="${EVAL_BASE_MODEL}" \
    LORA_ADAPTER="${ADAPTER_DIR}" \
    DIRECT_OPD_EVAL_ROOT="/data/peft-for-rl-runtime/datasets/Direct-OPD/datasets/eval" \
    CHECKPOINT_NAME="${EVAL_NAME}" \
    OUTPUT_DIR="${EVAL_DIR}" \
    BENCHMARK_SNAPSHOT_RECORDS="${BENCHMARK_SNAPSHOT_RECORDS}" \
    BENCHMARKS="aime24,aime25,amc23,hmmt_feb,math500,minerva" \
    SEED=42 \
    TEMPERATURE=0.6 \
    TOP_P=0.95 \
    MAX_NEW_TOKENS=32768 \
    MAX_MODEL_LEN=34816 \
    SAMPLES_SMALL=32 \
    SAMPLES_LARGE=4 \
    LIMIT_PER_BENCHMARK= \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    bash "${REPO_ROOT}/tina_run/scripts/local/eval/run_full_bench_vllm_4gpu.sh"
fi

"${PYTHON_BIN}" scripts/analysis/verify_full_bench_run.py \
    --summary "${EVAL_SUMMARY}" \
    --records "${EVAL_RECORDS}" \
    --eval-dir "${EVAL_DIR}" \
    --eval-name "${EVAL_NAME}" \
    --adapter "${ADAPTER_DIR}" \
    --base-model "${EVAL_BASE_MODEL}" \
    --snapshot "${BENCHMARK_SNAPSHOT_RECORDS}" \
    --snapshot-sha256 "${BENCHMARK_SNAPSHOT_SHA256}" >/dev/null
echo "[$(date --iso-8601=seconds)] Phase-1 ${METHOD} pipeline complete: ${EVAL_SUMMARY}"
