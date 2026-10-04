#!/usr/bin/env bash
set -euo pipefail

if (( $# != 0 )); then
    echo "Usage: $0" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/node/anaconda3/envs/peft-for-rl/bin/python}"
PHASE1_ARTIFACT_ROOT="${PHASE1_ARTIFACT_ROOT:-${REPO_ROOT}/runs/phase1-signal-random-v1/analysis/training_allocations}"
RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/runs/phase1-signal-random-v1}"
ANALYSIS_DIR="${RUNTIME_ROOT}/analysis"
PREPARATION="${PHASE1_ARTIFACT_ROOT}/phase1_training_preparation.json"
LOCK_PATH="${RUNTIME_ROOT}/logs/phase1_signal_random_confirmation.lock"
PHASE1_RUN_REVISION="${PHASE1_RUN_REVISION:-2}"
if [[ ! "${PHASE1_RUN_REVISION}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PHASE1_RUN_REVISION must be a positive integer" >&2
    exit 2
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}"
if [[ ! -s "${PREPARATION}" ]]; then
    echo "Missing Phase-1 preparation. Inspect the Phase-0.6 diagnosis before preparing or training Phase 1: ${PREPARATION}" >&2
    exit 2
fi
mkdir -p "${ANALYSIS_DIR}" "$(dirname -- "${LOCK_PATH}")"
exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
    echo "Another Phase-1 confirmation pipeline already holds ${LOCK_PATH}" >&2
    exit 2
fi

record_path() {
    local method="$1"
    local seed="$2"
    local method_lower="${method,,}"
    local train_name="phase1_${method_lower}_uniform_r32_b64m16n8_step50_seed${seed}_v${PHASE1_RUN_REVISION}"
    local eval_name="${train_name}_fullbench_32768_seed42"
    printf '%s\n' "${RUNTIME_ROOT}/outputs/full_bench_eval/${eval_name}/records/${eval_name}.jsonl"
}

for method in I0 I8 I16 I32; do
    TRAIN_SEED=42 \
    PHASE1_RUN_REVISION="${PHASE1_RUN_REVISION}" \
    PHASE1_ARTIFACT_ROOT="${PHASE1_ARTIFACT_ROOT}" \
    RUNTIME_ROOT="${RUNTIME_ROOT}" \
    bash "${SCRIPT_DIR}/run_phase1_signal_random_step50_fullbench.sh" "${method}"
done

I0_SEED42_RECORDS="$(record_path I0 42)"
for method in I8 I16 I32; do
    prefix="${ANALYSIS_DIR}/I0_vs_${method}_step50_seed42_paired"
    "${PYTHON_BIN}" scripts/analysis/compare_paired_full_bench.py \
        --baseline-records "${I0_SEED42_RECORDS}" \
        --candidate-records "$(record_path "${method}" 42)" \
        --output-prefix "${prefix}"
done

SELECTION_PATH="${ANALYSIS_DIR}/phase1_seed42_selection.json"
"${PYTHON_BIN}" scripts/analysis/select_phase1_seed42_candidate.py \
    --i8-comparison "${ANALYSIS_DIR}/I0_vs_I8_step50_seed42_paired.json" \
    --i16-comparison "${ANALYSIS_DIR}/I0_vs_I16_step50_seed42_paired.json" \
    --i32-comparison "${ANALYSIS_DIR}/I0_vs_I32_step50_seed42_paired.json" \
    --output "${SELECTION_PATH}"

SELECTED="$(${PYTHON_BIN} - "${SELECTION_PATH}" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))["selected_method"]
print(value or "")
PY
)"
if [[ -z "${SELECTED}" ]]; then
    echo "[$(date --iso-8601=seconds)] no Phase-1 mixed candidate passed seed 42; extra seeds skipped"
    exit 0
fi
case "${SELECTED}" in
    I8|I16) ;;
    *) echo "Invalid selected Phase-1 mixed method: ${SELECTED}" >&2; exit 1 ;;
esac

for seed in 43 44; do
    for method in I0 "${SELECTED}"; do
        TRAIN_SEED="${seed}" \
        PHASE1_RUN_REVISION="${PHASE1_RUN_REVISION}" \
        PHASE1_ARTIFACT_ROOT="${PHASE1_ARTIFACT_ROOT}" \
        RUNTIME_ROOT="${RUNTIME_ROOT}" \
        bash "${SCRIPT_DIR}/run_phase1_signal_random_step50_fullbench.sh" "${method}"
    done
    "${PYTHON_BIN}" scripts/analysis/compare_paired_full_bench.py \
        --baseline-records "$(record_path I0 "${seed}")" \
        --candidate-records "$(record_path "${SELECTED}" "${seed}")" \
        --output-prefix "${ANALYSIS_DIR}/I0_vs_${SELECTED}_step50_seed${seed}_paired"
done

"${PYTHON_BIN}" scripts/analysis/aggregate_multiseed_full_bench.py \
    --baseline-records \
        "$(record_path I0 42)" \
        "$(record_path I0 43)" \
        "$(record_path I0 44)" \
    --candidate-records \
        "$(record_path "${SELECTED}" 42)" \
        "$(record_path "${SELECTED}" 43)" \
        "$(record_path "${SELECTED}" 44)" \
    --training-seeds 42 43 44 \
    --output "${ANALYSIS_DIR}/I0_vs_${SELECTED}_step50_multiseed.json"
echo "[$(date --iso-8601=seconds)] Phase-1 confirmation complete: I0 versus ${SELECTED}"
