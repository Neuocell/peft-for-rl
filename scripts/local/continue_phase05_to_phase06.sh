#!/usr/bin/env bash
set -euo pipefail

if (( $# != 3 )); then
    echo "Usage: $0 <phase05-watcher-pid> <phase05-trainer-sid> <source-artifact-dir>" >&2
    exit 2
fi

PHASE05_WATCHER_PID="$1"
PHASE05_TRAINER_SID="$2"
SOURCE_ARTIFACT_DIR="$3"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/node/anaconda3/envs/peft-for-rl/bin/python}"
REPLAY_EXP_NAME="${REPLAY_EXP_NAME:-phase06_crossfit_replay_d64c32a16_r32_seed42_v1}"
REPLAY_ARTIFACT_DIR="${REPLAY_ARTIFACT_DIR:-${REPO_ROOT}/runs/phase0-gradient-diagnostics-v1/analysis/${REPLAY_EXP_NAME}}"
STATE_DIR="${REPO_ROOT}/runs/phase0-gradient-diagnostics-v1/logs"
LOCK_PATH="${STATE_DIR}/phase05_to_phase06.lock"

mkdir -p "${STATE_DIR}"
cd "${REPO_ROOT}"
exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
    echo "Another Phase-0.5 continuation controller already holds ${LOCK_PATH}" >&2
    exit 2
fi

run_training_confirmation() {
    local phase="$1"
    local baseline="$2"
    local candidate="$3"
    local runner="$4"
    local training_root="$5"
    local seed method train_name eval_name
    local -a baseline_records=()
    local -a candidate_records=()

    for seed in 42; do
        TRAIN_SEED="${seed}" RUNTIME_ROOT="${training_root}" bash "${runner}" "${baseline}"
        TRAIN_SEED="${seed}" RUNTIME_ROOT="${training_root}" bash "${runner}" "${candidate}"
        for method in "${baseline}" "${candidate}"; do
            train_name="${phase}_${method,,}_uniform_r32_b64m16n8_step50_seed${seed}_v1"
            eval_name="${train_name}_fullbench_32768_seed42"
            if [[ "${method}" == "${baseline}" ]]; then
                baseline_records+=("${training_root}/outputs/full_bench_eval/${eval_name}/records/${eval_name}.jsonl")
            else
                candidate_records+=("${training_root}/outputs/full_bench_eval/${eval_name}/records/${eval_name}.jsonl")
            fi
        done
    done

    local analysis_dir="${training_root}/analysis"
    local seed42_prefix="${analysis_dir}/${baseline}_vs_${candidate}_step50_seed42_paired"
    "${PYTHON_BIN}" scripts/analysis/compare_paired_full_bench.py \
        --baseline-records "${baseline_records[0]}" \
        --candidate-records "${candidate_records[0]}" \
        --output-prefix "${seed42_prefix}"
    local screen_path="${analysis_dir}/${baseline}_vs_${candidate}_step50_seed42_screen.json"
    "${PYTHON_BIN}" scripts/analysis/evaluate_single_seed_screen.py \
        --comparison "${seed42_prefix}.json" \
        --output "${screen_path}"
    local screen_decision
    screen_decision="$(${PYTHON_BIN} - "${screen_path}" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["decision"])
PY
)"
    if [[ "${screen_decision}" != "advance_multiseed" ]]; then
        echo "[$(date --iso-8601=seconds)] ${phase} seed-42 screen=${screen_decision}; extra seeds skipped"
        return 0
    fi

    for seed in 43 44; do
        TRAIN_SEED="${seed}" RUNTIME_ROOT="${training_root}" bash "${runner}" "${baseline}"
        TRAIN_SEED="${seed}" RUNTIME_ROOT="${training_root}" bash "${runner}" "${candidate}"
        for method in "${baseline}" "${candidate}"; do
            train_name="${phase}_${method,,}_uniform_r32_b64m16n8_step50_seed${seed}_v1"
            eval_name="${train_name}_fullbench_32768_seed42"
            if [[ "${method}" == "${baseline}" ]]; then
                baseline_records+=("${training_root}/outputs/full_bench_eval/${eval_name}/records/${eval_name}.jsonl")
            else
                candidate_records+=("${training_root}/outputs/full_bench_eval/${eval_name}/records/${eval_name}.jsonl")
            fi
        done
        "${PYTHON_BIN}" scripts/analysis/compare_paired_full_bench.py \
            --baseline-records "${baseline_records[-1]}" \
            --candidate-records "${candidate_records[-1]}" \
            --output-prefix "${analysis_dir}/${baseline}_vs_${candidate}_step50_seed${seed}_paired"
    done
    "${PYTHON_BIN}" scripts/analysis/aggregate_multiseed_full_bench.py \
        --baseline-records "${baseline_records[@]}" \
        --candidate-records "${candidate_records[@]}" \
        --training-seeds 42 43 44 \
        --output "${analysis_dir}/${baseline}_vs_${candidate}_step50_multiseed.json"
}

echo "[$(date --iso-8601=seconds)] waiting for Phase-0.5 watcher ${PHASE05_WATCHER_PID}"
while [[ -r "/proc/${PHASE05_WATCHER_PID}/cmdline" ]]; do
    if ! tr '\0' ' ' <"/proc/${PHASE05_WATCHER_PID}/cmdline" \
        | grep -Fq -- "postprocess_phase05_hybrid_probe.sh"; then
        break
    fi
    sleep 60
done

for required in \
    probe_summary.json \
    candidate_tensor_validation.json \
    artifact_validation.json \
    phase05_gate_decision.json \
    phase05_training_preparation.json; do
    if [[ ! -s "${SOURCE_ARTIFACT_DIR}/${required}" ]]; then
        echo "Phase-0.5 postprocessing ended without ${required}" >&2
        exit 1
    fi
done

DECISION="$(${PYTHON_BIN} - "${SOURCE_ARTIFACT_DIR}/phase05_gate_decision.json" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8")).get("decision")
if value not in {"go", "no-go"}:
    raise SystemExit(f"Invalid Phase-0.5 decision: {value!r}")
print(value)
PY
)"
echo "[$(date --iso-8601=seconds)] Phase-0.5 decision=${DECISION}"

if [[ "${DECISION}" == "go" ]]; then
    SELECTED="$(${PYTHON_BIN} - "${SOURCE_ARTIFACT_DIR}/phase05_gate_decision.json" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["calibration_selected_method"])
PY
)"
    echo "[$(date --iso-8601=seconds)] Phase 0.6 skipped; running H0 versus ${SELECTED}"
    run_training_confirmation \
        phase05 H0 "${SELECTED}" \
        "${SCRIPT_DIR}/run_phase05_hybrid_step50_fullbench.sh" \
        "${REPO_ROOT}/runs/phase05-hybrid-training-v1"
    echo "[$(date --iso-8601=seconds)] Phase-0.5 training confirmation pipeline complete"
    exit 0
fi

echo "[$(date --iso-8601=seconds)] waiting for trainer session ${PHASE05_TRAINER_SID} to release resources"
while pgrep -s "${PHASE05_TRAINER_SID}" >/dev/null 2>&1; do
    sleep 30
done

MIN_AVAILABLE_MEMORY_GIB="${PHASE06_MIN_AVAILABLE_MEMORY_GIB:-190}"
while true; do
    MEM_AVAILABLE_KIB="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    if [[ -z "${MEM_AVAILABLE_KIB}" ]]; then
        echo "Unable to read MemAvailable while waiting to start Phase 0.6" >&2
        exit 1
    fi
    MEM_AVAILABLE_GIB="$((MEM_AVAILABLE_KIB / 1024 / 1024))"
    if (( MEM_AVAILABLE_GIB >= MIN_AVAILABLE_MEMORY_GIB )); then
        break
    fi
    echo "[$(date --iso-8601=seconds)] waiting for host memory: available=${MEM_AVAILABLE_GIB} GiB required=${MIN_AVAILABLE_MEMORY_GIB} GiB"
    sleep 30
done

echo "[$(date --iso-8601=seconds)] starting Phase 0.6 sequential cache replay"
SOURCE_ARTIFACT_DIR="${SOURCE_ARTIFACT_DIR}" \
EXP_NAME="${REPLAY_EXP_NAME}" \
GRADIENT_PROBE_OUTPUT_DIR="${REPLAY_ARTIFACT_DIR}" \
bash "${SCRIPT_DIR}/start_phase06_crossfit_replay_4gpu.sh"

echo "[$(date --iso-8601=seconds)] Phase 0.6 trainer exited; starting strict postprocessing"
bash "${SCRIPT_DIR}/postprocess_phase06_crossfit_replay.sh" \
    0 "${SOURCE_ARTIFACT_DIR}" "${REPLAY_ARTIFACT_DIR}"
echo "[$(date --iso-8601=seconds)] Phase 0.6 replay, audit and gate complete"

PHASE06_DECISION="$(${PYTHON_BIN} - "${REPLAY_ARTIFACT_DIR}/phase06_gate_decision.json" <<'PY'
import json
import sys

print(json.load(open(sys.argv[1], encoding="utf-8"))["decision"])
PY
)"
if [[ "${PHASE06_DECISION}" != "go" ]]; then
    echo "[$(date --iso-8601=seconds)] Phase-0.6 decision=${PHASE06_DECISION}; no training launched"
    exit 0
fi

read -r STANDARD SELECTED < <("${PYTHON_BIN}" - "${REPLAY_ARTIFACT_DIR}/phase06_gate_decision.json" <<'PY'
import json
import sys

gate = json.load(open(sys.argv[1], encoding="utf-8"))
selected = gate["calibration_selected_method"]
standard = gate["candidates"][selected]["matched_standard_method"]
print(standard, selected)
PY
)
echo "[$(date --iso-8601=seconds)] running Phase-0.6 ${STANDARD} versus ${SELECTED}"
run_training_confirmation \
    phase06 "${STANDARD}" "${SELECTED}" \
    "${SCRIPT_DIR}/run_phase06_crossfit_step50_fullbench.sh" \
    "${REPO_ROOT}/runs/phase06-crossfit-training-v1"
echo "[$(date --iso-8601=seconds)] Phase-0.6 training confirmation pipeline complete"
