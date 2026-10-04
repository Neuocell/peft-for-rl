#!/usr/bin/env bash
set -euo pipefail

if (( $# != 3 )); then
    echo "Usage: $0 <trainer-pid> <source-artifact-dir> <replay-artifact-dir>" >&2
    exit 2
fi

TRAINER_PID="$1"
SOURCE_ARTIFACT_DIR="$2"
REPLAY_ARTIFACT_DIR="$3"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/node/anaconda3/envs/peft-for-rl/bin/python}"

while [[ -r "/proc/${TRAINER_PID}/cmdline" ]]; do
    if ! tr '\0' ' ' <"/proc/${TRAINER_PID}/cmdline" | grep -Fq -- "${REPLAY_ARTIFACT_DIR}"; then
        break
    fi
    sleep 60
done

cd "${REPO_ROOT}"
if [[ ! -s "${REPLAY_ARTIFACT_DIR}/probe_summary.json" ]]; then
    echo "Phase 0.6 ended without ${REPLAY_ARTIFACT_DIR}/probe_summary.json" >&2
    exit 1
fi

"${PYTHON_BIN}" scripts/analysis/audit_phase06_artifact.py \
    --source-artifact "${SOURCE_ARTIFACT_DIR}" \
    --replay-artifact "${REPLAY_ARTIFACT_DIR}" \
    --expected-discovery 64 \
    --expected-calibration 32 \
    --expected-audit 16 \
    --expected-modules 196 \
    --expected-rank 32

"${PYTHON_BIN}" scripts/analysis/evaluate_phase06_crossfit_gate.py \
    --source-artifact "${SOURCE_ARTIFACT_DIR}" \
    --replay-artifact "${REPLAY_ARTIFACT_DIR}"

"${PYTHON_BIN}" scripts/analysis/prepare_phase06_training_artifacts.py \
    --source-artifact "${SOURCE_ARTIFACT_DIR}" \
    --replay-artifact "${REPLAY_ARTIFACT_DIR}"
