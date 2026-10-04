#!/usr/bin/env bash
set -euo pipefail

if (( $# != 3 )); then
    echo "Usage: $0 <trainer-pid> <experiment-name> <artifact-dir>" >&2
    exit 2
fi

TRAINER_PID="$1"
EXP_NAME="$2"
ARTIFACT_DIR="$3"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/node/anaconda3/envs/peft-for-rl/bin/python}"

while [[ -r "/proc/${TRAINER_PID}/cmdline" ]]; do
    if ! tr '\0' ' ' <"/proc/${TRAINER_PID}/cmdline" | grep -Fq -- "${EXP_NAME}"; then
        break
    fi
    sleep 60
done

cd "${REPO_ROOT}"
SUMMARY_PATH="${ARTIFACT_DIR}/probe_summary.json"
if [[ ! -s "${SUMMARY_PATH}" ]]; then
    echo "Probe process ended without ${SUMMARY_PATH}" >&2
    exit 1
fi

"${PYTHON_BIN}" - "${ARTIFACT_DIR}" <<'PY'
import json
import sys
from pathlib import Path

from verl.utils.full_gradient_rl_probe import validate_full_gradient_probe_artifact

artifact_dir = Path(sys.argv[1]).resolve()
summary = json.loads((artifact_dir / "probe_summary.json").read_text(encoding="utf-8"))
results = {
    method: validate_full_gradient_probe_artifact(
        artifact_dir, candidate_method=method
    )
    for method in summary["candidate_methods"]
}
output = artifact_dir / "candidate_tensor_validation.json"
output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(results, indent=2, sort_keys=True))
PY

"${PYTHON_BIN}" scripts/analysis/audit_phase05_artifact.py \
    --artifact-dir "${ARTIFACT_DIR}" \
    --expected-discovery 64 \
    --expected-calibration 32 \
    --expected-audit 16 \
    --expected-modules 196 \
    --expected-rank 32

"${PYTHON_BIN}" scripts/analysis/evaluate_phase05_hybrid_gate.py \
    --artifact-dir "${ARTIFACT_DIR}"

"${PYTHON_BIN}" scripts/analysis/prepare_phase05_training_artifacts.py \
    --artifact-dir "${ARTIFACT_DIR}"
