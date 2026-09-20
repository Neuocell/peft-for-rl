#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/runs}"
PYTHON_BIN="${PYTHON_BIN:-/home/wangls/miniconda3/envs/peft-for-rl/bin/python}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-60}"
SUMMARY_GRACE_SECONDS="${SUMMARY_GRACE_SECONDS:-600}"

EVAL_NAME="${EVAL_NAME:-dominant_atoms_v1_trainableA_pbudget8_step50_fullbench_32768_v1}"
EVAL_DIR="${EVAL_DIR:-${RUNTIME_ROOT}/outputs/full_bench_eval/${EVAL_NAME}}"
CANDIDATE_SUMMARY="${EVAL_DIR}/summary/${EVAL_NAME}.json"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-${RUNTIME_ROOT}/outputs/full_bench_eval/gradtop_probe12_r8to32_mean31p63_step50_fullbench_32768_v1/summary/gradtop_probe12_r8to32_mean31p63_step50_fullbench_32768_v1.json}"
DECISION_PREFIX="${DECISION_PREFIX:-${RUNTIME_ROOT}/analysis/dominant_atoms_v1_step50_fullbench_decision/decision}"
NEXT_EXP_NAME="${NEXT_EXP_NAME:-dominant_atoms_uniform_r8_trainableA_a2_b64m16n8_50_v1}"
NEXT_CKPT_DIR="${RUNTIME_ROOT}/ckpts/verl/DAPO-Math-17k/${NEXT_EXP_NAME}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}"

echo "Waiting for full-bench evaluation: ${EVAL_NAME}"
while pgrep -f "eval_full_bench_vllm.py.*${EVAL_NAME}" >/dev/null; do
    echo "$(date -u +%FT%TZ) evaluation shards are still running"
    sleep "${WAIT_INTERVAL_SECONDS}"
done

grace_deadline=$((SECONDS + SUMMARY_GRACE_SECONDS))
while [[ ! -s "${CANDIDATE_SUMMARY}" ]] && (( SECONDS < grace_deadline )); do
    echo "$(date -u +%FT%TZ) waiting for aggregate summary"
    sleep 10
done
if [[ ! -s "${CANDIDATE_SUMMARY}" ]]; then
    echo "Evaluation ended without a summary: ${CANDIDATE_SUMMARY}" >&2
    exit 2
fi
if [[ ! -s "${BASELINE_SUMMARY}" ]]; then
    echo "Missing comparison baseline: ${BASELINE_SUMMARY}" >&2
    exit 2
fi

"${PYTHON_BIN}" scripts/analysis/compare_full_bench_and_select_next.py \
    --candidate "${CANDIDATE_SUMMARY}" \
    --baseline "${BASELINE_SUMMARY}" \
    --output-prefix "${DECISION_PREFIX}"

if [[ -e "${NEXT_CKPT_DIR}" ]]; then
    echo "Refusing to overwrite existing next-run checkpoint directory: ${NEXT_CKPT_DIR}" >&2
    exit 2
fi
if pgrep -f "verl.trainer.main_ppo.*${NEXT_EXP_NAME}" >/dev/null; then
    echo "Next experiment is already running: ${NEXT_EXP_NAME}" >&2
    exit 2
fi

while true; do
    mapfile -t used_mib < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -4 | tr -d ' ')
    if [[ "${#used_mib[@]}" -eq 4 ]] && (( used_mib[0] < 1024 && used_mib[1] < 1024 && used_mib[2] < 1024 && used_mib[3] < 1024 )); then
        break
    fi
    echo "$(date -u +%FT%TZ) waiting for GPUs 0-3 to become free: ${used_mib[*]:-unavailable} MiB"
    sleep "${WAIT_INTERVAL_SECONDS}"
done

echo "$(date -u +%FT%TZ) launching ${NEXT_EXP_NAME}"
export EXP_NAME="${NEXT_EXP_NAME}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
exec bash scripts/local/start_dominant_atoms_uniform_r8_4gpu.sh
