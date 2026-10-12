#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd -P)"
python_bin="${PYTHON_BIN:-/home/wangls/miniconda3/envs/peft-for-rl/bin/python}"
artifact_root="${ARTIFACT_ROOT:-${repo_root}/runs/delta-stability-v1}"
storage_root="${STORAGE_ROOT:-/media/nfs_share/wangls/peft-for-rl-delta-stability-v1}"
base_model="${repo_root}/runs/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base"
calibration_records="${repo_root}/runs/phase1-signal-random-v1/imported-runtime/outputs/full_bench_eval/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42/records/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42.jsonl"
analysis_parent="${storage_root}/analysis"
analysis_dir="${analysis_parent}/multiscale_delta_w"
work_dir="${analysis_parent}/.multiscale_delta_w.work"
lock_path="${artifact_root}/locks/delta_stability_analysis.lock"
log_path="${artifact_root}/logs/delta_stability_analysis.log"

mkdir -p "${analysis_parent}" "$(dirname -- "${lock_path}")" "$(dirname -- "${log_path}")"
exec 9>"${lock_path}"
flock 9
exec > >(tee -a "${log_path}") 2>&1
cd "${repo_root}"
export PYTHONPATH="${repo_root}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-32}"

if [[ -s "${analysis_dir}/summary.json" ]]; then
    echo "[$(date --iso-8601=seconds)] DeltaW analysis already complete: ${analysis_dir}"
    exit 0
fi
for verify in \
    "${artifact_root}/analysis/run_verification/delta_stability_s0_standard_r32_seed42_v1.json" \
    "${artifact_root}/analysis/run_verification/delta_stability_o0_random_orthogonal_r32_seed42_v1.json"; do
    jq -e '.status == "verified" and .snapshot_count == 101' "${verify}" >/dev/null
done

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -e "${analysis_dir}" ]]; then
    mv "${analysis_dir}" "${analysis_dir}.incomplete-${stamp}"
fi
if [[ -e "${work_dir}" ]]; then
    mv "${work_dir}" "${work_dir}.incomplete-${stamp}"
fi

echo "[$(date --iso-8601=seconds)] starting exact DeltaW analysis on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
"${python_bin}" scripts/analysis/analyze_delta_stability.py \
    --s0-snapshots "${storage_root}/snapshots/delta_stability_s0_standard_r32_seed42_v1" \
    --o0-snapshots "${storage_root}/snapshots/delta_stability_o0_random_orthogonal_r32_seed42_v1" \
    --base-model "${base_model}" \
    --calibration-records "${calibration_records}" \
    --out-dir "${work_dir}" \
    --device cuda:0

for required in summary.json group_summary.csv module_interval_spectra.csv adjacent_period_stability.csv s0_o0_multiscale_comparison.png; do
    [[ -s "${work_dir}/${required}" ]] || { echo "Missing analysis artifact: ${required}" >&2; exit 1; }
done
mv "${work_dir}" "${analysis_dir}"
echo "[$(date --iso-8601=seconds)] DELTA_STABILITY_ANALYSIS_COMPLETE ${analysis_dir}"
