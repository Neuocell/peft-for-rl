#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd -P)"
python_bin="${PYTHON_BIN:-/home/wangls/miniconda3/envs/peft-for-rl/bin/python}"
artifact_root="${ARTIFACT_ROOT:-${repo_root}/runs/delta-stability-v1}"
storage_root="${STORAGE_ROOT:-/media/nfs_share/wangls/peft-for-rl-delta-stability-v1}"
base_model="${repo_root}/runs/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base"
calibration_records="${repo_root}/runs/phase1-signal-random-v1/imported-runtime/outputs/full_bench_eval/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42/records/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42.jsonl"
analysis_dir="${storage_root}/analysis/multiscale_delta_w"
comparison_dir="${artifact_root}/analysis/final_comparison"
pipeline_log="${artifact_root}/logs/delta_stability_pipeline.log"

mkdir -p "${artifact_root}/logs" "${artifact_root}/analysis/run_verification" "${storage_root}"
if [[ ! -e "${artifact_root}/storage" ]]; then
    ln -s "${storage_root}" "${artifact_root}/storage"
fi
exec > >(tee -a "${pipeline_log}") 2>&1
cd "${repo_root}"
export PYTHONPATH="${repo_root}"
export ARTIFACT_ROOT="${artifact_root}" STORAGE_ROOT="${storage_root}" PYTHON_BIN="${python_bin}"

for method in s0 o0; do
    if [[ "${method}" == s0 ]]; then
        name="delta_stability_s0_standard_r32_seed42_v1"
    else
        name="delta_stability_o0_random_orthogonal_r32_seed42_v1"
    fi
    verify="${artifact_root}/analysis/run_verification/${name}.json"
    if [[ ! -s "${verify}" ]]; then
        echo "[$(date --iso-8601=seconds)] starting ${method^^} formal training"
        CUDA_VISIBLE_DEVICES=0,1,2,3 RESUME_MODE=auto \
            bash "${script_dir}/run_delta_stability_training_lane.sh" "${method}"
    fi
done

if [[ ! -s "${analysis_dir}/summary.json" ]]; then
    echo "[$(date --iso-8601=seconds)] starting exact multi-timescale DeltaW analysis"
    CUDA_VISIBLE_DEVICES=0 "${python_bin}" scripts/analysis/analyze_delta_stability.py \
        --s0-snapshots "${storage_root}/snapshots/delta_stability_s0_standard_r32_seed42_v1" \
        --o0-snapshots "${storage_root}/snapshots/delta_stability_o0_random_orthogonal_r32_seed42_v1" \
        --base-model "${base_model}" --calibration-records "${calibration_records}" \
        --out-dir "${analysis_dir}" --device cuda:0
fi

for method in s0 o0; do
    for step in 25 50 100; do
        CUDA_VISIBLE_DEVICES=0,1,2,3 bash "${script_dir}/run_delta_stability_fullbench.sh" "${method}" "${step}"
    done
done

"${python_bin}" scripts/analysis/summarize_delta_stability.py \
    --analysis-dir "${analysis_dir}" \
    --eval-root "${storage_root}/outputs/full_bench_eval" \
    --out-dir "${comparison_dir}"
echo "[$(date --iso-8601=seconds)] DELTA_STABILITY_PIPELINE_COMPLETE ${comparison_dir}"
