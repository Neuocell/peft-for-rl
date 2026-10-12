#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd -P)"
python_bin="${PYTHON_BIN:-/home/wangls/miniconda3/envs/peft-for-rl/bin/python}"
artifact_root="${ARTIFACT_ROOT:-${repo_root}/runs/delta-stability-v1}"
storage_root="${STORAGE_ROOT:-/media/nfs_share/wangls/peft-for-rl-delta-stability-v1}"
analysis_dir="${storage_root}/analysis/multiscale_delta_w"
comparison_dir="${artifact_root}/analysis/final_comparison"
eval_root="${storage_root}/outputs/full_bench_eval"
lock_path="${artifact_root}/locks/delta_stability_eval.lock"
log_path="${artifact_root}/logs/delta_stability_eval_lane.log"

mkdir -p "${eval_root}" "$(dirname -- "${lock_path}")" "$(dirname -- "${log_path}")"
exec 9>"${lock_path}"
flock 9
exec > >(tee -a "${log_path}") 2>&1
cd "${repo_root}"
export PYTHONPATH="${repo_root}"
export ARTIFACT_ROOT="${artifact_root}" STORAGE_ROOT="${storage_root}" PYTHON_BIN="${python_bin}"

for method in s0 o0; do
    if [[ "${method}" == s0 ]]; then
        train_name="delta_stability_s0_standard_r32_seed42_v1"
    else
        train_name="delta_stability_o0_random_orthogonal_r32_seed42_v1"
    fi
    for step in 25 50 100; do
        eval_name="${train_name}_step${step}_fullbench_32768_seed42"
        postverify="${artifact_root}/analysis/evaluation_verification/${eval_name}.json"
        eval_dir="${eval_root}/${eval_name}"
        if [[ ! -s "${postverify}" && -e "${eval_dir}" ]]; then
            stamp="$(date -u +%Y%m%dT%H%M%SZ)"
            echo "[$(date --iso-8601=seconds)] archiving incomplete evaluation ${eval_dir}"
            mv "${eval_dir}" "${eval_dir}.incomplete-${stamp}"
        fi
        echo "[$(date --iso-8601=seconds)] evaluating ${method^^} step ${step} on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
        bash "${script_dir}/run_delta_stability_fullbench.sh" "${method}" "${step}"
    done
done

while [[ ! -s "${analysis_dir}/summary.json" ]]; do
    echo "[$(date --iso-8601=seconds)] waiting for DeltaW analysis before final summary"
    sleep 60
done
"${python_bin}" scripts/analysis/summarize_delta_stability.py \
    --analysis-dir "${analysis_dir}" \
    --eval-root "${eval_root}" \
    --out-dir "${comparison_dir}"
echo "[$(date --iso-8601=seconds)] DELTA_STABILITY_POSTTRAIN_COMPLETE ${comparison_dir}"
