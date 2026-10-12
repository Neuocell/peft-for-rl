#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )) || [[ "$1" != "s0" && "$1" != "o0" ]]; then
    echo "Usage: $0 <s0|o0>" >&2
    exit 2
fi
method="$1"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd -P)"
artifact_root="${ARTIFACT_ROOT:-${repo_root}/runs/delta-stability-v1}"
storage_root="${STORAGE_ROOT:-/media/nfs_share/wangls/peft-for-rl-delta-stability-v1}"
python_bin="${PYTHON_BIN:-/home/wangls/miniconda3/envs/peft-for-rl/bin/python}"

if [[ "${method}" == "s0" ]]; then
    name="delta_stability_s0_standard_r32_seed42_v1"
else
    name="delta_stability_o0_random_orthogonal_r32_seed42_v1"
fi
audit="${artifact_root}/analysis/initialization/${name}.json"
snapshots="${storage_root}/snapshots/${name}"
checkpoints="${storage_root}/ckpts/verl/DAPO-Math-17k/${name}"
verification="${artifact_root}/analysis/run_verification/${name}.json"

mkdir -p "$(dirname -- "${verification}")" "${artifact_root}/logs"
cd "${repo_root}"
export PYTHONPATH="${repo_root}"
export ARTIFACT_ROOT="${artifact_root}" STORAGE_ROOT="${storage_root}"
export RESUME_MODE="${RESUME_MODE:-auto}"

echo "[$(date --iso-8601=seconds)] ${method^^} lane starting on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
bash "${script_dir}/start_delta_stability_uniform_r32_4gpu.sh" "${method}"

verify_args=(
    --method "${method}"
    --audit "${audit}"
    --snapshots "${snapshots}"
    --checkpoints "${checkpoints}"
    --output "${verification}"
)
if [[ "${method}" == "o0" ]]; then
    verify_args+=(--s0-audit "${artifact_root}/analysis/initialization/delta_stability_s0_standard_r32_seed42_v1.json")
fi
"${python_bin}" scripts/analysis/verify_delta_stability_run.py "${verify_args[@]}"
echo "[$(date --iso-8601=seconds)] ${method^^} lane complete: ${verification}"
