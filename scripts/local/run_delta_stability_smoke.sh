#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd -P)"
smoke_root="${repo_root}/runs/delta-stability-smoke-v1"
exp_name="delta_stability_o0_random_orthogonal_r32_seed42_smoke_v1"

[[ ! -e "${smoke_root}" ]] || { echo "Refusing existing smoke output: ${smoke_root}" >&2; exit 2; }
mkdir -p "${smoke_root}"

ARTIFACT_ROOT="${smoke_root}" \
STORAGE_ROOT="${smoke_root}/storage" \
EXP_NAME="${exp_name}" \
TOTAL_TRAINING_STEPS=1 \
SAVE_FREQ=0 \
SAVE_STEPS=null \
MAX_ACTOR_CKPT_TO_KEEP=1 \
LORA_SNAPSHOT_EVERY=1 \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
RAY_TEMP_DIR="/dev/shm/ray-delta-o0-smoke-v1" \
TMPDIR="/dev/shm/tmp-delta-o0-smoke-v1" \
bash "${script_dir}/start_delta_stability_uniform_r32_4gpu.sh" o0

for step in 000 001; do
    test -s "${smoke_root}/storage/snapshots/${exp_name}/step_${step}/adapter_model.safetensors"
    test -s "${smoke_root}/storage/snapshots/${exp_name}/step_${step}/manifest.json"
done
test -s "${smoke_root}/analysis/initialization/${exp_name}.json"
echo "DELTA_STABILITY_SMOKE_OK ${smoke_root}"
