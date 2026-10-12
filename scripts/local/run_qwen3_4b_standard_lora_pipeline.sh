#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd -P)"
python_bin="${PYTHON_BIN:-/home/wangls/miniconda3/envs/peft-for-rl/bin/python}"
artifact_root="${ARTIFACT_ROOT:-${repo_root}/runs/qwen3-4b-standard-lora-v1}"
storage_root="${STORAGE_ROOT:-/media/nfs_share/wangls/peft-for-rl-external-validity-v1/qwen3-4b-standard-lora-v1}"
train_name="${TRAIN_NAME:-qwen3_4b_standard_lora_r32a64_b64m16n8_step50_seed42_v1}"
eval_name="${EVAL_NAME:-${train_name}_fullbench_32768_seed42}"
base_model="${MODEL_PATH:-/home/wangls/models/Qwen3-4B}"
train_file="${TRAIN_FILE:-${repo_root}/runs/phase1-signal-random-v1/imported-runtime/datasets/verl_dapo/data/dapo-math-17k-boxed.parquet}"
cuda_devices="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
ngpus_per_node="${NGPUS_PER_NODE:-4}"
ray_num_cpus="${RAY_NUM_CPUS:-$((ngpus_per_node * 8))}"
ray_temp_dir="${RAY_TEMP_DIR:-/dev/shm/ray-qwen3-standard-lora-${ngpus_per_node}gpu-v1}"
training_tmpdir="${TRAINING_TMPDIR:-/dev/shm/tmp-qwen3-standard-lora-${ngpus_per_node}gpu-v1}"
snapshot="${repo_root}/runs/phase1-signal-random-v1/imported-runtime/outputs/full_bench_eval/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42/records/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42.jsonl"
ckpt_dir="${storage_root}/ckpts/verl/DAPO-Math-17k/${train_name}"
adapter_dir="${ckpt_dir}/global_step_50/actor/peft_adapter"
adapter_file="${adapter_dir}/adapter_model.safetensors"
audit="${artifact_root}/analysis/initialization/${train_name}.json"
contract="${artifact_root}/analysis/training_manifests/${train_name}.json"
eval_dir="${storage_root}/outputs/full_bench_eval/${eval_name}"
summary="${eval_dir}/summary/${eval_name}.json"
records="${eval_dir}/records/${eval_name}.jsonl"
postverify="${artifact_root}/analysis/verification/${train_name}_postverify.json"
pipeline_log="${artifact_root}/logs/${train_name}_pipeline.log"
smoke_result="${repo_root}/runs/qwen3-4b-standard-lora-smoke-v1/analysis/qwen3_4b_standard_lora_r32a64_b64m16n8_smoke_seed42_v1_result.json"
direct_opd_eval_root="/home/wangls/Direct-OPD/datasets/eval"

IFS=',' read -r -a gpu_ids <<< "${cuda_devices}"
[[ "${#gpu_ids[@]}" -eq "${ngpus_per_node}" ]] || {
    echo "CUDA_VISIBLE_DEVICES exposes ${#gpu_ids[@]} GPUs, expected ${ngpus_per_node}" >&2
    exit 2
}

"${python_bin}" - "${smoke_result}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file() or json.loads(path.read_text()).get("status") != "pass":
    raise SystemExit(f"Qwen3-4B smoke has not passed: {path}")
PY
for path in "${contract}" "${ckpt_dir}" "${eval_dir}" "${postverify}" "${audit}"; do
    [[ ! -e "${path}" ]] || { echo "Refusing existing formal output: ${path}" >&2; exit 2; }
done
mkdir -p "${artifact_root}/logs" "${artifact_root}/analysis/training_manifests"
exec > >(tee -a "${pipeline_log}") 2>&1
cd "${repo_root}"
export PYTHONPATH="${repo_root}"

echo "[$(date --iso-8601=seconds)] creating Qwen3-4B standard-LoRA contract"
"${python_bin}" scripts/analysis/qwen3_standard_lora_training_contract.py create \
    --experiment-name "${train_name}" --training-seed 42 \
    --model "${base_model}" --train-file "${train_file}" --snapshot "${snapshot}" \
    --audit "${audit}" --contract "${contract}" >/dev/null

echo "[$(date --iso-8601=seconds)] starting Qwen3-4B standard-LoRA training on ${ngpus_per_node} GPUs (${cuda_devices})"
ARTIFACT_ROOT="${artifact_root}" STORAGE_ROOT="${storage_root}" EXP_NAME="${train_name}" \
CKPTS_DIR="${ckpt_dir}" MODEL_PATH="${base_model}" TRAIN_FILE="${train_file}" \
LORA_INIT_AUDIT_PATH="${audit}" CUDA_VISIBLE_DEVICES="${cuda_devices}" \
NGPUS_PER_NODE="${ngpus_per_node}" RAY_NUM_CPUS="${ray_num_cpus}" \
RAY_TEMP_DIR="${ray_temp_dir}" TMPDIR="${training_tmpdir}" \
bash scripts/local/start_qwen3_4b_standard_lora_4gpu.sh
[[ -s "${adapter_file}" ]] || { echo "Training ended without adapter: ${adapter_file}" >&2; exit 1; }

"${python_bin}" scripts/analysis/qwen3_standard_lora_training_contract.py finalize \
    --contract "${contract}" --adapter "${adapter_file}" >/dev/null
"${python_bin}" scripts/analysis/qwen3_standard_lora_training_contract.py verify \
    --contract "${contract}" --adapter "${adapter_file}" >/dev/null

echo "[$(date --iso-8601=seconds)] starting fixed six-benchmark Qwen3-4B evaluation"
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
PROJECT_ROOT="${repo_root}/tina_run" PYTHON_BIN="${python_bin}" \
BASE_MODEL="${base_model}" LORA_ADAPTER="${adapter_dir}" \
DIRECT_OPD_EVAL_ROOT="${direct_opd_eval_root}" \
CHECKPOINT_NAME="${eval_name}" OUTPUT_DIR="${eval_dir}" \
BENCHMARK_SNAPSHOT_RECORDS="${snapshot}" \
BENCHMARKS="aime24,aime25,amc23,hmmt_feb,math500,minerva" \
SEED=42 TEMPERATURE=0.6 TOP_P=0.95 MAX_NEW_TOKENS=32768 MAX_MODEL_LEN=34816 \
SAMPLES_SMALL=32 SAMPLES_LARGE=4 CUDA_VISIBLE_DEVICES="${cuda_devices}" \
bash tina_run/scripts/local/eval/run_full_bench_vllm_4gpu.sh

"${python_bin}" scripts/analysis/qwen3_standard_lora_training_contract.py postverify \
    --contract "${contract}" --adapter "${adapter_file}" \
    --summary "${summary}" --records "${records}" --eval-dir "${eval_dir}" \
    --eval-name "${eval_name}" --output "${postverify}" >/dev/null
echo "[$(date --iso-8601=seconds)] Qwen3-4B standard-LoRA baseline pipeline complete"
