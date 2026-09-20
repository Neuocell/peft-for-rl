#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
TINA_ROOT="${REPO_ROOT}/tina_run"
RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/runs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MODEL="${BASE_MODEL:-${RUNTIME_ROOT}/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUNTIME_ROOT}/outputs/linear_extrapolation_screen}"
BENCHMARKS="${BENCHMARKS:-aime24,aime25,amc23,hmmt_feb,math500,minerva}"

names=(
  gradtop_step100_270_chord_lambda0_screen48x4
  gradtop_step100_270_chord_lambda0p5_screen48x4
  gradtop_step100_270_chord_lambda1_screen48x4
  gradtop_step100_270_chord_lambda1p25_screen48x4
)
adapters=(
  "${RUNTIME_ROOT}/initializers/gradtop_step100_270_chord_lambda0"
  "${RUNTIME_ROOT}/initializers/gradtop_step100_270_chord_lambda0p5"
  "${RUNTIME_ROOT}/initializers/gradtop_step100_270_chord_lambda1"
  "${RUNTIME_ROOT}/initializers/gradtop_step100_270_chord_lambda1p25"
)
gpus=(0 1 2 3)

mkdir -p "${OUTPUT_ROOT}"
cd "${TINA_ROOT}"

pids=()
for index in "${!names[@]}"; do
  name="${names[$index]}"
  adapter="${adapters[$index]}"
  gpu="${gpus[$index]}"
  output_dir="${OUTPUT_ROOT}/${name}"
  log_file="${output_dir}/logs/${name}.log"
  mkdir -p "${output_dir}/logs"
  echo "Launching ${name} on GPU ${gpu}; log: ${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/local/eval/eval_full_bench_vllm.py \
    --base_model "${BASE_MODEL}" \
    --lora_adapter "${adapter}" \
    --checkpoint_name "${name}" \
    --output_dir "${output_dir}" \
    --benchmarks "${BENCHMARKS}" \
    --seed 42 \
    --temperature 0.6 \
    --top_p 0.95 \
    --max_new_tokens 8192 \
    --max_model_len 9216 \
    --samples_small 4 \
    --samples_large 4 \
    --num_shards 1 \
    --shard_index 0 \
    --limit_per_benchmark 8 \
    --gpu_memory_utilization 0.85 \
    > "${log_file}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "At least one chord-screen worker failed; inspect ${OUTPUT_ROOT}/*/logs" >&2
  exit "${status}"
fi

for name in "${names[@]}"; do
  output_dir="${OUTPUT_ROOT}/${name}"
  "${PYTHON_BIN}" scripts/local/eval/eval_full_bench_vllm.py \
    --base_model "${BASE_MODEL}" \
    --checkpoint_name "${name}" \
    --output_dir "${output_dir}" \
    --benchmarks "${BENCHMARKS}" \
    --num_shards 1 \
    --aggregate_only \
    > "${output_dir}/logs/${name}.aggregate.log" 2>&1
done

echo "Chord screen complete: ${OUTPUT_ROOT}"
