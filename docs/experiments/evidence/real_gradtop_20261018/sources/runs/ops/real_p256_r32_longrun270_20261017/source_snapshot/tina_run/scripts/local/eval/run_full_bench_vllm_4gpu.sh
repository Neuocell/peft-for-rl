#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUNTIME_ROOT="${RUNTIME_ROOT:-${PROJECT_ROOT}/runs}"

BASE_MODEL="${BASE_MODEL:-${RUNTIME_ROOT}/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base}"
LORA_ADAPTER="${LORA_ADAPTER:-}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:?CHECKPOINT_NAME is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
BENCHMARKS="${BENCHMARKS:-aime24,aime25,amc23,hmmt_feb,math500,minerva}"
SEED="${SEED:-42}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-34816}"
SAMPLES_SMALL="${SAMPLES_SMALL:-32}"
SAMPLES_LARGE="${SAMPLES_LARGE:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
LIMIT_PER_BENCHMARK="${LIMIT_PER_BENCHMARK:-}"
DRY_RUN_NO_MODEL="${DRY_RUN_NO_MODEL:-0}"
BENCHMARK_SNAPSHOT_RECORDS="${BENCHMARK_SNAPSHOT_RECORDS:-}"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_SHARDS="${#GPU_IDS[@]}"
if [[ "${NUM_SHARDS}" -lt 1 ]]; then
  echo "No GPUs specified in CUDA_VISIBLE_DEVICES" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}/logs"

common_args=(
  --base_model "${BASE_MODEL}"
  --checkpoint_name "${CHECKPOINT_NAME}"
  --output_dir "${OUTPUT_DIR}"
  --benchmarks "${BENCHMARKS}"
  --seed "${SEED}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --max_model_len "${MAX_MODEL_LEN}"
  --samples_small "${SAMPLES_SMALL}"
  --samples_large "${SAMPLES_LARGE}"
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}"
  --num_shards "${NUM_SHARDS}"
)

if [[ -n "${LORA_ADAPTER}" ]]; then
  common_args+=(--lora_adapter "${LORA_ADAPTER}")
fi
if [[ -n "${LIMIT_PER_BENCHMARK}" ]]; then
  common_args+=(--limit_per_benchmark "${LIMIT_PER_BENCHMARK}")
fi
if [[ "${DRY_RUN_NO_MODEL}" == "1" ]]; then
  common_args+=(--dry_run_no_model)
fi
if [[ -n "${BENCHMARK_SNAPSHOT_RECORDS}" ]]; then
  common_args+=(--benchmark_snapshot_records "${BENCHMARK_SNAPSHOT_RECORDS}")
fi

pids=()
for shard in "${!GPU_IDS[@]}"; do
  gpu="${GPU_IDS[$shard]}"
  log_file="${OUTPUT_DIR}/logs/${CHECKPOINT_NAME}.shard-${shard}-of-${NUM_SHARDS}.log"
  echo "Launching shard ${shard}/${NUM_SHARDS} on GPU ${gpu}; log: ${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
    scripts/local/eval/eval_full_bench_vllm.py \
    "${common_args[@]}" \
    --shard_index "${shard}" \
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
  echo "At least one shard failed. See ${OUTPUT_DIR}/logs" >&2
  exit "${status}"
fi

if [[ "${DRY_RUN_NO_MODEL}" != "1" ]]; then
  "${PYTHON_BIN}" scripts/local/eval/eval_full_bench_vllm.py \
    --base_model "${BASE_MODEL}" \
    --checkpoint_name "${CHECKPOINT_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --benchmarks "${BENCHMARKS}" \
    --num_shards "${NUM_SHARDS}" \
    --aggregate_only \
    > "${OUTPUT_DIR}/logs/${CHECKPOINT_NAME}.aggregate.log" 2>&1
fi

echo "Done: ${OUTPUT_DIR}"
