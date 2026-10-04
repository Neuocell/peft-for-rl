#!/usr/bin/env bash
set -euo pipefail

repo_root="/root/peft-for-rl"
tina_root="${repo_root}/tina_run"
python_bin="/home/node/anaconda3/envs/peft-for-rl/bin/python"
train_name="phase1_i0_uniform_r32_b64m16n8_step50_seed43_v3"
eval_name="${train_name}_fullbench_32768_seed42"
eval_dir="${repo_root}/runs/phase1-signal-random-v1/outputs/full_bench_eval/${eval_name}"
adapter="${repo_root}/runs/phase1-signal-random-v1/ckpts/verl/DAPO-Math-17k/${train_name}/global_step_50/actor/peft_adapter"
base_model="/data/peft-for-rl-runtime/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base"
snapshot="/data/peft-for-rl-runtime/outputs/full_bench_eval/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42/records/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42.jsonl"
snapshot_sha256="3416ba93286b324a9663777fa472d0b568af5446319f27f876a63b59b3e863da"
preparation="${repo_root}/runs/phase1-signal-random-v1/analysis/training_allocations/phase1_training_preparation.json"
preparation_sha256="d60fc87775a5b1a0e1d27f922b6c4f2b100f0e8614faeb261afaf2c1c88dd386"
contract="${repo_root}/runs/phase1-signal-random-v1/analysis/training_manifests/${train_name}.json"
contract_verifier="${repo_root}/scripts/analysis/phase1_training_contract.py"
contract_verifier_sha256="7660390028739a1db83c1fa4de96235faf754e4fcbad604d946c0aab283bf80c"
evaluator="${tina_root}/scripts/local/eval/eval_full_bench_vllm.py"
evaluator_sha256="09e67123d5f345a5899b5299a1f046859d199a8d12dec2e25d03e0605b2e404a"
verifier="${repo_root}/scripts/analysis/verify_full_bench_run.py"
verifier_sha256="4bcd33a1a336397c56778eb866d39fc65b70229c7cb2e9fdc80b412745dea3f9"
summary="${eval_dir}/summary/${eval_name}.json"
records="${eval_dir}/records/${eval_name}.jsonl"
dry_run=0

usage() {
    cat <<'EOF'
Usage: resume_seed43_i0_eval_missing_shards.sh [--dry-run]

Validates and reuses complete seed-43 I0 evaluation shards, launches only
shards whose record and manifest files are both absent, aggregates all four
shards, and runs the frozen verifier. Any partial shard state fails closed.
CUDA_VISIBLE_DEVICES may provide a comma-separated list of available GPUs;
it defaults to 0,1,2,3. This script never starts training or another method.
EOF
}

if (( $# > 1 )); then
    usage >&2
    exit 2
fi
if (( $# == 1 )); then
    if [[ "$1" != "--dry-run" ]]; then
        usage >&2
        exit 2
    fi
    dry_run=1
fi

sha256_check() {
    local expected="$1"
    local path="$2"
    local actual
    actual="$(sha256sum "${path}" | awk '{print $1}')"
    if [[ "${actual}" != "${expected}" ]]; then
        echo "Pinned file changed: ${path} expected=${expected} actual=${actual}" >&2
        exit 2
    fi
}

[[ -x "${python_bin}" ]] || { echo "Missing Python: ${python_bin}" >&2; exit 2; }
[[ -s "${adapter}/adapter_model.safetensors" ]] || {
    echo "Missing seed-43 I0 step-50 adapter: ${adapter}" >&2
    exit 2
}
[[ -s "${snapshot}" ]] || { echo "Missing benchmark snapshot: ${snapshot}" >&2; exit 2; }
sha256_check "${snapshot_sha256}" "${snapshot}"
sha256_check "${preparation_sha256}" "${preparation}"
sha256_check "${contract_verifier_sha256}" "${contract_verifier}"
sha256_check "${evaluator_sha256}" "${evaluator}"
sha256_check "${verifier_sha256}" "${verifier}"

contract_output="$(
    cd "${repo_root}"
    "${python_bin}" "${contract_verifier}" verify \
        --preparation-kind phase1 \
        --preparation "${preparation}" \
        --method I0 \
        --training-seed 43 \
        --experiment-name "${train_name}" \
        --contract "${contract}" \
        --adapter "${adapter}/adapter_model.safetensors"
)"
printf '%s\n' "${contract_output}" | "${python_bin}" -c '
import json
import sys

value = json.load(sys.stdin)
expected = {
    "status": "verified",
    "candidate_method": "I0",
    "training_seed": 43,
    "experiment_name": "phase1_i0_uniform_r32_b64m16n8_step50_seed43_v3",
}
for field, expected_value in expected.items():
    if value.get(field) != expected_value:
        raise SystemExit(f"Training-contract verification changed: {field}")
if not value.get("adapter_sha256"):
    raise SystemExit("Training-contract verification omitted adapter_sha256")
'
echo "TRAINING_CONTRACT_AND_ADAPTER_VERIFIED $(printf '%s\n' "${contract_output}" | "${python_bin}" -c 'import json,sys; print(json.load(sys.stdin)["adapter_sha256"])')"

verify_complete_run() {
    "${python_bin}" "${verifier}" \
        --summary "${summary}" \
        --records "${records}" \
        --eval-dir "${eval_dir}" \
        --eval-name "${eval_name}" \
        --adapter "${adapter}" \
        --base-model "${base_model}" \
        --snapshot "${snapshot}" \
        --snapshot-sha256 "${snapshot_sha256}"
}

if [[ -s "${summary}" ]]; then
    verify_complete_run
    echo "Evaluation is already complete and verified: ${summary}"
    exit 0
fi

if pgrep -af "[e]val_full_bench_vllm.py.*${eval_name}" >/dev/null; then
    echo "Refusing recovery while the target evaluation is still running" >&2
    pgrep -af "[e]val_full_bench_vllm.py.*${eval_name}" >&2 || true
    exit 3
fi
if pgrep -af "[V]LLM::EngineCore" >/dev/null; then
    echo "Refusing recovery while a residual vLLM EngineCore exists" >&2
    pgrep -af "[V]LLM::EngineCore" >&2 || true
    exit 3
fi

validate_shard() {
    local shard_index="$1"
    local shard_tag
    local shard_path
    local manifest_path
    shard_tag="$(printf '%02d' "${shard_index}")"
    shard_path="${eval_dir}/shards/${eval_name}.shard-${shard_tag}-of-04.jsonl"
    manifest_path="${eval_dir}/manifests/${eval_name}.shard-${shard_tag}-of-04.json"

    "${python_bin}" - \
        "${shard_path}" "${manifest_path}" "${shard_index}" \
        "${snapshot}" "${snapshot_sha256}" "${eval_name}" \
        "${adapter}" "${base_model}" <<'PY'
import json
import sys
from pathlib import Path

(
    shard_arg,
    manifest_arg,
    shard_index_arg,
    snapshot_arg,
    snapshot_sha256,
    eval_name,
    adapter_arg,
    base_model_arg,
) = sys.argv[1:]
shard_path = Path(shard_arg).resolve()
manifest_path = Path(manifest_arg).resolve()
snapshot_path = Path(snapshot_arg).resolve()
adapter = Path(adapter_arg).resolve()
base_model = Path(base_model_arg).resolve()
shard_index = int(shard_index_arg)
benchmarks = ("aime24", "aime25", "amc23", "hmmt_feb", "math500", "minerva")
benchmark_sizes = {
    "aime24": 30,
    "aime25": 30,
    "amc23": 40,
    "hmmt_feb": 30,
    "math500": 500,
    "minerva": 272,
}
small = {"aime24", "aime25", "amc23", "hmmt_feb"}


def fail(message: str) -> None:
    raise SystemExit(message)


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"Invalid JSON in {path} at line {line_number}: {exc}")


if not shard_path.is_file() or shard_path.stat().st_size <= 0:
    fail(f"Missing shard: {shard_path}")
if not manifest_path.is_file() or manifest_path.stat().st_size <= 0:
    fail(f"Missing manifest for existing shard: {manifest_path}")

source = {}
for _, record in iter_jsonl(snapshot_path):
    benchmark = str(record.get("benchmark", ""))
    if benchmark not in benchmark_sizes:
        continue
    key = (benchmark, int(record["problem_index"]))
    invariant = (str(record["id"]), str(record["problem"]), str(record["gold_answer"]))
    previous = source.setdefault(key, invariant)
    if previous != invariant:
        fail(f"Snapshot invariant changed within problem: {key}")
for benchmark, size in benchmark_sizes.items():
    keys = sorted(index for name, index in source if name == benchmark)
    if keys != list(range(size)):
        fail(f"Snapshot problem indices changed for {benchmark}")

expected = set()
position = 0
for benchmark in benchmarks:
    samples = 32 if benchmark in small else 4
    for problem_index in range(benchmark_sizes[benchmark]):
        for sample_index in range(samples):
            if position % 4 == shard_index:
                expected.add((benchmark, problem_index, sample_index))
            position += 1
if len(expected) != 1812:
    fail(f"Internal expected shard size is {len(expected)} instead of 1812")

actual = set()
row_count = 0
for line_number, record in iter_jsonl(shard_path):
    row_count += 1
    try:
        key = (
            str(record["benchmark"]),
            int(record["problem_index"]),
            int(record["sample_index"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        fail(f"Invalid key in shard {shard_index} line {line_number}: {exc}")
    if key in actual:
        fail(f"Duplicate key in shard {shard_index}: {key}")
    actual.add(key)
    benchmark, problem_index, _ = key
    samples = 32 if benchmark in small else 4
    fixed = {
        "checkpoint": eval_name,
        "checkpoint_type": "lora_adapter",
        "seed": 42,
        "shard_index": shard_index,
        "num_shards": 4,
        "temperature": 0.6,
        "top_p": 0.95,
        "max_new_tokens": 32768,
        "max_model_len": 34816,
        "samples_per_problem": samples,
    }
    for field, value in fixed.items():
        if record.get(field) != value:
            fail(f"Record field {field} changed in shard {shard_index} line {line_number}")
    if Path(str(record.get("base_model", ""))).resolve() != base_model:
        fail(f"Record base model changed in shard {shard_index} line {line_number}")
    if Path(str(record.get("lora_adapter", ""))).resolve() != adapter:
        fail(f"Record adapter changed in shard {shard_index} line {line_number}")
    invariant = (
        str(record.get("id", "")),
        str(record.get("problem", "")),
        str(record.get("gold_answer", "")),
    )
    if invariant != source.get((benchmark, problem_index)):
        fail(f"Problem invariant changed in shard {shard_index}: {key}")
    if type(record.get("correct")) is not bool or type(record.get("format_valid")) is not bool:
        fail(f"Non-Boolean score field in shard {shard_index} line {line_number}")
    if not isinstance(record.get("completion"), str):
        fail(f"Missing completion in shard {shard_index} line {line_number}")
    if not isinstance(record.get("completion_length"), int) or record["completion_length"] < 0:
        fail(f"Invalid completion length in shard {shard_index} line {line_number}")
if row_count != 1812:
    fail(f"Shard {shard_index} has {row_count} rows instead of 1812")
if actual != expected:
    fail(
        f"Shard {shard_index} request partition changed: "
        f"missing={len(expected - actual)} extra={len(actual - expected)}"
    )

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
expected_manifest = {
    "checkpoint": eval_name,
    "benchmarks": list(benchmarks),
    "total_requests": 7248,
    "shard_requests": 1812,
    "num_shards": 4,
    "shard_index": shard_index,
    "seed": 42,
    "temperature": 0.6,
    "top_p": 0.95,
    "max_new_tokens": 32768,
    "max_model_len": 34816,
    "samples_small": 32,
    "samples_large": 4,
    "limit_per_benchmark": None,
    "benchmark_sizes": benchmark_sizes,
    "benchmark_snapshot_sha256": snapshot_sha256,
}
for field, value in expected_manifest.items():
    if manifest.get(field) != value:
        fail(f"Manifest field changed for shard {shard_index}: {field}")
for field, expected_path in (
    ("base_model", base_model),
    ("lora_adapter", adapter),
    ("benchmark_snapshot_records", snapshot_path),
):
    if Path(str(manifest.get(field, ""))).resolve() != expected_path:
        fail(f"Manifest path changed for shard {shard_index}: {field}")
print(f"VERIFIED_COMPLETE_SHARD {shard_index} rows=1812")
PY
}

missing_shards=()
for shard_index in 0 1 2 3; do
    shard_tag="$(printf '%02d' "${shard_index}")"
    shard_path="${eval_dir}/shards/${eval_name}.shard-${shard_tag}-of-04.jsonl"
    manifest_path="${eval_dir}/manifests/${eval_name}.shard-${shard_tag}-of-04.json"
    if [[ ! -e "${shard_path}" && ! -e "${manifest_path}" ]]; then
        missing_shards+=("${shard_index}")
        continue
    fi
    [[ -e "${shard_path}" && -e "${manifest_path}" ]] || {
        echo "Refusing partial shard state ${shard_index}: shard and manifest must both exist or both be absent" >&2
        exit 2
    }
    [[ -s "${shard_path}" && -s "${manifest_path}" ]] || {
        echo "Refusing empty shard state ${shard_index}: existing shard and manifest must both be non-empty" >&2
        exit 2
    }
    validate_shard "${shard_index}"
done

IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if (( ${#missing_shards[@]} > ${#gpu_ids[@]} )); then
    echo "Need ${#missing_shards[@]} GPUs but only ${#gpu_ids[@]} were provided" >&2
    exit 2
fi

echo "Verified complete shards: $((4 - ${#missing_shards[@]}))"
if (( ${#missing_shards[@]} > 0 )); then
    echo "Missing shards: ${missing_shards[*]}"
fi
for position in "${!missing_shards[@]}"; do
    echo "Recovery mapping: shard ${missing_shards[$position]} -> GPU ${gpu_ids[$position]}"
done
if (( dry_run == 1 )); then
    echo "DRY_RUN no shard launched and no aggregate written"
    exit 0
fi

mkdir -p "${eval_dir}/logs"
pids=()
for position in "${!missing_shards[@]}"; do
    shard_index="${missing_shards[$position]}"
    gpu="${gpu_ids[$position]}"
    log="${eval_dir}/logs/${eval_name}.shard-${shard_index}-of-4.log"
    {
        echo "[$(date --iso-8601=seconds)] RECOVERY_LAUNCH shard=${shard_index} gpu=${gpu}"
    } >>"${log}"
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    DIRECT_OPD_EVAL_ROOT="/data/peft-for-rl-runtime/datasets/Direct-OPD/datasets/eval" \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    "${python_bin}" "${evaluator}" \
        --base_model "${base_model}" \
        --lora_adapter "${adapter}" \
        --checkpoint_name "${eval_name}" \
        --output_dir "${eval_dir}" \
        --benchmarks "aime24,aime25,amc23,hmmt_feb,math500,minerva" \
        --seed 42 \
        --temperature 0.6 \
        --top_p 0.95 \
        --max_new_tokens 32768 \
        --max_model_len 34816 \
        --samples_small 32 \
        --samples_large 4 \
        --gpu_memory_utilization 0.85 \
        --num_shards 4 \
        --shard_index "${shard_index}" \
        --benchmark_snapshot_records "${snapshot}" \
        >>"${log}" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done
if (( status != 0 )); then
    echo "At least one recovery shard failed; preserving all outputs for audit" >&2
    exit 1
fi

for shard_index in 0 1 2 3; do
    validate_shard "${shard_index}"
done

cd "${tina_root}"
"${python_bin}" "${evaluator}" \
    --base_model "${base_model}" \
    --checkpoint_name "${eval_name}" \
    --output_dir "${eval_dir}" \
    --benchmarks "aime24,aime25,amc23,hmmt_feb,math500,minerva" \
    --num_shards 4 \
    --aggregate_only \
    >"${eval_dir}/logs/${eval_name}.aggregate-recovery.log" 2>&1

verify_complete_run
echo "RECOVERY_COMPLETE_AND_VERIFIED ${summary}"
