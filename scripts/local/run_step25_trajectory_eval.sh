#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 [--preflight] <standard|i8|i32>" >&2
}

preflight_only=0
if [[ "${1:-}" == "--preflight" ]]; then
    preflight_only=1
    shift
fi
if (( $# != 1 )); then
    usage
    exit 2
fi
method="$1"
case "${method}" in
    standard|i8|i32) ;;
    *) usage; exit 2 ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd -P)"
python_bin="${PYTHON_BIN:-/home/wangls/miniconda3/envs/peft-for-rl/bin/python}"
contract="${repo_root}/runs/step25-trajectory-v1/analysis/step25_trajectory_evaluation_contract_20261006.json"
log_root="${repo_root}/runs/step25-trajectory-v1/logs"
direct_opd_eval_root="$(cd -- "${repo_root}/.." && pwd -P)/Direct-OPD/datasets/eval"

fail() {
    echo "STEP25_TRAJECTORY_FAILED method=${method} $*" >&2
    exit 2
}

for tool in "${python_bin}" jq sha256sum; do
    if [[ "${tool}" == */* ]]; then
        [[ -x "${tool}" ]] || fail "missing executable: ${tool}"
    else
        command -v "${tool}" >/dev/null || fail "missing executable: ${tool}"
    fi
done
[[ -s "${contract}" ]] || fail "missing contract: ${contract}"
jq -e '.status == "authorized"' "${contract}" >/dev/null || fail "contract is not authorized"

entry="$(jq -ce --arg method "${method}" '.evaluations[] | select(.method == $method)' "${contract}")"
[[ -n "${entry}" ]] || fail "method is absent from contract"
base_model="$(jq -r '.base_model.path' "${contract}")"
base_sha256="$(jq -r '.base_model.weights_sha256' "${contract}")"
snapshot="$(jq -r '.benchmark_snapshot.path' "${contract}")"
snapshot_sha256="$(jq -r '.benchmark_snapshot.sha256' "${contract}")"
adapter_dir="$(jq -r '.adapter_path' <<<"${entry}")"
adapter_sha256="$(jq -r '.adapter_sha256' <<<"${entry}")"
eval_name="$(jq -r '.evaluation_name' <<<"${entry}")"
eval_dir="$(jq -r '.output_dir' <<<"${entry}")"
postverify="$(jq -r '.postverify_path' <<<"${entry}")"
gpu_group="$(jq -r '.gpu_group' <<<"${entry}")"
adapter_file="${adapter_dir}/adapter_model.safetensors"
adapter_config="${adapter_dir}/adapter_config.json"
summary="${eval_dir}/summary/${eval_name}.json"
records="${eval_dir}/records/${eval_name}.jsonl"

for required in "${base_model}/model.safetensors" "${snapshot}" "${adapter_file}" "${adapter_config}" "${direct_opd_eval_root}"; do
    [[ -e "${required}" ]] || fail "missing required input: ${required}"
done
[[ "$(sha256sum "${base_model}/model.safetensors" | awk '{print $1}')" == "${base_sha256}" ]] || fail "base-model hash mismatch"
[[ "$(sha256sum "${snapshot}" | awk '{print $1}')" == "${snapshot_sha256}" ]] || fail "snapshot hash mismatch"
[[ "$(sha256sum "${adapter_file}" | awk '{print $1}')" == "${adapter_sha256}" ]] || fail "adapter hash mismatch"
jq -e '
    .r == 32 and
    .lora_alpha == 64 and
    .lora_dropout == 0 and
    .bias == "none" and
    ((.target_modules | sort) == (["down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"] | sort))
' "${adapter_config}" >/dev/null || fail "adapter config is not protocol-matched r32/alpha64/dropout0/all-linear"
[[ ! -e "${eval_dir}" ]] || fail "refusing existing evaluation output: ${eval_dir}"
[[ ! -e "${postverify}" ]] || fail "refusing existing postverify: ${postverify}"

if (( preflight_only == 1 )); then
    echo "STEP25_TRAJECTORY_PREFLIGHT_OK method=${method} gpu_group=${gpu_group} adapter_sha256=${adapter_sha256} output=${eval_dir}"
    exit 0
fi

mkdir -p "${log_root}" "$(dirname -- "${postverify}")"
exec > >(tee -a "${log_root}/${eval_name}.log") 2>&1
cd "${repo_root}"
export PYTHONPATH="${repo_root}"
echo "[$(date --iso-8601=seconds)] starting method=${method} eval=${eval_name} GPUs=${gpu_group}"

HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
PROJECT_ROOT="${repo_root}/tina_run" PYTHON_BIN="${python_bin}" \
BASE_MODEL="${base_model}" LORA_ADAPTER="${adapter_dir}" \
DIRECT_OPD_EVAL_ROOT="${direct_opd_eval_root}" \
CHECKPOINT_NAME="${eval_name}" OUTPUT_DIR="${eval_dir}" \
BENCHMARK_SNAPSHOT_RECORDS="${snapshot}" \
BENCHMARKS="aime24,aime25,amc23,hmmt_feb,math500,minerva" \
SEED=42 TEMPERATURE=0.6 TOP_P=0.95 MAX_NEW_TOKENS=32768 MAX_MODEL_LEN=34816 \
SAMPLES_SMALL=32 SAMPLES_LARGE=4 LIMIT_PER_BENCHMARK= \
CUDA_VISIBLE_DEVICES="${gpu_group}" \
bash tina_run/scripts/local/eval/run_full_bench_vllm_4gpu.sh

verification_json="$(
    "${python_bin}" scripts/analysis/verify_full_bench_run.py \
        --summary "${summary}" \
        --records "${records}" \
        --eval-dir "${eval_dir}" \
        --eval-name "${eval_name}" \
        --adapter "${adapter_dir}" \
        --base-model "${base_model}" \
        --snapshot "${snapshot}" \
        --snapshot-sha256 "${snapshot_sha256}"
)"

postverify_tmp="$(mktemp "${postverify}.tmp.XXXXXX")"
trap 'rm -f "${postverify_tmp}"' EXIT
jq -n \
    --arg created_at "$(date --iso-8601=seconds)" \
    --arg method "${method}" \
    --arg eval_name "${eval_name}" \
    --arg contract "${contract}" \
    --arg contract_sha256 "$(sha256sum "${contract}" | awk '{print $1}')" \
    --arg adapter "${adapter_file}" \
    --arg adapter_sha256 "${adapter_sha256}" \
    --arg adapter_config "${adapter_config}" \
    --arg adapter_config_sha256 "$(sha256sum "${adapter_config}" | awk '{print $1}')" \
    --arg records "${records}" \
    --arg records_sha256 "$(sha256sum "${records}" | awk '{print $1}')" \
    --arg summary "${summary}" \
    --arg summary_sha256 "$(sha256sum "${summary}" | awk '{print $1}')" \
    --argjson verification "${verification_json}" \
    '{
      schema_version: 1,
      status: "verified",
      method: "step25_trajectory_fullbench_postverification_v1",
      candidate_method: $method,
      created_at: $created_at,
      evaluation_name: $eval_name,
      verification: $verification,
      artifacts: {
        contract: {path: $contract, sha256: $contract_sha256},
        adapter: {path: $adapter, sha256: $adapter_sha256},
        adapter_config: {path: $adapter_config, sha256: $adapter_config_sha256},
        records: {path: $records, sha256: $records_sha256},
        summary: {path: $summary, sha256: $summary_sha256}
      }
    }' > "${postverify_tmp}"
mv "${postverify_tmp}" "${postverify}"
trap - EXIT
echo "[$(date --iso-8601=seconds)] STEP25_TRAJECTORY_COMPLETE method=${method} postverify=${postverify}"
