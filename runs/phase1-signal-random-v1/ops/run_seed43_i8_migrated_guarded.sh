#!/usr/bin/env bash
set -euo pipefail

canonical_repo_root="/root/peft-for-rl"
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
runtime_root="${canonical_repo_root}/runs/phase1-signal-random-v1"
python_bin="/home/node/anaconda3/envs/peft-for-rl/bin/python"
method_launcher="${canonical_repo_root}/scripts/local/run_phase1_signal_random_step50_fullbench.sh"
postverify_script="${runtime_root}/ops/seed43_i0_postverify.sh"
i8_name="phase1_i8_uniform_r32_b64m16n8_step50_seed43_v3"
i8_eval_name="${i8_name}_fullbench_32768_seed42"
minimum_root_free_bytes=16500000000
execute=0
payload_only=0

usage() {
    cat <<'EOF'
Usage: run_seed43_i8_migrated_guarded.sh [--payload-only | --execute]

With no option, validate the complete migrated seed-43 I8 preflight without
starting a process. --payload-only verifies repository, payload, base-model,
and initialization hashes but skips runtime/storage gates. --execute launches
only seed-43 I8 revision 3, postverifies it, and then exits.
EOF
}

fail() {
    echo "SEED43_I8_MIGRATED_START_REFUSED $*" >&2
    exit 2
}

while (( $# > 0 )); do
    case "$1" in
        --execute)
            execute=1
            shift
            ;;
        --payload-only)
            payload_only=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            fail "unknown argument: $1"
            ;;
    esac
done
(( execute == 0 || payload_only == 0 )) || fail "--execute and --payload-only are mutually exclusive"

[[ "${repo_root}" == "${canonical_repo_root}" ]] || {
    fail "checkout must resolve to ${canonical_repo_root}; use a bind mount for relocated storage (actual=${repo_root})"
}
cd "${canonical_repo_root}"

hash_matches() {
    local expected="$1"
    local path="$2"
    local actual
    [[ -f "${path}" && -s "${path}" ]] || fail "missing required file: ${path}"
    actual="$(sha256sum "${path}" | awk '{print $1}')"
    [[ "${actual}" == "${expected}" ]] || {
        fail "SHA-256 changed: ${path} expected=${expected} actual=${actual}"
    }
}

hash_matches e87909adbf9fcf985d4890b9dfd208aa53f700d1cba04e67b1a5979b291e0085 "${method_launcher}"
hash_matches 288c21a83028e92a60f17a2ee696ce4a989eaebd07db003f8dca181394e0617a scripts/local/start_phase1_signal_random_uniform_r32_4gpu.sh
hash_matches fec360a1d9f664244518d036de7308be1b03fce5bd08511b5060a034faa572e8 scripts/analysis/prepare_phase1_signal_random_artifacts.py
hash_matches 7660390028739a1db83c1fa4de96235faf754e4fcbad604d946c0aab283bf80c scripts/analysis/phase1_training_contract.py
hash_matches 4bcd33a1a336397c56778eb866d39fc65b70229c7cb2e9fdc80b412745dea3f9 scripts/analysis/verify_full_bench_run.py
hash_matches c25685daa65bb66b5f5974857e3e3cc32e61a81359e90983cc1dcd6f542ffaa0 "${postverify_script}"

hash_matches 92d44f339e0129c81c43fd691395c3d87399a17c7fbbe9b42f34149fcf2cc65d "${runtime_root}/analysis/phase1_two_seed_scope_amendment.json"
hash_matches 02978fcdc0f6ce43682440582deb0ff427fdfeda189d457ad45cf31f13a3931b "${runtime_root}/analysis/phase1_two_seed_paper_baseline_amendment.json"
hash_matches 247fc05892648c41815252cecccb70c0152eddab940be642dc5d29fea65c27c8 "${runtime_root}/analysis/phase1_seed42_selection.json"

current_branch="$(git branch --show-current)"
[[ "${current_branch}" == "experiments/spar-lora-v0" ]] || fail "wrong Git branch: ${current_branch:-detached}"
git merge-base --is-ancestor 9f1fb738b0f1cd9f1aa105a10caa1dce24626da6 HEAD || {
    fail "HEAD does not contain the audited migration baseline 9f1fb738b0f1cd9f1aa105a10caa1dce24626da6"
}

"${python_bin}" - "${runtime_root}/analysis/phase1_seed42_selection.json" <<'PY' || exit 2
import json
import sys

selection = json.load(open(sys.argv[1], encoding="utf-8"))
expected = ("advance_multiseed", "I8", "macro_delta_avg_at_k")
actual = (
    selection.get("decision"),
    selection.get("selected_method"),
    selection.get("selection_metric"),
)
if actual != expected:
    raise SystemExit(f"seed-42 selection changed: expected={expected!r} actual={actual!r}")
PY

hash_matches d60fc87775a5b1a0e1d27f922b6c4f2b100f0e8614faeb261afaf2c1c88dd386 "${runtime_root}/analysis/training_allocations/phase1_training_preparation.json"
hash_matches 2f20e70333a66f979d7630c83823304d9b9c196d65e86e74ce329200a87478f5 "${runtime_root}/analysis/training_allocations/I8_uniform_r32/rank_map.json"
hash_matches 0c7ba14adca289fd1d88377d6c91afaab0f1b26914c03b088b0bddf78e55e131 "${runtime_root}/analysis/training_allocations/I8_uniform_r32/allocation_summary.json"
hash_matches 63842778dad6064beeae7c5de6a497e2bfd58d83ee93c2f5f567592045950135 "${runtime_root}/analysis/training_allocations/I8_uniform_r32/subspaces.safetensors"

hash_matches 8e3c9314db8b83c61ab62a3dc85e0a704dcc5f3a9d404d236943f719095ce82f /data/peft-for-rl-runtime/datasets/verl_dapo/data/dapo-math-17k-boxed.parquet
hash_matches 3416ba93286b324a9663777fa472d0b568af5446319f27f876a63b59b3e863da /data/peft-for-rl-runtime/outputs/full_bench_eval/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42/records/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42.jsonl
hash_matches a5d8f67ae6461f6577e4891a9049b5d8f37e6f71b1ec5bf0f9152b4ac02224da "${runtime_root}/outputs/full_bench_eval/phase1_i0_uniform_r32_b64m16n8_step50_seed43_v3_fullbench_32768_seed42/records/phase1_i0_uniform_r32_b64m16n8_step50_seed43_v3_fullbench_32768_seed42.jsonl"
hash_matches 179cdf9e3c293ce5b1b0461447b38a34490e5212dda67cff63940828856ef261 "${runtime_root}/ckpts/verl/DAPO-Math-17k/phase1_i0_uniform_r32_b64m16n8_step50_seed43_v3/global_step_50/actor/peft_adapter/adapter_config.json"
hash_matches 0037992fca30100e435c48f6e03f67f09031949324869c20524eb2dee96b7ec5 "${runtime_root}/ckpts/verl/DAPO-Math-17k/phase1_i0_uniform_r32_b64m16n8_step50_seed43_v3/global_step_50/actor/peft_adapter/adapter_model.safetensors"

base_model=/data/peft-for-rl-runtime/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base
hash_matches 37bd455e9679d2959536270fed49d25cc7c290a64f6e52abb97c71345a9cee41 "${base_model}/config.json"
hash_matches 72cbec1015da9ed03ad025483005cbf6481403abf947adfd54e504d1a66a2126 "${base_model}/generation_config.json"
hash_matches 88145e3c3249adc2546ede277e9819d6e405e19072456e4b521cbc724bd60773 "${base_model}/tokenizer.json"
hash_matches 8ac8c85fb242563c2260baec0909debd69d718af6a0b3d90e6cab62b4d341cd5 "${base_model}/tokenizer_config.json"
hash_matches 58858233513d76b8703e72eed6ce16807b523328188e13329257fb9594462945 "${base_model}/model.safetensors"

PYTHONPATH="${canonical_repo_root}" "${python_bin}" scripts/analysis/prepare_phase1_signal_random_artifacts.py \
    --output-root "${runtime_root}/analysis/training_allocations" \
    --verify-method I8 >/dev/null
echo "HANDOFF_PAYLOAD_AND_I8_INITIALIZATION_VERIFIED"

if (( payload_only == 1 )); then
    echo "PAYLOAD_ONLY_COMPLETE no runtime or storage gate evaluated"
    exit 0
fi

if pgrep -af '[r]un_phase1_signal_random_confirmation.sh' >/dev/null; then
    pgrep -af '[r]un_phase1_signal_random_confirmation.sh' >&2 || true
    fail "a Phase-1 controller is running; this migration entrypoint must run independently"
fi
if pgrep -af '[e]val_full_bench_vllm.py' >/dev/null; then
    pgrep -af '[e]val_full_bench_vllm.py' >&2 || true
    fail "an evaluator is running"
fi
if pgrep -af '[V]LLM::EngineCore' >/dev/null; then
    pgrep -af '[V]LLM::EngineCore' >&2 || true
    fail "a residual vLLM EngineCore is running"
fi
if find "${runtime_root}" -path '*seed44*' -print -quit | grep -q .; then
    fail "seed-44 artifacts already exist"
fi

i8_contract="${runtime_root}/analysis/training_manifests/${i8_name}.json"
i8_ckpt="${runtime_root}/ckpts/verl/DAPO-Math-17k/${i8_name}"
i8_eval="${runtime_root}/outputs/full_bench_eval/${i8_eval_name}"
i8_postverify="${runtime_root}/analysis/verification/${i8_name}_postverify.json"
for path in "${i8_contract}" "${i8_ckpt}" "${i8_eval}" "${i8_postverify}"; do
    [[ ! -e "${path}" ]] || fail "partial or complete seed-43 I8 state already exists: ${path}"
done
[[ ! -e "${runtime_root}/analysis/I0_vs_I8_step50_seed43_paired.json" ]] || fail "seed-43 paired comparison already exists"

available_bytes="$(df -B1 --output=avail /root | tail -n 1 | tr -d ' ')"
[[ "${available_bytes}" =~ ^[0-9]+$ ]] || fail "could not determine available bytes on /root"
(( available_bytes >= minimum_root_free_bytes )) || {
    fail "insufficient /root capacity: available=${available_bytes} required=${minimum_root_free_bytes}"
}
echo "SEED43_I8_MIGRATED_PREFLIGHT_READY available_bytes=${available_bytes}"

if (( execute == 0 )); then
    echo "DRY_RUN no training, evaluation, or runtime file was started"
    exit 0
fi

mkdir -p "${runtime_root}/logs"
exec 9>"${runtime_root}/logs/phase1_seed43_i8_migrated.lock"
flock -n 9 || fail "another migrated seed-43 I8 launcher holds the lock"

TRAIN_SEED=43 PHASE1_RUN_REVISION=3 bash "${method_launcher}" I8
bash "${postverify_script}" I8
[[ -s "${i8_postverify}" ]] || fail "I8 pipeline ended without its postverify artifact"
echo "SEED43_I8_MIGRATED_PIPELINE_COMPLETE postverify=${i8_postverify}"
