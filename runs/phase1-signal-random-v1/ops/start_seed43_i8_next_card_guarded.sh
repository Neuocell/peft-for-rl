#!/usr/bin/env bash
set -euo pipefail

repo_root="/root/peft-for-rl"
runtime_root="${repo_root}/runs/phase1-signal-random-v1"
jq_bin="/home/node/anaconda3/bin/jq"
controller_script="${repo_root}/scripts/local/run_phase1_signal_random_confirmation.sh"
controller_script_sha256="b6d4bdfc7af77019fbdfad8bec76187ba0254db6e24a2e5acd632ac5ed6aaa0f"
method_launcher="${repo_root}/scripts/local/run_phase1_signal_random_step50_fullbench.sh"
method_launcher_sha256="e87909adbf9fcf985d4890b9dfd208aa53f700d1cba04e67b1a5979b291e0085"
paired_comparator="${repo_root}/scripts/analysis/compare_paired_full_bench.py"
paired_comparator_sha256="5764621d26742238f77a4e8d47d5c37b14ca80ae2b65e6ea4bd3865cf2041fc8"
seed42_selector="${repo_root}/scripts/analysis/select_phase1_seed42_candidate.py"
seed42_selector_sha256="ed2b580b637d70e0f7fb25a7868c264c15643647132ef1764140c7d7d6355989"
postverify_script="${runtime_root}/ops/seed43_i0_postverify.sh"
postverify_script_sha256="c25685daa65bb66b5f5974857e3e3cc32e61a81359e90983cc1dcd6f542ffaa0"
scope_amendment="${runtime_root}/analysis/phase1_two_seed_scope_amendment.json"
scope_amendment_sha256="92d44f339e0129c81c43fd691395c3d87399a17c7fbbe9b42f34149fcf2cc65d"
i0_name="phase1_i0_uniform_r32_b64m16n8_step50_seed43_v3"
i8_name="phase1_i8_uniform_r32_b64m16n8_step50_seed43_v3"
i0_postverify="${runtime_root}/analysis/verification/${i0_name}_postverify.json"
seed42_selection="${runtime_root}/analysis/phase1_seed42_selection.json"
i8_eval_name="${i8_name}_fullbench_32768_seed42"
i8_summary="${runtime_root}/outputs/full_bench_eval/${i8_eval_name}/summary/${i8_eval_name}.json"
seed43_comparison="${runtime_root}/analysis/I0_vs_I8_step50_seed43_paired.json"
lock_path="${runtime_root}/logs/phase1_signal_random_confirmation.lock"
controller_log="${runtime_root}/logs/phase1_seed43_i8_guarded_controller.log"
stop_unit="peft-phase1-stop-after-seed43"
postverify_unit="peft-phase1-seed43-i8-postverify"
minimum_root_free_bytes=16500000000
mode=""
controller_pid=""
execute=0

cd "${repo_root}"

usage() {
    cat <<'EOF'
Usage:
  start_seed43_i8_next_card_guarded.sh --mode existing --controller-pid PID [--execute]
  start_seed43_i8_next_card_guarded.sh --mode replacement [--execute]

Without --execute, validate all currently available preconditions and print the
action that would be taken. The existing mode resumes an already stopped
controller. The replacement mode is only for a rebooted instance with no
controller and no partial seed-43 I8 state; it stages a new controller in the
stopped state, deploys transient guards bound to its PGID, then resumes it.
EOF
}

fail() {
    echo "SEED43_I8_GUARDED_START_REFUSED $*" >&2
    exit 2
}

while (( $# > 0 )); do
    case "$1" in
        --mode)
            (( $# >= 2 )) || fail "--mode requires existing or replacement"
            mode="$2"
            shift 2
            ;;
        --controller-pid)
            (( $# >= 2 )) || fail "--controller-pid requires a PID"
            controller_pid="$2"
            shift 2
            ;;
        --execute)
            execute=1
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

case "${mode}" in
    existing)
        [[ "${controller_pid}" =~ ^[1-9][0-9]*$ ]] || fail "existing mode requires a numeric --controller-pid"
        ;;
    replacement)
        [[ -z "${controller_pid}" ]] || fail "replacement mode does not accept --controller-pid"
        ;;
    *)
        usage >&2
        fail "--mode must be existing or replacement"
        ;;
esac

hash_matches() {
    local expected="$1"
    local path="$2"
    local actual
    [[ -f "${path}" && -s "${path}" ]] || fail "missing pinned file: ${path}"
    actual="$(sha256sum "${path}" | awk '{print $1}')"
    [[ "${actual}" == "${expected}" ]] || {
        fail "pinned file changed: ${path} expected=${expected} actual=${actual}"
    }
}

hash_matches "${controller_script_sha256}" "${controller_script}"
hash_matches "${method_launcher_sha256}" "${method_launcher}"
hash_matches "${paired_comparator_sha256}" "${paired_comparator}"
hash_matches "${seed42_selector_sha256}" "${seed42_selector}"
hash_matches "${postverify_script_sha256}" "${postverify_script}"
hash_matches "${scope_amendment_sha256}" "${scope_amendment}"
[[ -x "${jq_bin}" ]] || fail "missing jq: ${jq_bin}"

[[ -s "${seed42_selection}" ]] || fail "missing seed-42 selection artifact: ${seed42_selection}"
"${jq_bin}" -e \
    '.decision == "advance_multiseed" and
     .selected_method == "I8" and
     .selection_metric == "macro_delta_avg_at_k"' \
    "${seed42_selection}" >/dev/null || fail "seed-42 selection no longer advances I8"

[[ -s "${i0_postverify}" ]] || fail "seed-43 I0 postverify is not ready: ${i0_postverify}"
"${jq_bin}" -e \
    --arg train_name "${i0_name}" \
    --arg operational_sha "${postverify_script_sha256}" \
    '.status == "verified" and
     .candidate_method == "I0" and
     .training_seed == 43 and
     .train_name == $train_name and
     .counts.merged_records == 7248 and
     .counts.shard_records == [1812, 1812, 1812, 1812] and
     .verification.training_contract.status == "verified" and
     .verification.full_benchmark.status == "verified" and
     .artifacts.operational_script.sha256 == $operational_sha and
     (.artifacts.manifests | length) == 4' \
    "${i0_postverify}" >/dev/null || fail "seed-43 I0 postverify schema or status changed"

while IFS=$'\t' read -r path expected_sha256; do
    hash_matches "${expected_sha256}" "${path}"
done < <(
    "${jq_bin}" -r \
        '.artifacts | [.records, .summary, .adapter, .training_contract, .operational_script, .manifests[]] | .[] | [.path, .sha256] | @tsv' \
        "${i0_postverify}"
)

[[ ! -e "${seed43_comparison}" ]] || fail "seed-43 comparison already exists: ${seed43_comparison}"
if find "${runtime_root}" -path '*seed44*' -print -quit | grep -q .; then
    fail "seed-44 artifacts already exist"
fi

ensure_idle_evaluation_runtime() {
    if pgrep -af '[e]val_full_bench_vllm.py' >/dev/null; then
        pgrep -af '[e]val_full_bench_vllm.py' >&2 || true
        fail "an evaluation process is still running"
    fi
    if pgrep -af '[V]LLM::EngineCore' >/dev/null; then
        pgrep -af '[V]LLM::EngineCore' >&2 || true
        fail "a residual vLLM EngineCore is still running"
    fi
}

ensure_storage_capacity() {
    local available_bytes
    available_bytes="$(df -B1 --output=avail /root | tail -n 1 | tr -d ' ')"
    [[ "${available_bytes}" =~ ^[0-9]+$ ]] || fail "could not determine available bytes on /root"
    (( available_bytes >= minimum_root_free_bytes )) || {
        fail "insufficient /root capacity for seed-43 I8 step-25, step-50, evaluation, and margin: available=${available_bytes} required=${minimum_root_free_bytes}"
    }
    echo "STORAGE_CAPACITY_VERIFIED available=${available_bytes} required=${minimum_root_free_bytes}"
}

verify_stopped_controller() {
    local pid="$1"
    local pgid
    local stat
    local cwd
    local args
    [[ -d "/proc/${pid}" ]] || fail "controller PID does not exist: ${pid}"
    pgid="$(ps -o pgid= -p "${pid}" | tr -d ' ')"
    stat="$(ps -o stat= -p "${pid}" | tr -d ' ')"
    args="$(ps -o args= -p "${pid}")"
    cwd="$(readlink -f "/proc/${pid}/cwd")"
    [[ "${pgid}" == "${pid}" ]] || fail "controller PID ${pid} is not its process-group leader: pgid=${pgid}"
    [[ "${stat}" == *T* ]] || fail "controller PID ${pid} is not stopped: stat=${stat}"
    [[ "${cwd}" == "${repo_root}" ]] || fail "controller cwd changed: ${cwd}"
    [[ "${args}" == *"run_phase1_signal_random_confirmation.sh"* ]] || {
        fail "PID ${pid} is not the Phase-1 controller: ${args}"
    }
}

unit_load_state() {
    systemctl show "$1" --property=LoadState --value 2>/dev/null || true
}

verify_guard_units() {
    local pid="$1"
    local stop_path_unit="${stop_unit}.path"
    local postverify_path_unit="${postverify_unit}.path"
    local stop_definition
    local postverify_definition
    stop_definition="$(systemctl cat "${stop_path_unit}")"
    postverify_definition="$(systemctl cat "${postverify_path_unit}")"
    [[ "${stop_definition}" == *"PathExists=${seed43_comparison}"* ]] || fail "stop guard trigger path changed"
    [[ "${stop_definition}" == *"kill -STOP -- -${pid}"* ]] || fail "stop guard is not bound to PGID ${pid}"
    [[ "${postverify_definition}" == *"PathExists=${i8_summary}"* ]] || fail "I8 postverify trigger path changed"
    [[ "${postverify_definition}" == *"${postverify_script}"*"I8"* ]] || fail "I8 postverify action changed"
    [[ "$(systemctl is-active "${stop_path_unit}")" == "active" ]] || fail "stop guard is not active"
    [[ "$(systemctl is-active "${postverify_path_unit}")" == "active" ]] || fail "I8 postverify watcher is not active"
}

deploy_guard_units() {
    local pid="$1"
    local stop_path_unit="${stop_unit}.path"
    local postverify_path_unit="${postverify_unit}.path"
    local stop_command
    if [[ "$(unit_load_state "${stop_path_unit}")" != "not-found" || \
          "$(unit_load_state "${postverify_path_unit}")" != "not-found" ]]; then
        verify_guard_units "${pid}"
        return
    fi
    stop_command="date '+GUARD_TRIGGERED %Y-%m-%d %H:%M:%S %z'; kill -STOP -- -${pid}; echo STOP_SENT_TO_PGID_${pid}; ps -eo pid,ppid,pgid,sid,stat,etime,comm,cmd --sort=pid | awk '\$3==${pid} || \$1==${pid}'; find '${runtime_root}' -path '*seed44*' -maxdepth 8 -print 2>/dev/null; df -B1 /root"
    systemd-run --quiet \
        --unit="${stop_unit}" \
        --property=RemainAfterExit=yes \
        --path-property="PathExists=${seed43_comparison}" \
        /bin/bash -lc "${stop_command}"
    systemd-run --quiet \
        --unit="${postverify_unit}" \
        --property=RemainAfterExit=yes \
        --path-property="PathExists=${i8_summary}" \
        "${postverify_script}" I8
    verify_guard_units "${pid}"
}

ensure_idle_evaluation_runtime
ensure_storage_capacity

if [[ "${mode}" == "existing" ]]; then
    verify_stopped_controller "${controller_pid}"
    verify_guard_units "${controller_pid}"
    echo "READY existing stopped controller PID/PGID ${controller_pid}"
    if (( execute == 0 )); then
        echo "DRY_RUN no signal sent; rerun with --execute to send SIGCONT"
        exit 0
    fi
else
    if pgrep -af '[r]un_phase1_signal_random_confirmation.sh' >/dev/null; then
        pgrep -af '[r]un_phase1_signal_random_confirmation.sh' >&2 || true
        fail "a Phase-1 controller already exists"
    fi
    if ! flock -n "${lock_path}" -c true; then
        fail "the Phase-1 controller lock is held"
    fi
    if find "${runtime_root}" -path '*phase1_i8_uniform_r32_b64m16n8_step50_seed43_v3*' -print -quit | grep -q .; then
        fail "partial or complete seed-43 I8 state already exists; replacement mode only supports a fresh I8 start"
    fi
    [[ "$(unit_load_state "${stop_unit}.path")" == "not-found" ]] || fail "stale ${stop_unit}.path is loaded"
    [[ "$(unit_load_state "${postverify_unit}.path")" == "not-found" ]] || fail "stale ${postverify_unit}.path is loaded"
    echo "READY replacement controller can be staged with revision 3"
    if (( execute == 0 )); then
        echo "DRY_RUN no controller launched, no unit deployed, and no signal sent"
        exit 0
    fi
    mkdir -p "$(dirname "${controller_log}")"
    setsid bash -c \
        'kill -STOP "$$"; exec env PHASE1_RUN_REVISION=3 bash "$1"' \
        seed43-i8-guarded "${controller_script}" \
        >>"${controller_log}" 2>&1 &
    controller_pid="$!"
    for _ in $(seq 1 100); do
        if [[ -d "/proc/${controller_pid}" ]] && \
           [[ "$(ps -o stat= -p "${controller_pid}" | tr -d ' ')" == *T* ]]; then
            break
        fi
        sleep 0.1
    done
    verify_stopped_controller "${controller_pid}"
    deploy_guard_units "${controller_pid}"
fi

kill -CONT "${controller_pid}"
sleep 1
[[ -d "/proc/${controller_pid}" ]] || fail "controller exited immediately after SIGCONT; inspect ${controller_log}"
[[ "$(ps -o stat= -p "${controller_pid}" | tr -d ' ')" != *T* ]] || fail "controller remained stopped after SIGCONT"
if [[ "${mode}" == "replacement" ]]; then
    echo "SEED43_I8_CONTROLLER_RESUMED pid=${controller_pid} pgid=${controller_pid} log=${controller_log}"
else
    echo "SEED43_I8_CONTROLLER_RESUMED pid=${controller_pid} pgid=${controller_pid} log=existing-controller-owned"
fi
echo "The seed-44 stop guard and I8 postverify watcher are active."
