#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )) || [[ "$1" != "a" && "$1" != "b" ]]; then
    echo "Usage: $0 <a|b>" >&2
    exit 2
fi
lane="$1"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd -P)"
lock_root="${repo_root}/runs/step25-trajectory-v1/locks"
mkdir -p "${lock_root}"
exec 9>"${lock_root}/lane-${lane}.lock"
flock -n 9 || { echo "Step-25 trajectory lane ${lane} is already running" >&2; exit 2; }

if [[ "${lane}" == "a" ]]; then
    "${script_dir}/run_step25_trajectory_eval.sh" standard
    "${script_dir}/run_step25_trajectory_eval.sh" i32
else
    "${script_dir}/run_step25_trajectory_eval.sh" i8
fi
