#!/usr/bin/env python3
"""Record per-GPU used memory during one training process."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def sample() -> list[tuple[int, int, int]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return [
        (int(index), int(memory), int(utilization))
        for index, memory, utilization in csv.reader(output.splitlines())
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args()
    if args.pid <= 0 or args.interval <= 0:
        parser.error("pid and interval must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("timestamp_utc", "gpu_index", "used_mib", "utilization_percent"))
        while process_exists(args.pid):
            timestamp = datetime.now(timezone.utc).isoformat()
            for index, memory, utilization in sample():
                writer.writerow((timestamp, index, memory, utilization))
            output.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
