#!/usr/bin/env python3
"""Rebuild DeltaW aggregate tables and plots from completed per-interval CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analyze_delta_stability import aggregate, plot_comparison, write_csv


INTERVAL_NUMERIC = {
    "effective_rank",
    "end",
    "frobenius_norm",
    "h",
    "layer",
    "rank90",
    "rank95",
    "t",
}
ADJACENCY_NUMERIC = {
    "activation_full_capture_95",
    "activation_future_energy",
    "activation_left_capture_95",
    "activation_right_capture_95",
    "end",
    "full_capture_95",
    "future_energy",
    "h",
    "layer",
    "left_capture_95",
    "left_overlap_95",
    "middle",
    "right_capture_95",
    "right_overlap_95",
    "t",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path, numeric: set[str]) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in numeric:
            value = row.get(key)
            if value not in (None, ""):
                row[key] = float(value)
    return rows


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir.resolve()
    intervals = load_rows(analysis_dir / "module_interval_spectra.csv", INTERVAL_NUMERIC)
    adjacency = load_rows(analysis_dir / "adjacent_period_stability.csv", ADJACENCY_NUMERIC)

    summaries = []
    for method in ("S0", "O0"):
        summaries.extend(
            aggregate(
                method,
                [row for row in intervals if row["method"] == method],
                [row for row in adjacency if row["method"] == method],
            )
        )

    artifact_path = analysis_dir / "summary.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    artifact["summary"] = summaries
    artifact["rows"]["group_summary"] = len(summaries)
    artifact["summary_rebuilt_at_utc"] = datetime.now(timezone.utc).isoformat()
    artifact.setdefault("corrections", []).append(
        {
            "name": "sort_concatenated_singular_values_before_global_rank_at_energy",
            "scope": "group_summary rank90/rank95 global fields only",
            "raw_interval_and_stability_rows_changed": False,
        }
    )

    with tempfile.TemporaryDirectory(prefix="delta-summary-", dir=analysis_dir) as temp_name:
        temp_dir = Path(temp_name)
        write_csv(temp_dir / "group_summary.csv", summaries)
        plot_comparison(summaries, temp_dir)
        (temp_dir / "summary.json").write_text(
            json.dumps(artifact, indent=2, allow_nan=True) + "\n", encoding="utf-8"
        )
        for name in ("group_summary.csv", "s0_o0_multiscale_comparison.png", "summary.json"):
            os.replace(temp_dir / name, analysis_dir / name)

    print(f"DELTA_STABILITY_SUMMARY_REBUILT {analysis_dir}")


if __name__ == "__main__":
    main()
