#!/usr/bin/env python3
"""Combine stability and fixed full-benchmark results into comparison tables/report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def earliest_stable(rows: list[dict], method: str) -> int | None:
    selected = sorted(
        (row for row in rows if row["method"] == method and row["group"] == "all"), key=lambda row: row["h"]
    )
    for row in selected:
        if all(
            float(row[field]) >= 0.80
            for field in ("left_overlap_95", "right_overlap_95", "full_capture_95")
        ):
            return int(row["h"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stability = load_rows(args.analysis_dir / "group_summary.csv")
    eval_rows = []
    task_rows = []
    for method, train_name in (
        ("S0", "delta_stability_s0_standard_r32_seed42_v1"),
        ("O0", "delta_stability_o0_random_orthogonal_r32_seed42_v1"),
    ):
        for step in (25, 50, 100):
            name = f"{train_name}_step{step}_fullbench_32768_seed42"
            path = args.eval_root / name / "summary" / f"{name}.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            eval_rows.append(
                {
                    "method": method,
                    "step": step,
                    "macro_avg_at_k": summary["macro_avg_at_k_over_benchmarks"],
                    "macro_pass_at_k": summary["macro_pass_at_k_over_benchmarks"],
                    "parse_rate": summary["parse_rate_overall"],
                    "hit_max_rate": summary["hit_max_rate_overall"],
                    "mean_length": summary["length_mean_overall"],
                }
            )
            for task in summary["benchmarks"]:
                task_rows.append({"method": method, "step": step, "benchmark": task["benchmark"], "avg_at_k": task["avg_at_k"]})

    def write_csv(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(args.out_dir / "fullbench_comparison.csv", eval_rows)
    write_csv(args.out_dir / "six_task_avg_at_k.csv", task_rows)
    earliest = {method: earliest_stable(stability, method) for method in ("S0", "O0")}
    all_rows = {(row["method"], int(row["h"])): row for row in stability if row["group"] == "all"}
    side_means = {
        method: {
            "input": statistics.fmean(float(all_rows[(method, horizon)]["right_overlap_95"]) for horizon in (1, 5, 10, 25, 50)),
            "output": statistics.fmean(float(all_rows[(method, horizon)]["left_overlap_95"]) for horizon in (1, 5, 10, 25, 50)),
            "functional": statistics.fmean(float(all_rows[(method, horizon)]["full_capture_95"]) for horizon in (1, 5, 10, 25, 50)),
        }
        for method in ("S0", "O0")
    }
    early_prediction = {
        method: statistics.fmean(float(all_rows[(method, horizon)]["full_capture_95"]) for horizon in (1, 5, 10))
        for method in ("S0", "O0")
    }
    lines = [
        "# S0 / O0 多时间尺度 DeltaW 稳定性总结",
        "",
        "稳定出现判据预设为：全模型 energy-weighted `left overlap@95`、`right overlap@95` 和 "
        "`full cross-capture@95` 同时不低于 0.80。activation 指标使用同一批固定 prompt 在 frozen base model 上的输入激活。",
        "",
        "## 稳定性",
        "",
        f"- S0 最早满足判据的时间尺度：`{earliest['S0'] if earliest['S0'] is not None else '未出现'}` trainer steps。",
        f"- O0 最早满足判据的时间尺度：`{earliest['O0'] if earliest['O0'] is not None else '未出现'}` trainer steps。",
        "",
        "| Method | h | Input overlap | Output overlap | Full cross-capture | Effective rank | Activation full capture |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in ("S0", "O0"):
        for horizon in (1, 5, 10, 25, 50):
            row = all_rows[(method, horizon)]
            lines.append(
                f"| {method} | {horizon} | {row['right_overlap_95']:.4f} | {row['left_overlap_95']:.4f} | "
                f"{row['full_capture_95']:.4f} | {row['effective_rank_module_mean']:.2f} | "
                f"{row['activation_full_capture_95']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## 正式评测",
            "",
            "| Method | Step | Macro Avg@k | Pass@k | Parse | Hit max | Mean len |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in eval_rows:
        lines.append(
            f"| {row['method']} | {row['step']} | {row['macro_avg_at_k']:.6f} | "
            f"{row['macro_pass_at_k']:.6f} | {row['parse_rate']:.6f} | {row['hit_max_rate']:.6f} | "
            f"{row['mean_length']:.1f} |"
        )
    step100 = {row["method"]: row for row in eval_rows if row["step"] == 100}
    evaluation_delta = step100["O0"]["macro_avg_at_k"] - step100["S0"]["macro_avg_at_k"]
    geometry_delta = statistics.fmean(
        float(all_rows[("O0", horizon)]["full_capture_95"]) - float(all_rows[("S0", horizon)]["full_capture_95"])
        for horizon in (1, 5, 10, 25, 50)
    )
    consistent = geometry_delta == 0 or evaluation_delta == 0 or math.copysign(1, geometry_delta) == math.copysign(1, evaluation_delta)
    lines.extend(
        [
            "",
            "## 结论提示",
            "",
            f"- O0-S0 的平均 full cross-capture 差为 `{geometry_delta:+.6f}`。",
            f"- step-100 Macro Avg@k 差为 `{evaluation_delta:+.6f}`。",
            f"- 两者方向{'一致' if consistent else '不一致'}；这只是两点方法对照，不作相关性或因果性外推。",
            "",
            "## 对研究问题的直接回答",
            "",
            f"- 稳定性最早出现：S0 为 `{earliest['S0'] if earliest['S0'] is not None else '未达到预设阈值'}`，"
            f"O0 为 `{earliest['O0'] if earliest['O0'] is not None else '未达到预设阈值'}` trainer steps。",
            f"- 稳定侧别：S0 的输入/输出 overlap 与完整 cross-capture 五尺度均值分别为 "
            f"`{side_means['S0']['input']:.4f}/{side_means['S0']['output']:.4f}/{side_means['S0']['functional']:.4f}`；"
            f"O0 为 `{side_means['O0']['input']:.4f}/{side_means['O0']['output']:.4f}/{side_means['O0']['functional']:.4f}`。"
            "输入/输出侧取均值更高者视为更稳定；完整更新必须单独看 functional capture，不能由单侧 overlap 代替。",
            f"- random-orthogonal 是否更早或更稳定：最早尺度按上述阈值比较；O0-S0 的五尺度完整 capture 均值差为 "
            f"`{geometry_delta:+.6f}`。",
            f"- 早期子空间能否预测后续更新：h=1/5/10 的完整 cross-capture 均值，S0=`{early_prediction['S0']:.4f}`、"
            f"O0=`{early_prediction['O0']:.4f}`；达到 0.80 才按本报告判为有较强预测力。",
            f"- 几何稳定性与正式评测是否一致：{'方向一致' if consistent else '方向不一致'}；"
            f"几何差 `{geometry_delta:+.6f}`，step-100 Macro Avg@k 差 `{evaluation_delta:+.6f}`。",
            "- 所有结论均使用 DeltaW；A/B 参数距离不参与判定。",
            "",
        ]
    )
    (args.out_dir / "summary_zh.md").write_text("\n".join(lines), encoding="utf-8")
    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "earliest_stable_h": earliest,
                "mean_o0_minus_s0_full_capture": geometry_delta,
                "step100_o0_minus_s0_macro_avg_at_k": evaluation_delta,
                "geometry_evaluation_direction_consistent": consistent,
                "side_means": side_means,
                "early_full_cross_capture_h1_h5_h10": early_prediction,
                "evaluations": eval_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
