"""Judge-independent fullbench aggregation for saved complete samples."""
from collections import Counter, defaultdict
import statistics


def summarize(rows):
    by_bench = defaultdict(list)
    for row in rows:
        by_bench[row["benchmark"]].append(row)
    tasks = []
    for benchmark, group in sorted(by_bench.items()):
        problems = defaultdict(list)
        for row in group:
            problems[str(row["id"])].append(row)
        expected = {int(row["samples_per_problem"]) for row in group}
        if len(expected) != 1 or any(len(g) != next(iter(expected)) for g in problems.values()):
            raise ValueError(f"Incomplete/unequal sampling: {benchmark}")
        if len({(r["id"], r["sample_index"]) for r in group}) != len(group):
            raise ValueError(f"Duplicate samples: {benchmark}")
        tasks.append({
            "benchmark": benchmark, "num_problems": len(problems), "num_samples": len(group),
            "samples_per_problem": next(iter(expected)),
            "avg_at_k": sum(r["correct"] for r in group) / len(group),
            "macro_avg_at_k": sum(r["correct"] for r in group) / len(group),
            "pass_at_k": sum(any(r["correct"] for r in g) for g in problems.values()) / len(problems),
            "parse_rate": sum(r["format_valid"] for r in group) / len(group),
            "hit_max_rate": sum(r["hit_max_new_tokens"] for r in group) / len(group),
            "length_mean": statistics.mean(r["completion_length"] for r in group),
            "length_median": statistics.median(r["completion_length"] for r in group),
            "source_correct": sum(r["source_judgment"]["correct"] for r in group),
            "new_correct": sum(r["correct"] for r in group),
            "wrong_to_correct": sum(not r["source_judgment"]["correct"] and r["correct"] for r in group),
            "correct_to_wrong": sum(r["source_judgment"]["correct"] and not r["correct"] for r in group),
            "all_wrong_groups": sum(not any(r["correct"] for r in g) for g in problems.values()),
            "all_correct_groups": sum(all(r["correct"] for r in g) for g in problems.values()),
            "judge_status_counts": dict(Counter(r["judge_status"] for r in group)),
        })
    return {
        "checkpoint": rows[0]["checkpoint"], "num_samples": len(rows),
        "num_problems": sum(t["num_problems"] for t in tasks), "benchmarks": tasks,
        "response_weighted_avg_at_k": sum(r["correct"] for r in rows) / len(rows),
        "source_response_weighted_avg_at_k": sum(r["source_judgment"]["correct"] for r in rows) / len(rows),
        "macro_avg_at_k_over_benchmarks": sum(t["avg_at_k"] for t in tasks) / len(tasks),
        "macro_pass_at_k_over_benchmarks": sum(t["pass_at_k"] for t in tasks) / len(tasks),
        "parse_rate_overall": sum(r["format_valid"] for r in rows) / len(rows),
        "hit_max_rate_overall": sum(r["hit_max_new_tokens"] for r in rows) / len(rows),
        "length_mean_overall": statistics.mean(r["completion_length"] for r in rows),
    }
