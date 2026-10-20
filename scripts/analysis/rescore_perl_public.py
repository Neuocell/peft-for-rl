#!/usr/bin/env python3
"""Rescore saved generations with the original PeRL paper-release grader."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tina_run"))
from tina.analysis.perl_public_judge import judge_math_answer, judge_metadata
from fullbench_statistics import summarize

DEFAULT_MINERVA = ROOT / "configs/evaluation/assets/minerva_mathai.jsonl"
MINERVA_SHA256 = "e04ec2bc1d8f80a0091b4b1989f1ffb366f3583f5bae381546a066d4ccc26e3d"


def score(row):
    result = judge_math_answer(row["completion"], row["judge_gold_answer"])
    return {**row, "correct": result.correct, "format_valid": result.format_valid,
            "extracted_answer": result.answer, "judge_status": result.status}


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minerva-source", type=Path, default=DEFAULT_MINERVA)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    metadata = judge_metadata()
    gold_raw = args.minerva_source.read_bytes()
    if hashlib.sha256(gold_raw).hexdigest() != MINERVA_SHA256:
        raise ValueError("Minerva gold differs from the pinned PeRL dataset snapshot")
    minerva = {row["question"]: str(row["answer"]) for row in map(json.loads, gold_raw.splitlines())}
    if len(minerva) != 272:
        raise ValueError("Expected 272 unique Minerva questions")
    gold_source = {"dataset": "math-ai/minervamath", "revision": "ee46ddc498933b1977577953250ca5c66be64f96",
                   "sha256": MINERVA_SHA256, "path": str(args.minerva_source.resolve())}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".rescore.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for source in args.records:
            raw = source.read_bytes()
            source_hash = hashlib.sha256(raw).hexdigest()
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
            if not rows or len({r["checkpoint"] for r in rows}) != 1:
                raise ValueError("Expected one nonempty checkpoint")
            checkpoint = rows[0]["checkpoint"]
            destination = args.output_dir / "summary" / f"{checkpoint}.json"
            if destination.exists():
                existing = json.loads(destination.read_text())
                if (existing["source_sha256"] != source_hash or existing["judge"] != metadata
                        or existing["gold_source"] != gold_source):
                    raise ValueError(f"Existing source/protocol differs: {destination}")
                print(f"Already verified: {destination}", flush=True)
                continue
            print(f"Scoring {checkpoint}: {len(rows)} responses", flush=True)
            for row in rows:
                row["source_judgment"] = {key: row.get(key) for key in (
                    "correct", "format_valid", "extracted_answer", "judge_version", "judge_source_sha256", "judge_status")}
                row["judge_gold_answer"] = minerva[row["problem"]] if row["benchmark"] == "minerva" else str(row["gold_answer"])
                row["judge_version"] = metadata["version"]
                row["judge_source_sha256"] = metadata["source_sha256"]
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                corrected = list(pool.map(score, rows, chunksize=16))
            summary = summarize(corrected)
            summary.update({"judge": metadata, "source": str(source.resolve()), "source_sha256": source_hash,
                            "gold_source": gold_source, "source_judges": sorted({r.get("judge_version") or "legacy_unversioned" for r in [row["source_judgment"] for row in rows]}),
                            "minerva_exact_matched_questions": len({r["problem"] for r in rows if r["benchmark"] == "minerva"}),
                            "created_at_utc": datetime.now(timezone.utc).isoformat()})
            for suffix, selected in (("records", corrected), ("changes", [r for r in corrected if r["correct"] != r["source_judgment"]["correct"]]),
                                     ("unextracted", [r for r in corrected if r["judge_status"] != "extracted"])):
                target = args.output_dir / suffix / f"{checkpoint}.jsonl"
                target.parent.mkdir(exist_ok=True)
                temp = target.with_suffix(".jsonl.tmp")
                with temp.open("w") as handle:
                    for row in selected:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                temp.replace(target)
            atomic_json(destination, summary)
            print(json.dumps({"checkpoint": checkpoint, "avg": summary["response_weighted_avg_at_k"]}), flush=True)


if __name__ == "__main__":
    main()
