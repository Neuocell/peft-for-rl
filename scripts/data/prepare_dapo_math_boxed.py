#!/usr/bin/env python3
"""Deduplicate the public DAPO parquet and align prompts with boxed reward."""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PREFIX = (
    "Solve the following math problem step by step. The last line of your response should be of the form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
)
SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'
BOXED_PREFIX = (
    "Solve the following math problem efficiently and clearly.\n"
    "The last line of your response must be exactly of the form:\n"
    "Therefore, the final answer is: $\\boxed{ANSWER}$. I hope it is correct\n\n"
)


def convert_prompt(value: list[dict[str, str]]) -> list[dict[str, str]]:
    if len(value) != 1 or value[0].get("role") != "user":
        raise ValueError(f"Unexpected DAPO prompt structure: {value!r}")
    content = value[0]["content"]
    if not content.startswith(PREFIX) or not content.endswith(SUFFIX):
        raise ValueError("Unexpected DAPO prompt template")
    problem = content[len(PREFIX) : -len(SUFFIX)].strip()
    return [{"role": "user", "content": BOXED_PREFIX + problem}]


def prepare(source: Path, output: Path) -> tuple[int, int]:
    table = pq.read_table(source)
    rows = table.to_pylist()
    seen: set[str] = set()
    converted = []
    for row in rows:
        identifier = str(row["extra_info"]["index"])
        if identifier in seen:
            continue
        seen.add(identifier)
        row["prompt"] = convert_prompt(row["prompt"])
        converted.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = pa.Table.from_pylist(converted, schema=table.schema)
    pq.write_table(result, output, compression="zstd")
    return len(rows), len(converted)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_rows, output_rows = prepare(args.source.resolve(), args.output.resolve())
    print(f"source_rows={source_rows} output_rows={output_rows} output={args.output.resolve()}")


if __name__ == "__main__":
    main()
