"""Unmodified PeRL release grader with explicit dependency/source provenance."""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from importlib.metadata import version
from pathlib import Path

JUDGE_VERSION = "perl_release_judge_v1"
UPSTREAM_COMMIT = "a5ec6c36089cb3def58952d8544a69b93b30db62"
UPSTREAM_SHA256 = "b8790bb7ec815e0338905f411d2e8fa7c3eb3e55cca7e852523b7ca34163461f"
PINS = {"sympy": "1.14.0", "pylatexenc": "2.10"}


@lru_cache(maxsize=1)
def runtime():
    for name, expected in PINS.items():
        if version(name) != expected:
            raise RuntimeError(f"PeRL judge requires {name}=={expected}; use a separate CPU process")
    from tina.analysis.vendor.perl_release import grader
    if hashlib.sha256(Path(grader.__file__).read_bytes()).hexdigest() != UPSTREAM_SHA256:
        raise RuntimeError("PeRL release grader differs from the pinned upstream source")
    return grader


def judge_metadata():
    runtime()
    return {"version": JUDGE_VERSION, "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "upstream_commit": UPSTREAM_COMMIT, "upstream_source_sha256": UPSTREAM_SHA256,
            "reference": f"https://github.com/MikaStars39/PeRL/blob/{UPSTREAM_COMMIT}/perl/eval/grader.py",
            "entrypoint": "grade_answer_perl", "dependencies": PINS,
            "dependency_scope": "Release requirements are unpinned; these are local reproducibility pins, not a claim about the paper runtime.",
            "gold_policy": "MATH500 answer; Minerva exact-question matched math-ai/minervamath answer; other tasks original answer",
            "generation_scope": "Judge-only rescore; saved Tina-style prompts and original sampling are retained."}


@dataclass(frozen=True)
class Judgment:
    correct: bool
    format_valid: bool
    answer: str | None
    status: str


def judge_math_answer(completion: str, gold: str) -> Judgment:
    grader = runtime()
    score, format_score = grader.grade_answer_perl(completion, gold)
    # Diagnostics follow the release extraction, without changing the score.
    answer = None
    if r"\boxed" in completion:
        answer = grader.extract_boxed_answer(completion)
    elif "Answer:" in completion:
        last = completion.strip().split("\n")[-1]
        if last.startswith("Answer:"):
            answer = last[7:].strip().removesuffix(".")
            if answer.startswith("$") and answer.endswith("$"):
                answer = answer[1:-1]
    status = "extracted" if answer else ("invalid_box" if format_score else "format_rejected")
    return Judgment(bool(score), bool(format_score), answer, status)
