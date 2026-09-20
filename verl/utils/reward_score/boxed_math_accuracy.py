# Copyright 2026 Individual contributors.
#
# Boxed-answer math outcome reward aligned with local fullbench evaluation.

import re
from typing import Any, Optional


def extract_boxed_answer(text: str) -> str | None:
    text = text or ""
    last_start = text.rfind(r"\boxed{")
    if last_start < 0:
        return None
    i = last_start + len(r"\boxed{")
    depth = 1
    chars = []
    while i < len(text):
        char = text[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
        chars.append(char)
        i += 1
    return None


def extract_final_answer_text(text: str) -> str | None:
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed
    patterns = [
        r"final answer is\s*:?\s*\$?\\?boxed\{?([^}\n]+)",
        r"final answer is\s*:?\s*\$?([^\n$.]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text or "", flags=re.IGNORECASE)
        if matches:
            return str(matches[-1]).strip().strip("$.")
    return None


def normalize_answer(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\\boxed\s*\{([^}]*)\}", r"\1", text)
    text = text.replace("$", "")
    text = re.sub(r"\s+", "", text)
    return text


def evaluate_math_answer(completion: str, gold: str) -> tuple[str | None, bool, bool]:
    pred = extract_final_answer_text(completion)
    if pred is None:
        return None, False, False

    try:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify

        gold_parsed = parse(str(gold), extraction_mode="first_match")
        answer_parsed = parse(
            completion,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        if gold_parsed:
            return pred, True, bool(verify(gold_parsed, answer_parsed))
    except Exception:
        pass

    return pred, True, normalize_answer(pred) == normalize_answer(gold)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
) -> dict:
    """Return binary accuracy reward using the fullbench boxed-answer parser."""

    del data_source, extra_info, strict_box_verify, pause_tokens_index

    pred, parse_success, correct = evaluate_math_answer(solution_str, ground_truth)
    has_boxed = extract_boxed_answer(solution_str) is not None
    has_answer_colon = bool(re.search(r"(?i)\bAnswer\s*:", solution_str or ""))
    has_final_answer_is = bool(re.search(r"(?i)final answer is", solution_str or ""))

    return {
        "score": 1.0 if correct else 0.0,
        "acc": bool(correct),
        "parse_success": bool(parse_success),
        "has_boxed": bool(has_boxed),
        "has_answer_colon": bool(has_answer_colon),
        "has_final_answer_is": bool(has_final_answer_is),
        "pred": pred or "",
    }
