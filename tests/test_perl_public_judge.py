import hashlib
from pathlib import Path
import pytest
from tina.analysis.perl_public_judge import judge_math_answer, runtime, UPSTREAM_SHA256


@pytest.mark.parametrize("prediction,gold,expected", [
    (r"\boxed{8\sqrt{10}}", r"8\sqrt{10}", True),
    (r"\boxed{8}", r"8\sqrt{10}", False),
    (r"\boxed{14+4\sqrt{37}}", r"14+4\sqrt{37}", True),
    (r"\boxed{14}", r"14+4\sqrt{37}", False),
    (r"\boxed{\frac43}", r"\frac{4}{3}", True),
    (r"\boxed{\sqrt2}", r"\sqrt{2}", True),
    ("Work\nAnswer: $42$.", "42", True),
    ("ANSWER: 42", "42", False),
    ("Final Answer: 42", "42", False),
    (r"\boxed {42}", "42", False),
    ("42", "42", False),
    (r"\boxed{43}", "42", False),
])
def test_original_peft_paper_protocol(prediction, gold, expected):
    result = judge_math_answer(prediction, gold)
    assert result.correct == expected
    assert (result.correct, result.format_valid) == tuple(map(bool, runtime().grade_answer_perl(prediction, gold)))


def test_original_source_is_unchanged():
    assert hashlib.sha256(Path(runtime().__file__).read_bytes()).hexdigest() == UPSTREAM_SHA256
