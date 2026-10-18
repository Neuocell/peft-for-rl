import json
from pathlib import Path

import pytest

from scripts.local import continue_real_probe32 as continuation


def test_restore_adapter_after_checkpoint_retention(tmp_path):
    backup = tmp_path / "backup"
    backup.mkdir()
    (backup / "adapter_model.safetensors").write_bytes(b"saved adapter")
    plan = {"step50_adapter_backup": str(backup), "checkpoint_root": str(tmp_path / "ckpts")}
    continuation.restore_step50_adapter(plan)
    destination = Path(plan["checkpoint_root"]) / "global_step_50/actor/peft_adapter/adapter_model.safetensors"
    assert destination.read_bytes() == b"saved adapter"
    continuation.restore_step50_adapter(plan)
    destination.write_bytes(b"unexpected different adapter")
    with pytest.raises(RuntimeError, match="unexpectedly changed"):
        continuation.restore_step50_adapter(plan)


def test_resume_refuses_wrong_latest_checkpoint(tmp_path):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("100")
    with pytest.raises(RuntimeError, match="Expected latest complete checkpoint 50"):
        continuation.validate_resume({"checkpoint_root": str(tmp_path)}, 50)


def test_progress_report_compares_against_step50(tmp_path):
    def summary(step, correct_task):
        output = tmp_path / str(step)
        (output / "summary").mkdir(parents=True)
        rows = [{"benchmark": k, "num_samples": n, "avg_at_k": float(k == correct_task),
                 "pass_at_k": float(k == correct_task)} for k, n in continuation.BENCHMARKS.items()]
        (output / "summary/report.json").write_text(json.dumps({"checkpoint": str(step), "benchmarks": rows}))
        return str(output)
    baseline = summary(50, "aime24")
    step100 = summary(100, "math500")
    continuation.write_results({"job_dir": str(tmp_path), "step50_eval_dir": baseline}, {100: step100})
    report = json.loads((tmp_path / "results.json").read_text())
    assert report["delta_vs_step50_pp"]["100"] == pytest.approx(100 * (2000 - 960) / 7248)
    assert "| 100 |" in (tmp_path / "results.md").read_text()
