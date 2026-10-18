import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from scripts.local import real_probe_cache as cache
from verl.utils.peft_real import RealSubspaceAccumulator


def collector():
    acc = RealSubspaceAccumulator(rank=2, input_width=4, seed=42)
    generator = torch.Generator().manual_seed(7)
    for _ in range(3):
        acc.add("model.layers.0.q_proj", torch.randn(4, 6, generator=generator))
    return SimpleNamespace(
        accumulator=acc, observations=3, prompt_ids=["p0", "p1", "p2"],
        module_shapes={"model.layers.0.q_proj": (5, 6)}, rank=2,
        input_width=4, output_width=4, seed=42, scaling_ratio=2.0,
        config={"path": "/model"}, output_dir=Path("/probe").resolve(),
        target_steps=64, target_prompts=0, finish_trainer_step=lambda step: {"step": step},
    )


def test_saved_prefix_matches_online_basis_after_accumulator_advances(tmp_path):
    c = collector()
    expected, _ = c.accumulator.snapshot()
    snapshot = cache.save_snapshot(c, tmp_path / "cache", 2, reserve_bytes=0)
    frozen_y = {name: value.clone() for name, value in c.accumulator.sketches.items()}
    c.accumulator.add("model.layers.0.q_proj", torch.ones(4, 6))
    output = tmp_path / "replayed"
    cache.export_snapshot(snapshot["snapshot"], output)
    for name, value in load_file(str(output / "subspaces.safetensors")).items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    for name, value in load_file(str(Path(snapshot["snapshot"]) / "y_sum.safetensors")).items():
        torch.testing.assert_close(value, frozen_y[name], rtol=0, atol=0)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["trainer_step"] == summary["requested_observation_steps"] == 2
    assert summary["prompt_groups"] == 3
    assert summary["optimizer_updates"] == 0
    assert summary["subspace_path"] == str(output / "subspaces.safetensors")


def test_cache_rejects_incomplete_module_counts(tmp_path):
    c = collector()
    c.accumulator.counts["model.layers.0.q_proj"] = 2
    with pytest.raises(ValueError, match="boundary"):
        cache.save_snapshot(c, tmp_path, 2, reserve_bytes=0)


def test_snapshot_corruption_is_detected_before_export(tmp_path):
    saved = cache.save_snapshot(collector(), tmp_path / "cache", 2, reserve_bytes=0)
    path = Path(saved["snapshot"]) / "y_sum.safetensors"
    with path.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        cache.export_snapshot(saved["snapshot"], tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_live_hook_preserves_metrics_and_only_saves_selected_steps(tmp_path, monkeypatch):
    c = collector()
    worker = SimpleNamespace(rank=0, actor=SimpleNamespace(_real_probe_collector=c))
    monkeypatch.setenv("REAL_PIPELINE_STAGE_TOKEN", "owned")
    # Exercise the observer independently of free space on the test host.
    monkeypatch.setattr(cache.shutil, "disk_usage", lambda path: SimpleNamespace(free=64 * 1024**3))
    before = {name: value.clone() for name, value in c.accumulator.sketches.items()}
    policy = cache.attach_worker_cache(worker, cache_dir=tmp_path, expected_output_dir="/probe",
                                       expected_stage_token="owned", steps=[32, 64])
    assert policy["initial"]["trainer_step"] is None
    assert c.finish_trainer_step(31) == {"step": 31}
    assert not (tmp_path / "step_031").exists()
    assert c.finish_trainer_step(32) == {"step": 32}
    assert (tmp_path / "step_032/metadata.json").is_file()
    for name, value in before.items():
        torch.testing.assert_close(c.accumulator.sketches[name], value, rtol=0, atol=0)
    def disk_error(*args, **kwargs):
        raise OSError("disk failure")
    monkeypatch.setattr(cache, "save_snapshot", disk_error)
    assert c.finish_trainer_step(64) == {"step": 64}
    assert "disk failure" in (tmp_path / "step_064.error.json").read_text()


def test_wrong_actor_cannot_attach(tmp_path, monkeypatch):
    monkeypatch.setenv("REAL_PIPELINE_STAGE_TOKEN", "different")
    worker = SimpleNamespace(rank=0, actor=SimpleNamespace(_real_probe_collector=collector()))
    with pytest.raises(ValueError, match="wrong experiment"):
        cache.attach_worker_cache(worker, cache_dir=tmp_path, expected_output_dir="/probe",
                                  expected_stage_token="owned", steps=[32])
    assert not list(tmp_path.iterdir())


def test_existing_cache_cannot_be_mixed_with_new_collector(tmp_path):
    cache.save_snapshot(collector(), tmp_path, 2, reserve_bytes=0)
    with pytest.raises(FileExistsError, match="mix"):
        cache.save_snapshot(collector(), tmp_path, 3, reserve_bytes=0)


def test_no_invented_step_for_live_group_prefix(tmp_path):
    saved = cache.save_snapshot(collector(), tmp_path / "cache", reserve_bytes=0)
    result = cache.export_snapshot(saved["snapshot"], tmp_path / "export")
    assert result["trainer_step"] is None
    summary = json.loads((tmp_path / "export/summary.json").read_text())
    assert summary["requested_prompt_groups"] == 3
    assert summary["requested_observation_steps"] is None
