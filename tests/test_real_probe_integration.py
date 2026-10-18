from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.workers.actor import dp_actor
from verl.workers.actor.dp_actor import DataParallelPPOActor


def rollout_batch(reward_groups: list[list[int]]) -> DataProto:
    rewards = torch.tensor([value for group in reward_groups for value in group], dtype=torch.float32)
    count = rewards.numel()
    mask = torch.ones(count, 4)
    scores = torch.zeros_like(mask)
    scores[:, 0] = rewards
    return DataProto.from_dict(
        tensors={
            "response_mask": mask,
            "token_level_scores": scores,
            # REAL recomputes population-normalized observation advantages.
            "advantages": torch.zeros_like(mask),
            "inputs": torch.arange(count * 8, dtype=torch.float32).reshape(count, 4, 2) / 10,
        },
        non_tensors={
            "extra_info": np.asarray(
                [{"index": group_id} for group_id, group in enumerate(reward_groups) for _ in group],
                dtype=object,
            )
        },
        meta_info={"temperature": 1.0, "global_steps": 1},
    )


def trainer(*, dp_size: int = 2, prompt_cap: int = 0):
    result = RayPPOTrainer.__new__(RayPPOTrainer)
    result.config = OmegaConf.create(
        {"actor_rollout_ref": {"model": {"peft_type": "real_probe", "real_probe_prompts": prompt_cap}}}
    )
    result.actor_rollout_wg = object()
    result._get_dp_size = lambda worker_group, mesh: dp_size
    return result


def test_selection_preserves_complete_balanced_groups_on_every_dp_rank():
    batch = rollout_batch([[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1], [1, 1, 1, 1]])
    worker = trainer()
    selected, metrics = worker._build_full_gradient_probe_batch(batch)

    assert selected is not None
    assert len(selected) == 16
    assert selected.non_tensor_batch["full_gradient_prompt_id"].tolist() == ["0"] * 4 + ["2"] * 4 + ["0"] * 4 + ["2"] * 4
    assert torch.equal(selected.batch["inputs"][:8], selected.batch["inputs"][8:])
    assert metrics["real_probe/batch_selected_prompts"] == 2
    assert metrics["real_probe/batch_reward_balance_rejected"] == 2

    # A step with no unseen eligible groups still enters the collector, which
    # may need to finish the observation budget and publish its artifacts.
    carrier, metrics = worker._build_full_gradient_probe_batch(batch)
    assert carrier is not None
    assert len(carrier) == 2
    assert carrier.meta_info["real_probe_skip_batch"] is True
    assert metrics["real_probe/batch_selected_prompts"] == 0


def test_selection_honors_group_cap_and_rejects_empty_responses():
    batch = rollout_batch([[1, 1, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0]])
    batch.batch["response_mask"][0].zero_()
    selected, metrics = trainer(dp_size=1, prompt_cap=1)._build_full_gradient_probe_batch(batch)
    assert selected.non_tensor_batch["full_gradient_prompt_id"].tolist() == ["1"] * 4
    assert metrics["real_probe/batch_empty_response_rejected"] == 1


def test_selection_rejects_nonbinary_observation_scores():
    batch = rollout_batch([[1, 1, 0, 0]])
    batch.batch["token_level_scores"][0, 0] = 0.5
    with pytest.raises(ValueError, match="binary correctness"):
        trainer()._build_full_gradient_probe_batch(batch)


class RecordingCollector:
    def __init__(self, model):
        self.model = model
        self.complete = False
        self.observations = []
        self.steps = []

    def capture_and_refresh(self, prompt_id):
        assert not self.model.training
        self.observations.append((prompt_id, self.model.weight.grad.detach().clone()))
        return {"real_probe/observed_prompts": float(len(self.observations))}

    def finish_trainer_step(self, step):
        self.steps.append(step)
        return {"real_probe/artifact_ready": 1.0}


def probe_actor(monkeypatch):
    monkeypatch.setattr(dp_actor, "get_device_id", lambda: "cpu")
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.actor_module = torch.nn.Linear(2, 1, bias=False)
    actor.actor_optimizer = torch.optim.SGD(actor.actor_module.parameters(), lr=1.0)
    actor._static_actor_metrics = {}
    actor._real_probe_collector = RecordingCollector(actor.actor_module)
    actor._forward_micro_batch = lambda inputs, **kwargs: (
        None,
        actor.actor_module(inputs["inputs"]).squeeze(-1),
    )
    return actor


def test_actor_accumulates_one_sequence_normalized_contrast_without_update(monkeypatch):
    actor = probe_actor(monkeypatch)
    batch = rollout_batch([[1, 1, 0, 0]])
    lengths = torch.tensor([1, 2, 3, 4])
    batch.batch["response_mask"] = (torch.arange(4)[None, :] < lengths[:, None]).float()
    batch.non_tensor_batch["full_gradient_prompt_id"] = np.asarray(["prompt"] * 4, dtype=object)
    before = actor.actor_module.weight.detach().clone()

    metrics = actor.run_full_gradient_probe(batch)

    mask = batch.batch["response_mask"]
    mean_inputs = (batch.batch["inputs"] * mask[..., None]).sum(dim=1) / lengths[:, None]
    expected_gradient = -(mean_inputs * torch.tensor([1, 1, -1, -1])[:, None]).mean(dim=0)
    observations = actor._real_probe_collector.observations
    assert len(observations) == 1
    assert observations[0][0] == "prompt"
    torch.testing.assert_close(observations[0][1], expected_gradient[None, :])
    torch.testing.assert_close(actor.actor_module.weight, before)
    assert actor.actor_module.weight.grad is None
    assert actor.actor_module.training
    assert actor._real_probe_collector.steps == [1]
    assert metrics["real_probe/artifact_ready"] == 1
    assert actor.actor_optimizer.state == {}


def test_actor_counts_steps_with_no_eligible_group(monkeypatch):
    actor = probe_actor(monkeypatch)
    batch = rollout_batch([[1, 1, 1, 1]])
    batch.meta_info["real_probe_skip_batch"] = True
    actor.run_full_gradient_probe(batch)
    assert actor._real_probe_collector.steps == [1]
    assert actor._real_probe_collector.observations == []


def test_actor_restores_mode_and_clears_gradients_after_invalid_group(monkeypatch):
    actor = probe_actor(monkeypatch)
    batch = rollout_batch([[1, 1, 1, 0]])
    batch.non_tensor_batch["full_gradient_prompt_id"] = np.asarray(["prompt"] * 4, dtype=object)
    with pytest.raises(ValueError):
        actor.run_full_gradient_probe(batch)
    assert actor.actor_module.training
    assert actor.actor_module.weight.grad is None
