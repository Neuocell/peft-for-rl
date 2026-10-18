from __future__ import annotations

import itertools
import math
from datetime import timedelta
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.full_gradient_rl_probe import build_probe_token_mask
from verl.utils.gradtop_objectives import (
    GradTopNormalization,
    build_gradtop_normalization,
    gradtop_policy_loss,
    gradtop_probe_variant,
    gradtop_prompt_signs,
    validate_gradtop_actor_config,
)
from verl.workers.actor import dp_actor
from verl.workers.actor.dp_actor import DataParallelPPOActor


def actor_config(variant="prompt_sign", *, dynamic=False):
    return OmegaConf.create({
        "model_config": {"peft_type": "grad_probe", "gradient_probe_variant": variant, "gradient_probe_sign_seed": 42},
        "policy_loss": {"loss_mode": "vanilla"},
        "clip_ratio": 0.2, "clip_ratio_low": 0.2, "clip_ratio_high": 0.2, "clip_ratio_c": 3.0,
        "global_batch_info": {}, "loss_agg_mode": "token-mean", "entropy_coeff": 0.0,
        "calculate_entropy": False, "use_kl_loss": False, "ppo_epochs": 1,
        "ulysses_sequence_parallel_size": 1, "biso_raw_l2_coef": 0.0,
        "ppo_mini_batch_size": 4, "ppo_micro_batch_size_per_gpu": 1,
        "use_dynamic_bsz": dynamic, "ppo_max_token_len_per_gpu": 64,
        "fsdp_config": {"seed": 42},
    })


def loss_inputs():
    # Unequal response lengths and clipping on both signs of advantage.
    return {
        "old_log_prob": torch.zeros(6, 4, dtype=torch.float64),
        "log_prob": torch.tensor([
            [0.4, -0.4, 0.1, 0.8], [0.0, 1.5, -0.3, 0.2],
            [-0.5, 0.5, 0.0, 0.1], [0.2, -0.1, 0.4, 0.3],
            [0.0, 0.1, -0.3, 0.4], [-0.2, 0.0, 0.1, 0.2],
        ], dtype=torch.float64),
        "advantages": torch.tensor([1., -1., 1.4, -1.4, 0., 0.], dtype=torch.float64)[:, None].expand(6, 4),
        "response_mask": (torch.arange(4)[None] < torch.tensor([1, 4, 2, 3, 4, 1])[:, None]).double(),
    }


def test_token_mean_matches_vanilla_dual_clipped_gradient():
    inputs = loss_inputs()
    inputs["log_prob"].requires_grad_()
    normalizer = build_gradtop_normalization(torch.arange(3).repeat_interleave(2), inputs["response_mask"], expected_rollouts=2)
    actual, _ = gradtop_policy_loss(**inputs, variant="token_mean", normalization=normalizer, config=actor_config())
    expected, _ = compute_policy_loss_vanilla(**inputs, config=actor_config())
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, inputs["log_prob"], retain_graph=True)[0],
                               torch.autograd.grad(expected, inputs["log_prob"])[0])


def test_signs_are_shared_by_prompt_and_stable_under_reordering_and_packing():
    ids = torch.tensor([100, 100, 101, 101, 102, 102])
    kwargs = dict(seed=42, trainer_step=7, minibatch_index=2)
    signs = gradtop_prompt_signs(ids, **kwargs)
    assert set(signs.tolist()) <= {-1., 1.}
    assert torch.equal(signs[::2], signs[1::2])
    order = torch.tensor([5, 0, 3, 1, 4, 2])
    assert torch.equal(gradtop_prompt_signs(ids[order], **kwargs), signs[order])
    assert torch.equal(torch.cat([gradtop_prompt_signs(ids[:1], **kwargs), gradtop_prompt_signs(ids[1:], **kwargs)]), signs)
    many_ids = torch.arange(100)
    assert not torch.equal(gradtop_prompt_signs(many_ids, **kwargs),
                           gradtop_prompt_signs(many_ids, **{**kwargs, "minibatch_index": 3}))


def test_signed_gradient_equals_signed_dense_prompt_gradients_and_exact_second_moment():
    inputs = loss_inputs()
    ids = torch.arange(3).repeat_interleave(2)
    normalizer = build_gradtop_normalization(ids, inputs["response_mask"], expected_rollouts=2)
    # Shared linear parameters produce nontrivial cross terms between prompts.
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(6, 4, 5, generator=generator, dtype=torch.float64)
    theta = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    log_prob = inputs["log_prob"] + features @ theta
    dense = []
    for prompt in range(3):
        rows = ids == prompt
        local = {key: value[rows] for key, value in inputs.items()}
        local["log_prob"] = log_prob[rows]
        one = GradTopNormalization(1, 2, int(local["response_mask"].sum()), 1, 2)
        objective, _ = gradtop_policy_loss(**local, variant="prompt_mean", normalization=one, config=actor_config())
        dense.append(torch.autograd.grad(objective, theta, retain_graph=True)[0])
    dense = torch.stack(dense)
    torch.testing.assert_close(dense[2], torch.zeros_like(dense[2]))  # retain zero-advantage groups in P
    outers = []
    for choice in itertools.product([-1., 1.], repeat=3):
        signs = torch.tensor(choice, dtype=torch.float64)
        signed, _ = gradtop_policy_loss(**{**inputs, "log_prob": log_prob}, variant="prompt_sign",
                                       normalization=normalizer, config=actor_config(), signs=signs.repeat_interleave(2))
        gradient = torch.autograd.grad(signed, theta, retain_graph=True)[0]
        expected = (dense * signs[:, None]).sum(0) / math.sqrt(3)
        torch.testing.assert_close(gradient, expected)
        outers.append(torch.outer(gradient, gradient))
    torch.testing.assert_close(torch.stack(outers).mean(0), dense.T @ dense / 3)


@pytest.mark.parametrize("variant", ["token_mean", "prompt_mean", "prompt_sign", "masked"])
def test_uneven_microbatch_splits_and_dp_average_preserve_gradient(variant):
    inputs = loss_inputs()
    inputs["log_prob"].requires_grad_()
    ids = torch.arange(3).repeat_interleave(2)
    normalizer = build_gradtop_normalization(ids, inputs["response_mask"], expected_rollouts=2)
    signs = torch.tensor([-1., -1., 1., 1., -1., -1.])
    mask = inputs["response_mask"].clone()
    mask[:, 1] = 0
    kwargs = dict(variant=variant, normalization=normalizer, config=actor_config(), signs=signs, token_keep_mask=mask)
    full, _ = gradtop_policy_loss(**inputs, **kwargs)
    full_grad = torch.autograd.grad(full, inputs["log_prob"], retain_graph=True)[0]
    packed = inputs["log_prob"].sum() * 0
    for rows in [torch.tensor([5]), torch.tensor([2, 0, 4]), torch.tensor([1, 3])]:
        local = {key: value[rows] for key, value in inputs.items()}
        part, _ = gradtop_policy_loss(**local, **{**kwargs, "signs": signs[rows], "token_keep_mask": mask[rows]})
        packed = packed + part
    torch.testing.assert_close(torch.autograd.grad(packed, inputs["log_prob"], retain_graph=True)[0], full_grad)
    # Emulate two DP ranks with globally reduced counts and FSDP's average.
    distributed = GradTopNormalization(3, 6, normalizer.valid_tokens, 2, 2)
    rank_losses = []
    for rows in [slice(0, 2), slice(2, 6)]:
        local = {key: value[rows] for key, value in inputs.items()}
        part, _ = gradtop_policy_loss(**local, **{**kwargs, "normalization": distributed,
                                                "signs": signs[rows], "token_keep_mask": mask[rows]})
        rank_losses.append(part)
    torch.testing.assert_close(torch.autograd.grad(sum(rank_losses) / 2, inputs["log_prob"])[0], full_grad)


def test_mask_removes_token_gradients_without_renormalizing_and_keeps_final_tokens():
    log_prob = -torch.arange(512, dtype=torch.float64).flip(0)[None].requires_grad_() / 100
    mask = torch.ones_like(log_prob)
    keep = build_probe_token_mask(log_prob, mask, mode="top_surprisal", keep_ratio=.5, min_keep=128, final_tokens=128)
    assert keep.sum() == 256
    assert torch.all(keep[:, :128] == 1) and torch.all(keep[:, -128:] == 1)
    assert torch.all(keep[:, 128:384] == 0)
    objective, _ = gradtop_policy_loss(old_log_prob=log_prob.detach(), log_prob=log_prob, advantages=mask,
                                     response_mask=mask, variant="masked", token_keep_mask=keep,
                                     normalization=GradTopNormalization(1, 1, 512, 1, 1), config=actor_config())
    gradient = torch.autograd.grad(objective, log_prob)[0]
    torch.testing.assert_close(gradient, -keep / 512)
    assert objective.item() == -.5  # dividing by kept tokens would incorrectly give -1


def test_negative_sign_applies_after_clipping_not_to_advantages():
    value = torch.tensor([[math.log(1.5)]], dtype=torch.float64, requires_grad=True)
    kwargs = dict(old_log_prob=torch.zeros_like(value), log_prob=value, response_mask=torch.ones_like(value),
                  normalization=GradTopNormalization(1, 1, 1, 1, 1), config=actor_config())
    signed, _ = gradtop_policy_loss(**kwargs, advantages=torch.ones_like(value), variant="prompt_sign", signs=torch.tensor([-1.]))
    wrong, _ = gradtop_policy_loss(**kwargs, advantages=-torch.ones_like(value), variant="prompt_mean")
    assert torch.autograd.grad(signed, value, retain_graph=True)[0].item() == 0
    assert torch.autograd.grad(wrong, value)[0].item() == pytest.approx(1.5)


def test_normalization_rejects_split_groups_and_empty_responses():
    with pytest.raises(ValueError, match="complete prompt groups"):
        build_gradtop_normalization(torch.tensor([0, 0, 1]), torch.ones(3, 2), expected_rollouts=2)
    with pytest.raises(ValueError, match="nonempty responses"):
        build_gradtop_normalization(torch.tensor([0, 0]), torch.tensor([[1., 0.], [0., 0.]]), expected_rollouts=2)


@pytest.mark.parametrize("key,value", [("entropy_coeff", .1), ("use_kl_loss", True), ("ppo_epochs", 2),
                                       ("ulysses_sequence_parallel_size", 2), ("biso_raw_l2_coef", .1)])
def test_probe_rejects_unsupported_actor_settings(key, value):
    config = actor_config()
    config[key] = value
    with pytest.raises(ValueError):
        validate_gradtop_actor_config(config)


def trainer_config():
    actor = actor_config()
    return OmegaConf.create({
        "actor_rollout_ref": {"actor": actor, "model": actor.model_config, "rollout": {"n": 2, "seed": 42}},
        "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": False, "rollout_correction": None},
        "data": {"train_batch_size": 8, "max_prompt_length": 2048, "max_response_length": 8192, "seed": 42},
    })


def test_trainer_assigns_stable_prompt_ids_and_preserves_groups_through_dp_split():
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = trainer_config()
    trainer.global_steps = 3
    trainer.use_legacy_worker_impl = "enable"
    trainer.actor_rollout_wg = object()
    trainer._get_dp_size = lambda *_: 2
    batch = DataProto.from_dict(tensors={"input_ids": torch.ones(8, 3, dtype=torch.int64)})
    assert trainer._prepare_gradtop_prompt_groups(batch)
    assert batch.batch["gradtop_prompt_id"].tolist() == list(range(16, 24))
    contract = batch.meta_info["gradtop_probe_contract"]
    assert contract["responses_per_prompt"] == 2 and contract["balance_batch"] is False
    repeated = batch.repeat(repeat_times=2, interleave=True)
    for shard in repeated.chunk(2):
        for mini in shard.split(4):
            assert torch.unique(mini.batch["gradtop_prompt_id"], return_counts=True)[1].tolist() == [2, 2]
    trainer.config.algorithm.rollout_correction = {"rollout_is": "token"}
    with pytest.raises(ValueError, match="correction disabled"):
        trainer._prepare_gradtop_prompt_groups(batch)


class RecordingCollector:
    def __init__(self, module):
        self.module = module
        self.observations = []
        self.contract = None

    def set_runtime_contract(self, contract):
        self.contract = contract

    def capture_and_refresh(self, *, observation_metadata):
        self.observations.append((self.module.weight.grad.detach().clone(), observation_metadata))
        return {}

    def finish_trainer_step(self, trainer_step):
        return {"gradient_probe/step": trainer_step}


def fake_actor(monkeypatch, variant="prompt_sign", *, dynamic=False):
    monkeypatch.setattr(dp_actor, "get_device_id", lambda: "cpu")
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.config = actor_config(variant, dynamic=dynamic)
    actor.actor_module = torch.nn.Linear(2, 1, bias=False).double()
    actor.actor_optimizer = torch.optim.SGD(actor.actor_module.parameters(), lr=.1)
    actor._adalora_model = None
    actor._gradient_probe_collector = RecordingCollector(actor.actor_module)
    actor._static_actor_metrics = {}
    actor.ulysses_sequence_parallel_size = 1
    actor.scaler = None
    actor._forward_micro_batch = lambda inputs, **_: (None, actor.actor_module(inputs["input_ids"]).squeeze(-1))
    actor._optimizer_step = lambda **_: pytest.fail("Explicit probe must never clip or invoke optimizer step")
    return actor


def fake_batch():
    features = torch.arange(24, dtype=torch.float64).reshape(4, 3, 2) / 30
    masks = (torch.arange(3)[None] < torch.tensor([1, 3, 2, 3])[:, None]).double()
    return DataProto.from_dict(tensors={
        "input_ids": features, "responses": torch.ones(4, 3, dtype=torch.long), "response_mask": masks,
        "attention_mask": torch.ones(4, 3), "position_ids": torch.ones(4, 3, dtype=torch.long),
        "old_log_probs": torch.zeros(4, 3, dtype=torch.float64),
        "advantages": torch.tensor([1., -1., 2., -2.])[:, None].expand(4, 3),
        "gradtop_prompt_id": torch.tensor([10, 10, 11, 11]),
    }, meta_info={"temperature": 1., "global_steps": 1, "gradtop_rollouts_per_prompt": 2,
                  "gradtop_probe_contract": {"test": True}})


@pytest.mark.parametrize("dynamic", [False, True])
def test_actor_accumulates_one_signed_observation_without_policy_update(monkeypatch, dynamic):
    actor = fake_actor(monkeypatch, dynamic=dynamic)
    if dynamic:
        monkeypatch.setattr(dp_actor, "prepare_dynamic_batch", lambda mini, **_: (
            [mini.select_idxs(torch.tensor([3, 0])), mini.select_idxs(torch.tensor([2])), mini.select_idxs(torch.tensor([1]))], None))
    batch = fake_batch()
    before = actor.actor_module.weight.detach().clone()
    actor.update_policy(batch)
    signs = gradtop_prompt_signs(batch.batch["gradtop_prompt_id"], seed=42, trainer_step=1, minibatch_index=0)
    means = (batch.batch["input_ids"] * batch.batch["response_mask"][..., None]).sum(1) / batch.batch["response_mask"].sum(1)[:, None]
    expected = -(means * (batch.batch["advantages"][:, 0] * signs)[:, None]).sum(0) / (2 * math.sqrt(2))
    observed, metadata = actor._gradient_probe_collector.observations[0]
    torch.testing.assert_close(observed, expected[None])
    assert metadata == {"prompt_groups": 2, "responses": 4, "valid_tokens": 9}
    assert len(actor._gradient_probe_collector.observations) == 1
    torch.testing.assert_close(actor.actor_module.weight, before)
    assert actor.actor_optimizer.state == {} and actor.actor_module.weight.grad is None


def test_ordinary_training_ignores_probe_variant_and_still_updates(monkeypatch):
    actor = fake_actor(monkeypatch)
    actor.config.model_config.peft_type = "lora"
    actor._gradient_probe_collector = None
    assert gradtop_probe_variant(actor.config.model_config) == "legacy"
    before = actor.actor_module.weight.detach().clone()
    def step(**kwargs):
        assert kwargs["apply_update"] is True
        actor.actor_optimizer.step()
        return torch.tensor(1.)
    actor._optimizer_step = step
    batch = fake_batch()
    del batch.batch["gradtop_prompt_id"]
    batch.meta_info.pop("gradtop_probe_contract")
    actor.update_policy(batch)
    assert not torch.equal(actor.actor_module.weight, before)


def _gloo_objective_worker(rank, rendezvous, output_dir):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo", rank=rank, world_size=2, init_method=f"file://{rendezvous}", timeout=timedelta(seconds=30)
    )
    try:
        inputs = loss_inputs()
        ids = torch.arange(3).repeat_interleave(2)
        rows = slice(0, 2) if rank == 0 else slice(2, 6)
        inputs = {key: value[rows] for key, value in inputs.items()}
        ids = ids[rows]
        normalization = build_gradtop_normalization(ids, inputs["response_mask"], expected_rollouts=2)
        features = torch.arange(6 * 4 * 3, dtype=torch.float64).reshape(6, 4, 3)[rows] / 20
        result = {"normalization": normalization.metadata(), "gradients": {}}
        for variant in ["token_mean", "prompt_mean", "prompt_sign", "masked"]:
            theta = torch.zeros(3, dtype=torch.float64, requires_grad=True)
            log_prob = inputs["log_prob"] + features @ theta
            signs = gradtop_prompt_signs(ids, seed=42, trainer_step=2, minibatch_index=0)
            mask = inputs["response_mask"].clone()
            mask[:, 1] = 0
            # Different microbatch schedules on the two ranks must not change
            # the shared normalization or require extra collectives.
            objective = theta.sum() * 0
            for start, end in ([(0, 1), (1, 2)] if rank == 0 else [(0, 3), (3, 4)]):
                part, _ = gradtop_policy_loss(
                    **{key: value[start:end] for key, value in inputs.items() if key != "log_prob"},
                    log_prob=log_prob[start:end], variant=variant, normalization=normalization,
                    config=actor_config(), signs=signs[start:end], token_keep_mask=mask[start:end],
                )
                objective = objective + part
            gradient = torch.autograd.grad(objective, theta)[0]
            torch.distributed.all_reduce(gradient)
            gradient /= 2  # FSDP's gradient reduction convention
            result["gradients"][variant] = gradient
        torch.save(result, Path(output_dir) / f"rank{rank}.pt")
    finally:
        torch.distributed.destroy_process_group()


def test_two_rank_cpu_gloo_normalization_and_gradient_reduction(tmp_path):
    torch.multiprocessing.spawn(
        _gloo_objective_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True
    )
    inputs = loss_inputs()
    ids = torch.arange(3).repeat_interleave(2)
    normalization = build_gradtop_normalization(ids, inputs["response_mask"], expected_rollouts=2)
    features = torch.arange(6 * 4 * 3, dtype=torch.float64).reshape(6, 4, 3) / 20
    results = [torch.load(tmp_path / f"rank{rank}.pt", weights_only=True) for rank in range(2)]
    for result in results:
        assert result["normalization"] == normalization.metadata()
    for variant in ["token_mean", "prompt_mean", "prompt_sign", "masked"]:
        theta = torch.zeros(3, dtype=torch.float64, requires_grad=True)
        signs = gradtop_prompt_signs(ids, seed=42, trainer_step=2, minibatch_index=0)
        mask = inputs["response_mask"].clone()
        mask[:, 1] = 0
        objective, _ = gradtop_policy_loss(
            **{**inputs, "log_prob": inputs["log_prob"] + features @ theta}, variant=variant,
            normalization=normalization, config=actor_config(), signs=signs, token_keep_mask=mask,
        )
        expected = torch.autograd.grad(objective, theta)[0]
        for result in results:
            torch.testing.assert_close(result["gradients"][variant], expected)
