"""A fresh FSDP worker must export the checkpoint adapter on its first rollout."""

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_custom_peft_fsdp_collection import _adapter_spec, _modified_model
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardedStateDictConfig, StateDictType
from transformers import Qwen2ForCausalLM

import verl.utils.fsdp_utils as fsdp_utils
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager


def _reference_merge(child, method):
    """Compute the saved adapter's dense weight without the production merge/export."""
    prefix = "nora_doft" if method == "nora_doft" else "nora_dora"
    weight = child.base_layer.weight.detach().float()
    a = getattr(child, f"{prefix}_A").detach().float()
    b = getattr(child, f"{prefix}_B").detach().float()
    magnitude = getattr(child, f"{prefix}_magnitude").detach().float()
    if method == "nora_doft":
        product = b @ a
        skew = (child.alpha / (2 * child.generator_rank)) * (product - product.transpose(-1, -2))
        identity = torch.eye(child.block_size).expand_as(skew)
        rotation = torch.linalg.solve(identity + skew, identity - skew)
        # Compute each block explicitly rather than using the adapter's einsum.
        blocks = [
            weight[:, index * child.block_size : (index + 1) * child.block_size] @ rotation[index].T
            for index in range(child.num_blocks)
        ]
        direction = torch.cat(blocks, dim=1)
        base_norm = child.nora_doft_base_row_norm.detach().float()
        scale = torch.where(base_norm > child.eps, magnitude / base_norm.clamp_min(child.eps), 1.0)
        return direction * scale[:, None]
    if method == "continuous":
        a = a / torch.linalg.vector_norm(a, dim=0, keepdim=True).clamp_min(child.eps)
    direction = weight + (child.alpha / child.rank) * (b @ a)
    return direction * (magnitude / torch.linalg.vector_norm(direction, dim=1).clamp_min(child.eps))[:, None]


def _wrap(model):
    policy = fsdp_utils.get_fsdp_wrap_policy(model, {"min_num_params": 0}, is_lora=True)
    wrapped = FSDP(
        model,
        auto_wrap_policy=policy,
        device_id=torch.device("cpu"),
        use_orig_params=True,
        device_mesh=init_device_mesh("cpu", (2,), mesh_dim_names=("fsdp",)),
    )
    FSDP.set_state_dict_type(wrapped, StateDictType.SHARDED_STATE_DICT, ShardedStateDictConfig())
    assert sum(isinstance(child, FSDP) for child in wrapped.modules()) > 1
    return wrapped


def _checkpoint_manager(wrapped):
    optimizer = torch.optim.AdamW([parameter for parameter in wrapped.parameters() if parameter.requires_grad], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return FSDPCheckpointManager(wrapped, optimizer, scheduler)


def _resume_worker(rank, rendezvous, checkpoint_root):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=rendezvous, timeout=timedelta(seconds=120))
    try:
        for method in ("standard", "init", "continuous", "nora_doft"):
            source, collect, base_weights, _ = _modified_model(method)
            wrapper_type, apply, _, adapter_config = _adapter_spec(method)
            prefix = "nora_doft" if method == "nora_doft" else "nora_dora"
            # All three trainable components differ from initialization. This
            # detects partial restores as well as an entirely missing adapter.
            with torch.no_grad():
                for child in source.modules():
                    if isinstance(child, wrapper_type):
                        getattr(child, f"{prefix}_A").mul_(1.04).add_(0.01)
            expected_adapter = {
                name: value.detach().clone() for name, value in source.named_parameters() if f".{prefix}_" in name
            }
            expected_weights = {
                f"{name}.weight": _reference_merge(child, method)
                for name, child in source.named_modules()
                if isinstance(child, wrapper_type)
            }
            assert len(expected_adapter) == 42
            assert len(expected_weights) == 14
            inputs = torch.tensor([[1, 4, 7, 3]])
            with torch.no_grad():
                saved_logits = source(inputs).logits

            # Recreate the production topology from the same base model, but
            # with its initially zero B and initial magnitude, before restoring.
            torch.manual_seed(42)
            fresh = Qwen2ForCausalLM(source.config).float().eval()
            apply(fresh, adapter_config)
            for name, child in fresh.named_modules():
                if isinstance(child, wrapper_type):
                    torch.testing.assert_close(child.base_layer.weight, base_weights[f"{name}.weight"], rtol=0, atol=0)
                    assert torch.count_nonzero(getattr(child, f"{prefix}_B")) == 0
                    for suffix in ("A", "B", "magnitude"):
                        assert not torch.equal(
                            getattr(child, f"{prefix}_{suffix}"), expected_adapter[f"{name}.{prefix}_{suffix}"]
                        )

            wrapped_source = _wrap(source)
            source_manager = _checkpoint_manager(wrapped_source)
            checkpoint_path = str(Path(checkpoint_root) / method / "global_step_20" / "actor")
            source_manager.save_checkpoint(checkpoint_path, global_step=20)
            del source_manager, wrapped_source, source
            dist.barrier()

            wrapped_fresh = _wrap(fresh)
            restored_manager = _checkpoint_manager(wrapped_fresh)
            restored_manager.load_checkpoint(checkpoint_path)

            # Deliberately export immediately after load, before any optimizer
            # update or forward call could repair a stale first-rollout policy.
            with patch.object(fsdp_utils, "get_torch_device", return_value=SimpleNamespace(empty_cache=lambda: None)):
                first_export = collect(wrapped_fresh)
            for name, expected in expected_weights.items():
                torch.testing.assert_close(first_export[name], expected, rtol=1e-6, atol=1e-7, msg=f"{method}: {name}")
                assert not torch.equal(first_export[name], base_weights[name]), f"{method}: base-only export {name}"

            with FSDP.summon_full_params(wrapped_fresh, recurse=True, writeback=False):
                restored_parameters = {
                    ".".join(part for part in name.split(".") if part != "_fsdp_wrapped_module"): parameter
                    for name, parameter in wrapped_fresh.named_parameters()
                }
                for name, expected in expected_adapter.items():
                    torch.testing.assert_close(restored_parameters[name], expected, rtol=0, atol=0, msg=name)

            dense = Qwen2ForCausalLM(fresh.config).float().eval()
            dense.load_state_dict(first_export, strict=True)
            with torch.no_grad():
                torch.testing.assert_close(dense(inputs).logits, saved_logits, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(wrapped_fresh(inputs).logits, saved_logits, rtol=1e-5, atol=1e-6)
            dist.barrier()
            del restored_manager, wrapped_fresh, fresh, dense
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU/Gloo distributed support is required")
def test_first_export_after_real_fsdp_checkpoint_resume_contains_saved_adapters(tmp_path):
    """Both ranks restore all adapters and emit their saved policy on the first export."""
    mp.spawn(
        _resume_worker,
        args=((tmp_path / "rendezvous").as_uri(), str(tmp_path / "checkpoints")),
        nprocs=2,
        join=True,
    )
