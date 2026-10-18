"""Opt-in tiny two-GPU verification of REAL's production FSDP1 collection path."""

from __future__ import annotations

import datetime
import json
import os
import socket
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import Linear as LoraLinear
from safetensors.torch import load_file
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision

from verl.utils.fsdp_utils import collect_peft_full_params
from verl.utils.peft_gradient_subspace import normalize_target_name
from verl.utils.peft_real import RealProbeCollector, initialize_real_probe, reward_contrast_coefficients


class TinyBlock(torch.nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.q_proj = torch.nn.Linear(width, width, bias=False)

    def forward(self, inputs):
        return torch.tanh(self.q_proj(inputs))


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([TinyBlock(), TinyBlock()])
        self.head = torch.nn.Linear(8, 1, bias=False)

    def forward(self, inputs):
        for layer in self.model.layers:
            inputs = layer(inputs)
        return self.head(inputs).squeeze(-1)


def _assert_replicated(tensor):
    replicas = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(replicas, tensor.contiguous())
    for replica in replicas:
        torch.testing.assert_close(replica, tensor, rtol=0, atol=0)


def _distributed_probe(rank: int, port: int, output_dir: str):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2,
        timeout=datetime.timedelta(seconds=90),
    )
    try:
        torch.manual_seed(912)
        device = torch.device("cuda", rank)
        base = TinyPolicy()
        base_state = {name: value.clone() for name, value in base.state_dict().items()}
        model = get_peft_model(
            base, LoraConfig(r=4, lora_alpha=4, lora_dropout=0, target_modules=["q_proj"], bias="none")
        )
        configuration = {
            "real_probe_rank": 2,
            "real_probe_output_width": 4,
            "real_probe_input_width": 8,
            "real_probe_prompts": 2,
            "real_probe_steps": 4,
            "real_probe_seed": 31,
            "gradient_probe_output_dir": output_dir,
        }
        initialize_real_probe(model, configuration)
        fsdp_kwargs = {
            "device_id": device,
            "use_orig_params": True,
            "sync_module_states": True,
            "mixed_precision": MixedPrecision(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.bfloat16,
            ),
        }
        # Keep A, B and each frozen base matrix within the same collection leaf,
        # matching the actor's transformer-layer wrapping policy.
        for index, layer in enumerate(base.model.layers):
            base.model.layers[index] = FSDP(layer, **fsdp_kwargs)
        wrapped = FSDP(model, **fsdp_kwargs)
        wrapped.eval()
        collector = RealProbeCollector(wrapped, configuration)
        projected_groups = {}
        previous_projections = {}
        with FSDP.summon_full_params(wrapped, writeback=False):
            for name, module in wrapped.named_modules():
                if isinstance(module, LoraLinear):
                    previous_projections[normalize_target_name(name)] = module.lora_B["default"].weight.detach().cpu().clone()

        generator = torch.Generator().manual_seed(503)
        lengths = torch.tensor([1, 3, 4, 2], device=device)
        mask = torch.arange(4, device=device)[None, :] < lengths[:, None]
        coefficients = reward_contrast_coefficients(torch.tensor([1, 1, 0, 0], device=device), mask)
        for group in range(2):
            inputs = torch.randn(4, 4, 8, generator=generator).to(device)
            wrapped.zero_grad(set_to_none=True)
            for response in range(4):
                loss = -coefficients[response] * (wrapped(inputs[response]) * mask[response]).sum()
                loss.backward()
            with FSDP.summon_full_params(wrapped, writeback=False, with_grads=True):
                for name, module in wrapped.named_modules():
                    if not isinstance(module, LoraLinear):
                        continue
                    stable_name = normalize_target_name(name)
                    a = module.lora_A["default"].weight
                    b = module.lora_B["default"].weight
                    assert a.requires_grad and not b.requires_grad
                    # FSDP1 materializes zero gradient views for frozen tensors
                    # sharing a flat buffer with A when summoning full grads.
                    for frozen in (b, module.base_layer.weight):
                        assert frozen.grad is None or torch.count_nonzero(frozen.grad) == 0
                    assert torch.count_nonzero(a) == 0
                    _assert_replicated(a.grad)
                    projected_groups.setdefault(stable_name, []).append(
                        a.grad.detach().float().cpu().clone() / module.scaling["default"]
                    )
            collector.capture_and_refresh(f"group-{group}")
            wrapped.zero_grad(set_to_none=True)
            with FSDP.summon_full_params(wrapped, writeback=False):
                for name, module in wrapped.named_modules():
                    if not isinstance(module, LoraLinear):
                        continue
                    stable_name = normalize_target_name(name)
                    a = module.lora_A["default"].weight
                    b = module.lora_B["default"].weight
                    assert not b.requires_grad and torch.count_nonzero(a) == 0
                    _assert_replicated(b)
                    assert not torch.equal(previous_projections[stable_name], b.detach().cpu())
                    previous_projections[stable_name] = b.detach().cpu().clone()

            # Exercise the actual full-weight rollout synchronization collector,
            # including PEFT merge and unmerge, between observation groups.
            merged = collect_peft_full_params(wrapped)
            assert set(merged) == set(base_state)
            for name, value in merged.items():
                torch.testing.assert_close(value, base_state[name], atol=0, rtol=0)
            for module in wrapped.modules():
                if isinstance(module, LoraLinear):
                    assert not module.merged
                    assert not module.lora_B["default"].weight.requires_grad

        if rank == 0:
            for name, groups in projected_groups.items():
                omega = collector.accumulator.omegas[name]
                expected = sum(h.T @ (h @ omega) for h in groups)
                torch.testing.assert_close(collector.accumulator.sketches[name], expected, atol=2e-7, rtol=2e-6)
            expected_bases, _ = collector.accumulator.snapshot()
        metrics = collector.finish_trainer_step(2)
        assert metrics["real_probe/artifact_ready"] == 1
        assert collector.complete and collector.exported
        _assert_replicated(torch.tensor([metrics["real_probe/prompt_groups"]], device=device))
        if rank == 0:
            exported = load_file(str(Path(output_dir) / "subspaces.safetensors"))
            assert set(exported) == set(projected_groups)
            for name, basis in exported.items():
                torch.testing.assert_close(basis, expected_bases[name], atol=0, rtol=0)
                torch.testing.assert_close(basis @ basis.T, torch.eye(2), atol=2e-6, rtol=2e-6)
            summary = json.loads((Path(output_dir) / "summary.json").read_text())
            assert summary["prompt_groups"] == 2
            (Path(output_dir) / "fsdp_test_memory.json").write_text(
                json.dumps({"max_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20})
            )
        dist.barrier()
    finally:
        # Let spawn report and terminate peers immediately on assertion errors;
        # NCCL teardown can otherwise wait for a peer's unmatched collective.
        if sys.exc_info()[0] is None:
            dist.destroy_process_group()


@pytest.mark.skipif(os.environ.get("RUN_REAL_FSDP_TEST") != "1", reason="set RUN_REAL_FSDP_TEST=1 for the tiny two-GPU test")
def test_real_probe_fsdp_bf16_capture_refresh_and_rollout_sync(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("REAL FSDP integration requires at least two CUDA devices")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    mp.spawn(_distributed_probe, args=(port, str(tmp_path)), nprocs=2, join=True)
