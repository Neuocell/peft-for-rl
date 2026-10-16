#!/usr/bin/env python
"""Exercise real FSDP export and vLLM reload without changing any training run.

Run ``torchrun --nproc_per_node=4 ... --phase fsdp``, then run the same command
with plain Python and ``--phase vllm``. Outputs include every adapted-layer
comparison and fixed-token probability comparisons, including the base-model
backend error floor. Probe inputs and inference limits are validation settings,
not changes to the RLVR training protocol.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["fsdp", "vllm"], required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def summarize_difference(first, second):
    a, b = torch.tensor(first, dtype=torch.float64), torch.tensor(second, dtype=torch.float64)
    delta = (a - b).abs()
    return {"count": delta.numel(), "mean_abs": delta.mean().item(), "max_abs": delta.max().item(),
            "p95_abs": torch.quantile(delta, .95).item(), "rms": delta.square().mean().sqrt().item()}


@torch.inference_mode()
def probe_hf(model, token_ids, vocab_size):
    result = {"logprobs_full_vocab": [], "logprobs_rollout_vocab": [], "next_token_ids": []}
    for ids in token_ids:
        inputs = torch.tensor([ids], device="cuda")
        logits = model(input_ids=inputs, use_cache=False).logits[0].float()
        labels = inputs[0, 1:]
        full = logits[:-1].log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)
        masked = logits[:-1, :vocab_size].log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)
        result["logprobs_full_vocab"].extend(full.cpu().tolist())
        result["logprobs_rollout_vocab"].extend(masked.cpu().tolist())
        result["next_token_ids"].append(logits[-1, :vocab_size].argmax().item())
    return result


def fsdp_phase(args):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardedStateDictConfig, StateDictType

    from verl.utils.fsdp_utils import collect_nora_dora_full_params, get_fsdp_wrap_policy
    from verl.utils.peft_nora_dora import NoRADoraLinear, apply_nora_dora_adapters, nora_dora_snapshot_config

    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(args.snapshot) / "manifest.json").read_text())
    method = manifest["method_config"]
    config = nora_dora_snapshot_config(method)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32,
                                                attn_implementation="flash_attention_2")
    stats = apply_nora_dora_adapters(model, config)
    snapshot_file = Path(args.snapshot) / "adapter_model.safetensors"
    snapshot = load_file(str(snapshot_file))
    required = {name for name, _ in model.named_parameters() if "nora_dora_" in name}
    assert set(snapshot) == required, "Snapshot does not cover all trainable adapter tensors"
    incompatible = model.load_state_dict(snapshot, strict=False)
    assert not incompatible.unexpected_keys
    assert stats.num_layers == 196 and method["rank"] == 32 and method["alpha"] == 64
    del snapshot

    expected, bases = {}, {}
    if rank == 0:
        # Independent formula, evaluated BEFORE wrapping. No collector/helper reuse.
        for name, layer in model.named_modules():
            if isinstance(layer, NoRADoraLinear):
                w = layer.base_layer.weight.detach().cuda().float()
                a, b = layer.nora_dora_A.detach().cuda().float(), layer.nora_dora_B.detach().cuda().float()
                if method["normalization"] == "continuous":
                    norm = torch.linalg.vector_norm(a, dim=0, keepdim=True)
                    if config["nora_dora_initialization_version"] == "legacy_v1":
                        a = a / norm.clamp_min(method["eps"])
                    else:
                        a = a / (norm + method["eps"])
                direction = w + (method["alpha"] / method["rank"]) * (b @ a)
                merged = direction * (layer.nora_dora_magnitude.detach().cuda().float() /
                                      direction.norm(dim=1).clamp_min(method["eps"]))[:, None]
                expected[name + ".weight"] = merged.cpu()
                bases[name + ".weight"] = layer.base_layer.weight.detach().clone()
        del w, a, b, direction, merged
        print("Independent full-model merge reference prepared", flush=True)

    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("fsdp",))
    policy = get_fsdp_wrap_policy(model, {"min_num_params": 0}, is_lora=True)
    model.eval()
    wrapped = FSDP(model, auto_wrap_policy=policy, device_id=torch.cuda.current_device(),
                   use_orig_params=True, device_mesh=mesh, sync_module_states=True,
                   mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                                  reduce_dtype=torch.float32, buffer_dtype=torch.float32))
    FSDP.set_state_dict_type(wrapped, StateDictType.SHARDED_STATE_DICT, ShardedStateDictConfig())
    collected = collect_nora_dora_full_params(wrapped)
    if rank == 0:
        checks = [{"name": name, "merged_max_abs": (collected[name] - expected[name]).abs().max().item(),
                   "base_max_abs": (collected[name] - bases[name]).abs().max().item(),
                   "merged_sha256": digest(collected[name]),
                   "expected_sha256": digest(expected[name])} for name in expected]
        assert len(checks) == 196 and all(x["merged_max_abs"] == 0 for x in checks), checks
        assert all(x["base_max_abs"] > 0 for x in checks), "An adapted layer still equals base"
        save_file({name: tensor.to(torch.bfloat16).contiguous() for name, tensor in collected.items()},
                  str(output / "collected_bf16.safetensors"))
        print("196/196 collected weights exactly match independent FP32 merge; all differ from base", flush=True)
    del expected, bases, collected
    gc.collect()
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    texts = ["Solve: If 3x + 5 = 20, what is x?\nLet us reason step by step.\n3x = 15, so x = 5.",
             "A bag contains 4 red balls and 6 blue balls. What is the probability of drawing a red ball?\n"
             "The probability is 4/10 = 2/5.",
             "Find the sum of the first 10 positive integers.\nUsing n(n+1)/2, the answer is 10*11/2 = 55."]
    token_ids = [tokenizer.encode(text, add_special_tokens=True) for text in texts]
    with model.disable_adapter():
        actor_base = probe_hf(wrapped, token_ids, len(tokenizer))
    actor = probe_hf(wrapped, token_ids, len(tokenizer))
    if rank == 0:
        report = {"phase": "four_gpu_fsdp", "world_size": dist.get_world_size(),
                  "model": str(Path(args.base_model).resolve()), "snapshot": str(Path(args.snapshot).resolve()),
                  "snapshot_sha256": hashlib.sha256(snapshot_file.read_bytes()).hexdigest(),
                  "torch_version": torch.__version__, "master_dtype": "float32", "compute_dtype": "bfloat16",
                  "attention": "flash_attention_2", "method_config": method, "adapter_count": len(checks),
                  "exact_merge_count": sum(x["merged_max_abs"] == 0 for x in checks),
                  "non_base_count": sum(x["base_max_abs"] > 0 for x in checks),
                  "collector_sha256": hashlib.sha256(
                      Path(inspect.getsourcefile(collect_nora_dora_full_params)).read_bytes()
                  ).hexdigest(),
                  "wrap_policy": str(policy), "layers": checks, "token_ids": token_ids,
                  "tokenizer_vocab_size": len(tokenizer), "actor_base": actor_base, "actor": actor,
                  "actor_effect": summarize_difference(actor["logprobs_full_vocab"], actor_base["logprobs_full_vocab"])}
        write_json(output / "fsdp_report.json", report)
        print(json.dumps({
            k: report[k] for k in ("adapter_count", "exact_merge_count", "non_base_count", "actor_effect")
        }))
    dist.barrier()
    dist.destroy_process_group()


def reload_and_inspect(model, weight_path):
    """Runs inside the real vLLM worker, exercises production model.load_weights."""
    state = load_file(weight_path)
    loaded = model.load_weights(iter(state.items()))
    live = dict(model.named_parameters())
    checks = []
    for name, expected in state.items():
        if not any(name.endswith(f".{part}.weight") for part in
                   ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")):
            continue
        if any(f".{part}." in name for part in ("q_proj", "k_proj", "v_proj")):
            part = name.rsplit(".", 2)[-2]
            prefix = name.rsplit(".", 2)[0]
            packed = live[prefix + ".qkv_proj.weight"]
            qn = state[prefix + ".q_proj.weight"].shape[0]
            kn = state[prefix + ".k_proj.weight"].shape[0]
            offset = {"q_proj": 0, "k_proj": qn, "v_proj": qn + kn}[part]
            actual = packed[offset:offset + expected.shape[0]]
        elif ".gate_proj." in name or ".up_proj." in name:
            part = name.rsplit(".", 2)[-2]
            prefix = name.rsplit(".", 2)[0]
            offset = 0 if part == "gate_proj" else expected.shape[0]
            actual = live[prefix + ".gate_up_proj.weight"][offset:offset + expected.shape[0]]
        else:
            actual = live[name]
        checks.append({"name": name, "max_abs": (actual.detach().cpu().float() - expected.float()).abs().max().item()})
    return {"loaded_parameter_count": len(loaded), "adapted_layer_count": len(checks),
            "exact_loaded_layer_count": sum(x["max_abs"] == 0 for x in checks), "layers": checks}


def mask_vllm_vocab(model, vocab_size):
    from verl.workers.rollout.vllm_rollout.vllm_rollout import _monkey_patch_compute_logits
    _monkey_patch_compute_logits(model, vocab_size)
    return {"masked_vocab_size": vocab_size}


def vllm_phase(args):
    from functools import partial
    # The local v1 worker RPC serializes the inspection callbacks with pickle.
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    import vllm
    from vllm import LLM, SamplingParams

    torch.set_num_threads(2)
    output = Path(args.output_dir)
    source = json.loads((output / "fsdp_report.json").read_text())
    token_ids, vocab_size = source["token_ids"], source["tokenizer_vocab_size"]
    reference = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16,
                                                    attn_implementation="flash_attention_2").cuda().eval()
    hf_base = probe_hf(reference, token_ids, vocab_size)
    weights = load_file(str(output / "collected_bf16.safetensors"))
    reference.load_state_dict(weights, strict=True)
    del weights
    hf_merged = probe_hf(reference, token_ids, vocab_size)
    del reference
    gc.collect()
    torch.cuda.empty_cache()
    # Production rollout uses TP=1; four independent replicas are unnecessary for
    # this loader check. Eager/short context bounds the validation resource use.
    llm = LLM(model=args.base_model, dtype="bfloat16", tensor_parallel_size=1,
              gpu_memory_utilization=.35, max_model_len=256, max_num_seqs=1,
              enforce_eager=True, enable_prefix_caching=False, seed=42)
    llm.apply_model(partial(mask_vllm_vocab, vocab_size=vocab_size))

    def probe_vllm():
        probabilities, next_tokens = [], []
        for ids in token_ids:
            generated = llm.generate([{"prompt_token_ids": ids}],
                                     SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1), use_tqdm=False)[0]
            probabilities.extend(generated.prompt_logprobs[i][token].logprob for i, token in enumerate(ids) if i)
            next_tokens.append(generated.outputs[0].token_ids[0])
        return {"logprobs_rollout_vocab": probabilities, "next_token_ids": next_tokens}

    vllm_base = probe_vllm()
    loaded = llm.apply_model(partial(
        reload_and_inspect, weight_path=str((output / "collected_bf16.safetensors").resolve())
    ))
    assert len(loaded) == 1 and loaded[0]["exact_loaded_layer_count"] == 196, loaded
    vllm_merged = probe_vllm()
    arrays = {"actor_base": source["actor_base"], "actor": source["actor"],
              "hf_base": hf_base, "hf_merged": hf_merged, "vllm_base": vllm_base, "vllm_merged": vllm_merged}
    pairs = [("hf_base", "vllm_base"), ("hf_merged", "vllm_merged"), ("actor_base", "hf_base"),
             ("actor", "hf_merged"), ("actor_base", "vllm_base"), ("actor", "vllm_merged"),
             ("vllm_base", "vllm_merged"), ("hf_base", "hf_merged")]
    comparisons = {f"{a}_vs_{b}": summarize_difference(arrays[a]["logprobs_rollout_vocab"],
                                                       arrays[b]["logprobs_rollout_vocab"]) for a, b in pairs}
    report = {"phase": "real_vllm_reload", "vllm_version": vllm.__version__, "tp": 1,
              "inference_only_settings": {"enforce_eager": True, "max_model_len": 256, "max_num_seqs": 1,
                                          "gpu_memory_utilization": .35, "temperature": 0},
              "scope": "Fixed-token logprobs; full 196 adapted-layer loaded-weight equality. "
                       "No RL update or long rollout tested.",
              "source_collector_sha256": source["collector_sha256"], "loaded": loaded,
              "comparisons": comparisons, "probes": arrays}
    write_json(output / "vllm_report.json", report)
    print(json.dumps(comparisons, indent=2), flush=True)


if __name__ == "__main__":
    arguments = parse_args()
    (fsdp_phase if arguments.phase == "fsdp" else vllm_phase)(arguments)
