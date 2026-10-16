"""Read back merged adapter weights before permitting the first rollout.

This check covers unsharded dense weights and Qwen/Llama-style packed QKV and
gate/up projections. Callers enable it for tensor parallel size one. Comparing
in the receiver's dtype accounts for the intended FP32 -> BF16 conversion.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence

import torch


def _loaded_weight(name, received, live):
    if name in live:
        return live[name]
    prefix, projection, suffix = name.rsplit(".", 2)
    if projection in {"q_proj", "k_proj", "v_proj"}:
        parts = ("q_proj", "k_proj", "v_proj")
        packed_name = f"{prefix}.qkv_proj.{suffix}"
    elif projection in {"gate_proj", "up_proj"}:
        parts = ("gate_proj", "up_proj")
        packed_name = f"{prefix}.gate_up_proj.{suffix}"
    else:
        raise RuntimeError(f"Cannot read back merged adapter weight: {name}")
    if packed_name not in live:
        raise RuntimeError(f"Missing packed rollout parameter for {name}: {packed_name}")
    part_names = [f"{prefix}.{part}.{suffix}" for part in parts]
    missing = set(part_names) - received.keys()
    if missing:
        raise RuntimeError(f"Missing projection references for {name}: {sorted(missing)}")
    sizes = [received[key].shape[0] for key in part_names]
    packed = live[packed_name]
    if packed.shape[0] != sum(sizes):
        raise RuntimeError(f"Unsupported sharded/packed shape for {name}: {tuple(packed.shape)}")
    index = parts.index(projection)
    offset = sum(sizes[:index])
    return packed.narrow(0, offset, sizes[index])


@torch.no_grad()
def verify_rollout_weight_values(model, received: Mapping, names: Sequence[str]) -> dict:
    """Fail on missing, stale, nonfinite or misloaded adapted-layer weights.

    ``received`` must be retained from the same payload passed to load_weights;
    the reference is never obtained from the receiver itself. A receipt is
    returned only after every requested tensor matches exactly in load dtype.
    """
    if not names or len(set(names)) != len(names):
        raise RuntimeError("Rollout verification requires nonempty unique adapter weight names")
    missing = set(names) - received.keys()
    if missing:
        raise RuntimeError(f"Missing merged adapter payload: {sorted(missing)}")
    live = dict(model.named_parameters())
    digest = hashlib.sha256()
    dtypes = set()
    for name in sorted(names):
        actual = _loaded_weight(name, received, live).detach()
        reference = received[name].detach()
        if actual.shape != reference.shape:
            raise RuntimeError(
                f"Rollout weight shape mismatch for {name}: {tuple(actual.shape)} != {tuple(reference.shape)}"
            )
        expected = reference.to(device=actual.device, dtype=actual.dtype)
        if not torch.isfinite(expected).all() or not torch.equal(actual, expected):
            raise RuntimeError(f"Rollout weight verification failed for {name}; refusing to generate")
        dtype = str(actual.dtype)
        dtypes.add(dtype)
        digest.update(json.dumps([name, list(actual.shape), dtype]).encode("utf-8"))
        digest.update(actual.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return {
        "status": "verified",
        "verification_version": "merged-adapter-readback-v1",
        "tensor_count": len(names),
        "load_dtypes": sorted(dtypes),
        "loaded_weights_sha256": digest.hexdigest(),
        "comparison": "exact_in_receiver_dtype",
    }
