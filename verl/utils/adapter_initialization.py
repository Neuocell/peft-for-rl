"""Source-aligned adapter initialization, independent of the training RNG.

References are pinned in docs/adapter_initialization.md. FP32 is the protocol's
adapter master precision; conversion to the base layer dtype happens afterward.
"""

from math import sqrt

import torch
from torch import nn

REFERENCE_INIT_VERSION = "peft_nora_reference_v2"
LEGACY_INIT_VERSION = "legacy_v1"
INIT_VERSIONS = (REFERENCE_INIT_VERSION, LEGACY_INIT_VERSION)


def nora_normalize_columns(weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """NoRA official code: normalize over rank with an additive epsilon."""
    weight = weight.float()
    return weight / (torch.linalg.vector_norm(weight, dim=0, keepdim=True) + eps)


def seeded_peft_lora_a(
    a_shape: torch.Size, b_shape: torch.Size, generator: torch.Generator
) -> torch.Tensor:
    """Match default PEFT nn.Linear A/B construction followed by A reset.

    A single CPU generator is shared in module traversal order. The two discarded
    constructor draws are needed for elementwise pairing with real PEFT LoRA.
    """
    initial_a = torch.empty(a_shape, device="cpu", dtype=torch.float32)
    constructor_b = torch.empty(b_shape, device="cpu", dtype=torch.float32)
    nn.init.kaiming_uniform_(initial_a, a=sqrt(5), generator=generator)
    nn.init.kaiming_uniform_(constructor_b, a=sqrt(5), generator=generator)
    nn.init.kaiming_uniform_(initial_a, a=sqrt(5), generator=generator)
    return initial_a
