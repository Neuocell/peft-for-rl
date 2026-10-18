"""Deterministic REAL observation with checkpointing for supported decoders."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


@contextmanager
def real_probe_mode(model: torch.nn.Module) -> Iterator[None]:
    """Disable dropout while retaining modern Qwen2 activation checkpointing.

    Transformers' Qwen2 decoder uses ``training`` on the decoder itself only
    to select its GradientCheckpointingLayer call path. Its attention and
    other children can remain in eval mode, yielding the same deterministic
    function with checkpoint recomputation. Set the decoder flag directly:
    calling ``train()`` here would re-enable dropout in all its children.

    Other decoder implementations retain ordinary eval behavior. In particular,
    older Transformers versions without GradientCheckpointingLayer do not gain
    checkpointing from this context. Do not opt arbitrary layer classes into
    this behavior without verifying their own uses of ``self.training``.
    """

    try:
        from transformers.modeling_layers import GradientCheckpointingLayer
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
    except ImportError:
        checkpoint_layer_types = ()
    else:
        checkpoint_layer_types = (
            (Qwen2DecoderLayer,) if issubclass(Qwen2DecoderLayer, GradientCheckpointingLayer) else ()
        )

    original_modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        for module, _ in original_modes:
            if type(module) in checkpoint_layer_types and getattr(module, "gradient_checkpointing", False):
                module.training = True
        yield
    finally:
        # A model may start with mixed modes; restoring only the root recursively
        # would erase those original flags and can also re-enable child dropout.
        for module, training in original_modes:
            module.training = training
