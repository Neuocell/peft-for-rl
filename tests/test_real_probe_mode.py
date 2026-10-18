from __future__ import annotations

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from verl.utils.real_probe_mode import real_probe_mode


def tiny_qwen2():
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=31,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
            attention_dropout=0.4,
            use_cache=False,
            _attn_implementation="eager",
        )
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def test_qwen2_probe_matches_eval_with_dropout_disabled_and_checkpoint_recomputation():
    torch.manual_seed(42)
    model = tiny_qwen2()
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    model.eval()
    reference = model(inputs, use_cache=False).logits
    reference.square().mean().backward()
    reference_gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
    model.zero_grad(set_to_none=True)

    model.train()
    # Check exact restoration even when the starting modes are not uniform.
    model.model.layers[0].self_attn.eval()
    original_modes = [module.training for module in model.modules()]
    calls = [0, 0]

    def count_forward(index):
        def hook(module, args):
            calls[index] += 1
        return hook

    handles = [layer.register_forward_pre_hook(count_forward(index)) for index, layer in enumerate(model.model.layers)]
    try:
        with real_probe_mode(model):
            assert not model.training
            for layer in model.model.layers:
                assert layer.training
                assert not layer.self_attn.training
                assert layer.self_attn.attention_dropout > 0
            observed = model(inputs, use_cache=False).logits
            torch.testing.assert_close(observed, reference, rtol=0, atol=0)
            observed.square().mean().backward()
        assert calls == [2, 2], "Both decoder layers must recompute during backward"
    finally:
        for handle in handles:
            handle.remove()

    assert [module.training for module in model.modules()] == original_modes
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, reference_gradients[name], rtol=1e-5, atol=1e-8)


def test_probe_mode_restores_mixed_modes_after_exception():
    model = tiny_qwen2()
    model.eval()
    model.model.layers[0].self_attn.train()
    original_modes = [module.training for module in model.modules()]
    with pytest.raises(RuntimeError, match="failed observation"):
        with real_probe_mode(model):
            assert model.model.layers[0].training
            assert not model.model.layers[0].self_attn.training
            raise RuntimeError("failed observation")
    assert [module.training for module in model.modules()] == original_modes


def test_probe_mode_does_not_enable_checkpointing_that_was_disabled():
    model = tiny_qwen2()
    model.gradient_checkpointing_disable()
    with real_probe_mode(model):
        assert all(not module.training for module in model.modules())


def test_unsupported_module_remains_in_eval_mode():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.5))
    model.gradient_checkpointing = True
    with real_probe_mode(model):
        assert all(not module.training for module in model.modules())
    assert all(module.training for module in model.modules())
