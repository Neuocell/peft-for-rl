from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from peft import AdaLoraConfig, TaskType, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

from verl.utils.fsdp_utils import collect_lora_params
from verl.utils.peft_adalora import (
    adalora_config_for_vllm,
    build_adalora_config,
    find_adalora_model,
    update_and_allocate,
)
from verl.workers.actor.dp_actor import DataParallelPPOActor, _aggregate_adalora_loss_metrics

REPO_ROOT = Path(__file__).resolve().parents[1]


def tiny_adalora_model(*, orth_reg_weight: float = 0.5):
    base = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    return get_peft_model(
        base,
        AdaLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj"],
            init_r=4,
            target_r=2,
            lora_alpha=8,
            tinit=1,
            tfinal=2,
            deltaT=2,
            total_step=10,
            orth_reg_weight=orth_reg_weight,
        ),
    )


def test_build_config_uses_official_fields_and_optimizer_step_schedule():
    config = SimpleNamespace(
        adalora_init_r=32,
        adalora_target_r=8,
        adalora_tinit=100,
        adalora_tfinal=200,
        adalora_delta_t=20,
        adalora_beta1=0.85,
        adalora_beta2=0.85,
        adalora_orth_reg_weight=0.5,
        adalora_total_step=1080,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules="all-linear",
        exclude_modules=None,
    )

    peft_config = build_adalora_config(config)

    assert isinstance(peft_config, AdaLoraConfig)
    assert peft_config.init_r == 32
    assert peft_config.target_r == 8
    assert peft_config.deltaT == 20
    assert peft_config.total_step == 1080
    # AdaLoRA does not use the inherited LoRA r field.
    assert peft_config.r == 8


def test_build_config_uses_rl_scale_orthogonal_default():
    config = SimpleNamespace(
        adalora_init_r=4,
        adalora_target_r=2,
        adalora_total_step=10,
    )

    peft_config = build_adalora_config(config)

    assert peft_config.orth_reg_weight == pytest.approx(1e-3)


def test_adalora_loss_ratio_is_aggregated_before_division():
    metrics = _aggregate_adalora_loss_metrics(
        weighted_regularization_sum=5.0,
        weighted_orthogonal_loss_sum=0.005,
        # One micro-batch may have zero PG loss; this is the weighted sum from
        # the complete optimizer update, including another non-zero batch.
        weighted_pg_abs_sum=0.05,
        loss_weight_sum=1.0,
    )

    assert metrics["adalora/orthogonal_regularization"] == pytest.approx(5.0)
    assert metrics["adalora/orthogonal_loss"] == pytest.approx(0.005)
    assert metrics["adalora/pg_loss_abs"] == pytest.approx(0.05)
    assert metrics["adalora/orthogonal_to_pg_abs_ratio"] == pytest.approx(0.1)


def test_official_forward_returns_orthogonal_loss_for_ignored_labels():
    model = tiny_adalora_model()
    input_ids = torch.randint(0, 32, (2, 6))

    output = model(
        input_ids=input_ids,
        labels=torch.full_like(input_ids, -100),
        num_items_in_batch=1,
    )
    output.loss.backward()

    assert torch.isfinite(output.loss)
    assert output.loss.item() > 0
    assert any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    )


def test_model_discovery_returns_real_adalora_model_not_peft_proxy():
    model = tiny_adalora_model()

    assert find_adalora_model(model) is model.base_model


def test_ppo_micro_batch_captures_official_orthogonal_loss(monkeypatch):
    model = tiny_adalora_model()
    monkeypatch.setattr(
        "verl.workers.actor.dp_actor.logprobs_from_logits",
        lambda logits, labels, **_: torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1),
    )
    actor = object.__new__(DataParallelPPOActor)
    actor.actor_module = model
    actor.use_remove_padding = False
    actor.use_fused_kernels = False
    actor.use_ulysses_sp = False
    actor.device_name = "cpu"
    actor.param_dtype = torch.bfloat16
    actor._last_adalora_orth_loss = None
    actor.config = SimpleNamespace(entropy_checkpointing=False)

    input_ids = torch.randint(0, 32, (2, 6))
    micro_batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "position_ids": torch.arange(input_ids.shape[1]).repeat(input_ids.shape[0], 1),
        "responses": input_ids[:, -2:],
    }
    _, log_probs = actor._forward_micro_batch(
        micro_batch,
        temperature=1.0,
        capture_adalora_orthogonal_loss=True,
    )

    assert log_probs.shape == (2, 2)
    assert actor._last_adalora_orth_loss is not None
    assert torch.isfinite(actor._last_adalora_orth_loss)
    actor._last_adalora_orth_loss.backward()
    assert any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    )


def test_allocator_advances_on_optimizer_steps_and_reports_mask_rank():
    model = tiny_adalora_model()
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-3)
    input_ids = torch.randint(0, 32, (2, 6))

    model(input_ids=input_ids, labels=input_ids).loss.backward()
    optimizer.step()
    first_metrics = update_and_allocate(model, optimizer_step=1)
    optimizer.zero_grad()

    model(input_ids=input_ids, labels=input_ids).loss.backward()
    optimizer.step()
    second_metrics = update_and_allocate(model, optimizer_step=2)

    assert "adalora/mask_applied" not in first_metrics
    assert second_metrics["adalora/mask_applied"] == 1
    assert first_metrics["adalora/optimizer_step"] == 1
    assert second_metrics["adalora/optimizer_step"] == 2
    assert second_metrics["adalora/masked_at_optimizer_step"] == 2
    assert second_metrics["adalora/masked_rank_mean"] == pytest.approx(second_metrics["adalora/scheduled_rank_mean"])
    assert "adalora/current_nonzero_rank_mean" in second_metrics
    assert "adalora/uncertainty_ema_mean" in second_metrics


def test_vllm_export_folds_e_into_a_and_restores_actor_parameters():
    model = tiny_adalora_model()
    svd_layer = next(module for module in model.modules() if hasattr(module, "lora_E"))
    with torch.no_grad():
        svd_layer.lora_A["default"].fill_(1.0)
        svd_layer.lora_E["default"].fill_(0.5)
        svd_layer.ranknum["default"].fill_(4.0)
    original_a = svd_layer.lora_A["default"].detach().clone()
    rollout_config = adalora_config_for_vllm(model.peft_config["default"])

    exported = collect_lora_params(model, layered_summon=False, base_sync_done=True, adalora=True)

    assert model.peft_config["default"].r == 8
    assert rollout_config.r == 4
    assert exported
    assert all("lora_E" not in name and "ranknum" not in name for name in exported)
    exported_a = next(value for name, value in exported.items() if name.endswith(".lora_A.weight"))
    exported_b = next(value for name, value in exported.items() if name.endswith(".lora_B.weight"))
    expected_scale = 0.5 * 4.0 / (4.0 + 1e-5)
    assert torch.allclose(exported_a, original_a * expected_scale)
    actor_delta = (
        (svd_layer.lora_B["default"] @ (original_a * svd_layer.lora_E["default"]))
        * svd_layer.scaling["default"]
        / (svd_layer.ranknum["default"] + 1e-5)
    )
    rollout_delta = (exported_b @ exported_a) * rollout_config.lora_alpha / rollout_config.r
    assert torch.allclose(rollout_delta, actor_delta)
    assert torch.equal(svd_layer.lora_A["default"], original_a)


def test_official_launcher_maps_270_trainer_steps_to_1080_optimizer_steps():
    launcher = (REPO_ROOT / "examples/verl_train/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh").read_text()
    wrapper = (REPO_ROOT / "examples/verl_train/run_dapo_math_boxed_official_adalora_1p5b_4gpu_8k.sh").read_text()

    assert "total_training_steps * actor_updates_per_step" in launcher
    assert "+actor_rollout_ref.model.adalora_total_step=${adalora_total_step}" in launcher
    assert 'export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-270}"' in wrapper
    assert 'export TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-64}"' in wrapper
    assert 'export TRAIN_PROMPT_MINI_BSZ="${TRAIN_PROMPT_MINI_BSZ:-16}"' in wrapper
    assert 'export ADALORA_TINIT="${ADALORA_TINIT:-100}"' in wrapper
    assert 'export ADALORA_TFINAL="${ADALORA_TFINAL:-200}"' in wrapper
    assert 'export ADALORA_DELTA_T="${ADALORA_DELTA_T:-20}"' in wrapper
    assert 'export ADALORA_ORTH_REG_WEIGHT="${ADALORA_ORTH_REG_WEIGHT:-1e-3}"' in wrapper
    assert "export FSDP_RESHARD_AFTER_FORWARD=False" in wrapper
    assert 'export RAY_TEMP_DIR="${RAY_TEMP_DIR:-${RUNTIME_ROOT}/ray}"' in wrapper
    assert 'export RAY_local_fs_capacity_threshold="${RAY_LOCAL_FS_CAPACITY_THRESHOLD:-0.98}"' in wrapper
    assert 'export ACTOR_PPO_MAX_TOKEN_LEN="${ACTOR_PPO_MAX_TOKEN_LEN:-12288}"' in wrapper
    assert 'export INFER_PPO_MAX_TOKEN_LEN="${INFER_PPO_MAX_TOKEN_LEN:-12288}"' in wrapper
    assert 'export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-256}"' in wrapper
    assert "actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs}" in launcher


def test_launcher_is_pinned_to_this_checkout_and_isolated_runtime():
    launcher = (REPO_ROOT / "examples/verl_train/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh").read_text()
    starter = (REPO_ROOT / "scripts/local/start_official_adalora_4gpu.sh").read_text()

    assert 'RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/runs}"' in launcher
    assert 'cd "${REPO_ROOT}"' in launcher
    assert 'export PYTHONPATH="${REPO_ROOT}"' in launcher
    assert "pathlib.Path(verl.__file__).resolve().parent" in launcher
    assert 'expected_verl_root="${REPO_ROOT}/verl"' in launcher
    assert '[[ ! -s "${TRAIN_FILE}" ]]' in launcher
    assert 'mkdir -p "${CKPTS_DIR}" "${RAY_TEMP_DIR}"' in launcher
    assert 'CONDA_ENV_NAME="${CONDA_ENV_NAME:-peft-for-rl}"' in starter
    assert 'LOG_DIR="${LOG_DIR:-${RUNTIME_ROOT}/logs/verl}"' in starter
    assert 'conda run --no-capture-output -n "${CONDA_ENV_NAME}"' in starter


def test_fsdp_worker_passes_model_peft_config_to_ppo_actor():
    worker_source = (REPO_ROOT / "verl/workers/fsdp_workers.py").read_text()

    assert "actor_cfg.model_config = omega_conf_to_dataclass(self.config.model)" in worker_source
