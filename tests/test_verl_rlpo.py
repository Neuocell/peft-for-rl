import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from peft import AdaLoraConfig, LoraConfig, get_peft_model

from verl.utils.peft_rlpo import apply_rlpo_initialization

REPO_ROOT = Path(__file__).resolve().parents[1]


class TinyModel(torch.nn.Module):
    def __init__(self, out_features: int = 5, in_features: int = 6):
        super().__init__()
        self.proj = torch.nn.Linear(in_features, out_features, bias=False)

    def forward(self, inputs):
        return self.proj(inputs)


def tiny_lora_model(rank: int = 2):
    base = TinyModel()
    with torch.no_grad():
        base.proj.weight.copy_(
            torch.tensor(
                [
                    [9.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 7.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                ]
            )
        )
    return get_peft_model(
        base,
        LoraConfig(r=rank, lora_alpha=4, lora_dropout=0.0, target_modules=["proj"], bias="none"),
    )


def get_lora_layer(model):
    return next(module for module in model.modules() if hasattr(module, "lora_A") and "default" in module.lora_A)


def test_rlpo_matches_paper_source_initialization_and_preserves_base_weight():
    model = tiny_lora_model(rank=2)
    layer = get_lora_layer(model)
    base_before = layer.base_layer.weight.detach().clone()

    stats = apply_rlpo_initialization(
        model,
        SimpleNamespace(lora_rank=2, rlpo_svd_device="cpu"),
    )

    a = layer.lora_A["default"].weight.detach().float()
    b = layer.lora_B["default"].weight.detach().float()
    _, _, vh = torch.linalg.svd(base_before.float(), full_matrices=False)

    assert torch.count_nonzero(b).item() == 0
    assert torch.count_nonzero(b @ a).item() == 0
    assert torch.allclose(a @ a.T, torch.eye(2), atol=1e-6, rtol=1e-6)
    # Compare projection matrices because singular-vector signs are arbitrary.
    assert torch.allclose(a.T @ a, vh[:2].T @ vh[:2], atol=1e-6, rtol=1e-6)
    assert torch.equal(layer.base_layer.weight, base_before)

    assert stats.num_layers == 1
    assert stats.num_parameters == base_before.numel()
    assert math.isfinite(stats.total_s)
    assert math.isfinite(stats.mean_s)
    assert math.isfinite(stats.max_s)
    assert stats.orthogonality_error_max <= 1e-6
    assert stats.b_abs_max == 0.0


def test_rlpo_remains_ordinary_trainable_two_factor_lora():
    model = tiny_lora_model(rank=2)
    apply_rlpo_initialization(model, {"lora_rank": 2, "rlpo_svd_device": "cpu"})
    layer = get_lora_layer(model)
    peft_config = model.peft_config["default"]

    assert isinstance(peft_config, LoraConfig)
    assert not isinstance(peft_config, AdaLoraConfig)
    assert not hasattr(layer, "lora_E")
    assert not hasattr(model.base_model, "rankallocator")
    assert layer.lora_A["default"].weight.requires_grad
    assert layer.lora_B["default"].weight.requires_grad


def test_rlpo_initial_adapter_output_is_exactly_zero():
    model = tiny_lora_model(rank=2)
    layer = get_lora_layer(model)
    apply_rlpo_initialization(model, {"lora_rank": 2, "rlpo_svd_device": "cpu"})

    inputs = torch.randn(4, 6)
    adapter_output = layer.lora_B["default"](layer.lora_A["default"](inputs))

    assert torch.count_nonzero(adapter_output).item() == 0


def test_rlpo_rejects_rank_larger_than_base_weight_dimension():
    model = tiny_lora_model(rank=7)

    with pytest.raises(ValueError, match=r"rank=7 exceeds max rank 5"):
        apply_rlpo_initialization(model, {"lora_rank": 7, "rlpo_svd_device": "cpu"})


def test_rlpo_accepts_layer_rank_below_configured_maximum():
    model = get_peft_model(
        TinyModel(),
        LoraConfig(
            r=4,
            lora_alpha=8,
            rank_pattern={"proj": 2},
            alpha_pattern={"proj": 4},
            lora_dropout=0.0,
            target_modules=["proj"],
            bias="none",
        ),
    )
    stats = apply_rlpo_initialization(model, {"lora_rank": 4, "rlpo_svd_device": "cpu"})
    layer = get_lora_layer(model)
    assert layer.lora_A["default"].weight.shape[0] == 2
    assert stats.rank_mean == 2
    assert (stats.rank_min, stats.rank_max) == (2, 2)
    assert layer.scaling["default"] == 2.0


def test_rlpo_launcher_and_fsdp_integration_are_isolated_and_lora_compatible():
    wrapper = (REPO_ROOT / "examples/verl_train/run_dapo_math_boxed_rlpo_init_1p5b_4gpu_8k.sh").read_text()
    starter = (REPO_ROOT / "scripts/local/start_rlpo_init_4gpu.sh").read_text()
    paper_clean_launcher = (REPO_ROOT / "scripts/local/run_dapo_boxed_paper_clean_1p5b_4gpu.sh").read_text()
    worker_source = (REPO_ROOT / "verl/workers/fsdp_workers.py").read_text()

    assert "export PEFT_TYPE=rlpo" in wrapper
    assert 'export LORA_RANK="${LORA_RANK:-32}"' in wrapper
    assert 'export LORA_ALPHA="${LORA_ALPHA:-64}"' in wrapper
    assert 'export LORA_DROPOUT="${LORA_DROPOUT:-0.05}"' in wrapper
    assert 'export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-270}"' in wrapper
    assert 'export SAVE_FREQ="${SAVE_FREQ:-50}"' in wrapper
    assert 'export RAY_TEMP_DIR="${RAY_TEMP_DIR:-${HOME}/rrlpo}"' in wrapper
    assert 'export RAY_local_fs_capacity_threshold="${RAY_LOCAL_FS_CAPACITY_THRESHOLD:-0.98}"' in wrapper
    assert 'export ACTOR_PPO_MAX_TOKEN_LEN="${ACTOR_PPO_MAX_TOKEN_LEN:-12288}"' in wrapper
    assert 'export INFER_PPO_MAX_TOKEN_LEN="${INFER_PPO_MAX_TOKEN_LEN:-12288}"' in wrapper
    assert 'export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-256}"' in wrapper
    assert "ADALORA_" not in wrapper
    assert "ORTH_REG" not in wrapper

    assert 'CONDA_ENV_NAME="${CONDA_ENV_NAME:-peft-for-rl}"' in starter
    assert 'RUNTIME_ROOT="${RUNTIME_ROOT:-${REPO_ROOT}/runs}"' in starter
    assert 'LOG_DIR="${LOG_DIR:-${RUNTIME_ROOT}/logs/verl}"' in starter
    assert 'export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"' in starter
    assert 'rlpo) PEFT_TYPE="rlpo" ;;' in paper_clean_launcher
    assert '"+actor_rollout_ref.model.rlpo_svd_device=${RLPO_SVD_DEVICE}"' in paper_clean_launcher

    for peft_type in ("lora", "rlpo", "adalora", "geora", "grad_probe", "grad_subspace"):
        assert f'"{peft_type}",' in worker_source
    assert 'self._peft_type not in ("geora", "rlpo", "grad_probe", "grad_subspace")' in worker_source
    assert 'elif self._peft_type == "rlpo":' in worker_source
    assert "apply_rlpo_initialization(actor_module, self.config.model)" in worker_source
    assert "if self.rank == 0:" in worker_source
    assert "sync_module_states=True" in worker_source

    base_launcher = (REPO_ROOT / "examples/verl_train/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh").read_text()
    assert 'if [[ "${DRY_RUN:-0}" == "1" ]]' in base_launcher


def test_chr_rlpo_launcher_uses_static_d1_map_and_preserves_checkpoints():
    wrapper = (
        REPO_ROOT / "examples/verl_train/run_dapo_math_boxed_chr_rlpo_v0_1p5b_4gpu_8k.sh"
    ).read_text()
    starter = (REPO_ROOT / "scripts/local/start_chr_rlpo_v0_4gpu.sh").read_text()
    worker_source = (REPO_ROOT / "verl/workers/fsdp_workers.py").read_text()

    assert "export PEFT_TYPE=rlpo" in wrapper
    assert "rank_map_activation_prefix.json" in wrapper
    assert 'export LORA_RANK="${LORA_RANK:-32}"' in wrapper
    assert 'export LORA_ALPHA="${LORA_ALPHA:-64}"' in wrapper
    assert 'export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-270}"' in wrapper
    assert 'export SAVE_FREQ="${SAVE_FREQ:-50}"' in wrapper
    assert 'export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-10}"' in wrapper
    assert "ADALORA_" not in wrapper
    assert "ORTH_REG" not in wrapper
    assert "start_chr_rlpo_v0_4gpu" not in wrapper
    assert "chr_rlpo_v0_d1act95" in starter
    assert "Applied static heterogeneous RLPO rank pattern" in worker_source
    assert 'rank_pattern_path = self.config.model.get("lora_rank_pattern_path")' in worker_source
