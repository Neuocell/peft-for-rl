# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING
from transformers import AutoConfig

from verl.base_config import BaseConfig
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.import_utils import import_external_libs
from verl.utils.model import get_generation_config, update_model_config

__all__ = ["HFModelConfig"]


@dataclass
class HFModelConfig(BaseConfig):
    # note that we separate model_path, model_config_path and tokenizer_path in case they are different
    _mutable_fields = {
        "hf_config_path",
        "tokenizer_path",
        "hf_config",
        "generation_config",
        "tokenizer",
        "processor",
        "local_path",
        "architectures",
        "local_hf_config_path",
        "local_tokenizer_path",
    }

    path: str = MISSING
    local_path: Optional[str] = None
    hf_config_path: Optional[str] = None
    local_hf_config_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    local_tokenizer_path: Optional[str] = None

    # whether to load tokenizer. This is useful when we only want to load model config
    load_tokenizer: bool = True

    hf_config: Any = None
    generation_config: Any = None
    tokenizer: Any = None
    processor: Any = None

    # whether to use shared memory
    use_shm: bool = False
    trust_remote_code: bool = False

    # custom chat template for the model
    custom_chat_template: Optional[str] = None

    external_lib: Optional[str] = None

    override_config: dict = field(default_factory=dict)

    enable_gradient_checkpointing: bool = True
    enable_activation_offload: bool = False

    use_remove_padding: bool = True

    # TODO: unify fsdp and megatron peft config.
    # "lora" preserves the original vLLM LoRA fast path. "oft" trains a PEFT
    # OFT actor and syncs merged full weights to rollout.
    peft_type: str = "lora"
    lora_rank: int = 0
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_freeze_a: bool = False
    lora_rank_pattern_path: Optional[str] = None
    rlpo_svd_device: str = "auto"
    adalora_init_r: int = 0
    adalora_target_r: int = 0
    adalora_tinit: int = 0
    adalora_tfinal: int = 0
    # verl uses snake_case; this maps to PEFT AdaLoraConfig.deltaT.
    adalora_delta_t: int = 1
    adalora_beta1: float = 0.85
    adalora_beta2: float = 0.85
    # PEFT's 0.5 default targets supervised loss scales. PPO policy losses in
    # this repository are much smaller, so use an RL-scale default.
    adalora_orth_reg_weight: float = 1e-3
    adalora_total_step: int = 0
    # Zero-function randomized LoRA probe and its exported RL-gradient basis.
    gradient_probe_width: int = 8
    gradient_probe_capacity: int = 64
    gradient_probe_target_energy: float = 0.95
    gradient_probe_rank_bins: list[int] = field(default_factory=lambda: [8, 12, 16, 20, 24, 28, 32])
    gradient_probe_seed: int = 42
    gradient_probe_min_steps: int = 6
    gradient_probe_max_steps: int = 12
    gradient_probe_stability_patience: int = 3
    gradient_probe_overlap_threshold: float = 0.98
    gradient_probe_rank_tolerance: float = 1.0
    gradient_probe_output_dir: Optional[str] = None
    gradient_probe_method: str = "energy"
    gradient_probe_window_size: int = 3
    gradient_probe_num_windows: int = 5
    gradient_probe_calibration_windows: int = 0
    gradient_probe_validation_windows: int = 0
    gradient_probe_target_mean_rank: float = 16.0
    gradient_probe_snr_ridge: float = 0.05
    gradient_probe_clip_factor: float = 2.5
    gradient_probe_normalize_rank_utility: bool = False
    gradient_probe_balance_rank_by_module_type: bool = False
    gradient_probe_local_atoms: int = 2
    gradient_probe_gap_cap: float = 4.0
    gradient_subspace_rank_map_path: Optional[str] = None
    gradient_subspace_path: Optional[str] = None
    gradient_subspace_scaling: float = 2.0
    # SPAR-LoRA v0 uses positive teacher-forced rollouts to discover and score
    # a fixed candidate atom set before ordinary heterogeneous LoRA training.
    spar_r_max: int = 32
    spar_discovery_samples: int = 32
    spar_calibration_samples: int = 32
    spar_sample_clip_factor: float = 2.5
    spar_score_eps: float = 1e-12
    # Full-weight prompt-group GRPO probe. It observes actual base-weight
    # gradients and exports mean, covariance and hybrid right subspaces.
    full_gradient_probe_mode: str = "legacy"
    full_gradient_probe_rank: int = 32
    full_gradient_probe_sketch_width: int = 40
    full_gradient_probe_svd_oversample: int = 8
    full_gradient_probe_svd_niter: int = 1
    full_gradient_probe_svd_device: str = "auto"
    full_gradient_probe_discovery_prompts: int = 16
    full_gradient_probe_calibration_prompts: int = 16
    full_gradient_probe_audit_prompts: int = 8
    full_gradient_probe_clip_factor: float = 2.5
    full_gradient_probe_confidence_z: float = 1.0
    full_gradient_probe_eps: float = 1e-12
    full_gradient_probe_min_advantage_rms: float = 1e-6
    full_gradient_probe_window_prompts: int = 16
    full_gradient_probe_discovery_windows: int = 8
    full_gradient_probe_calibration_windows: int = 2
    full_gradient_probe_audit_windows: int = 2
    full_gradient_probe_local_atoms: int = 4
    full_gradient_probe_adam_beta1: float = 0.9
    full_gradient_probe_adam_beta2: float = 0.999
    full_gradient_probe_adam_eps: float = 1e-8
    full_gradient_probe_future_lcb_z: float = 1.0
    full_gradient_probe_support_floor: float = 1e-4
    oft_rank: int = 0
    oft_block_size: int = 32
    oft_dropout: float = 0.0
    oft_coft: bool = False
    oft_eps: float = 6e-5
    oft_block_share: bool = False
    oft_use_cayley_neumann: bool = True
    oft_num_cayley_neumann_terms: int = 5
    skew_rank: int = 0
    skew_alpha: int = 16
    skew_dropout: float = 0.0
    skew_init_std: float = 0.01
    boet_rank: int = 0
    boet_alpha: int = 16
    boet_dropout: float = 0.0
    boet_init_std: float = 0.01
    boet_use_cayley_neumann: bool = False
    boet_num_cayley_neumann_terms: int = 5
    boet_cayley_neumann_eps: float = 0.9
    biso_block_size: int = 32
    biso_alpha: int = 16
    biso_init_std: float = 0.0
    biso_parameterization: str = "primitive_cn"
    biso_use_cayley_neumann: bool = True
    biso_num_cayley_neumann_terms: int = 5
    biso_cayley_neumann_eps: float = 0.9
    biso_selective_mode: str = "none"
    biso_selective_keep_ratio: float = 0.3
    biso_selective_topk: int = 16
    biso_selective_mask_quantile: float = 0.7
    biso_selective_seed: int = 42
    spo_block_size: int = 16
    spo_depth: int = 1
    spo_alpha: int = 8
    spo_init_std: float = 0.0
    spo_use_cayley_neumann: bool = True
    spo_num_cayley_neumann_terms: int = 5
    spo_cayley_neumann_eps: float = 0.9
    spo_seed: int = 42
    tinylora_rank: int = 2
    tinylora_projection_dim: int = 1
    tinylora_tie_factor: int = 16
    tinylora_tie_strategy: str = "tiled"
    tinylora_seed: int = 42
    tinylora_svd_device: str = "auto"
    tinylora_svd_method: str = "lowrank"
    tinylora_svd_oversample: int = 4
    tinylora_svd_niter: int = 2
    tinylora_projection_std: float = 1.0
    geora_sparsity_ratio: float = 0.2
    geora_oversample: int = 8
    geora_niter: int = 2
    geora_svd_device: str = "auto"
    geora_residual_anchor: bool = True
    geora_init_scale: float = 1.0
    geora_seed: int = 42
    target_modules: Optional[str] = "all-linear"

    exclude_modules: Optional[str] = None

    # megatron lora config
    lora: dict[str, Any] = field(default_factory=dict)

    # path to pre-trained LoRA adapter to load for continued training
    lora_adapter_path: Optional[str] = None
    use_liger: bool = False

    use_fused_kernels: bool = False
    fused_kernel_options: dict = field(default_factory=dict)

    # TiledMLP configuration for memory-efficient MLP computation
    tiled_mlp: dict = field(default_factory=lambda: {"enabled": False, "num_shards": 4})

    architectures: Optional[list[str]] = None

    def __post_init__(self):
        import_external_libs(self.external_lib)

        if self.hf_config_path is None:
            self.hf_config_path = self.path
        if self.tokenizer_path is None:
            self.tokenizer_path = self.path

        self.local_path = copy_to_local(self.path, use_shm=self.use_shm)

        # construct tokenizer
        if self.load_tokenizer:
            self.local_tokenizer_path = copy_to_local(self.tokenizer_path, use_shm=self.use_shm)
            self.tokenizer = hf_tokenizer(self.local_tokenizer_path, trust_remote_code=self.trust_remote_code)
            self.processor = hf_processor(self.local_tokenizer_path, trust_remote_code=self.trust_remote_code)

        if self.custom_chat_template is not None:
            if self.processor is not None:
                self.processor.chat_template = self.custom_chat_template
            else:
                self.tokenizer.chat_template = self.custom_chat_template

        self.local_hf_config_path = copy_to_local(self.hf_config_path, use_shm=self.use_shm)
        self.generation_config = get_generation_config(
            self.local_hf_config_path, trust_remote_code=self.trust_remote_code
        )

        # construct hf_config
        attn_implementation = self.override_config.get("attn_implementation", "flash_attention_2")
        self.hf_config = AutoConfig.from_pretrained(
            self.local_hf_config_path, trust_remote_code=self.trust_remote_code, attn_implementation=attn_implementation
        )

        override_config_kwargs = {}

        if self.tokenizer is not None:
            override_config_kwargs.update(
                {
                    "bos_token_id": self.tokenizer.bos_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                }
            )

        # TODO: (vermouth1992). self.config.model in megatron differs from that of fsdp in the override_config.
        override_config = (
            self.override_config["model_config"] if "model_config" in self.override_config else self.override_config
        )
        override_config_kwargs.update(override_config)
        update_model_config(self.hf_config, override_config_kwargs=override_config_kwargs)

        self.share_embeddings_and_output_weights = getattr(self.hf_config, "tie_word_embeddings", False)

        # get model architectures
        self.architectures = getattr(self.hf_config, "architectures", None)
        assert self.architectures is not None and len(self.architectures) == 1, (
            "Expect only one architecture, got {}".format(self.architectures)
        )

        # per model patch
        if getattr(self.hf_config, "model_type", None) == "kimi_vl":
            self.hf_config.text_config.topk_method = "greedy"

    def get_processor(self):
        return self.processor if self.processor is not None else self.tokenizer
