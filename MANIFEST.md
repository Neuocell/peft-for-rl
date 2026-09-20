# File Manifest

## Entry

- `scripts/local/run_dapo_boxed_paper_clean_1p5b_4gpu.sh`
- `scripts/local/start_chr_rlpo_v0_4gpu.sh`

## PEFT patches for verl

- `verl/workers/engine/fsdp/transformer_impl.py`
- `verl/workers/fsdp_workers.py`
- `verl/workers/config/model.py`
- `verl/workers/config/actor.py`
- `verl/workers/actor/dp_actor.py`

## PEFT implementations

- `examples/verl_train/run_dapo_math_boxed_chr_rlpo_v0_1p5b_4gpu_8k.sh`
- `verl/utils/peft_adalora.py`
- `verl/utils/peft_rlpo.py`
- `verl/utils/peft_geora.py`
- `verl/utils/peft_biso.py`
- `verl/utils/peft_boet.py`
- `verl/utils/peft_skew.py`
- `verl/utils/peft_spo.py`
- `verl/utils/peft_oft_compat.py`

## AdaLoRA validation and documentation

- `examples/verl_train/run_dapo_math_boxed_official_adalora_1p5b_4gpu_8k.sh`
- `tests/test_verl_adalora.py`
- `docs/verl_adalora_code_walkthrough_zh.md`

## Infrastructure

- `verl/trainer/config/data/legacy_data.yaml`
- `verl/utils/fsdp_utils.py`
- `verl/utils/checkpoint/fsdp_checkpoint_manager.py`
- `verl/model_merger/fsdp_model_merger.py`

## Analysis

- `scripts/analysis/analyze_lora_delta_rank.py`
- `scripts/analysis/diagnose_lora_subspace_trajectory.py`
- `scripts/analysis/diagnose_activation_weighted_rank.py`
- `scripts/analysis/diagnose_rlpo_init_subspace.py`
- `scripts/local/run_d1_activation_weighted_rank.sh`
- `tests/test_activation_weighted_rank.py`
- `tests/test_rlpo_init_subspace.py`
- `docs/rl_adaptive_rank_minimal_experiment_zh.md`
