# File Manifest

## Entry

- `scripts/local/run_dapo_boxed_paper_clean_1p5b_4gpu.sh`

## Training core

- `verl/workers/engine/fsdp/transformer_impl.py`
- `verl/workers/fsdp_workers.py`
- `verl/workers/config/model.py`
- `verl/workers/config/actor.py`
- `verl/workers/actor/dp_actor.py`

## PEFT implementations

- `verl/utils/peft_geora.py`
- `verl/utils/peft_biso.py`
- `verl/utils/peft_boet.py`
- `verl/utils/peft_skew.py`
- `verl/utils/peft_spo.py`
- `verl/utils/peft_oft_compat.py`

## Infrastructure

- `verl/utils/fsdp_utils.py`
- `verl/utils/checkpoint/fsdp_checkpoint_manager.py`
- `verl/model_merger/fsdp_model_merger.py`
