# PEFT for RL

This repository collects the local PEFT-for-RL training modifications used in
the DAPO-style RLVR experiments.

The code was extracted from the active CHERRL/verl workspace so the method
implementations, training integration points, and experiment entry script can be
reviewed without mixing them into the larger CHERRL tree.

## Layout

- `verl/utils/peft_*.py`: PEFT method implementations.
- `verl/workers/engine/fsdp/transformer_impl.py`: FSDP engine PEFT injection path.
- `verl/workers/fsdp_workers.py`: actor/rollout/ref worker PEFT injection, FSDP wrapping, optimizer setup.
- `verl/workers/config/*.py`: model and actor config fields for custom PEFT methods.
- `verl/workers/actor/dp_actor.py`: actor update path, including custom BISO regularization hooks.
- `verl/utils/fsdp_utils.py`: FSDP PEFT state handling helpers.
- `verl/utils/checkpoint/fsdp_checkpoint_manager.py`: checkpoint save/load support.
- `verl/model_merger/fsdp_model_merger.py`: model merge support.
- `scripts/local/run_dapo_boxed_paper_clean_1p5b_4gpu.sh`: paper-clean local training entry.

## Upstream Workspace

The live training workspace remains:

- `/home/wangls/CHERRL`

The extracted files in this repository are a standalone copy. Editing this
repository does not affect running CHERRL experiments unless changes are copied
back deliberately.

## Environment Used by the Original Experiments

- Conda environment: `/home/wangls/miniconda3/envs/cherrl`
- Imported `verl`: `/home/wangls/CHERRL/verl`
- Typical launch module: `python3 -m verl.trainer.main_ppo`

## Sync From CHERRL

To refresh this repository from the current local CHERRL workspace:

```bash
bash scripts/sync_from_cherrl.sh
```

Review the resulting diff before committing.
