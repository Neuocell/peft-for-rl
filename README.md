# PEFT for RL

This repository is a self-contained RLVR project built on top of `verl`.
It includes the training framework code, PEFT method implementations, launch
scripts, evaluation helpers, and reproducibility files needed to run the
experiments on a fresh machine.
Supported PEFT variants currently include LoRA, RLPO-init, RL gradient-subspace
initialization, AdaLoRA, OFT, GeoRA, BISO, BOET, Skew, and SPO.

## Layout

```text
peft-for-rl/
  verl/                 # integrated verl training framework
  examples/verl_train/  # stable LoRA/AdaLoRA/OFT training entry points
  tina_run/             # evaluation and data-prep helpers
  docs/                 # environment and experiment notes
  scripts/              # local launch helpers
  environment.yml       # conda environment definition
  requirements*.txt     # pip dependency layers
```

## Quick start

```bash
git clone git@github.com:<your-account>/peft-for-rl.git
cd peft-for-rl
conda env create -f environment.yml
conda activate peft-for-rl
pip install -e .
```

If you need the full CUDA / evaluation stack, install the matching extras:

```bash
pip install -r requirements-cuda.txt
pip install -r requirements-eval.txt
```

## Training

On this machine, the isolated runtime is rooted at `runs/`: model files,
datasets, checkpoints, Ray state, and logs no longer use
`/home/wangls/Tina_orthres_run`. The cloned training environment is named
`peft-for-rl`. Start the official AdaLoRA run from this checkout with:

```bash
cd /home/wangls/peft-for-rl
bash scripts/local/start_official_adalora_4gpu.sh
```

The default log is
`runs/logs/verl/adalora_r32_to_r8_a64_b64m16n8_270_4gpu_boxed_official.log`.
Override `EXP_NAME`, `CUDA_VISIBLE_DEVICES`, `LOG_FILE`, or any training
variable on the command line when a separate run is needed.
See [the local runtime inventory](docs/local_runtime_zh.md) for the cloned
environment, copied assets, output paths, and isolation checks.

The main launch script runs directly against the integrated `verl` tree:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_PATH=/path/to/DeepSeek-R1-Distill-Qwen-1.5B/base \
TRAIN_FILE=/path/to/dapo-math-17k-boxed.parquet \
RUNTIME_ROOT=/path/to/runtime \
TOTAL_TRAINING_STEPS=100 \
SAVE_FREQ=20 \
RESUME_MODE=disable \
bash examples/verl_train/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh
```

The official PEFT AdaLoRA integration (`peft==0.19.1`) has a dedicated
270-step launcher. It maps the 64/16 PPO mini-batch split to 1080 AdaLoRA
optimizer steps and uses PEFT's own orthogonal loss and RankAllocator:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_PATH=/path/to/DeepSeek-R1-Distill-Qwen-1.5B/base \
TRAIN_FILE=/path/to/dapo-math-17k-boxed.parquet \
RUNTIME_ROOT=/path/to/runtime \
bash examples/verl_train/run_dapo_math_boxed_official_adalora_1p5b_4gpu_8k.sh
```

See [the Chinese verl/AdaLoRA code walkthrough](docs/verl_adalora_code_walkthrough_zh.md)
for the end-to-end training path, official algorithm mapping, FSDP constraints,
rollout synchronization, metrics, and known resume limitation.

RLPO-init follows the public paper repository: it initializes ordinary LoRA
`A` with the top-r right singular vectors of each frozen base weight and sets
`B=0`. It adds no orthogonal loss, singular-value parameter, rank allocator,
or dynamic pruning. The isolated 4-GPU configuration is available at:

```bash
cd /home/wangls/peft-for-rl
bash scripts/local/start_rlpo_init_4gpu.sh
```

See [the Chinese RLPO-init integration notes](docs/verl_rlpo_init_zh.md) for
the source mapping, initialization formula, and verl execution path.

The RL gradient-subspace experiment first runs a zero-function rank-8 LoRA
probe on 6-12 small GRPO batches, incrementally estimates each module's
policy-gradient input subspace, and exports an independent energy-based rank
and orthonormal basis. A separate 270-step launcher then initializes ordinary
heterogeneous LoRA from that artifact:

```bash
bash scripts/local/start_gradient_probe_4gpu.sh
bash scripts/local/start_gradient_subspace_init_4gpu.sh
```

See [the Chinese design and implementation notes](docs/verl_rl_gradient_subspace_init_zh.md)
for the gradient identity, stability criterion, FSDP boundary, artifact format,
overhead, and first-run checks.

## Evaluation

Evaluation helpers live under `tina_run/` and expect the same repository
checkout plus a separate evaluation Python environment when needed.

## Reproducibility

The repository includes:

- `environment.yml`
- `requirements.txt`
- `requirements-cuda.txt`
- `requirements-eval.txt`
- `requirements_sglang.txt`

Those files are intended to capture the dependency stack needed to recreate the
training and evaluation setup on another machine.
