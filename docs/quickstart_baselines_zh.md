# 快速启动：PeRL/Tina 对齐的 verl 基线

本文档记录当前推荐迁移到新机器继续跑的最小稳定基线。项目定位是复现和对齐 PeRL/Tina 风格的数学 RL + PEFT 实验，训练框架使用本仓库内独立放置的 `verl`。

## 目录结构

```text
peft-for-rl/
  verl/                         # 本仓库内的 verl 训练框架
  examples/verl_train/          # stable LoRA/OFT 训练脚本
  tina_run/scripts/data/        # DAPO-Math boxed 数据准备
  tina_run/scripts/local/eval/  # vLLM fullbench 评测与 OFT merge
  tina_run/tina/analysis/       # Tina parser 最小依赖
  docs/                         # 环境、配置、结果
```

关键入口：

```text
examples/verl_train/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh
examples/verl_train/run_dapo_math_boxed_stable_oft_1p5b_4gpu_8k.sh
verl/utils/reward_score/boxed_math_accuracy.py
tina_run/scripts/data/prepare_dapo_math_boxed_verl.py
tina_run/scripts/local/eval/run_full_bench_vllm_4gpu.sh
tina_run/scripts/local/eval/eval_full_bench_vllm.py
tina_run/scripts/local/eval/merge_oft_wrapped_hf.py
tina_run/scripts/local/run_oft_stable_v1_train_eval_300.sh
```

## 环境

训练环境见 [environment_train.md](environment_train.md)，评测环境见 [environment_eval.md](environment_eval.md)。两者可以是同一个 conda 环境，但文档上分开，是因为训练侧主要要求 `verl + FSDP + Ray + PEFT`，评测侧主要要求 `vLLM + benchmark datasets + Tina parser`。

最小安装：

```bash
cd /path/to/peft-for-rl
pip install -e .
```

如果复用本机已跑通环境，可以继续用原 conda env 名称；但仓库文档统一写成 `peft-for-rl`。环境名不是项目身份。

## 运行资产

仓库不包含模型、数据、checkpoint、生成输出和日志。建议在一个独立运行根目录下放置：

```text
RUNTIME_ROOT=/path/to/runtime
MODEL_PATH=/path/to/DeepSeek-R1-Distill-Qwen-1.5B/base
TRAIN_FILE=/path/to/dapo-math-17k-boxed.parquet
CKPTS_DIR=/path/to/ckpts/verl/DAPO-Math-17k/<exp_name>
OUTPUT_DIR=/path/to/outputs/full_bench_eval/<eval_name>
```

## 稳定训练配置

```text
Framework: verl
Base model: DeepSeek-R1-Distill-Qwen-1.5B
Training data: DAPO-Math-17k boxed
Reward: boxed outcome accuracy only
No cosine shaping
No format reward
ENABLE_OVERLONG_BUFFER=False
max_prompt_length=1024
max_response_length=8192
train_prompt_bsz=16
n_resp_per_prompt=8
rollouts/update=128
ppo_mini_batch_size=16
LR=1e-6
warmup=10
weight_decay=0.1
clip_ratio_low=0.2
clip_ratio_high=0.28
KL off
loss_agg_mode=token-mean
save_freq=20
max_actor_ckpt_to_keep=2
```

正式评测配置：

```text
benchmarks=aime24,aime25,amc23,hmmt_feb,math500,minerva
temperature=0.6
top_p=0.95
max_new_tokens=32768
max_model_len=34816
samples_small=32
samples_large=4
small benchmarks: Avg@32 / Pass@32
large benchmarks: Avg@4 / Pass@4
```

注意：训练生成长度固定为 8K，评测生成长度固定为 32K。这是为了避免训练阶段 overlong hacking，同时保留推理时足够长的数学解题 budget。

## 启动 LoRA

```bash
cd /path/to/peft-for-rl

CUDA_VISIBLE_DEVICES=0,1,2,3 \
RUNTIME_ROOT=/path/to/runtime \
MODEL_PATH=/path/to/DeepSeek-R1-Distill-Qwen-1.5B/base \
TRAIN_FILE=/path/to/dapo-math-17k-boxed.parquet \
EXP_NAME=dapo_math_boxed_stable_lora_1p5b_4gpu_8k_v1 \
CKPTS_DIR=/path/to/ckpts/verl/DAPO-Math-17k/dapo_math_boxed_stable_lora_1p5b_4gpu_8k_v1 \
RAY_TEMP_DIR=/path/to/ray_stable_lora \
TOTAL_TRAINING_STEPS=100 \
SAVE_FREQ=20 \
MAX_ACTOR_CKPT_TO_KEEP=2 \
RESUME_MODE=disable \
bash examples/verl_train/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh
```

续训：

```bash
TOTAL_TRAINING_STEPS=200 RESUME_MODE=auto bash examples/verl_train/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh
TOTAL_TRAINING_STEPS=300 RESUME_MODE=auto bash examples/verl_train/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh
```

## 启动 OFT

```bash
cd /path/to/peft-for-rl

CUDA_VISIBLE_DEVICES=0,1,2,3 \
RUNTIME_ROOT=/path/to/runtime \
MODEL_PATH=/path/to/DeepSeek-R1-Distill-Qwen-1.5B/base \
TRAIN_FILE=/path/to/dapo-math-17k-boxed.parquet \
EXP_NAME=dapo_math_boxed_stable_oft_1p5b_4gpu_8k_v1 \
CKPTS_DIR=/path/to/ckpts/verl/DAPO-Math-17k/dapo_math_boxed_stable_oft_1p5b_4gpu_8k_v1 \
RAY_TEMP_DIR=/path/to/ray_stable_oft \
TOTAL_TRAINING_STEPS=100 \
SAVE_FREQ=20 \
MAX_ACTOR_CKPT_TO_KEEP=2 \
RESUME_MODE=disable \
bash examples/verl_train/run_dapo_math_boxed_stable_oft_1p5b_4gpu_8k.sh
```

OFT 与 LoRA 的差别：

```text
PEFT_TYPE=oft
LORA_RANK=0
OFT_BLOCK_SIZE=32
OFT_RANK=0
OFT_DROPOUT=0.0
```

OFT 评测前需要 merge：

1. verl FSDP actor checkpoint -> OFT-wrapped HF checkpoint
2. OFT-wrapped HF checkpoint -> plain HF model for vLLM

推荐使用 controller：

```bash
cd /path/to/peft-for-rl/tina_run

CUDA_DEVICES=0,1,2,3 \
REPO_ROOT=/path/to/peft-for-rl \
RUN_ROOT=/path/to/runtime \
CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh \
CONDA_ENV=peft-for-rl \
BASE_MODEL=/path/to/DeepSeek-R1-Distill-Qwen-1.5B/base \
CKPTS_DIR=/path/to/ckpts/verl/DAPO-Math-17k/dapo_math_boxed_stable_oft_1p5b_4gpu_8k_v1 \
RAY_TEMP_DIR=/path/to/ray_stable_oft \
TARGETS="100 200 300" \
bash scripts/local/run_oft_stable_v1_train_eval_300.sh
```

controller 会等待指定 GPU 空闲，评测失败会重试，评测成功后会删除 OFT 临时 merged model 以节省磁盘。

## 已跑核心结果

LoRA stable v1 的正式 fullbench 32K 结果：

| ckpt | macro avg | macro pass | parse | hit max | mean len |
|---|---:|---:|---:|---:|---:|
| base | 0.315514 | 0.571642 | 0.804084 | 0.040977 | 9241.7 |
| step100 | 0.331179 | 0.588252 | 0.806567 | 0.036700 | 9137.2 |
| step200 | 0.340859 | 0.586039 | 0.816915 | 0.036010 | 9005.7 |
| step300 | 0.346207 | 0.611590 | 0.837610 | 0.041115 | 9118.6 |

OFT 当前状态：

```text
stable OFT global_step_100 actor checkpoint 已在原机器跑出；
step100 eval 的 controller bug 已修复；
step200/step300 建议在新机器按相同 controller 继续补齐。
```
