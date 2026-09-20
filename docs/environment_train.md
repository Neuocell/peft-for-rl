# 训练环境

本仓库训练侧使用 `verl`。之前本机成功运行时复用了一个已有 conda 环境，但迁移时应该按 `verl + PEFT + Ray + FSDP/vLLM rollout` 单独安装。

## 推荐环境

```text
Python: 3.10 或 3.11
CUDA: 与本机 torch/vLLM/flash-attn 匹配
GPU: 4 x A6000 级别显存可跑当前 1.5B LoRA/OFT 8K 配置
训练框架: 本仓库根目录的 verl 包
```

安装：

```bash
cd /path/to/peft-for-rl
conda create -n peft-for-rl python=3.10 -y
conda activate peft-for-rl

pip install -U pip
pip install -r requirements.txt
pip install -r requirements-cuda.txt
pip install -e .
```

实际 torch、vLLM、flash-attn 的组合要以机器 CUDA 版本为准。如果已有能跑通的 verl 环境，可以直接复用，但不要在文档或脚本中把环境名当成项目身份。

## 训练侧关键依赖

```text
torch
ray[default]
transformers
datasets
accelerate
peft==0.19.1
vllm
flash-attn
liger-kernel
tensordict
hydra-core
```

OFT 训练还依赖本仓库 `verl` 中已加入的 PEFT/OFT 兼容逻辑。LoRA 和 OFT 共享同一个训练脚本，OFT wrapper 通过环境变量切换：

```text
PEFT_TYPE=oft
LORA_RANK=0
OFT_BLOCK_SIZE=32
OFT_RANK=0
OFT_DROPOUT=0.0
```

## 运行目录

建议把运行资产放到仓库外或 `runs/` 下，避免误提交：

```text
RUNTIME_ROOT=/path/to/runtime
MODEL_PATH=${RUNTIME_ROOT}/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base
TRAIN_FILE=${RUNTIME_ROOT}/datasets/verl_dapo/data/dapo-math-17k-boxed.parquet
CKPTS_DIR=${RUNTIME_ROOT}/ckpts/verl/DAPO-Math-17k/<exp_name>
RAY_TEMP_DIR=${RUNTIME_ROOT}/ray_<exp_name>
```

`.gitignore` 已排除常见运行目录；模型、checkpoint、日志、Ray 临时文件和生成输出都不应进入 git。

## 训练 smoke check

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
RUNTIME_ROOT=/path/to/runtime \
MODEL_PATH=/path/to/base \
TRAIN_FILE=/path/to/dapo-math-17k-boxed.parquet \
TOTAL_TRAINING_STEPS=1 \
SAVE_FREQ=1 \
RESUME_MODE=disable \
bash examples/verl_train/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh
```

smoke check 只用于验证环境、Ray、rollout、reward 和 checkpoint 写入链路，不能作为训练结论。
