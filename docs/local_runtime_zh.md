# 本机独立运行目录

`/home/wangls/peft-for-rl` 已作为后续训练的独立代码和运行根目录。
新实验不需要读取 `/home/wangls/Tina_orthres_run` 或
`/home/wangls/CHERRL`。

## 环境

- Conda 环境：`peft-for-rl`
- 环境前缀：`/home/wangls/miniconda3/envs/peft-for-rl`
- Python：3.12.13
- PyTorch：2.8.0+cu128
- PEFT：0.19.1
- vLLM：0.11.0
- Ray：2.56.0
- editable 项目：`/home/wangls/peft-for-rl`

该环境由原 `cherrl` 克隆，随后删除了指向
`/home/wangls/CHERRL` 的 `verl` 和评测框架 editable 安装，再安装本仓库。
环境变量 `PYTHONPATH` 和 `PEFT_FOR_RL_ROOT` 均固定为本仓库路径。

## 运行资产

```text
runs/
  ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base/
  datasets/verl_dapo/data/
  ckpts/verl/                  # 新 checkpoint
  logs/verl/                   # 新控制台日志
  ray/                         # 新 Ray 临时状态（刻意保持短路径）
```

训练数据使用非空文件：

```text
runs/datasets/verl_dapo/data/dapo-math-17k-boxed.parquet
```

源数据目录中的 `dapo-math-17k.parquet` 是 0 字节镜像文件，不能作为训练输入。
启动脚本会拒绝空的训练或验证文件。

## 启动

从仓库根目录运行：

```bash
cd /home/wangls/peft-for-rl
bash scripts/local/start_official_adalora_4gpu.sh
```

RLPO-init 使用独立入口，所有资产仍位于本仓库的 `runs/` 下：

```bash
cd /home/wangls/peft-for-rl
bash scripts/local/start_rlpo_init_4gpu.sh
```

它按论文源码用基础权重的 top-r 右奇异向量初始化普通 LoRA `A`，并将
`B` 置零；训练期间不使用正交 loss 或 AdaLoRA 动态裁秩。默认日志为
`runs/logs/verl/rlpo_init_r32a64_b64m16n8_270_4gpu_boxed.log`。Ray 临时目录
使用短路径 `/home/wangls/rrlpo`，避免 Unix socket 路径超限。

默认使用 GPU 0-3，并把日志写到：

```text
runs/logs/verl/adalora_r32_to_r8_a64_b64m16n8_270_4gpu_boxed_official.log
```

新实验应使用新的 `EXP_NAME`，例如：

```bash
EXP_NAME=adalora_official_r32_to_r8_v2 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/local/start_official_adalora_4gpu.sh
```

launcher 会在训练前确认实际导入的 `verl` 位于本仓库，同时确认模型权重、
模型配置以及 parquet 文件存在。checkpoint、日志和 Ray 状态默认只写入 `runs/`。
Ray 使用较短的 `runs/ray`，避免其内部 session/socket 后缀超过 Unix socket 路径限制。

旧 Tina 目录中的日志、checkpoint、模型和数据暂时保留，以免破坏仍在运行或待复查的
历史实验。它们不是新启动链的依赖。
