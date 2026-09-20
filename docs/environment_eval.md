# 评测环境

评测侧使用 `vLLM` 做多 shard 采样，并使用 Tina 中抽出的最小 parser 计算数学答案正确率。评测环境可以和训练环境相同，但建议独立记录版本，因为 vLLM/torch 的组合对稳定性影响很大。

## 推荐环境

```text
Python: 3.10 或 3.11
CUDA: 与 vLLM wheel 匹配
推理框架: vLLM
parser: tina_run/tina/analysis/rollout_utils.py
```

最小安装：

```bash
conda create -n peft-for-rl-eval python=3.10 -y
conda activate peft-for-rl-eval

pip install -U pip
pip install transformers datasets pandas pyarrow peft
pip install vllm
pip install math-verify
```

如果训练环境中的 vLLM 已经稳定，也可以直接使用同一个环境：

```bash
PYTHON_BIN=/path/to/env/bin/python
```

## Benchmark 数据

当前 formal eval 固定为：

```text
aime24,aime25,amc23,hmmt_feb,math500,minerva
```

脚本会优先从 HuggingFace datasets 读取：

```text
HuggingFaceH4/aime_2024
yentinglin/aime_2025
knoveleng/AMC-23
HuggingFaceH4/MATH-500
knoveleng/Minerva-Math
```

`hmmt_feb` 以及部分 fallback 使用本地 Direct-OPD parquet 路径。迁移机器上需要同步这些评测 parquet，并设置：

```bash
export DIRECT_OPD_EVAL_ROOT=/path/to/Direct-OPD/datasets/eval
```

## 评测协议

```text
temperature=0.6
top_p=0.95
max_new_tokens=32768
max_model_len=34816
samples_small=32  # AIME24/AIME25/AMC23/HMMT Feb
samples_large=4   # MATH500/Minerva
```

指标：

```text
Avg@k: 所有采样的平均正确率
Pass@k: 每题 k 次采样中至少一次正确的比例
parse_rate: parser 能抽取 final answer 的比例
hit_max_rate: 生成达到 max_new_tokens 上限的比例
maxed_unparseable_rate: 到达长度上限且未能解析答案的比例
```

## 运行示例

LoRA adapter 评测：

```bash
cd /path/to/peft-for-rl/tina_run

CUDA_VISIBLE_DEVICES=0,1,2,3 \
PROJECT_ROOT="$PWD" \
PYTHON_BIN=/path/to/eval/env/bin/python \
BASE_MODEL=/path/to/DeepSeek-R1-Distill-Qwen-1.5B/base \
LORA_ADAPTER=/path/to/global_step_100/actor/lora_adapter \
CHECKPOINT_NAME=dapo_boxed_stable_lora_step100_fullbench_32768 \
OUTPUT_DIR=/path/to/outputs/full_bench_eval/dapo_boxed_stable_lora_step100_fullbench_32768 \
bash scripts/local/eval/run_full_bench_vllm_4gpu.sh
```

Base model 评测时不传 `LORA_ADAPTER`：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PROJECT_ROOT="$PWD" \
PYTHON_BIN=/path/to/eval/env/bin/python \
BASE_MODEL=/path/to/DeepSeek-R1-Distill-Qwen-1.5B/base \
CHECKPOINT_NAME=base_fullbench_32768 \
OUTPUT_DIR=/path/to/outputs/full_bench_eval/base_fullbench_32768 \
bash scripts/local/eval/run_full_bench_vllm_4gpu.sh
```
