# 实验配置

这个配置是当前后续实验的稳定起点。它参考 PeRL/Tina 的数学 RL + PEFT 设定，但训练框架使用本仓库的 `verl`。

## 数据

```text
Training data: DAPO-Math-17k boxed
Prompt key: prompt
Reward key: data_source
Ground truth: reward_model.ground_truth
```

数据转换脚本：

```text
tina_run/scripts/data/prepare_dapo_math_boxed_verl.py
```

## Prompt

评测 prompt 固定为 boxed final-answer 风格：

```text
Solve the following math problem efficiently and clearly.
The last line should contain:
Therefore, the final answer is: $\boxed{ANSWER}$. I hope it is correct
```

训练数据也对齐 boxed answer parser，避免再混用 think-tag、cosine shaping、单独格式奖励等不稳定因素。

## Reward

当前稳定版只使用 outcome reward：

```text
Reward file: verl/utils/reward_score/boxed_math_accuracy.py
Reward manager: dapo
Correct boxed answer: positive reward
Wrong/unparseable answer: no correctness reward
Cosine reward: off
Format reward: off
Overlong buffer: off
```

关闭 overlong buffer 是当前经验结论。之前开启 `ENABLE_OVERLONG_BUFFER=True` 且训练生成长度为 8K 时，模型明显学习到长度相关 hacking 行为；因此稳定基线中固定：

```text
ENABLE_OVERLONG_BUFFER=False
```

## 训练超参

```text
Base model: DeepSeek-R1-Distill-Qwen-1.5B
Framework: verl
Algorithm: GRPO-style, adv_estimator=grpo
max_prompt_length=1024
max_response_length=8192
train_prompt_bsz=16
n_resp_per_prompt=8
rollouts/update=128
ppo_mini_batch_size=16
LR=1e-6
lr_warmup_steps=10
weight_decay=0.1
clip_ratio_low=0.2
clip_ratio_high=0.28
KL reward: off
KL loss: off
loss_agg_mode=token-mean
save_freq=20
max_actor_ckpt_to_keep=2
```

LoRA：

```text
PEFT_TYPE=lora
LORA_RANK=32
LORA_ALPHA=64
LORA_DROPOUT=0.05
target_modules=all-linear
```

OFT：

```text
PEFT_TYPE=oft
LORA_RANK=0
OFT_BLOCK_SIZE=32
OFT_RANK=0
OFT_DROPOUT=0.0
target_modules=all-linear
```

## Checkpoint 选择

后续比较不建议只按训练 reward 选最优 ckpt。训练 reward 容易受长度、parser、题目分布和 rollout 方差影响。当前协议固定：

```text
每 100 step 做一次 formal eval
主要比较 base / step100 / step200 / step300
如果继续训练，沿用每 100 step 的固定评测节奏
```

如果要做方法比较，必须让 normal LoRA 和新 PEFT 方法共享同一数据、reward、prompt、长度、batch、LR、save/eval 频率和 parser。

## Formal Eval

```text
benchmarks=aime24,aime25,amc23,hmmt_feb,math500,minerva
temperature=0.6
top_p=0.95
max_new_tokens=32768
max_model_len=34816
samples_small=32
samples_large=4
```

小 benchmark 使用 Avg@32 / Pass@32，大 benchmark 使用 Avg@4 / Pass@4。所有结果同时报告 parse、hit max、平均长度，避免把长度/格式行为误解成数学能力提升。

## 当前训练经验

1. 训练 8K、评测 32K 是当前资源约束下比较稳的折中。训练 32K 成本太高，训练 3K 又会放大短 budget 偏差。
2. 对数学 RLVR，reward 应先 outcome-first。格式 reward 或长度 shaping 权重过大时，模型容易优先优化可控表面行为。
3. 对难题数据，group 内全错会导致 correctness advantage 稀疏。解决方向应优先考虑数据难度、采样数、curriculum 或 critique/auxiliary signal，而不是把格式奖励调大。
4. 评测必须多样本聚合。单 seed 或少量题的波动很大，尤其是 AIME/HMMT 这类小集合。
5. 新 PEFT 方法先在 stable LoRA 对照上只改参数化本身。不要同时改 reward、prompt、长度和数据，否则无法归因。

