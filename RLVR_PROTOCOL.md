# RLVR 实验协议（强制）

> **生效日期：2026-10-12。状态：新实验唯一有效协议。**
>
> 从本协议生效后，所有可进入主对比表的 `DeepSeek-R1-Distill-Qwen-1.5B`
> 数学 RLVR 实验必须使用本文件定义的配置和固定数据加载顺序。科学变量只允许：
> **学习率、具体 LoRA/PEFT 方法、rank**。其他变化必须另立协议，且不得与本协议结果
> 标为 matched comparison。

机器可读版本：
[`configs/experiments/rlvr_dapo_math_1p5b_protocol_v1.json`](configs/experiments/rlvr_dapo_math_1p5b_protocol_v1.json)

唯一推荐训练入口：
[`scripts/local/run_rlvr_protocol_1p5b_4gpu.sh`](scripts/local/run_rlvr_protocol_1p5b_4gpu.sh)

## 1. 允许变化的科学变量

| 变量 | 允许值/规则 |
| --- | --- |
| `LR` | `1e-6`、`5e-6`、`1e-5` |
| `METHOD` | 启动器已支持且预先声明的 LoRA/PEFT 方法 |
| `LORA_RANK` | 正整数；每个实验名必须包含 rank |

`TOTAL_TRAINING_STEPS`、`RESUME_MODE`、GPU 编号、日志路径属于执行控制，不是可扫的
科学变量。目标 step 必须是 `10` 的倍数且不超过 `270`；断点续跑必须恢复 model、
optimizer、RNG、scheduler 和 data state，累计总预算仍不得超过 270 steps。

**Step 计数口径（2026-10-14 核对）：** 本协议的 `step`、`trainer.global_steps` 和
`global_step_*` checkpoint 名称均表示一次 rollout/trainer iteration：采样64个prompt，
每题8条回答。正式训练将这批数据分成4个16-prompt PPO minibatch；`ppo_epochs=1`，
每个minibatch梯度累积结束后更新一次参数。因此每个正式trainer step对应4次逻辑
optimizer update，step50/100/150分别对应200/400/600次更新。多卡同步不再乘GPU数量，
显存限制导致的microbatch拆分也不增加更新次数。非有限梯度导致的跳过更新须单独记录。
REAL观测阶段的step只计采样批次，optimizer update恒为0；观测与正式训练预算分别报告。
与外部论文比较时必须同时列出采样prompt数、rollout数、optimizer update数及观测预算，
不能只依据同名step判断等预算。已核对标准LoRA step100/150 checkpoint中的Adam计数
分别为400/600；旧snapshot manifest的`optimizer_update`因启动器标注系数写成1而少计，
该历史字段不能代替真实optimizer state。

方法实现不可避免的内部参数必须在方法首次实验前固定并记录，不能在同一轮结果出来后
继续搜索。除方法定义和 rank 直接决定的结构外，共享训练配置不得变化。

2026-10-13 注册 `METHOD=real`：先从 base policy 观察 64 个 batch，逐 prompt 使用
二元正确性 reward contrast、每 response 长度归一化、population std，至少两正确两错误，
输入/输出 sketch 均为 `4*rank`、seed42、无 optimizer 更新和梯度裁剪；导出均匀 rank
子空间，随后 A 固定、B 从零开始训练。观察成本与正式 optimizer steps 分开报告。
训练仍使用本协议的 token-mean GRPO、alpha64 和原始 seed42 sampler 起点。
启动时显式设置 `REAL_ARTIFACT_DIR`；验证器检查完整模型模块覆盖、正交性和探测元数据，
将三个初始化文件的 SHA256 记录到 experiment contract。

2026-10-14 注册两个 GradTop 候选及三个机制对照：`gradtop_prompt_sign`
（8次 observation、完整 prompt 组随机符号）、`gradtop_masked`（16次 observation、
历史 surprisal50%/min128/final128 selector）；对照为 `gradtop`（16次全局token mean）、
`gradtop_short`（8次全局token mean）和 `gradtop_prompt_mean`（8次等prompt/response mean）。
所有 observation 使用 width8/capacity64、seed42、完整题组、不更新参数、不裁剪梯度，
固定预算结束后导出均匀rank，正式训练A/B均可训练、B为零、alpha64。
8/16次 observation 分别为2/4个64-prompt采样step，不能按8/16个trainer step启动。
Prompt等权归一化和原token-mean不同，必须使用同归一化无符号对照判断sign贡献。
新probe的分母在microbatch拆分前固定；它不逐位复现旧microbatch加权的GradTop。
缺少信号支持的方向沿用确定性QR补齐，artifact及contract明确披露补齐维数。
使用 `GRADTOP_ARTIFACT_DIR` 指定产物，验证器校验完整模块、实际观察设置/预算及三个文件哈希。
正式训练共享配置及sampler起点保持本协议规定。
入口与完整设计见 [GradTop备选实验](docs/experiments/gradtop_candidates_20261014_zh.md)。

## 2. 固定训练配置

```text
task                 = RLVR math reasoning
base model           = DeepSeek-R1-Distill-Qwen-1.5B
objective            = DAPO-style GRPO
GPUs                  = 4
precision             = bfloat16
train file            = original 17,917-row boxed parquet

train prompts/step    = 64
PPO minibatch         = 16
responses/prompt      = 8
PPO epochs            = 1
max steps             = 270
evaluation snapshot   = every 10 steps (adapter only, not resumable)
resume checkpoint     = every 50 steps and planned terminal step
resume ckpts retained = latest 2

rank                  = scientific variable
alpha                 = 64
adapter dropout       = 0
target modules        = all-linear

prompt length         = 1024
completion length     = 8192
loss aggregation      = token-mean

LR                    = one of {1e-6, 5e-6, 1e-5}
LR scheduler          = constant
warmup steps          = 0
weight decay          = 0

clip epsilon low      = 0.20
clip epsilon high     = 0.28
clip ratio C          = 10.0
KL reward beta        = 0
actor KL loss         = off (coefficient 0)
entropy coefficient   = 0
overlong shaping      = off

rollout temperature   = 1.0
rollout top-p         = 1.0
rollout top-k         = -1
```

这里的 “DAPO-style” 精确定义为：`adv_estimator=grpo`、DAPO reward manager、
asymmetric clipping (`0.20/0.28`)、token-mean policy loss、无 KL reward、无 actor KL
loss、无额外 format/length reward。不得仅凭名称替换这些开关。

## 3. 固定数据顺序和 seed

训练直接读取原始 `DAPO-Math-17K boxed` parquet，实际为 `17,917` 条：

```text
runs/datasets/verl_dapo/data/dapo-math-17k-boxed.parquet
```

固定加载规则如下：

1. `StatefulDataLoader` 使用 torchdata `RandomSampler`，`shuffle=True`、data seed `42`。
2. batch size 固定为 `64`，训练预算最多 `270` steps，因此最多消费 permutation 前
   `64 * 270 = 17,280` 条。
3. 第一个 epoch 的剩余 `637` 条在 270-step 预算内自然不会被消费；它们不定义为 held-out，
   也不用于 validation 或模型选择。
4. `TEST_FILE` 与 `TRAIN_FILE` 指向同一原始文件，但 `val_before_train=False`、`test_freq=0`，
   训练过程不会运行该 validation loader。正式模型选择只使用第 5 节六榜。
5. PPO actor minibatch 不二次 shuffle；data、PPO dataloader、rollout、FSDP 和方法初始化
   seed 均固定为 `42`。

`runs/datasets/verl_dapo/data/dapo_math_boxed_heldout637_seed42_v1/train_indices.csv`
只作为 sampler-prefix 审计证据保留，不再作为独立训练集：其 17,280 个索引已经验证为
当前 torchdata `RandomSampler(seed=42)` 在原始文件上的前 17,280 项，与旧 seed42
dataloader 前 270 个 batch 逐项一致。启动器会检查源 parquet 和索引的哈希，并使用训练
所依赖的实际 sampler 类重新计算这一前缀顺序。

不得改为加载已按该 permutation 排列的 17,280-row parquet 后再设置 `shuffle=True`；那会发生
第二次打乱。实测二次打乱后只有 1/17,280 个位置相同，270/270 个 batch 的顺序均不同。

## 4. Checkpoint 选择

- 训练预算上限为 270 steps，不以训练 reward 选择最终 checkpoint。
- 每 10 steps 保存一份方法完备的轻量 adapter snapshot 和 manifest，用于正式评测与
  best-checkpoint 选择；它不包含 optimizer、scheduler、RNG 或 dataloader state，**不得用于
  精确续训**。
- 每 50 steps 保存一份完整可恢复 checkpoint，内容必须包括 FSDP model、optimizer、
  scheduler/RNG extra state 和 dataloader state。计划终止步无论是否为 50 的倍数也必须保存，
  因而单独跑到 step10/50 或最终 step270 后都可以从该终止点原地续跑。
- 完整恢复点只保留最近 2 个；轻量 adapter snapshot 保留完整的每 10-step 轨迹。以标准
  rank-32 LoRA 的现有实测计，单个 snapshot 约 148 MB，而完整 FSDP 恢复点约 7.2 GiB。
- 训练中的 checkpoint 和 snapshot 必须写本机 `runs/` 热存储；NFS 不进入同步保存热路径，
  只允许在训练结束或暂停后做归档，避免四个 rank 等待共享盘写入。
- 正式模型选择使用固定六榜协议，可按每 10 steps 或每 50 steps 的预声明网格评测。
- 比较不同 LR/方法/rank 时，候选 checkpoint 网格必须一致。
- 选择并报告 best checkpoint 时，同时报告整个已评测轨迹，禁止只展示最优点。

固定落盘位置：

```text
full resume checkpoints = runs/ckpts/verl/peft-for-rl/<experiment>/global_step_*
evaluation snapshots    = runs/snapshots/rlvr_protocol/<experiment>/step_*
NFS                     = post-run archive only
```

每种允许进入协议的方法都必须让 evaluation snapshot 覆盖该方法推理所需的全部可训练状态；
标准 LoRA/NoRA/RLPO 为完整 adapter 权重，带 magnitude 或正交参数的方法还必须包含对应状态。
如果某个新方法尚无方法完备的轻量导出器，不得静默保存不完整 snapshot；应先补齐导出和
加载验证，或将该实验声明为协议外实验。

## 5. 固定正式评测

```text
benchmarks       = aime24,aime25,amc23,hmmt_feb,math500,minerva
AIME24/AIME25    = Avg@32
AMC23/HMMT Feb   = Avg@32
MATH500/Minerva  = Avg@4
evaluation seed  = 42
temperature      = 0.6
top_p            = 0.95
max_new_tokens   = 32768
max_model_len    = 34816
total records    = 7248
```

每次评测必须保存逐样本 records、四份 shard manifest 和 verification JSON；比较时优先使用
problem-cluster paired bootstrap，并同时报告 Avg@k、Pass@k、parse rate、hit-max rate 和
平均输出长度。

## 6. 与 2026-10-12 前最近实现的差异

| 项目 | 最近 NoRA/Matched-LoRA 实现 | 本协议 |
| --- | --- | --- |
| 模型、4 GPU、bf16 | 相符 | 固定 |
| batch/minibatch/responses | `64/16/8`，相符 | 固定 |
| rank/alpha/dropout | `32/64/0`，相符 | rank 可变，其余固定 |
| prompt/completion | `1024/8192`，相符 | 固定 |
| clip high、KL beta | `0.28/0`，相符 | 固定 |
| LR/schedule/warmup | NoRA `1e-5`，LoRA `1e-6`；constant/0，相符 | 只扫三档 LR |
| seed | 最近主线为 42，相符 | 全链路固定 42 |
| 最大训练预算 | 最近 NoRA 实际到 step100 | 上限 270，按 checkpoint 早停 |
| 训练数据 | 完整 17,917 条文件 | 同一原始文件，固定 seed42 sampler 前缀 |
| 270-step 数据成员 | seed42 时为 permutation 前 17,280 条 | 相符，并额外校验逐项顺序 |
| `TEST_FILE` | 指回训练文件 | 保持相同；训练中禁用 validation |

因此，协议生效前且同样使用原始 parquet、`shuffle=True`、data seed42、batch64 的实验，
其 prompt 顺序与本协议兼容，可以继续作为数据口径一致的历史证据；是否构成严格 matched
baseline 仍需同时核对 LR、dropout、rank、seed、checkpoint、rollout 和其余训练参数。
seed43、不同 batch/sampler 或不同数据文件的实验不属于这一统一数据顺序。

## 7. 启动示例

先做只读检查和命令预览：

```bash
LR=1e-5 METHOD=lora LORA_RANK=32 TARGET_STEP=50 \
  bash scripts/local/run_rlvr_protocol_1p5b_4gpu.sh
```

实际启动：

```bash
DRY_RUN=0 CUDA_VISIBLE_DEVICES=4,5,6,7 \
LR=1e-5 METHOD=lora LORA_RANK=32 TARGET_STEP=50 \
  bash scripts/local/run_rlvr_protocol_1p5b_4gpu.sh
```

从最近保留的完整 checkpoint 原地续到下一目标步：

```bash
DRY_RUN=0 RESUME_MODE=auto TARGET_STEP=100 \
EXP_NAME=<same-experiment-name> \
LR=1e-5 METHOD=lora LORA_RANK=32 \
  bash scripts/local/run_rlvr_protocol_1p5b_4gpu.sh
```
