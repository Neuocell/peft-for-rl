# 本机 LoRA 梯度子空间实验汇总

更新时间：2026-10-04 13:31（Asia/Shanghai）

本文整理当前机器上围绕 SPAR-LoRA、random-B、full-gradient probe、token mask、梯度协方差和 rank 的主要实验。内容以现存启动脚本、Hydra 配置、训练日志、probe artifact、checkpoint 和 full-benchmark JSON 为准，不把仅存在于会话记忆中的数字当作正式结果。

## 1. 结论摘要

截至本次整理，统一 6-benchmark 评测中的最好结果是：

| 方法 | step | Macro Avg@k |
|---|---:|---:|
| token-mask centered covariance adaptive eq-r28 | 100 | **0.382131** |
| full-gradient mean / Gain-LCB uniform-r32 | 150 | 0.375749 |
| token-mask centered covariance adaptive eq-r28 | 50 | 0.375451 |
| token-mask uncentered covariance adaptive eq-r28 | 100 | 0.369779 |
| original random-B 复现，近似 r32 | 50 | 0.367304 |

当前可以较有把握地记录以下现象：

1. 直接训练标准 LoRA r8 表现最差，step 50 为 `0.323111`，step 100 为 `0.315783`。
2. 相同 r8 下，random-B 探测初始化达到 `0.358323`，比标准 LoRA r8 高 `0.035212`。初始化方向明显重要。
3. original random-B 从 r8 提高到平均 rank `31.6122` 后，step 50 只比 random-B uniform-r8 高 `0.008981`。rank 不是唯一解释。
4. centered covariance 在 step 50/100 都高于 uncentered，差值分别为 `+0.024681` 和 `+0.012352`；但两组的 calibration mask scope 不同，step 100 的优化器连续性也不同，因此这仍是强相关证据，不是严格单变量因果结论。
5. full-gradient mean uniform-r32 训练到 step 150 后评测为 `0.375749`，高于 mean-r8 的 step 50/100，但仍未超过 centered covariance step 100。由于没有 r32 step 50/100 评测，这个差异不能纯归因于 rank。
6. 训练 reward、训练 parse success 与 full-benchmark 泛化并不同步。多个实验训练 reward 持续升高，但评测持平或下降。
7. 当前所有主要 `grad_subspace` 正式训练均为 `LORA_FREEZE_A=False`。探测得到的 A 是可训练初始化，不是永久固定子空间；静态不变的是训练前生成的 rank map。
8. original random-B rank map 的 196 个模块中，192 个为 r32，只有 3 个 r8、1 个 r28，不能把它当作“自适应 rank 有效”的证据。
9. Phase 1 seed 42 中，I8 相对匹配 I0 的 macro delta 为 `+0.035999`，I16 为 `-0.016665`，signal-only I32 为 `+0.046317`；I8 仍是预注册的 mixed candidate。为缩短当前模型/任务结论的周转时间，执行范围已于 `2026-10-03T18:55:11+08:00` 由 seeds 42/43/44 显式修订为 seeds 42/43，seed 44 不启动。该修订只能支持“在一个 held-out 训练 seed 上复现”的固定设置结论，不能支持“跨训练 seed 稳定”或跨模型/任务泛化表述。
10. Phase 1 seed-43 I0 已完成训练、7,248-record 六基准聚合和独立 postverify，Macro Avg@k 为 `0.327747`；seed-43 I8 尚未启动，因此 I8-vs-I0 的 seed-43 comparison 和两-seed 最终 gate 都还不能计算。两个 I0 seed 的点估计相差约 `-0.002415`，只说明匹配随机基线在这两个训练轨迹上的宏平均接近，不能据此推断 I8 提升可复现。

Phase 1 当前状态如下；这里的 `verified` 同时要求 training contract 与固定 full-benchmark 协议通过独立验收：

| training seed | I0 Macro Avg@k | I8 Macro Avg@k | 当前状态 |
|---:|---:|---:|---|
| 42 | 0.330162 | 0.366161 | I0/I8 均 verified；I8-I0 为 `+0.035999` |
| 43 | 0.327747 | 未运行 | I0 verified；I8 尚未启动 |
| 两-seed 最终 gate | - | - | 输入不全，blocked on seed-43 I8 |

## 2. 代码与设备快照

### 2.1 硬件

```text
GPU: 4 x NVIDIA L40S
单卡显存: 46068 MiB
Driver: 610.57.04
```

### 2.2 仓库状态

初始实验整理快照保留如下：

```text
Repository: /root/peft-for-rl
Git HEAD: 926909c94d59c5950d201dc9d68616beea986bff
```

整理开始时工作树有 34 项未提交修改或新增文件，其中包含 token-mask covariance、续训、评测和 original random-B 复现脚本。因此只记录 Git HEAD 不足以完整复现实验，必须同时保留本文列出的脚本和 artifact。

截至本次迁移整理前，源码、实验配置、小型文本证据和最小 I8 初始化已归档并推送：

```text
Migration baseline commit: 0cca95911fc36271480318718c9f4d634d8d4d0f
Remote: ssh://git@ssh.github.com:443/Neuocell/peft-for-rl.git
Remote branch: origin/experiments/spar-lora-v0
```

该迁移基线不包含 base model、数据集、checkpoint、adapter 或 raw evaluation records；具体边界见 19.2.45--19.2.47。上面的 `926909c` 是实验整理开始时的历史快照，不是当前迁移分支 HEAD。

### 2.3 主要存储位置

```text
代码仓库:              /root/peft-for-rl
共享运行目录:          /data/peft-for-rl-runtime
本机根盘实验目录:      /root/peft-for-rl/runs
历史 random-B 日志:    /root/gradtop_probe12_r8to32_mean31p63_a2_b64m16n8_270_v1.log
基础模型:              /data/peft-for-rl-runtime/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base
训练集:                /data/peft-for-rl-runtime/datasets/verl_dapo/data/dapo-math-17k-boxed.parquet
评测根目录:            /data/peft-for-rl-runtime/outputs/full_bench_eval
```

## 3. 公共训练协议

除早期小规模筛选外，主要对照使用以下公共配置：

| 配置项 | 值 |
|---|---|
| Base model | DeepSeek-R1-Distill-Qwen-1.5B |
| Train data | DAPO-Math-17k boxed |
| Reward | boxed outcome accuracy only |
| Algorithm | GRPO |
| Target modules | `all-linear`，共 196 个 q/k/v/o/gate/up/down 模块 |
| Global prompt batch | 64 |
| PPO prompt mini-batch | 16 |
| Rollout per prompt | 8 |
| PPO epochs | 1 |
| Train max prompt | 1024 |
| Train max response | 8192 |
| Actor/inference token budget | 12288 |
| vLLM max sequences | 256 |
| Learning rate | `1e-6`，constant，0 warmup |
| Weight decay | 0 |
| PPO clip | low 0.2，high 0.28 |
| KL reward/loss | disabled，系数 0 |
| Entropy coefficient | 0 |
| Gradient clip | 1.0 |
| Loss aggregation | token mean |
| LoRA scaling | `alpha/r = 2` |
| Seeds | data=42，PPO dataloader=42 |
| GPUs | 4 |

训练 launcher 为 [`examples/verl_train/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh`](../examples/verl_train/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh)。训练 rollout 使用 temperature 1.0、top-p 1.0；评测使用独立的采样配置。

需要注意的非严格对齐项：

- 标准 LoRA r8、random-B r8、centered/uncentered adaptive 和 original random-B 使用 LoRA dropout 0.05。
- full-gradient mean uniform-r8/r32 与 windowed consensus 使用 dropout 0.0。
- vLLM 异步 rollout 没有 request-level 完全固定随机性。即使 data/PPO seed 相同，单 seed 训练轨迹仍不能视作逐请求配对。

## 4. 统一评测协议

评测脚本为 [`tina_run/scripts/local/eval/run_full_bench_vllm_4gpu.sh`](../tina_run/scripts/local/eval/run_full_bench_vllm_4gpu.sh) 和 [`eval_full_bench_vllm.py`](../tina_run/scripts/local/eval/eval_full_bench_vllm.py)。所有主表结果使用同一套 benchmark snapshot。

| 项目 | 配置 |
|---|---|
| Benchmarks | AIME24、AIME25、AMC23、HMMT Feb、MATH500、Minerva |
| AIME24/AIME25/HMMT | 各 30 题，每题 32 samples |
| AMC23 | 40 题，每题 32 samples |
| MATH500 | 500 题，每题 4 samples |
| Minerva | 272 题，每题 4 samples |
| 总生成数 | 7248 |
| Temperature | 0.6 |
| top-p | 0.95 |
| max new tokens | 32768 |
| max model length | 34816 |
| seed | 42 |

`Macro Avg@k` 是 6 个 benchmark 的 Avg@k 等权平均。7248 条 generation 不是 7248 个独立问题，因此当前表格是点估计，尚未做按问题 paired bootstrap 置信区间。

## 5. 方法定义

### 5.1 标准 LoRA r8

标准 `PEFT_TYPE=lora`，每层 rank 8、alpha 16，A/B 均训练。没有训练前梯度探测。

启动脚本：[`scripts/local/start_standard_lora_r8_b64_50_4gpu.sh`](../scripts/local/start_standard_lora_r8_b64_50_4gpu.sh)。

### 5.2 original random-B energy probe

对一层完整梯度 `G`，probe 令 `A=0`、`B=R`，因此模型函数不变，但：

```text
grad_A = R^T G
S = grad_A^T = G^T R
E[S S^T | G] = G^T G
```

这里 `R[i,j] ~ N(0,1/r)`。累计随机 sketch 后做压缩/SVD，按目标能量 0.95 从 `[8,12,16,20,24,28,32]` 中选择 rank。正式训练将选出的右子空间写入 A、B 清零，然后 A/B 均继续训练。

复现参数：probe width 8、capacity 64、min/max step 6/12、stability patience 3、overlap threshold 0.98、seed 42。启动脚本：[`scripts/local/run_original_random_b_reproduction.sh`](../scripts/local/run_original_random_b_reproduction.sh)。

### 5.3 random-B uniform-r8

复用 random-B energy probe 的候选方向，但把每层统一截断到 rank 8、alpha 16。该实验用于把“探测初始化”与“大 rank”部分解耦。

启动脚本：[`scripts/local/run_random_b_energy_uniform_r8_control.sh`](../scripts/local/run_random_b_energy_uniform_r8_control.sh)。

### 5.4 full-gradient mean / Gain-LCB

直接读取完整权重梯度 `G_p`。对 discovery prompt 求均值：

```text
M = mean_p G_p
```

候选 A 是 `M` 的 top right singular vectors。calibration 上用 discovery 更新方向与 held-out 梯度的一阶 gain 下置信界筛 atom。uniform-r8/r32 分别保留 8/32 个方向。

基础 probe 使用 16 discovery、16 calibration、8 audit 有效 prompt；每个 prompt 8 rollout。详细设计见 [`full_gradient_rl_probe_v1_zh.md`](full_gradient_rl_probe_v1_zh.md)。

### 5.5 SPAR positive-probe adaptive eq-r8

先采样 base policy rollout，只保留 reward>=0.5、parse 成功且 prompt 唯一的正样本。32 条用于 discovery，32 条用于 calibration。teacher-forced CE probe 使用 random-B sketch 产生 rank-32 候选，再用 calibration 的 `U=P*R` 分数分配与 uniform-r8 等参数预算的 rank。

本次 probe 共检查 112 prompt/896 rollout，其中 315 条正 rollout，正样本率 0.3516。adaptive rank 范围 2 到 32，平均 rank 13.7398；参数 9,230,848，对照 uniform-r8 为 9,232,384。详见 [`spar_lora_v0_static_zh.md`](spar_lora_v0_static_zh.md)。

### 5.6 windowed Adam consensus

使用 8 个 discovery、2 个 calibration、2 个 audit 时间窗口，每窗口 16 prompt，共 512 prompt/4096 rollout。每个 prompt 从 8 条 response 中稳定抽取一条并做无偏缩放。对 full-weight 梯度建立 CPU 虚拟 Adam，再从 raw momentum、Adam update 和跨窗口 consensus hybrid 中产生候选。

正式实验选择 `consensus_hybrid` uniform-r8。其 calibration energy capture 为 0.867181，U-score capture 为 0.962896。详见 [`windowed_adam_consensus_probe_zh.md`](windowed_adam_consensus_probe_zh.md)。

### 5.7 token-mask covariance

完整 GRPO prompt-group 梯度按 response token 聚合。`top_surprisal` mask 使用当前策略的负 log-probability 作为分数，保留每条 response 的 top 50% token，同时至少保留 128 个 token，并保留末尾 128 token。

候选 rank 32，Nyström sketch width 40；有效 split 为 64 discovery、32 calibration、16 audit prompt。adaptive allocator 以 uniform-r28 的参数预算分配，r_min=16、r_max=32。centered 和 uncentered 两组均有：

```text
trainable parameters = 32,311,296
rank min/mean/max = 16 / 29.7551 / 32
```

centered 使用：

```text
C = mean_p (G_p - mean(G))^T (G_p - mean(G))
mask scope = discovery only
calibration energy capture = 0.986011
U-score capture = 0.984049
```

uncentered 使用：

```text
M2 = mean_p G_p^T G_p
mask scope = discovery + calibration
calibration energy capture = 0.988920
U-score capture = 0.995739
```

两组 rank 结构几乎相同，但 atom 和 calibration 数据处理不同。启动脚本分别为 [`run_token_mask_covariance_adaptive_experiment.sh`](../scripts/local/run_token_mask_covariance_adaptive_experiment.sh) 和 [`run_token_mask_uncentered_covariance_adaptive_270.sh`](../scripts/local/run_token_mask_uncentered_covariance_adaptive_270.sh)。

## 6. Full-benchmark 总表

| 方法 | step | Macro Avg | Macro Pass | Parse | Hit max | Mean len |
|---|---:|---:|---:|---:|---:|---:|
| token-mask centered covariance adaptive eq-r28 | 100 | **0.382131** | 0.591042 | **0.895557** | 0.019454 | 7471.6 |
| full-gradient mean / Gain-LCB uniform-r32 | 150 | 0.375749 | **0.633608** | 0.887417 | **0.013935** | **6609.4** |
| token-mask centered covariance adaptive eq-r28 | 50 | 0.375451 | 0.628655 | 0.870999 | 0.027180 | 8295.7 |
| token-mask uncentered covariance adaptive eq-r28 | 100 | 0.369779 | 0.610990 | 0.877759 | 0.016694 | 7382.6 |
| original random-B r8→32 复现 | 50 | 0.367304 | 0.632931 | 0.862307 | 0.031181 | 8592.4 |
| full-gradient mean / Gain-LCB uniform-r8 | 50 | 0.358565 | 0.602812 | 0.846164 | 0.030767 | 8655.5 |
| random-B energy uniform-r8 | 50 | 0.358323 | 0.598933 | 0.853201 | 0.028698 | 8518.4 |
| token-mask uncentered covariance adaptive eq-r28 | 50 | 0.350770 | 0.630265 | 0.842577 | 0.026490 | 8352.7 |
| SPAR positive-probe adaptive eq-r8 | 50 | 0.348220 | 0.624363 | 0.834437 | 0.030767 | 8698.2 |
| windowed consensus hybrid uniform-r8 | 50 | 0.348194 | 0.641042 | 0.845199 | 0.029525 | 8598.9 |
| full-gradient mean / Gain-LCB uniform-r8 | 100 | 0.340163 | 0.616144 | 0.841887 | 0.024972 | 8251.1 |
| standard LoRA r8 | 50 | 0.323111 | 0.572593 | 0.810982 | 0.034906 | 9014.7 |
| standard LoRA r8 | 100 | 0.315783 | 0.551141 | 0.812224 | 0.039597 | 9260.4 |

作为协议参考，仓库既有 stable baseline 文档记录 base model 为 `0.315514`，普通 stable LoRA step 100/200/300 分别为 `0.331179/0.340859/0.346207`。这组是已有基线记录，不属于本轮 probe 方法的严格同批训练。

## 7. 分 benchmark Avg@k

| 方法 | step | AIME24 | AIME25 | AMC23 | HMMT | MATH500 | Minerva |
|---|---:|---:|---:|---:|---:|---:|---:|
| centered cov eq-r28 | 100 | 0.2760 | 0.2344 | **0.7445** | 0.1042 | **0.7140** | 0.2197 |
| full-grad mean r32 | 150 | 0.2885 | 0.2260 | 0.7039 | 0.0948 | 0.7050 | **0.2362** |
| centered cov eq-r28 | 50 | **0.3135** | **0.2375** | 0.6727 | 0.1000 | 0.7075 | 0.2215 |
| uncentered cov eq-r28 | 100 | 0.2875 | 0.2219 | 0.7023 | 0.0875 | 0.6860 | 0.2335 |
| original random-B near-r32 | 50 | 0.3021 | 0.2115 | 0.6938 | 0.0990 | 0.6825 | 0.2151 |
| full-grad mean r8 | 50 | 0.3115 | 0.2125 | 0.6523 | 0.1010 | 0.6700 | 0.2040 |
| random-B r8 | 50 | 0.2812 | 0.2302 | 0.6703 | 0.0906 | 0.6735 | 0.2040 |
| uncentered cov eq-r28 | 50 | 0.2573 | 0.2219 | 0.6570 | 0.0990 | 0.6645 | 0.2050 |
| SPAR adaptive eq-r8 | 50 | 0.2854 | 0.1958 | 0.6359 | **0.1063** | 0.6600 | 0.2059 |
| windowed consensus r8 | 50 | 0.2708 | 0.2031 | 0.6453 | 0.0979 | 0.6615 | 0.2105 |
| full-grad mean r8 | 100 | 0.2625 | 0.1917 | 0.6359 | 0.0792 | 0.6695 | 0.2022 |
| standard LoRA r8 | 50 | 0.2625 | 0.1979 | 0.5555 | 0.0958 | 0.6275 | 0.1994 |
| standard LoRA r8 | 100 | 0.2219 | 0.1698 | 0.5883 | 0.0938 | 0.6335 | 0.1875 |

centered step 100 的总体领先主要来自 AMC23 和 MATH500，并不是所有 benchmark 都提高：相对 centered step 50，AIME24 从 0.3135 降到 0.2760，Macro Pass 也从 0.6287 降到 0.5910。继续训练提高了平均正确率和 parse，但降低了一部分采样覆盖度。

full-gradient mean r32 step 150 在 Minerva 上最高，平均长度也最短；但 HMMT 和 AIME 并没有随训练 reward 同比例提高。

## 8. 三组重点评测明细

### 8.1 Centered covariance，step 100

| Benchmark | Avg@k | Pass@k | Parse | Hit max | Mean len |
|---|---:|---:|---:|---:|---:|
| AIME24 | 0.2760 | 0.6333 | 0.8990 | 0.0458 | 11793.7 |
| AIME25 | 0.2344 | 0.5000 | 0.9240 | 0.0323 | 10903.4 |
| AMC23 | 0.7445 | 0.9500 | 0.9391 | 0.0141 | 6313.6 |
| HMMT Feb | 0.1042 | 0.3333 | 0.9365 | 0.0312 | 12403.6 |
| MATH500 | 0.7140 | 0.7840 | 0.9710 | 0.0065 | 3622.5 |
| Minerva | 0.2197 | 0.3456 | 0.6415 | 0.0046 | 4716.3 |

### 8.2 Uncentered covariance，step 100

| Benchmark | Avg@k | Pass@k | Parse | Hit max | Mean len |
|---|---:|---:|---:|---:|---:|
| AIME24 | 0.2875 | 0.7000 | 0.8708 | 0.0302 | 11166.7 |
| AIME25 | 0.2219 | 0.5000 | 0.9031 | 0.0406 | 11293.4 |
| AMC23 | 0.7023 | 0.9250 | 0.9094 | 0.0109 | 6135.8 |
| HMMT Feb | 0.0875 | 0.4000 | 0.9177 | 0.0292 | 12501.0 |
| MATH500 | 0.6860 | 0.7880 | 0.9465 | 0.0040 | 3492.2 |
| Minerva | 0.2335 | 0.3529 | 0.6627 | 0.0028 | 4694.9 |

### 8.3 Full-gradient mean uniform-r32，step 150

| Benchmark | Avg@k | Pass@k | Parse | Hit max | Mean len |
|---|---:|---:|---:|---:|---:|
| AIME24 | 0.2885 | 0.8000 | 0.8385 | 0.0385 | 10489.7 |
| AIME25 | 0.2260 | 0.5333 | 0.9104 | 0.0229 | 9655.6 |
| AMC23 | 0.7039 | 0.9500 | 0.9117 | 0.0117 | 5492.9 |
| HMMT Feb | 0.0948 | 0.3667 | 0.9479 | 0.0177 | 10832.5 |
| MATH500 | 0.7050 | 0.7840 | 0.9565 | 0.0035 | 3202.2 |
| Minerva | 0.2362 | 0.3676 | 0.7013 | 0.0028 | 4348.4 |

该 rank-32 评测原本因实例释放只留下 4 个完整 shard，没有 summary。整理时确认每个 shard 都有 1812 条记录，并用相同 evaluator 的 `--aggregate_only` 离线生成了 JSON/CSV；没有重新运行推理。

## 9. 训练动态

下表是日志中的 10-step 窗口均值。训练 reward 即 boxed accuracy；parse 为训练 rollout 的 parse success。

| 实验/窗口 | Reward | Parse | Entropy | Grad norm | Response len |
|---|---:|---:|---:|---:|---:|
| standard r8, 1-10 | 0.3482 | 0.5359 | 0.8931 | 0.0036 | 6439.5 |
| standard r8, 41-50 | 0.3574 | 0.5436 | 0.8933 | 0.0035 | 6420.0 |
| standard r8, 91-100 | 0.4023 | 0.5826 | 0.8753 | 0.0037 | 6080.3 |
| random-B r8, 41-50 | 0.3924 | 0.5822 | 0.9175 | 0.0631 | 6424.1 |
| full-grad mean r8, 41-50 | 0.3859 | 0.5711 | 0.9140 | 0.0589 | 6433.9 |
| full-grad mean r8, 91-100 | 0.4510 | 0.6469 | 0.9309 | 0.0624 | 6006.6 |
| centered cov eq-r28, 41-50 | 0.3854 | 0.5676 | 0.8592 | 0.0711 | 6382.0 |
| centered cov eq-r28, 91-100 | 0.4588 | 0.6600 | 0.8078 | 0.0727 | 5937.7 |
| uncentered cov eq-r28, 41-50 | 0.3867 | 0.5803 | 0.9027 | 0.0712 | 6360.7 |
| uncentered cov eq-r28, 91-100 | 0.4621 | 0.6719 | 0.8901 | 0.0715 | 5907.5 |
| full-grad mean r32, 91-100 | 0.4586 | 0.6826 | 0.9145 | 0.0737 | 5807.5 |
| full-grad mean r32, 141-150 | 0.4145 | 0.6922 | 0.9423 | 0.0695 | 5931.6 |
| original random-B repro, 1-10 | 0.3570 | 0.5439 | 0.9068 | 0.0686 | 6431.1 |
| original random-B repro, 41-50 | 0.3926 | 0.5783 | 0.9383 | 0.0661 | 6447.1 |
| historical random-B, 1-10 | 0.3162 | 0.4645 | 0.9476 | 0.0697 | 6543.1 |
| historical random-B, 91-100 | 0.4488 | 0.6465 | 0.9138 | 0.0722 | 5875.7 |
| historical random-B, 261-270 | 0.4463 | 0.7707 | 0.9300 | 0.0771 | 5420.9 |

关键训练现象：

- standard r8 在 step 51-100 的训练 reward 明显上升，但 full-benchmark 从 0.3231 降到 0.3158，属于最清楚的“训练进步、泛化退化”。
- full-gradient mean r8 的 91-100 reward 达 0.4510，评测却从 0.3586 降到 0.3402。梯度均值子空间内可以继续优化训练目标，但未带来测试收益。
- centered 与 uncentered 的训练 reward 几乎无法区分；uncentered 后期 reward 甚至略高，但评测更低。训练 reward 不能用于选择 covariance estimator。
- centered entropy 到 91-100 降至约 0.808，但没有数值发散，parse 和 Macro Avg 都提高；当前不能把单纯 entropy 下降等同于 collapse。
- full-gradient mean r32 到 150 步仍无 KL、clip fraction、gradient norm 异常；其局限更像泛化目标不匹配，而不是优化器不稳定。
- latest original random-B 复现的早期 reward 比历史日志更高，但 step-50 评测没有复现历史机器上“明显领先”的印象，说明 rollout/probe 随机性或旧机器未知配置仍是重要变量。

## 10. Probe 与 rank 结构

### 10.1 Original random-B 复现

```text
modules = 196
observations/module = 12
rank counts = {8: 3, 28: 1, 32: 192}
rank min/mean/max = 8 / 31.6122 / 32
overlap mean / p10 / min = 0.84935 / 0.78756 / 0.76686
retained energy mean / min = 0.74886 / 0.49957
```

历史旧机器 rank sum 为 6200、mean 31.6327；本次复现 rank sum 为 6196、mean 31.6122。rank allocation 高度复现，但实际子空间 overlap 只有约 0.85，不能认为具体方向也被精确复现。

### 10.2 Token-mask covariance adaptive eq-r28

centered 与 uncentered 的平均 rank 都为 29.7551。attention 的 k/q/v/o 基本都分到 r32，rank 压缩主要发生在 MLP gate/up/down：

| family | centered mean rank | uncentered mean rank |
|---|---:|---:|
| down_proj | 29.000 | 28.857 |
| gate_proj | 26.107 | 26.214 |
| up_proj | 25.179 | 25.214 |
| k/q/v/o | 32.000 | 32.000 |

这说明 allocator 的“自适应”主要是在参数成本高的 MLP 上减少 rank。它比 original random-B 更有结构差异，但仍然接近高 rank。

### 10.3 SPAR adaptive eq-r8

SPAR 的平均 rank 13.7398 看起来高于 8，但严格满足 uniform-r8 的参数预算。原因是不同模块每增加一个 rank 的参数成本不同：allocator 给便宜的 attention k/v 更多 rank，给昂贵的 MLP 更低 rank。

```text
rank min/mean/max = 2 / 13.7398 / 32
calibration energy capture = 0.880820
uniform-r8 capture = 0.860903
adaptive reward AUC1:50 = 0.374336
uniform reward AUC1:50 = 0.373477
```

训练 AUC 只高 0.000859，full-benchmark 只有 adaptive 的结果，缺少同协议 uniform full-benchmark，因此不能证明 adaptive allocation 改善泛化。

## 11. 关键对比与中间判断

### 11.1 初始化方向与 rank

```text
random-B r8 - standard LoRA r8, step50 = +0.035212
original random-B near-r32 - random-B r8, step50 = +0.008981
```

这组结果不支持“退化完全由 rank-8 导致”。更合理的描述是：标准 LoRA r8 的随机初始方向和早期路径依赖较差；探测初始化在相同 rank 下已经恢复大部分性能，增大 rank 再提供较小增益。

### 11.2 Mean gradient 与二阶统计

mean-gradient 依据 `mu^T mu`，其中 `mu=mean(G)`；不同 prompt 上符号变化的方向会抵消。uncentered covariance 依据 `mean(G^T G)`，保留反复出现但符号变化的梯度能量；centered covariance 再减去 `mu^T mu`，强调 prompt 间变化。

full-gradient mean r32 step 150 达到 0.375749，说明 mean 子空间并非不可训练；但训练到 150 步仍只与 centered step 50 相当，且低于 centered step 100。这支持“均值方向更容易拟合共同训练模式，而 covariance 更可能覆盖任务多样性”的解释，但 r32 与 r8 缺少同 step 评测，整个比较也尚未达到多 seed 因果标准。

### 11.3 Centered 与 uncentered

```text
step50: 0.375451 - 0.350770 = +0.024681
step100: 0.382131 - 0.369779 = +0.012352
```

这是当前最强的候选信号，但必须保留两个限定：

1. centered 只 mask discovery，uncentered mask discovery+calibration；两组不是纯 centering ablation。
2. centered step 50→100 没有 optimizer shard，续训恢复权重、global step、RNG 和 data position后重建优化器；uncentered 1→100 保存并恢复完整 optimizer，是连续轨迹。

因此可以把 centered 作为下一轮主候选，不能把差值全部归因于数学中心化。

### 11.4 Token mask

当前没有“centered covariance + no mask”和“centered covariance + 同密度随机 mask”的严格对照。token mask 的因果贡献尚未被隔离。保留它的理由是机制合理且当前最好方法使用了它，不是因为已有实验已独立证明它有效。

### 11.5 长度与 parse

高分方法通常伴随更高 parse、更低平均长度和更低 hit-max，但并非单调对应。full-gradient mean r32 step 150 的长度最短、hit-max 最低，Macro Avg 仍低于 centered step 100；因此格式改善解释不了全部能力差异。

## 12. Checkpoint 与优化器连续性

| 实验 | 保留 step | optimizer shard | 续训解释 |
|---|---|---|---|
| original random-B reproduction | 50 | **有，4 份** | 可从 step 50 完整恢复 |
| token-mask uncentered covariance | 50、100 | **有，4 份** | 1→100 为连续优化器轨迹 |
| token-mask centered covariance | 50、75、100 | 无 | 50→100 重建 optimizer |
| standard LoRA r8 | 50、75、100 | 无 | 50→100 重建 optimizer |
| full-gradient mean r8 | 50、100 | 无 | 续训包含 optimizer reset 混杂 |
| full-gradient mean r32 | 25、50、75、100、125、150 | 无 | 1→150 单进程连续训练，未中途恢复 |
| random-B r8 | 50 | 无 | 仅用于 50-step 对照 |
| windowed consensus r8 | 50 | 有 | 保存 model/optimizer/extra |

original random-B step 50 的专用恢复脚本为 [`scripts/local/resume_original_random_b_from_step50.sh`](../scripts/local/resume_original_random_b_from_step50.sh)，已检查 4 份 model、optimizer、extra shard 和 `data.pt`。

## 13. Artifact 索引

### 13.1 训练日志

```text
/root/peft-for-rl/runs/standard-lora-rank8-v1/logs/verl/
/root/peft-for-rl/runs/random-b-rank8-v1/logs/verl/
/root/peft-for-rl/runs/random-b-original-repro-v1/logs/verl/
/root/peft-for-rl/runs/full-gradient-v1/logs/verl/
/root/peft-for-rl/runs/full-gradient-rank32-v1/logs/verl/
/root/peft-for-rl/runs/windowed-adam-consensus-v1/logs/verl/
/data/peft-for-rl-runtime/logs/verl/
```

### 13.2 Probe artifact

```text
/root/peft-for-rl/runs/random-b-original-repro-v1/analysis/rl_gradient_probe_b16n8_w8_cap64_e95_s6to12_seed42_original_repro_v1/
/root/peft-for-rl/runs/random-b-rank8-v1/analysis/random_b_energy_probe_b16n8_w8_s6to12_seed42_v2/
/root/peft-for-rl/runs/full-gradient-v1/analysis/full_gradient_signed_grpo_probe_r32_d16c16a8_seed42/
/root/peft-for-rl/runs/windowed-adam-consensus-v1/analysis/windowed_adam_consensus_probe_b64n8_seed42_v2/
/data/peft-for-rl-runtime/analysis/spar_v0_positive_probe_seed42/
/data/peft-for-rl-runtime/analysis/full_gradient_tokenmask_top50_cov_centered_r32_d64c32a16_seed42_v1/
/data/peft-for-rl-runtime/analysis/full_gradient_tokenmask_top50_cov_uncentered_r32_d64c32a16_seed42_v1/
```

### 13.3 评测结果

除 original random-B 复现外，统一 summary 位于：

```text
/data/peft-for-rl-runtime/outputs/full_bench_eval/<eval-name>/summary/<eval-name>.json
```

original random-B 复现 summary 位于：

```text
/root/peft-for-rl/runs/random-b-original-repro-v1/outputs/full_bench_eval/
  gradtop_original_random_b_r8to32_step50_fullbench_32768_seed42_repro_v1/summary/
  gradtop_original_random_b_r8to32_step50_fullbench_32768_seed42_repro_v1.json
```

每个 summary 旁保留 CSV；records/shards 用于重新聚合或做 paired bootstrap。

## 14. 尚不能下的结论

1. 不能说“rank-8 本身必然退化”。现有证据同时包含初始化方向、dropout、probe 和续训状态差异。
2. 不能说 token mask 已独立有效，因为没有相同 covariance/rank 下的 no-mask 和 random-mask 对照。
3. 不能说 centered 的全部优势来自去均值，因 calibration mask scope 和 optimizer continuity 未完全对齐。
4. 不能说当前 adaptive rank 优于 uniform equal-budget。original random-B 几乎是 uniform-r32；token-mask covariance 没有 equal-budget uniform-r28/r30 full-benchmark；SPAR 缺少 uniform-r8 full-benchmark。
5. 不能把旧机器 historical random-B 的单次强结果视作稳定可复现结论。本机重现了 rank map，没有重现明显领先的评测优势。
6. 不能用训练 reward、训练 parse 或单点 entropy 代替 full-benchmark 泛化评测。

## 15. 文献检索前的最小对照建议

以下是仅依据本机实验得到的初步矩阵。结合近期研究后的正式优先级见第 19 节；第 19 节保留这里的严格控制原则，但进一步拆分了 token selector、初始化覆盖与动态 rank，后续执行以第 19 节为准。

下一轮应先固定 uniform-r32，取消 allocator 混杂，只比较：

| 组 | covariance | token scope |
|---|---|---|
| A | centered | top-surprisal |
| B | centered | 同密度随机 mask |
| C | centered | no mask |
| D | uncentered | top-surprisal |

四组必须使用相同 probe prompt、mask scope、dropout、训练 seed、完整 optimizer checkpoint 和 step 50 评测。确定最佳 estimator/mask 后，再比较 uniform r8/r16/r24/r32，最后才比较 equal-parameter adaptive allocation。

所有新实验都应从启动时设置总目标 270，并在 25/50/100/150/270 保存 `model + optimizer + extra + data`。即使中途评测或停止，也只能从完整状态继续，避免再次引入 optimizer reset。

## 16. 近期相关研究跟进

本节检索截至 2026-09-30。优先引用论文原文和正式会议版本；2026 年 9 月刚发布、尚缺少独立复现的工作明确标为预印本，主要用于提出可检验假设，不直接视为已经成立的结论。

### 16.1 初始化不是实现细节，而是在选择早期优化几何

| 工作 | 状态 | 核心观点 | 对本项目的直接意义 |
|---|---|---|---|
| [PiSSA](https://arxiv.org/abs/2404.02948) | NeurIPS 2024 Spotlight | 用预训练权重的主奇异方向初始化可训练部分 | 说明初始化方向足以改变收敛速度和终点，但 weight-SVD 方向不一定等于 RL 任务方向 |
| [LoRA-GA](https://arxiv.org/abs/2407.05000) | 2024 预印本 | 用初始 full gradient 使 LoRA 的首步更新逼近全参梯度 | 与本机 full-gradient probe 最接近，但论文主要使用一次任务梯度，没有解决小 probe 集合的泛化误差 |
| [LoRA-One](https://arxiv.org/abs/2502.01235) | ICML 2025 Oral | 理论上分析 LoRA 因子向一步全梯度奇异子空间对齐，并使用谱初始化和预条件 | 支持“早期子空间与条件数影响后续路径”，也提示只取噪声梯度的 top-k 可能把模型送入错误子空间 |
| [EVA](https://arxiv.org/abs/2410.07170) | NeurIPS 2025 | 对下游激活做增量 SVD，用 explained variance 初始化并跨层重分配 rank | 提供一个不依赖 reward 梯度符号、样本效率更高的对照；其 rank 分配依据是激活覆盖，不是训练 reward |
| [TaRA](https://arxiv.org/abs/2609.02639) | EMNLP 2026 Main | 直接优化低秩因子诱导梯度与 full-weight 梯度的保真度 | 最新证据仍把 gradient fidelity 视为关键，但论文结论尚未在 GRPO 高方差梯度上验证 |

这些工作与本机结果方向一致：standard LoRA r8 到 random-B probe r8 的 `+0.035212`，比 random-B r8 到近似 r32 的 `+0.008981` 更大，表明“从什么方向开始”至少和“有多少方向”同样重要。不过，文献中的 full gradient 通常来自监督损失或较稳定的批梯度，本项目使用的是带组相对优势的 on-policy RL 梯度，方差结构不同，不能直接照搬一次 SVD top-k。

### 16.2 随机投影、换基与优化器状态

| 工作 | 状态 | 核心观点 | 对本项目的直接意义 |
|---|---|---|---|
| [Flora](https://arxiv.org/abs/2402.03293) | ICML 2024 | 早期 LoRA 动态可近似为梯度的随机投影；周期重采样投影可累积高秩更新 | 为 random-B 的稳健性提供机制解释：随机基不依赖一个小 probe 的经验 top eigenspace，并能较均匀覆盖方向 |
| [GaLore](https://arxiv.org/abs/2403.03507) | ICML 2024 Oral | 周期性更新低秩梯度投影，减少优化器状态内存 | 支持“子空间不应永久锁死”，但它训练 full weight，不等同于当前可训练 LoRA A/B |
| [COAP](https://arxiv.org/abs/2412.00071) | CVPR 2025 | 更新投影时显式考虑相邻投影间相关性 | 说明换基本身会破坏累计信息，和本机反复关注的 optimizer continuity 是同一个问题 |
| [LoRA-Pro](https://arxiv.org/abs/2407.18242) | ICLR 2025 | 将 LoRA 优化解释成低秩 weight-space 梯度，并校正因子梯度去逼近全参梯度 | 提醒我们不能只看初始化 A 的谱，实际由 A/B 因子优化诱导出的 weight-space 更新也可能失真 |
| [AltLoRA](https://arxiv.org/abs/2505.12455) | NeurIPS 2025 | 交替投影以更稳定地近似全梯度并容纳 momentum | 提供了比同时更新两个因子更可控的优化思路，但属于下一阶段优化器改造，不应混入首轮初始化消融 |
| [LoFT](https://arxiv.org/abs/2505.21289) | 2025 预印本，2026 修订 | 不仅投影梯度，也校准 Adam 一、二阶矩在变化子空间中的表示 | 直接支持本机观察：清空 optimizer 或改变子空间却不搬运 moments，会产生独立于方法本身的性能混杂 |

Flora 尤其值得重视。若写成 `Delta W = B A`、标准初始化为 `B_0=0`，首步有

```text
grad_B = G A_0^T
grad_A = B_0^T G = 0
Delta W_1 approximately equals -eta G A_0^T A_0
```

随机 `A_0` 使 `A_0^T A_0` 在期望上接近各向同性缩放；它不是准确找到某个任务方向，而是以较小的结构偏见压缩梯度。相反，probe-SVD 初始化用的是经验投影器 `P_hat`。若 probe 数量小、rollout 噪声大，则 `G P_hat` 可能精确拟合 probe 的主方向，却系统性丢掉未在 probe 中稳定出现的方向。当前 A 后续可训练，所以这不是永久约束，但首批更新、Adam moments 和后续采样分布已经产生路径依赖。

因此，random-B 的优势不一定来自“随机方向比梯度方向更正确”，更可能来自以下三点：

1. 各向同性随机投影的估计偏差低，虽然方差较高；小样本 top-k 的方差和选择偏差都可能很高。
2. random-B 保留宽谱覆盖，而 hard top-k 会放大经验协方差最大特征值对应的噪声。
3. on-policy 训练会改变数据分布；静态 probe 子空间只适配启动分布，随机基对分布漂移没有如此强的先验承诺。

### 16.3 nominal rank、有效 rank 与自适应分配

| 工作 | 状态 | 核心观点 | 对本项目的直接意义 |
|---|---|---|---|
| [AdaLoRA](https://arxiv.org/abs/2303.10512) | ICLR 2023 | 用 SVD 参数化和重要度分数逐步裁剪奇异分量 | 支持“先给足容量，再渐进裁剪”，而不是从很小 rank 启动 |
| [rsLoRA](https://arxiv.org/abs/2312.03732) | 2023 | 高 rank 下 `alpha/r` 会使学习信号随 rank 衰减，建议 `alpha/sqrt(r)` | rank 对照必须同时记录实际 scaling；本项目保持 `alpha/r=2`，因此已有 r8/r32 对照没有该种随 rank 衰减，但动态减 rank 时不能突然改变 scale |
| [What Does the Rank Buy?](https://arxiv.org/abs/2609.32002) | 2026-09 预印本 | 在逐因子范数约束下，rank 不直接控制最坏情况复杂度，而主要控制可达更新和抵消谱方向的成本 | 与本机“r8 不是唯一原因”一致；不能把 nominal rank 简单解释成泛化容量 |
| [Tight Sample Complexity for LoRA](https://arxiv.org/abs/2607.27680) | 2026 预印本 | 在局部二次、未正则 ERM 假设下复杂度可随 `r*d/n` 增长；若先做 nuclear-norm 收缩再截断，过量 rank 可以无害 | 与上一工作并不矛盾，而是约束假设不同；启发是高 rank 启动必须配合显式收缩/门控，而不是无约束保留所有分量 |
| [ISO-LoRA](https://arxiv.org/abs/2609.12123) | 2026-09 预印本 | AdamW 可能把同一 nominal rank 的单步更新压成低有效 rank，优化器决定 rank 是否真正被利用 | 本项目必须开始记录有效 rank；仅报告 `r=32` 不能证明模型使用了 32 个方向 |
| [`ell_p`-LoRA](https://arxiv.org/abs/2609.28998) | 2026-09 预印本 | 对 rank-one 分量能量施加 `0<p<1` 的稀疏正则，使冗余分量通过近端阈值消失 | 为“保持最大 shape 和 optimizer state、用软门控逐渐稀疏”提供了比物理删 tensor 更合适的实现方向 |

最新理论的共同点不是“rank 越大越好”或“越小越泛化”，而是需要区分三件事：

```text
nominal rank: 配置中允许的最大 rank
effective rank: 实际 Delta W 或单步更新的奇异谱宽度
statistical rank: 在有限数据和正则下能稳定估计的方向数
```

本机 original random-B 的 mean rank 为 `31.6122`，但目前没有记录 `Delta W` 的 stable rank 或谱熵 rank，因此尚不能判断它实际是否只使用了少数方向。后续应至少记录：

```text
stable_rank(M) = ||M||_F^2 / ||M||_2^2
entropy_rank(M) = exp(-sum_j p_j log p_j),  p_j = sigma_j / sum_k sigma_k
```

其中 `M` 同时取累计权重更新 `Delta W` 和当前 optimizer-induced weight update。若 nominal r32 的 entropy rank 长期只有 6--10，继续讨论静态 r28/r32 分配的意义有限，应先解决 rank utilization。

### 16.4 Token 选择：top-surprisal 不是充分统计量

当前实现对每条 response 保留固定比例 token，先保留结尾 token，再按 sampled-token surprisal `-log p(y_t)` 从高到低补齐。该方法可复现且成本低，但近期 RLVR 研究指出两个具体问题。

| 工作 | 状态 | 核心观点 | 对当前 mask 的启发 |
|---|---|---|---|
| [NAT](https://arxiv.org/abs/2603.06619) | 2026 预印本 | token 子采样配合 Horvitz-Thompson 权重可得到无偏的 partial-token policy gradient | 同密度随机 mask 不只是弱 baseline，也是区分“稀疏化效果”和“语义选择效果”的必要对照 |
| [RSI-S](https://arxiv.org/abs/2606.31575) | 2026 预印本 | 只看 sampled-token probability 或 entropy 都不充分；过滤低信息 token 的同时应排除极端高 surprisal 尾部 | 当前 top-surprisal 恰好会优先保留 RSI-S 认为可能不稳定的尾部 token |
| [GMTS](https://arxiv.org/abs/2608.30632) | EMNLP 2026 Findings | token 梯度为 `omega_t * grad log p_t`，重要度近似应包含 advantage、clip/KL 状态与 entropy，而不是只看不确定度 | 当前每条 response 固定 token 配额会忽略不同 rollout 的优势幅度；零/小优势 response 仍占相同 token 预算 |
| [STARE](https://arxiv.org/abs/2606.19236) | 2026 预印本 | surprisal 与 advantage 符号共同决定 token 对 entropy 的局部作用 | mask 应按正负优势分层审计，避免只收集推动 entropy 单向变化的梯度 |
| [Locked at the Entrance](https://arxiv.org/abs/2608.29188) | 2026 预印本 | 多个数学 RLVR 实验的覆盖度收缩主要发生在推理早期分支 token | 当前强制保留结尾 token 有利于答案格式，却可能低估决定解题路径的早期分支方向 |

GMTS 将 token 梯度写成：

```text
grad l_t = omega_t * grad log pi(y_t | x, y_<t)
```

在本项目 KL 关闭、probe 时新旧策略接近且未触发 clipping 的近似下，`omega_t approximately equals advantage`。因此只使用 `-log p(y_t)` 不能反映真实梯度强度。一个更贴近本项目、但仍便宜的 selector 应至少同时看：

```text
abs(advantage) * token_entropy
```

并对极端 sampled surprisal 做上尾截断。更完整的 RSI 需要分布 entropy `H_t`，其定义为：

```text
RSI_t = 1 + log p(y_t) / H_t
```

首轮不建议把这些 mask 直接用于正式 GRPO 更新，否则会同时改变训练目标。它们只应用于 probe discovery；calibration/audit 继续使用完整 token 梯度，从而检验“选择后的子空间能否解释未筛选梯度”。

### 16.5 RL 动态文献只用于诊断，不并入首轮方法

[Dr. GRPO](https://arxiv.org/abs/2503.20783) 指出 GRPO 存在响应长度方向的优化偏差；[OPEFO](https://arxiv.org/abs/2605.11491) 从 token entropy flow 解释 entropy collapse，并按 entropy-increasing/decreasing 更新做平衡。这与本机“reward 升高但 full-benchmark 不同步”和部分实验 entropy 更快下降相符。

但下一轮目标是隔离初始化子空间和 adaptive rank，暂不修改 GRPO loss、优势估计、entropy bonus 或采样策略。否则即使评测提高，也无法判断来自子空间还是 RL 算法变化。当前只新增 entropy-flow 分层日志：按 advantage 正负、token 位置分桶记录被 mask 的比例、surprisal、entropy 和梯度能量。

## 17. 统一数学解释与新的核心假设

### 17.1 Mean、uncentered covariance 与 centered covariance 分别在优化什么

对模块 `l`，记第 `i` 个 prompt group 产生的矩阵梯度为 `G_i`：

```text
mu = E[G_i]
C_unc = E[G_i^T G_i]
C_ctr = E[(G_i-mu)^T (G_i-mu)] = C_unc - mu^T mu
```

mean 方法取 `mu` 的 top singular directions，偏好跨 prompt 同号的共同下降方向；uncentered 同时包含 mean energy 和变化能量；centered 去掉共同均值后，偏好 prompt 间变化最大的方向。它可以增加任务覆盖，但“变化大”也可能只是 rollout 噪声大，所以 centered 并不天然等于泛化更好。

本机 centered 最好结果提出的新假设应表述为：

> 数学任务的可迁移更新可能分布在跨 prompt 的多样方向中，单一均值会发生符号抵消；但需要从 centered covariance 中进一步剔除 rollout 噪声，才能判断其是否是真实的 prompt-level 信号。

### 17.2 用 GRPO 的组结构分离信号与噪声

每个 prompt 已有 8 个 rollout，可以把它们随机分成两个独立半组，构造 `G_i^(a)` 和 `G_i^(b)`。建议新增对称 cross-fit covariance：

```text
C_cross = (1 / 2N) * sum_i [
    (G_i^(a)-mu_a)^T (G_i^(b)-mu_b)
  + (G_i^(b)-mu_b)^T (G_i^(a)-mu_a)
]
```

若两个半组的 rollout 噪声近似条件独立，交叉项会压低只在单个半组出现的噪声，保留可跨 rollout 重现的 prompt-level 方向。这里不是严格独立：同组 rollout 共享归一化 advantage。实现时应先用完整 8-response group 固定 advantage，再重复多种确定性 4/4 拆分；只有跨拆分稳定的谱才视为信号。`C_cross` 不保证半正定，因此只取稳定的正特征值，并用独立 calibration prompts 检查 held-out capture。

另一个等价诊断是分别估计 between-prompt 和 within-prompt 能量：

```text
C_between = Cov_i(E_j[G_ij | prompt_i])
C_within  = E_i[Cov_j(G_ij | prompt_i)]
```

通过广义特征值 `C_between v = lambda (C_within + epsilon I) v` 选择高 signal-to-noise 方向。首版可先实现 cross-fit，计算和存储更简单；广义特征值版本作为后续升级。

### 17.3 不在 random 与 top-k 之间二选一

当前实验把 random-B 和 deterministic top covariance 当成两个离散方案，更合理的是构造连续谱。设 cross-fit covariance 的稳定特征向量为 `U`，最大 rank 为 32：

```text
A_0 = concat(U[:, :k]^T, R_perp),  k in {0, 8, 16, 24, 32}
```

`R_perp` 是在 `U[:, :k]` 正交补中生成的随机基。于是：

- `k=0` 是纯 random-B；
- `k=32` 是纯数据子空间；
- 中间值保留可重复信号，同时用随机方向覆盖 probe 未见分布。

这比把协方差 eigenvalue 与单位阵做普通 shrinkage 更有效，因为 `C + lambda I` 不改变 eigenvectors，不能缓解 hard top-k 的方向选择误差。混合基或按谱温度随机采样才能真正改变覆盖范围。

### 17.4 初始化质量应看 held-out fidelity，不看 discovery energy

每个候选子空间 `P` 至少报告以下三个指标：

```text
discovery_capture = ||G_discovery P||_F^2 / ||G_discovery||_F^2
audit_capture     = ||G_audit P||_F^2 / ||G_audit||_F^2
capture_gap       = discovery_capture - audit_capture
```

同时报告不同 prompt split、rollout half 和 probe seed 之间的 principal-angle overlap。高 discovery capture、低 audit capture 或 overlap 不稳定，正是 probe 子空间过拟合，不能进入正式 270-step 训练。random-B 应用相同 audit 指标，作为各向同性覆盖的基准，而不是只和 top-k 比训练 reward。

## 18. 建议的方法设计

### 18.1 初始化：cross-fit signal + random complement

下一版初始化的最小实现应包含：

1. 仍以 prompt group 为统计单位，避免把同一题的 8 个高度相关 rollout 当成 8 个独立样本。
2. discovery 使用 advantage-aware、截断极端尾部的 token mask；calibration 和 audit 始终 no-mask。
3. 同一 prompt 的 rollout 拆半，估计 `C_cross`，只保留跨半组同号的正谱。
4. 用 held-out audit capture 和 split overlap 决定信号分量数 `k`，剩余 rank 用正交 random complement 补齐到 r32。
5. A 保持可训练；不把该方法描述成固定子空间。正式训练期间记录 A 相对初始化子空间的 principal angles，观察它何时以及多快旋转。

这个设计同时保留当前两个最有价值的组件：centered covariance 对任务多样性的覆盖，以及 random-B 对小样本估计误差的鲁棒性。

### 18.2 Adaptive rank：高 rank 启动，门控而非删 tensor

首版动态 rank 不建议在 step 50 物理改变 LoRA tensor shape。物理裁剪会要求重新构建参数和 Adam moments，重新引入已经确认的 optimizer-reset 混杂。建议保持每层最大 r32，并给每个 rank-one atom 增加 gate `z_lj`：

```text
Delta W_l = sum_j z_lj * b_lj * a_lj^T,  z_lj in [0, 1]
```

训练阶段按以下时间表渐进收缩：

```text
step 1-50:   所有 z=1，允许高 rank 探索
step 51-75:  根据 EMA utility 开始软衰减低分 atom
step 76-100: 达到目标参数预算，但保留完整 tensor 和 optimizer state
step >100:   gate 固定；只在最终导出时物理压缩
```

utility 不使用训练 reward，而组合三个量：

```text
u_lj = heldout_gradient_capture_lj
       * cross_window_stability_lj
       * function_space_energy_lj
       / parameter_cost_l
```

其中 `function_space_energy` 在 calibration activation 上测量关闭该 atom 对 `Delta W h` 的影响。为避免 LoRA 因子存在 `B A = (B R)(R^-1 A)` 的 gauge ambiguity，atom 分数至少采用对 reciprocal scaling 不变的 `||b_lj|| * ||a_lj||`，并定期记录 `Delta W` 的奇异谱。若门控分数仍因因子旋转不稳定，再升级为 AdaLoRA 风格的 SVD 参数化；不要在首版同时更换参数化和 allocator。

动态 rank 改变 active rank 时，LoRA scaling 固定按最大 rank 的启动值，不随 active rank 重算，避免 gate 变化同时引入整个 adapter 的幅度跳变。

### 18.3 这一轮明确不加入的组件

以下方法有研究价值，但首轮不加入：

- 不修改 GRPO advantage、clip、KL、entropy bonus 或采样温度；
- 不采用 LoRA-Pro、AltLoRA、LoFT 或 Muon 等新优化器；
- 不周期性整体重采样已经承载 Adam 状态的 LoRA 基；
- 不依据 full-benchmark 测试结果选择 probe 超参数；
- 不同时改变 covariance、mask、rank budget 和 optimizer continuity。

这些限制是为了保证下一轮能够回答因果问题，而不是因为上述组件无效。

## 19. 修订后的下一步实验计划

### 19.1 Phase 0：只做 probe 诊断，不启动正式训练

用相同 64 个 discovery prompts、32 calibration、16 audit 和固定 rollout 缓存，生成以下诊断：

| 编号 | estimator | discovery token selector | calibration/audit |
|---|---|---|---|
| P0 | centered | no mask | no mask |
| P1 | centered | 同密度随机 mask | no mask |
| P2 | centered | 当前 top-surprisal + final tokens | no mask |
| P3 | centered cross-fit | advantage-aware + RSI stable-band | no mask |

必须先输出 `audit capture`、`capture gap`、跨 split overlap、cross-half overlap 和谱有效 rank。P3 若不能同时优于 P2 的 audit capture 和稳定性，不进入训练。uncentered 可由相同未中心化统计离线导出，避免重复 rollout；先用它确认严格相同 mask/split 下 centering 的纯影响。

#### 19.1.1 正式结果（2026-09-30）

正式 run 使用 `all-linear`、candidate rank 32、sketch width 40、3 个 cross-fit splits、8 responses/prompt 和固定 seed 42，完整收集了 64 discovery、32 calibration、16 audit，共 112 份不相交的 rollout cache。产物位于：

```text
runs/phase0-gradient-diagnostics-v1/analysis/
  phase0_shared_rollout_d64c32a16_r32_seed42_v1/
```

目录大小约 824 MiB。`probe_summary.json` 的 SHA-256 为 `331fcf20de4f543ffecf307d643eb8a945878d429eed741fd3eaf713ef12b8cc`；`candidates_P3.safetensors` 的 SHA-256 为 `32cb617bb05c7a2f2261a2fb9af2afc9ac4277f295a77a15f40b028c6be1c2f7`。7 个候选均通过 artifact validator，每个包含 196 个模块，最大正交误差为 `3.34e-6`。三个 prompt split 完全不相交，112 份 cache 均存在，正式 diagnostics 没有 NaN 或 Inf。

| 方法 | discovery capture（Nystrom） | calibration capture | audit capture | capture gap | prompt-split mean cosine | stable rank | entropy rank | positive rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 | 0.512337 | 0.571205 | 0.549364 | -0.037027 | 0.717728 | 2.788 | 9.919 | 32.000 |
| P0 uncentered | 0.512378 | 0.571248 | 0.549398 | -0.037019 | 0.718630 | 2.784 | 9.901 | 32.000 |
| P1 | 0.505897 | 0.570243 | 0.549561 | -0.043663 | 0.707911 | 2.763 | 10.035 | 32.000 |
| P1 uncentered | 0.505937 | 0.570317 | 0.549604 | -0.043667 | 0.708836 | 2.761 | 10.021 | 32.000 |
| P2 | 0.564348 | 0.571859 | 0.549244 | 0.015105 | 0.723547 | 2.675 | 9.157 | 32.000 |
| P2 uncentered | 0.564387 | 0.571889 | 0.549286 | 0.015101 | 0.724443 | 2.672 | 9.142 | 32.000 |
| P3 | 0.200877 | 0.211234 | 0.207643 | -0.006766 | 0.318512 | 2.525 | 6.003 | 10.389 |

P3 的 cross-half mean cosine 为 `0.833855`（mean squared cosine `0.741604`），但 response-split mean cosine 只有 `0.247558`（mean squared cosine `0.134307`）。这个 cross-half 指标不能抵消其 held-out fidelity 和 prompt-split 稳定性的显著下降：P3 audit capture 比 P2 低约 62.2%，prompt-split mean cosine 也从 `0.723547` 降到 `0.318512`。P3 较小的绝对 capture gap 主要来自 discovery 和 audit capture 同时很低，不能解释为更好的可泛化子空间。

同 mask、同 split 下，P0/P1/P2 的 centered 与 uncentered audit capture 差异均小于 `4.4e-5`，因此这批数据不支持 centering 是主要差异来源。P0/P1/P2 的 audit capture 非常接近，P2 虽有最高 discovery capture，却没有提高 audit capture。

预注册判定为 **no-go**：P3 没有同时优于 P2 的 audit capture 和稳定性，因此不启动 Phase 1。Phase 2 的前提是另有证据表明 Phase 0 指标不能预测训练；本次只得到 probe 层面的明确否定结果，并没有该证据，所以也不以事后改规则的方式直接启动 Phase 2/3。

#### 19.1.2 Phase 0.5：P3 支持方向与 P2 补空间（预注册）

对正式 Phase 0 artifact 做事后诊断，但不从 audit 选择候选。诊断脚本为 `scripts/analysis/analyze_phase0_crossfit_support.py`，结果写入原 artifact 目录的 `phase0_crossfit_support_analysis.json`。P3 consensus 谱按

```text
spectrum > max(max(spectrum) * 1e-7, 1e-12)
```

定义数值支持。需要区分两种 rank：单个 response split 的正谱秩均值是 `10.389`，三路 split 合并后的 consensus 支持秩均值是 `25.755`（中位数 32）。后者较高不代表方向跨 split 稳定，可能只是三路低重合子空间的并集；这与 response-split mean cosine 仅 `0.247558` 一致。

离线结果显示，P3 支持方向贡献了 P3 自身 `97.73%` 的 calibration 能量和 `97.71%` 的 audit 能量，零谱完成方向确实主要是无效补全。然而，把每个模块的 P3 支持前缀与相同长度的 P2 前缀比较，P3 只有 P2 的 `37.30%` calibration 能量和 `38.20%` audit 能量；195 个可比较模块中，P3 在 calibration、audit 或两者同时胜过 P2 的模块数均为 0。P3 支持方向投影到 P2 rank-32 子空间内的平均能量比例为 `24.46%`，说明它们多数位于 P2 之外，但 held-out fidelity 较弱。

因此 Phase 0.5 不再测试纯 P3，而是在同一次 shared-rollout run 中预注册以下固定候选：

| 候选 | P3 consensus 前缀 | rank-32 其余方向 |
|---|---:|---|
| H0 | 0 | P2，且必须与 P2 artifact 位级一致 |
| H2 | 至多 2 个正谱支持方向 | 按 P2 discovery 顺序残差化补满 |
| H4 | 至多 4 个正谱支持方向 | 同上 |
| H8 | 至多 8 个正谱支持方向 | 同上 |
| H16 | 至多 16 个正谱支持方向 | 同上 |

若某模块的 P3 支持秩小于 `k`，只取实际支持方向；零谱完成方向不得计入 P3 信号。所有候选保持 rank 32，逐行二次正交化，并在相同无 mask calibration/audit 梯度上精确计分。H0 是管线一致性控制，若其 candidate、capture 或 atom score 与 P2 不一致，则本轮作废。

选择与停止规则在运行前固定如下：

1. 对 `k in {2,4,8,16}` 全部报告，不删除负结果；
2. calibration admissibility 要求 capture 不低于 H0 超过 `0.002`，且 prompt-split mean cosine 不低于 H0 超过 `0.02`；
3. 只在 admissible 集合中按 calibration capture 选择单个全局 `k*`；差值不超过 `5e-4` 时选更小的 `k`；
4. audit 不参与 `k*` 选择；只有 `k*` 同时比 H0 提高至少 `0.001` 的 calibration capture 和 `0.001` 的 audit capture，且 audit capture 不出现非有限值，才进入 step-50 训练；
5. 若没有 nonzero `k` 通过，则判定 P3 方向没有可用互补增益，不用 audit 另选 `k`，下一步转向 estimator/token-mask 正交消融。

正式重跑保持 Phase 0 的模型、seed、64/32/16 split、8 responses/prompt、rank/sketch width 32/40 和 rollout 配置不变。新增候选只发生在 discovery 统计完成后，不增加 backward 次数。

正式 run 结束后，`scripts/local/postprocess_phase05_hybrid_probe.sh` 依次执行 artifact 校验、上述预注册 gate 和训练 artifact 准备。`candidate_tensor_validation.json` 保存逐候选的形状、有限性和正交性轻量检查；随后 `scripts/analysis/audit_phase05_artifact.py` 还会逐个核验 112 个 cache、64/32/16 phase 顺序、prompt ID 不交叠、196 个模块、所有有限指标、hybrid rank 组成以及 H0/P2 张量与 capture 精确不变量，只有全部通过才写 `artifact_validation.json` 并进入 gate。若 gate 为 `no-go`，只写出 `phase05_training_preparation.json` 的跳过记录；若为 `go`，则由 `scripts/analysis/prepare_phase05_training_artifacts.py` 在 `training_allocations/` 下同时生成 `H0_uniform_r32` 与校准阶段选中的 `Hk_uniform_r32`。准备清单固定记录 gate、rank map 和 subspace 文件的 SHA-256；该步骤不启动训练，也不允许根据 audit 改选另一个 `k`。

通过 gate 后的 step-50 对照统一由 `scripts/local/start_phase05_hybrid_uniform_r32_4gpu.sh` 启动。脚本只接受 `H0` 或 gate 选中的非零 `Hk`，启动前重新验证严格审计状态、gate 哈希、allocation/rank-map/subspace 哈希、196 个 uniform-r32 模块、`alpha=64` 和正交性。训练固定使用 `b64/m16/n8`、`alpha/r=2`、dropout 0、step 25/50 checkpoint，并保存 optimizer state。为使采样 RNG 不再依赖异步请求先后，正式对照显式设置 per-trajectory rollout seed；seed 由训练 seed、global step、dataset index、rollout 序号和 train/validation 标志稳定派生。它不构成 CUDA/vLLM 跨进程逐字节确定性的保证，因此仍需多 seed 统计。默认不配置 rollout seed 的旧实验保持原采样行为。

##### 19.1.2.1 正式结果（2026-10-01）

正式 run 为 `phase05_hybrid_shared_rollout_d64c32a16_r32_seed42_v3`。严格审计确认了 `64/32/16` 个 discovery/calibration/audit cache、112 个唯一且不交叠的 prompt ID、196 个模块、896 个 cache tensor 以及 26022 个有限数值。H0 与 P2 的 candidate 和 atom-score 文件分别逐字节相同，calibration/audit capture 也精确相同。`probe_summary.json` 的 SHA-256 为 `f1bc8c3fc23b56d4853de84756bf7e6090136e7087f7fe0cf044e0ee8b6ce1bb`，`artifact_validation.json` 的 SHA-256 为 `ad583fb11b8302cb7ef1c7abd2b8279160027e88366660636b52aade32913144`。

| 候选 | calibration capture | audit capture | prompt-split mean cosine | 平均实际 P3 方向 | 相对 H0 calibration | 相对 H0 audit |
|---|---:|---:|---:|---:|---:|---:|
| H0 | 0.582496 | 0.546506 | 0.719763 | 0.000 | 0.000000 | 0.000000 |
| H2 | 0.581875 | 0.545901 | 0.687171 | 1.796 | -0.000621 | -0.000605 |
| H4 | 0.581079 | 0.545073 | 0.659721 | 3.520 | -0.001417 | -0.001433 |
| H8 | 0.578780 | 0.542790 | 0.612564 | 6.668 | -0.003716 | -0.003717 |
| H16 | 0.570143 | 0.533719 | 0.534702 | 12.429 | -0.012353 | -0.012787 |

五个候选的平均 P3 consensus 支持秩均为 `21.230`，但随着实际纳入的 P3 方向增多，calibration capture、audit capture 和 prompt-split overlap 全部单调下降。H2 的 capture 损失仍在 `0.002` 容差内，但 overlap 相对 H0 下降 `0.032592`，超过预注册容差 `0.02`；H4 也因 overlap 失败，H8/H16 同时因 capture 和 overlap 失败。因此 calibration admissible 集合为空，没有选择 `k*`，audit 未参与选择。预注册 gate 结论为 **no-go**，`phase05_training_preparation.json` 正确记录 `skipped_no_go`，不生成训练 allocation。这说明 P3 支持方向不是只因 rank-32 中的零谱补全而失败；即使只注入少量正谱方向，也没有可用的互补增益。

#### 19.1.3 Phase 0.6：cross-fit estimator 与 token mask 正交消融（仅 Phase 0.5 no-go 时执行）

若 Phase 0.5 为 `no-go`，不重新采样，也不同时常驻多套 cross-fit 统计。直接顺序重放 v3 的同一组 112 份 cache，每完成一种 discovery selector 就导出候选并释放 accumulator，最后只重放一遍 32/16 calibration/audit 梯度给全部候选评分。这样保证 prompt、response、advantage 和 held-out 梯度完全共享，同时把峰值主机内存限制在单套 cross-fit accumulator；当前单套 P3 的主 rank 已占约 153 GiB RSS，三套并存不满足内存安全边界。

预注册候选如下：

| 候选 | estimator | discovery token mask | 对照问题 |
|---|---|---|---|
| C0 | symmetric centered cross-fit | no mask | 与同 rollout 的 P0 比较，隔离 cross-fit estimator 本身 |
| C2 | symmetric centered cross-fit | top-surprisal + final 128 | 与同 rollout 的 P2 比较，隔离 estimator；与 C0 比较 top-surprisal mask |
| C3 | symmetric centered cross-fit | advantage-entropy stable-band | 必须复现原 P3；与 C2 比较 stable-band 的增量影响 |

C3 使用和 P3 完全相同的 split、mask、归一化、Nyström seed 与 consensus seed。若 C3 candidate、atom score 或全局 calibration/audit capture 未在预先声明的数值容差内复现 P3，则 replay run 作废。固定容差为 candidate 每行允许独立符号翻转后的最大绝对误差 `3e-5`、atom score `atol=1e-6, rtol=1e-5`、全局 capture 绝对误差 `1e-6`。P0/P2 直接引用 v3 source artifact，不从 audit 重建或改选。预注册判定实现为 `scripts/analysis/evaluate_phase06_crossfit_gate.py`。

所有 C 候选继续报告 calibration/audit capture、capture gap、prompt-split overlap、cross-half overlap、response-split overlap、支持秩和有效秩。诊断与停止规则在看到 Phase 0.5 audit 结果前固定如下：

1. C0 与 P0、C2 与 P2 分别给出 calibration/audit delta；C3 与 P3 只作为 replay invariant，不计作新收益；
2. cross-fit 候选若 calibration capture 低于对应 standard 候选超过 `0.002`，或 prompt-split mean cosine 低于超过 `0.02`，判为不可接受；
3. 在可接受的 C0/C2 中只按 calibration capture 选择，差值不超过 `5e-4` 时选 token mask 更简单的 C0；audit 不参与选择；
4. 只有 calibration 选中的候选同时比对应 standard 候选提高至少 `0.001` calibration capture 和 `0.001` audit capture，才允许进入 uniform-r32 step-50 训练；
5. 若 C0 相对 P0 和 C2 相对 P2 都显著下降，则证据指向 cross-fit estimator，而不是 stable-band mask；停止继续微调 cross-fit mask，转向 standard covariance signal + random complement 或训练中动态门控；
6. 若 C0 接近 P0、C2 接近 P2，但 C3 明显下降，则证据指向 stable-band mask；后续只保留 no-mask/top-surprisal，不再使用 stable-band。

顺序 replay 的 discovery backward 数为 `64 × 8 × 3 = 1536`，held-out backward 为 `48 × 8 = 384`，总计约 1920 次；比重新运行三次完整 Phase 0 明显更省，并且没有跨 run rollout 混杂。

实现入口为 `scripts/local/start_phase06_crossfit_replay_4gpu.sh`。启动器只接受已通过 `artifact_validation.json` 严格审计、且 `phase05_gate_decision.json` 明确为 `no-go`、`phase05_training_preparation.json` 明确跳过训练的正式 v3 artifact；它拒绝覆盖已有输出，并在启动前要求至少 190 GiB 可用主机内存。replay 不依赖新 rollout 的 advantage，trainer 中的实时 batch 仅作为一次 RPC/温度元数据载体，C0/C2/C3 的梯度全部来自固定 cache。

结束后由 `scripts/local/postprocess_phase06_crossfit_replay.sh` 调用 `scripts/analysis/audit_phase06_artifact.py`，严格核验 64/32/16 prompt、prompt ID 与 cache manifest 完全复用、196 个模块、rank/sketch/cross-fit 配置、C0/C2/C3 的预注册 method-mask 映射、token-mask 参数、score labels、scaling、shared-rollout 标志、张量形状/有限性/正交性以及全部诊断有限性；随后才调用预注册 gate。method-mask 映射或任一配置被改动时 audit 直接失败，不允许只依赖 launcher 声明。小模型顺序 replay、严格审计、gate 选择隔离和相关完整测试集已通过。

若 Phase 0.6 gate 为 `go`，`scripts/analysis/prepare_phase06_training_artifacts.py` 只生成 calibration 选中项及其匹配 standard baseline：C0 对 P0，C2 对 P2；两者均为 uniform-r32。准备清单固定 source/replay validation、gate、rank map、subspace 和 allocation summary 的 SHA-256，`scripts/local/start_phase06_crossfit_uniform_r32_4gpu.sh` 会在训练前重新核验。若 gate 为 `no-go`，准备清单只记录跳过且不生成训练 allocation。

自动训练分支另使用独立的 `phase06_training_provenance_contract_v1`：训练前原子写入并锁定 preparation、source/replay validation、gate、rank map、subspace、allocation、训练 seed 与固定超参，step 50 后记录 adapter 路径和 SHA-256。已有 adapter 只有在完整 contract 与当前 preparation、adapter 字节都匹配时才可复用。六基准评测复用还必须同时通过固定 snapshot SHA-256、7,248 条无重复 records、每集精确样本数、精确四个 shard manifest、adapter/base-model/snapshot 路径、request 数、基准集合与完整采样协议校验；采样协议固定为 eval seed 42、temperature 0.6、top-p 0.95、短集 32 samples、长集 4 samples、最大生成 32,768 tokens。manifest 文件名按评测器实际的 `shard-00-of-04` 至 `shard-03-of-04` 检查。

#### 19.1.4 Phase 0.6 正式运行失效记录（2026-10-01）

正式顺序 replay 完成了全部 `64/32/16` prompt，并导出 C0/C2/C3 的 196 个模块；`artifact_validation.json` 对 prompt ID、cache manifest、配置、有限性、张量形状和正交性检查为 `valid`。但是预注册 gate 在任何候选选择前检查 P3/C3 replay invariant 时失败，因此本轮没有 `phase06_gate_decision.json`，也没有生成训练 allocation 或启动 C 方法训练。

失效不是轻微越界：P3/C3 candidate 的 196 个模块中有 181 个超过逐行符号不变容差 `3e-5`，最大误差 `1.189915`；1372 个 atom-score 张量中有 1023 个未通过 `atol=1e-6, rtol=1e-5`，最大绝对误差 `0.531989`。全局 calibration capture 从 `0.1722970351` 变为 `0.1724865860`，差 `+1.8955e-4`；audit capture 从 `0.1678894802` 变为 `0.1679613174`，差 `+7.1837e-5`，也都超过预注册绝对容差 `1e-6`。因此不能在看到结果后放宽容差、改成 projector invariant，或把该轮追认为正式 `no-go`。

根因审计发现两个可复现性风险。首先，P3 的有效谱秩远低于名义 rank 32，弱谱/补全方向对微小梯度扰动非常敏感；P3/C3 的 64 个原始梯度范数平均相对差约 `2.44e-4`，但会放大为 basis 和逐 atom score 的明显差异。其次，Phase 0.5 source 运行期间 `verl/utils/full_gradient_rl_probe.py` 在磁盘上被修改，而 artifact 没有保存代码哈希；虽然已运行进程通常继续使用导入时的旧代码，但现有产物无法证明 source 与 replay 的实现字节一致。后续 probe artifact 必须记录代码文件哈希，不能只锁配置和数据 manifest。

`scripts/analysis/record_phase06_invalidation.py` 重跑结构审计并生成不可覆盖的 `phase06_invalidation.json`，明确记录 `status=invalid`、`decision=unavailable`、`phase06_training_authorized=false`。它没有伪造 gate。为决定是否允许既定的 P2+random fallback，它只使用预注册的 calibration capture 和 prompt-split overlap 容差；audit delta 仅报告，不参与授权。诊断值如下：

| replay | 对照 | calibration capture delta | audit capture delta（仅报告） | prompt-split mean cosine delta | calibration admissible |
|---|---|---:|---:|---:|---|
| C0 | P0 | -0.149727 | -0.147074 | -0.416451 | 否 |
| C2 | P2 | -0.309416 | -0.287585 | -0.413317 | 否 |

两种 mask 下的 cross-fit estimator 都远低于对应 standard estimator，且幅度远超容差，因此保守诊断为 `crossfit_estimator_failure`。这只是失效 run 的诊断证据，不是正式 Phase 0.6 gate 结果。作为明确标注的协议偏离，后续只允许使用已经严格审计的 source P2 构造 I0/I8/I16/I32；所有 C 方法永久禁止进入训练。Phase 1 preparation 和训练 contract 会锁定 invalidation 路径与 SHA-256，任何篡改都会在训练前被拒绝。

正式文件哈希为：

```text
probe_summary.json:
4d3c8189a65f85a5d2b85afa7880877e9b5aa02b99bd84a088e5e0343052cb92

artifact_validation.json:
b4e188ac2157c80abe42c7dad96372f5adee061211af72d0bda6dfb24ccb82c6

phase06_invalidation.json:
7eac808c53ad02adbc0a90ea5470736705bb8bbfb777c82a3cc8e6a614c4c5f4
```

### 19.2 Phase 1：uniform-r32 初始化覆盖消融

固定最佳 estimator/mask，取消 allocator，只改变 32 个初始化方向中数据方向的数量：

| 组 | data-stable directions `k` | random complement | 目的 |
|---|---:|---:|---|
| I0 | 0 | 32 | 确定性正交随机 A 子空间控制 |
| I8 | 8 | 24 | 弱数据信号、强覆盖 |
| I16 | 16 | 16 | 平衡方案 |
| I32 | 32 | 0 | hard data top-k 控制 |

这里的实现坐标必须写清：本仓库 `grad_subspace` 把 rank 个输入方向写入 LoRA `A`，将 `B` 置零，随后 A/B 都可训练。因此 I0 是与数据子空间同样行归一化的随机正交 `A`，不是历史 original random-B probe reproduction 的复用；后者还带有 probe-derived 方向、近似 adaptive rank 和 dropout 等差异，只能作为外部历史参照。所有 I 组保持 uniform-r32、`alpha/r=2`、dropout 0、数据顺序、rollout seed 和完整 optimizer checkpoint 一致。`scripts/analysis/prepare_phase1_signal_random_artifacts.py` 只接受两条显式授权路径：Phase 0.6 正式 `no-go` gate，或严格校验的 `crossfit_estimator_failure` 失效记录；本轮使用后者，且只允许从已审计的 Phase 0.5 source P2 构造四组 allocation，绝不引用 C 候选。随机补空间 seed 固定为 42，I32 必须与 P2 位级一致。该准备器不自动启动训练，须先读取 Phase 0.6 的 estimator 诊断。先跑到 step 50，统一六集评测；I0 和最佳混合组至少再用两个训练 seed 重复。只有多 seed 的提升超过评测 bootstrap 区间，才认为 random+signal 混合有效。

以下是 Phase 1 在看到 Phase 0.5/0.6 gate 和任何 Phase-1 训练结果前冻结的**原始三-seed预注册协议**。`scripts/local/run_phase1_signal_random_confirmation.sh` 要求人工判读 Phase 0.6 diagnosis 后已经生成 preparation manifest，不自动接到 Phase 0.5/0.6 controller。它先用 seed 42 依次训练和评测 `I0/I8/I16/I32`，再以 I0 为 baseline 生成三份 paired problem-bootstrap comparison。只有混合组 `I8/I16` 有资格进入多 seed；I32 只作为 hard signal-only endpoint，不参与“最佳混合组”选择。I8/I16 各自必须先满足 macro Avg@k delta `>= +0.005` 且 `P(delta > 0) >= 0.90`，过线者按 seed-42 macro delta 最大者选择，精确并列时选择信号方向更少的 I8。若两者均未过线，Phase 1 停止，不因 I32 或单 benchmark 的亮点追加训练。若有选中项，只对 I0 和该项运行 seeds 43/44，再对固定 seeds 42/43/44 应用下述分层 bootstrap 门槛。

该原始协议没有被删除或回溯改写，但当前执行范围已按 19.2.27--19.2.28 前瞻性修订为 seeds `42/43`，seed 44 不启动。原 `aggregate_multiseed_full_bench.py` 仍严格实现 seeds `42/43/44`，不能用于当前两-seed 最终 gate；两-seed 专用 analyzer 只能在 seed-43 I8 完成并独立验收后，按冻结 execution spec 实现和测试。

对应入口为 `scripts/local/start_phase1_signal_random_uniform_r32_4gpu.sh`、`scripts/local/run_phase1_signal_random_step50_fullbench.sh`、`scripts/analysis/select_phase1_seed42_candidate.py`。每次训练前重新核验 Phase 0.5/0.6 validation 与 gate 哈希、P2 signal 哈希、allocation/rank-map/subspace 哈希、196 个 uniform-r32 模块、`alpha=64`、正交性以及 I8/I16/I32 的 P2 前缀逐元素精确一致。这样 Phase 1 即使稍后人工启动，也不能在看到结果后替换 signal、complement seed 或候选集合。

六基准评测固定复用 7248 条 snapshot records，其 SHA-256 为 `3416ba93286b324a9663777fa472d0b568af5446319f27f876a63b59b3e863da`。每个评测 shard 的 manifest 必须记录该哈希、完整的 7248 requests、固定 seed 42、预期 adapter 路径和六个基准；即使训练或评测产物已存在，wrapper 也会重新核验 allocation、summary、records 行数和四个 manifest。多 seed 聚合器还会在统计入口再次要求每个 seed 恰好 7248 条匹配记录，避免缺失样本或陈旧产物被误判为稳定泛化。

每个 Phase 1 训练 run 还必须有独立的 `phase1_training_provenance_contract_v1`。contract 在训练前原子写入，锁定 preparation、Phase 0.5/0.6 validation 与 outcome、P2 signal、rank map、subspace、allocation summary、训练 seed 和固定超参的哈希/值；同时记录 `verl`、训练入口、Phase 1 分析器和六基准评测代码的逐文件 SHA-256 及聚合哈希，并在复用结果时按当前字节重算。step-50 训练成功后再写入 adapter 路径与 SHA-256。Phase 1 专用 launcher 显式固定 learning rate/scheduler/weight decay、PPO epochs、clip/KL、rollout temperature/top-p/top-k 和 shuffle 设置，不允许调用者的 shell 环境暗中改变原预注册 seed 对照。任何已存在的 adapter 只有在 complete contract 与当前 preparation、代码清单及 adapter 字节全部匹配时才能复用，避免旧 checkpoint 与新 allocation 同名而被误评测。

Phase 0.5/0.6 的 gate 通过项也遵守同一训练验证协议。`scripts/local/run_phase05_hybrid_step50_fullbench.sh` 和 `scripts/local/run_phase06_crossfit_step50_fullbench.sh` 串行执行 step-50 训练与固定 snapshot 六集评测；每个 summary 必须包含 7248 条采样、六个预定 benchmark。`scripts/analysis/compare_paired_full_bench.py` 要求 baseline/candidate 的 `(benchmark, problem_index, sample_index)` 逐项一致，并以 problem 为 cluster 做 10000 次 paired bootstrap。`scripts/analysis/evaluate_single_seed_screen.py` 固化 seed-42 晋级门槛，`scripts/analysis/aggregate_multiseed_full_bench.py` 对固定 seeds 42/43/44 做 training-seed × problem 分层 bootstrap。最近一次完整相关测试为 `45 passed, 1 warning`；warning 来自 Ray 的 deprecation warning，不是测试失败。seed-42 输出仍只是筛选证据，不替代训练 seed 重复。

#### 19.2.1 Phase 1 allocation 准备（2026-10-01）

正式 allocation 已从 `phase05_hybrid_shared_rollout_d64c32a16_r32_seed42_v3` 的 P2 生成，Phase 0.6 授权来源为 `phase06_invalidation.json`，其 SHA-256 为 `7eac808c53ad02adbc0a90ea5470736705bb8bbfb777c82a3cc8e6a614c4c5f4`。四组均经独立 verifier 确认为 196 个模块、uniform-r32、`alpha=64`、scaling 2；最大正交误差不超过 `9.54e-7`，I8/I16/I32 的 P2 前缀逐元素误差均为 0，I32 与 `candidates_P2.safetensors` 位级一致。

```text
phase1_training_preparation.json:
d60fc87775a5b1a0e1d27f922b6c4f2b100f0e8614faeb261afaf2c1c88dd386

I0  rank_map / subspace / summary:
9276de94a217c1fb1297d8531236b4d8a7113bafaa8f9482c31cb7f496c95927
30e0a7eba1ff06b6798805c2feeb3eb21501c79ea53a71906889ea7051a2e276
7d793a718f4640e54d6e10ca8201352a4eb2c06f80f61b7192fcc935621ecc88

I8  rank_map / subspace / summary:
2f20e70333a66f979d7630c83823304d9b9c196d65e86e74ce329200a87478f5
63842778dad6064beeae7c5de6a497e2bfd58d83ee93c2f5f567592045950135
0c7ba14adca289fd1d88377d6c91afaab0f1b26914c03b088b0bddf78e55e131

I16 rank_map / subspace / summary:
cd40feb4847364f0c946450757a8cba4af9b27f00011b268ec831eb415705c89
9452bbe57076291a4f92b833277d2263b36c3d491091c5d9794680a22115ed35
f4648ec41a7148ed822bb094ec16ac344351cf47253596cc05d18f3545c4bafe

I32 rank_map / subspace / summary:
c6c7fbd5e765f2367d745c8380da862b3e6a84e34fb89a59989ef1fb3472182d
a9e33ccadf14a0e6253ae22919fac86a4a812df6d4db27ee22a26ca852c4b9dc
028045d33910be3ace15fc373ce9efadc6f5be58b48431f7bc51a9fd928ddcc8
```

#### 19.2.2 v1/v2 启动前工程失效（2026-10-01）

首次 I0 seed-42 v1 启动在 optimizer step 0 之前失效。四个 worker 已正确加载 196 层 uniform-r32 子空间，日志显示 rank mean/range 为 `32/[32,32]`、最大正交误差 `8.345e-7`、LoRA B 最大绝对值为 0；随后 `dp_actor.py` 把通用 `grad_subspace` allocation summary 误当成旧 SPAR summary，强制读取不存在的 `allocation["structure"]`，四个 worker 同时以 `KeyError: 'structure'` 退出。该尝试没有产生 checkpoint 或训练 step，不属于方法结果。

v1 失败产物保留且不覆盖：training contract SHA-256 为 `fb4884689781cc6c5007541439dcb87e745d8b450a893481b496596473f4cbfe`，训练日志 SHA-256 为 `38de5dec80d9f1cf9b3e1ddd53014b419122caf9fad972e19194277fd369e952`，controller 日志 SHA-256 为 `6cbb934c0be6a07b91fce87469a272d81d19c64d4862cc26928588762e49612c`。修复后 consumer 同时支持通用 allocation 和带 `structure` 的 SPAR allocation：两者都记录参数量与 rank，只有后者记录 SPAR 专属 capture/family/segment 指标。新增回归测试确认通用 schema 不再要求 `structure`。

v2 在 `ray.init` 时、worker 创建前因 Ray 临时目录过长而退出：完整 experiment name 进入 `_temp_dir` 后，使 plasma-store Unix socket 路径超过 Linux 107 字节上限。v2 同样没有 worker、训练 step 或 checkpoint，只是启动工程失效。v2 training contract、训练日志和 controller 日志 SHA-256 分别为 `3d219586af8d8d3dc520cd55703327f759127b3433c152b4a88cb9131b97469e`、`e9ae2ad177bea9c7355180ab264aee603907486326f951b142cd9c1adc0fd0ff` 和 `b6cb82be5b4910a9b066d7df8a18759bd6b60b13d6ad1cad20937447f626a587`。后续 Ray 根目录固定为短且按 method/seed/revision 唯一的 `/tmp/ray-p1-<method>-s<seed>-r<revision>`，正式运行递增为 v3；v1/v2 contract 与日志均不复用、不覆盖。

#### 19.2.3 v3 正式运行前三步验证（2026-10-01）

I0 seed-42 v3 使用代码 provenance 聚合 SHA-256 `7d79cbfac7f801b30a7fe05dcd9ce4cc1a68766d5702a0523c529d668ccdb14f` 启动。四个 actor worker 正确应用 196 层 uniform-r32 初始化：rank mean/range 为 `32/[32,32]`、最大正交误差为 `8.345e-7`、LoRA B 最大绝对值为 0。`global_step=1/2/3` 均已完成，证明固定 rollout、reward、old log-prob、GRPO advantage、actor backward 和 optimizer update 的完整训练链路可执行；截至第三步没有 traceback、OOM、aborted response 或非有限指标。

首步实际处理 64 prompts x 8 responses，即 512 条 trajectory、总计 3,331,076 tokens。response 长度均值为 `6356.54`，达到 8192 上限的比例为 `0.51758`，因此不能把“64 prompts”理解为只处理 64 条短样本。首步计时如下：

| 子阶段 | 时间（秒） |
|---|---:|
| rollout generation | 302.00 |
| old log-prob | 54.25 |
| advantage | 0.03 |
| actor update | 169.24 |
| 完整 step | 525.70 |

其余首步健康指标包括 score mean `0.33594`、parse success `0.52148`、entropy `0.86690`、gradient norm `0.01194`、PPO KL `-2.06e-6` 和 clip fraction `3.65e-5`。第二步完整耗时为 `507.04` 秒，处理 3,214,564 tokens；score mean `0.38281`、parse success `0.54297`、entropy `0.88515`、gradient norm `0.01211`、PPO KL `1.07e-5`、clip fraction `2.13e-5`。第三步完整耗时为 `507.43` 秒，处理 3,240,867 tokens；score mean `0.39453`、parse success `0.57031`、entropy `0.91272`、gradient norm `0.01286`、PPO KL `8.08e-6`、clip fraction `3.76e-5`。第二、三步完整指标存在于 Ray TaskRunner 原始日志；外层 Ray log monitor 只转发了进度行，这属于日志转发差异，不是指标丢失或训练失败。

这些值只证明训练链路健康，不构成性能结论。前三步耗时为 `525.70/507.04/507.43` 秒，均值 `513.39` 秒；若后续保持该速度，单个 50-step 训练本体约为 `7.13` 小时，seed-42 的 I0/I8/I16/I32 四个训练本体约为 `28.5` 小时。三个没有重试混杂的历史同协议 7248-record 六基准评测，从 controller 创建到 summary 写入分别约为 `2:18`、`2:31` 和 `2:54`；据此四组 seed-42 训练加评测约需 `38--41` 小时，尚未计入少量模型启动和 checkpoint 保存时间。若 I8/I16 晋级，seeds 43/44 还要顺序运行 I0 与胜出方法共四组，约再增加 `38--41` 小时。controller 串行运行所有组，因此总墙钟时间不能按单个 64-prompt step 的约 9 分钟估算。

#### 19.2.4 v3 运行中容量审计（2026-10-01）

第二步完成时根文件系统可用 `39.5 GiB`，`/data` 仅余 `2.8 GiB`，且没有未分配磁盘空间。历史同规模 rank-32 完整 checkpoint 实测约 `7.0 GiB`；当前每组配置保留 step 25/50 两个 checkpoint，四个 seed-42 方法理论上需要约 `56 GiB`，还未包括评测 records，因此现有持久空间不足以原样保留全部八个完整 checkpoint。该问题不会影响当前 I0 的计算，但若不处置，可能在第三或第四组保存时失败。

本次审计没有删除、移动或覆盖任何历史/当前 artifact，也没有修改锁定代码。首次 step-25 保存后必须用真实大小替换上述历史估算；在第三组启动前，必须额外提供至少约 `20 GiB` 持久空间，或仅在某方法 step-50 adapter、final contract 和 7248-record 评测全部验证后，明确记录并执行该方法 step-25 中间 FSDP shard 的保留策略。不得为了释放空间删除 step-50 adapter、contract、评测 records、source allocation 或尚可能用于恢复当前训练的 checkpoint。

training contract 预注册 `checkpoint_steps=[25,50]`，但 finalize/verify 的持久结果只绑定 step-50 `adapter_path` 与 `adapter_sha256`，不会读取完整 FSDP model/optimizer shard。因此，完成方法的 step-50 contract 和 7248-record 评测均通过后，迁移该方法的 step-25 完整恢复状态不会改变 adapter、contract 或评测验证结果；这只属于存储保留策略，不得被描述成训练协议或性能结果的变化。为保留故障恢复能力，迁移优先于删除，并且必须记录源路径、目标路径、字节数和逐文件 SHA-256。

#### 19.2.5 I0 seed-42 step-50 与六基准正式结果（2026-10-02）

I0 seed-42 v3 完成全部 50 个 optimizer step，累计处理 `166,378,936` tokens。训练 step 总耗时 `26,009.16` 秒，即约 `7.23` 小时；单步均值/最小/最大为 `520.18/462.99/569.35` 秒。50 步平均 score、parse success、entropy 和 gradient norm 分别为 `0.363164`、`0.548555`、`0.890928` 和 `0.011917`，gradient norm 范围为 `[0.010197, 0.014281]`；最大绝对 PPO KL 为 `4.29e-5`，最大 clip fraction 为 `5.59e-5`，aborted ratio 始终为 0。这些指标只证明 I0 训练稳定，不是泛化收益证据。

step-50 contract 状态为 `complete`，独立 verifier 返回 `status=verified`。正式 adapter SHA-256 为 `6f46a787faea8d65c259840024924a4f0f3b834d47b0fe8af867fcacad5164fc`，contract SHA-256 为 `832effc6dfa710ad3c125babe1fd2a64fb90a5cf2d398b7735fceac2f4d5516b`，代码 provenance 聚合 SHA-256 仍为 `7d79cbfac7f801b30a7fe05dcd9ce4cc1a68766d5702a0523c529d668ccdb14f`。

固定 snapshot 六基准评测包含精确的 7,248 条 records，四个 shard 各 1,812 条；正式 verifier 再次确认六个 benchmark 的样本数、四份 manifest、adapter/base-model/snapshot 路径、采样协议和 snapshot SHA-256 全部匹配。summary 与 records SHA-256 分别为 `442e93b3855ea50d233197136ba50d34954340f57ad5eed759727bac2ae49733` 和 `7ea31a89f04ab52e3044f079e3f191e402639b52667fa85cf7b88b79236fd9ad`。结果如下：

| benchmark | Avg@k | Pass@k | Parse | Hit max | Mean length |
|---|---:|---:|---:|---:|---:|
| AIME24 | 0.253125 | 0.533333 | 0.821875 | 0.058333 | 13871.6 |
| AIME25 | 0.212500 | 0.566667 | 0.837500 | 0.075000 | 13517.1 |
| AMC23 | 0.589844 | 0.925000 | 0.818750 | 0.039062 | 8109.0 |
| HMMT Feb | 0.088542 | 0.400000 | 0.903125 | 0.048958 | 14760.0 |
| MATH500 | 0.632000 | 0.772000 | 0.880000 | 0.012500 | 4292.5 |
| Minerva | 0.204963 | 0.345588 | 0.602941 | 0.010110 | 5932.0 |
| **Macro/overall** | **0.330162** | **0.590431** | **0.817329** | **0.036010** | **9089.6** |

I0 是本轮相同 rank、scaling、dropout、训练 seed 和评测协议下的随机正交 A 基线。它低于若干历史 run，但历史 run 仍含初始化、训练轨迹和 request-level 采样差异；正式结论只能来自后续 I8/I16/I32 与本次 I0 的固定 paired comparison。controller 已于 `2026-10-02T08:14:00+08:00` 自动进入 I8 seed-42 训练。

#### 19.2.6 I0 评测退出阶段卡死与恢复（2026-10-02）

四个评测 shard 都已写出精确 1,812 行且日志末尾明确记录 `Wrote 1812 rows`，但 shard 0/1/3 的 Python 父进程在主逻辑结束后仍阻塞于 `do_wait`，对应三个 vLLM EngineCore 子进程保持约 `40.6 GiB/GPU` 显存、GPU 利用率为 0；shard 2 正常退出。该清理阶段持续约 3 小时，导致 4-GPU wrapper 无法执行 aggregate，也阻塞 controller 进入 I8。

处置前确认了全部 7,248 行、四份 manifest、四份 shard 结束标记、无 traceback，并确认三个父进程只在等待各自唯一的 EngineCore 子进程。随后只向 EngineCore PID `184894/184882/184876` 发送 `SIGTERM`，没有终止 shard 父进程、wrapper 或 controller。三个父进程随即自然返回，wrapper 以成功状态执行 aggregate；contract verifier 和 full-benchmark verifier 均通过，正式 records 没有重采样或改写。当前 controller 的代码 provenance 已锁定，因此在 I0/I8/I16/I32 链路结束前不修改评测代码；若后续 shard 再出现相同退出卡死，只能在完整性条件全部满足后做同样的精确子进程清理，并完整记录。

#### 19.2.7 I0 checkpoint 真实容量与保留处置（2026-10-02）

I0 的 step-25 和 step-50 checkpoint 各占 `7,715,573,760` 和 `7,715,553,280` 磁盘字节，两份合计 `15,431,135,232` 字节。每份包含四个约 `1.815 GB` 的 model shard、四个约 `74.3 MB` 的 optimizer shard、约 `147.8 MB` 的 PEFT adapter、tokenizer/Hugging Face 配置、extra state 和 data state。该实测值高于先前约 `7.0 GiB` 的粗估；I0 两份写入后根盘只剩约 `25 GiB`，不足以让后续三组都保留两份完整状态。

I0 step-50 contract 和 7,248-record 六基准评测独立验证通过后，对 step-25 的 22 个文件逐一记录路径、字节数和 SHA-256。逐文件 inventory SHA-256 为 `7b6d1f6f6d538c7c4ae2d58cf11cbfd0993ac0fb107b9185cb40b47c3af50fa0`；因没有可用的持久迁移盘，随后只删除 I0 `global_step_25`，释放 `7,715,573,760` 字节，根盘恢复到约 `32 GiB`。step-50 完整 checkpoint、正式 adapter、contract、summary 和 records 均保留，删除后两个 verifier 再次通过。

最终删除 manifest 位于 `analysis/checkpoint_retention/phase1_i0_uniform_r32_b64m16n8_step50_seed42_v3_step25_deletion_manifest.json`，SHA-256 为 `5c23243de7efebfc7b2f8773dda4a136a9cdad6b97a7c3541543d0c0933895a1`，其中明确记录 `status=deleted`、源目录、逐文件 inventory、释放字节数和所有保留正式 artifact 的哈希。该 step-25 恢复状态已不可本地恢复，如需重建只能重新运行 I0 seed 42 至 step 25；不得把本次存储处置描述成训练协议或性能结果变化。后续方法只有在各自 step-50 contract 与完整评测都验证后，才可应用相同保留边界。

#### 19.2.8 I8 seed-42 step-25 checkpoint 审计（2026-10-02）

I8 seed-42 v3 已完成 step 25 并保存第一份完整恢复状态。checkpoint 包含四个 model shard、四个 optimizer shard、四个 extra-state shard、PEFT adapter、Hugging Face/tokenizer 配置和 data state，`latest_checkpointed_iteration.txt` 精确为 `25`；目录实占 `7,715,557,376` 字节，与 I0 同阶段容量一致。I8 training contract 保持预期的 `status=prepared`，正式 adapter 路径与 SHA-256 要到 step 50 后才允许写入；代码 provenance 聚合 SHA-256 仍为 `7d79cbfac7f801b30a7fe05dcd9ce4cc1a68766d5702a0523c529d668ccdb14f`。

审计时 controller、trainer 和 TaskRunner 均存活，没有 traceback、OOM、NCCL error/timeout 或异常退出；根盘可用约 `25 GiB`，足够写入 I8 step-50 checkpoint 和评测 records。当前 step-25 是 I8 在 step 50 正式 contract 与六基准评测完成前的唯一恢复点，不移动、不删除，也不因中途训练指标选择或停止方法。

在看到正式 step-50 结果前，预注册训练验证门槛如下：

1. seed 42 只有在六集 macro Avg@k delta 至少 `+0.005`，且 problem-cluster bootstrap 的 `P(delta > 0) >= 0.90` 时，才进入额外训练 seed；否则停止该候选，不用单个 benchmark 的亮点改写结论；
2. 额外训练 seed 固定为 `43` 和 `44`，rollout seed 与训练 seed 相同；评测数据 snapshot、评测采样 seed 42、温度、top-p、每题采样数和最大长度保持不变；
3. “稳定改进”要求三个训练 seed 的 macro delta 均值至少 `+0.005`、至少 2/3 seed 为正、最差 seed 不低于 `-0.002`，并且跨 seed/problem 的分层 bootstrap 95% CI 下界大于 0；
4. “跨 benchmark 泛化”另外要求六集里至少 4 个 benchmark 的三 seed 平均 delta 为正，且任何 benchmark 不得低于 `-0.02`；
5. 未同时通过第 3、4 条时，只能报告为 probe 或单 seed 现象，不能作为稳定可泛化方法的主结论。

#### 19.2.9 I8 seed-42 step-50 训练与 contract 审计（2026-10-02）

I8 seed-42 v3 已完成全部 50 个 optimizer step，累计处理 `166,816,089` tokens。训练 step 总耗时 `25,991.46` 秒，即约 `7.22` 小时。50 步平均 score、parse success、entropy 和 gradient norm 分别为 `0.376758`、`0.562227`、`0.868708` 和 `0.066184`，gradient norm 范围为 `[0.053866, 0.078736]`；最大绝对 PPO KL 为 `3.76e-5`，最大 clip fraction 为 `5.42e-5`，aborted ratio 始终为 0。I8 的 gradient norm 高于 I0，但仍远低于 `grad_clip=1.0`，且 KL、clip fraction、aborted ratio 和有限性检查均正常；这些值只证明训练健康，不构成性能或泛化结论。

step-50 checkpoint 实占 `7,715,487,815` 字节，包含四个 model shard、四个 optimizer shard、四个 extra-state shard、PEFT adapter、Hugging Face/tokenizer 配置和 data state；`latest_checkpointed_iteration.txt` 精确为 `50`。training contract 状态已从 `prepared` 转为 `complete`，独立 verifier 返回 `status=verified`。正式 adapter SHA-256 为 `8d4dfe3ff8b93f75f777f0a774d812ffb23f06c5be78a46112f2a6a69325585a`，contract SHA-256 为 `6d6ea833c79281fbf24f2d9673635d61d33a0ae2d89f2f47302ea09bba4038ed`，代码 provenance 聚合 SHA-256 仍为 `7d79cbfac7f801b30a7fe05dcd9ce4cc1a68766d5702a0523c529d668ccdb14f`。

controller 随后自动启动固定 snapshot 的四 shard 六基准评测；正式结果和处置见下一节。

#### 19.2.10 I8 seed-42 六基准结果与 paired screen（2026-10-02）

I8 评测包含精确的 7,248 条 records，四个 shard 各 1,812 条；独立 full-benchmark verifier 返回 `status=verified`，确认六个 benchmark 的样本数、四份 manifest、adapter/base-model/snapshot 路径、采样协议和 snapshot SHA-256 全部匹配。summary 与 records SHA-256 分别为 `213cfd2c2f8669a2dc66f1d928799ce18b865d2b0fa33c61b12e74ff9b193ede` 和 `9e1c31b6f4aace7408362fbc612b1ad6f22be8291e7b73c89f270f72b97892b6`。正式结果如下：

| benchmark | Avg@k | Pass@k | Parse | Hit max | Mean length |
|---|---:|---:|---:|---:|---:|
| AIME24 | 0.278125 | 0.633333 | 0.871875 | 0.063542 | 13271.6 |
| AIME25 | 0.225000 | 0.533333 | 0.904167 | 0.053125 | 12927.2 |
| AMC23 | 0.699219 | 0.975000 | 0.925781 | 0.019531 | 7220.2 |
| HMMT Feb | 0.097917 | 0.366667 | 0.896875 | 0.064583 | 14672.0 |
| MATH500 | 0.694500 | 0.800000 | 0.943500 | 0.011000 | 4028.1 |
| Minerva | 0.202206 | 0.327206 | 0.590993 | 0.012868 | 5593.9 |
| **Macro/overall** | **0.366161** | **0.605923** | **0.866584** | **0.032423** | **8639.7** |

预注册的 matched-problem cluster bootstrap 使用 10,000 次重采样和 seed 42。I8 相对 I0 的 macro Avg@k delta 为 `+0.035999`，95% CI 为 `[+0.020847,+0.051859]`，`P(delta>0)=1.0`，精确匹配 902 个 problem 和 7,248 条 record；comparison SHA-256 为 `1a2155ea87b90ba607a5d76c4b5f1facd52e9399e8e08f88e88249c09f3ce0cc`。逐集结果如下：

| benchmark | I0 Avg@k | I8 Avg@k | delta | 95% CI | P(delta>0) |
|---|---:|---:|---:|---:|---:|
| AIME24 | 0.253125 | 0.278125 | +0.025000 | [-0.030208,+0.081250] | 0.8208 |
| AIME25 | 0.212500 | 0.225000 | +0.012500 | [-0.018750,+0.058333] | 0.6857 |
| AMC23 | 0.589844 | 0.699219 | +0.109375 | [+0.063262,+0.158594] | 1.0000 |
| HMMT Feb | 0.088542 | 0.097917 | +0.009375 | [-0.013542,+0.038542] | 0.7504 |
| MATH500 | 0.632000 | 0.694500 | +0.062500 | [+0.043500,+0.082000] | 1.0000 |
| Minerva | 0.204963 | 0.202206 | -0.002757 | [-0.023897,+0.018382] | 0.3917 |

I8 明确通过 seed-42 晋级门槛 `delta>=+0.005` 和 `P(delta>0)>=0.90`，但仍只是单训练 seed screen。稳健性检查表明结果并非仅由一个 benchmark 决定：去掉 AMC23 后 macro delta 仍为 `+0.021324`，去掉 MATH500 后为 `+0.030699`，同时去掉两者后其余四集均值仍为 `+0.011029`。I8 的 overall parse rate 相对 I0 提高约 `+0.0493`，但在只看成功解析样本的近似条件正确率 `Avg@k/Parse` 时，AIME24、AMC23、HMMT Feb、MATH500 和 Minerva 仍为正，只有 AIME25 约为 `-0.0049`；因此不能把全部收益归因于答案解析率。不过该条件比值不是预注册主指标，只用于排查解释。

评测退出阶段重现 I0 的 vLLM 清理问题：全部四份 records、manifest 和 `Wrote 1812 rows` 标记都完整且无 traceback 后，shard 0/2 正常退出，shard 1/3 的父进程阻塞于 `do_wait`，对应 EngineCore GPU 利用率为 0。只向 EngineCore PID `367861/367867` 发送 `SIGTERM` 后，父进程自然返回，wrapper 正常聚合且 verifier 通过；没有终止 shard 父进程、controller，也没有重采样或改写正式结果。

#### 19.2.11 I8 checkpoint 保留处置（2026-10-02）

I8 step-50 contract 与正式评测独立验证通过后，对 step-25 的 22 个文件逐一记录路径、字节数和 SHA-256，所有哈希均反向校验成功。逐文件 inventory SHA-256 为 `2bbe198899ea46da74f3b909b3e0c62a624bc8cd4c915846ba3c4e38c7ff1346`；随后只删除 I8 `global_step_25`，释放 `7,715,487,815` 字节，根盘可用空间从约 `16 GiB` 回升到约 `23 GiB`。I8 step-50 完整 checkpoint、正式 adapter、contract、summary、records 和 paired comparison 均保留，删除后 contract 与 full-benchmark verifier 再次通过。

最终删除 manifest 位于 `analysis/checkpoint_retention/phase1_i8_uniform_r32_b64m16n8_step50_seed42_v3_step25_deletion_manifest.json`，SHA-256 为 `abe26cfdbd93e3b1e7834a0cac134e8699d08bc29ac29b245e85bd1341fc2494`。该 step-25 恢复状态已不可本地恢复，如需重建只能重新运行 I8 seed 42 至 step 25；本次处置没有改变训练协议或性能结果。

#### 19.2.12 I8 结果后的继续训练决策（2026-10-02）

I8 verifier 完成后 controller 自动启动 I16；在人工作出后续决策前，整个 controller 进程组以 `SIGSTOP` 可逆暂停，确认 GPU 利用率为 0 且 I16 尚未生成 checkpoint。审查结论是继续预注册的 I16/I32，而不是看到 I8 的正结果后直接跳到 I8 seeds 43/44：I8/I16 都是有资格晋级的 mixed candidate，预注册选择规则要求先比较二者并选择 seed-42 macro delta 较高者；I32 虽不参与晋级，但作为 signal-only endpoint 是解释剂量效应和排除“任意 P2 比例都相同”的必要控制。此时跳过它们会构成 outcome-adaptive protocol change，削弱论文证据。

因此后续仍按 `I16 -> evaluation -> I32 -> evaluation -> fixed selection` 推进；只有固定规则选择出的 I8 或 I16 才与 I0 一起运行训练 seeds 43/44。当前 I8 结果可以表述为“强单 seed 候选”，不能表述为稳定提升、跨 seed 泛化或 COLING 方法已经成立。

#### 19.2.13 I16 seed-42 step-50 训练与 contract 审计（2026-10-03）

I16 seed-42 v3 已完成全部 50 个 optimizer step，累计处理 `166,235,271` tokens。训练 step 总耗时 `26,227.52` 秒，即约 `7.29` 小时。50 步平均 score、parse success、entropy 和 gradient norm 分别为 `0.373203`、`0.559414`、`0.927497` 和 `0.070362`，gradient norm 范围为 `[0.059180,0.082731]`；最大绝对 PPO KL 为 `3.47e-5`，最大 clip fraction 为 `6.02e-5`，aborted ratio 始终为 0。全部标量有限，gradient norm 远低于 `grad_clip=1.0`，没有 traceback、OOM、NCCL error/timeout 或异常退出。因此后续性能下降不能解释成训练崩坏。

step-50 checkpoint 包含四个 model shard、四个 optimizer shard、四个 extra-state shard、PEFT adapter、Hugging Face/tokenizer 配置和 data state；`latest_checkpointed_iteration.txt` 精确为 `50`。training contract 状态已从 `prepared` 转为 `complete`，独立 verifier 返回 `status=verified`。正式 adapter SHA-256 为 `df66bf8756b6888b1641b4b4f8766d33fcd1c1699f3c3aeb991faae83cdaffb6`，contract SHA-256 为 `42c414f1a1b137d470b2067f7f05ee0d6b3d05cd60f566a974ad107027fef06c`，代码 provenance 聚合 SHA-256 仍为 `7d79cbfac7f801b30a7fe05dcd9ce4cc1a68766d5702a0523c529d668ccdb14f`。

#### 19.2.14 I16 seed-42 六基准结果与 paired screen（2026-10-03）

I16 评测包含精确的 7,248 条 records，四个 shard 各 1,812 条；独立 full-benchmark verifier 返回 `status=verified`，确认六个 benchmark 的样本数、四份 manifest、adapter/base-model/snapshot 路径、采样协议和 snapshot SHA-256 全部匹配。summary 与 records SHA-256 分别为 `f9db8e856767aacd1391859e437385744af541d27f49e68cc4407c4dacfb2759` 和 `5ec73279993a0df7e8a0d450b111c06928c63bae800300bb57867243c7b09281`。正式结果如下：

| benchmark | Avg@k | Pass@k | Parse | Hit max | Mean length |
|---|---:|---:|---:|---:|---:|
| AIME24 | 0.208333 | 0.733333 | 0.746875 | 0.059375 | 13051.1 |
| AIME25 | 0.175000 | 0.433333 | 0.815625 | 0.052083 | 12766.1 |
| AMC23 | 0.565625 | 0.950000 | 0.793750 | 0.027344 | 7391.0 |
| HMMT Feb | 0.097917 | 0.400000 | 0.898958 | 0.059375 | 14235.7 |
| MATH500 | 0.636500 | 0.786000 | 0.891500 | 0.007500 | 3858.4 |
| Minerva | 0.197610 | 0.327206 | 0.582721 | 0.002757 | 5293.4 |
| **Macro/overall** | **0.313498** | **0.604979** | **0.799669** | **0.029939** | **8469.6** |

预注册的 matched-problem cluster bootstrap 使用 10,000 次重采样和 seed 42。I16 相对 I0 的 macro Avg@k delta 为 `-0.016665`，95% CI 为 `[-0.032627,-0.001820]`，`P(delta>0)=0.0133`，精确匹配 902 个 problem 和 7,248 条 record；comparison SHA-256 为 `62b97995fe8614ec6503611e9a69158fef930ce9530c6cf06e4a2312ed7c9616`。逐集结果如下：

| benchmark | I0 Avg@k | I16 Avg@k | delta | 95% CI | P(delta>0) |
|---|---:|---:|---:|---:|---:|
| AIME24 | 0.253125 | 0.208333 | -0.044792 | [-0.116667,+0.015625] | 0.0843 |
| AIME25 | 0.212500 | 0.175000 | -0.037500 | [-0.084375,+0.000000] | 0.0256 |
| AMC23 | 0.589844 | 0.565625 | -0.024219 | [-0.057813,+0.007813] | 0.0753 |
| HMMT Feb | 0.088542 | 0.097917 | +0.009375 | [-0.009375,+0.031250] | 0.8217 |
| MATH500 | 0.632000 | 0.636500 | +0.004500 | [-0.013500,+0.022500] | 0.6901 |
| Minerva | 0.204963 | 0.197610 | -0.007353 | [-0.029412,+0.013787] | 0.2479 |

I16 明确未通过 seed-42 晋级门槛：macro delta 不仅低于 `+0.005`，而且为负，`P(delta>0)` 也远低于 `0.90`。六个 benchmark 中四个下降，仅 HMMT Feb 和 MATH500 小幅上升。结合 I8 的 `+0.035999` 与 I16 的 `-0.016665`，signal slot 比例呈明显非单调性：`8/32` 的 mixed 初始化在本 seed 有效，增加到 `16/32` 反而劣于全随机 I0。I16 因此淘汰，不进入 seeds 43/44；这一结果同时说明不能把“加入更多 P2 signal”当成单调改进规律。

#### 19.2.15 I16 评测退出阶段卡死与恢复（2026-10-03）

四个评测 shard 都已写出精确 1,812 行，四份 manifest 和日志末尾结束标记齐全，且无 traceback、OOM 或 NCCL 异常。shard 3 正常退出；shard 0/1/2 的 Python 父进程已完成正式记录写入，但仍等待各自唯一的 vLLM EngineCore 子进程退出。确认父子关系、GPU 状态和全部正式 artifact 完整性后，只向 EngineCore PID `580956/580938/580950` 发送 `SIGTERM`，没有终止 shard 父进程、4-GPU wrapper 或 controller。

三个父进程随后自然返回，wrapper 正常执行 aggregate，contract verifier 与 full-benchmark verifier 均返回 `status=verified`；正式 records 没有重采样或改写。该处置严格沿用 I0/I8 已记录的清理边界：只有在四个 shard 的精确行数、manifest、结束标记和无异常条件全部满足后，才允许只清理残留 EngineCore。

#### 19.2.16 I16 checkpoint 保留处置与 I32 接续（2026-10-03）

I16 step-50 contract、正式评测和 paired comparison 独立验证通过后，对 step-25 的 22 个文件逐一记录路径、字节数和 SHA-256。逐文件 inventory SHA-256 为 `c6fea4b1de1e331b67c4d2bc860cf2f1f2edb2936056d1cbb4943addd05f4242`；随后只删除 I16 `global_step_25`，释放 `7,715,487,815` 磁盘字节。该次删除完成时，I16 step-50 完整 checkpoint、正式 adapter、contract、summary、records 和 paired comparison 均仍保留，删除前后 contract 与 full-benchmark verifier 均返回 `status=verified`。

最终删除 manifest 位于 `analysis/checkpoint_retention/phase1_i16_uniform_r32_b64m16n8_step50_seed42_v3_step25_deletion_manifest.json`，SHA-256 为 `6e9079d5e6478c53b97b249a1c04b9f706e5d9a341be0b718f6a14546ef7fe86`。该 step-25 恢复状态已不可本地恢复，如需重建只能重新运行 I16 seed 42 至 step 25；本次存储处置没有改变训练协议或性能结果。

进一步审查 controller 后确认：I32 完成评测和 seed-42 selection 后会无人工停顿地直接启动 seed-43 I0，而当时约 `18.6 GB` 可用空间不足以容纳 I32 的两份 checkpoint 后再写 seed-43 的两份 checkpoint。I16 已按预注册门槛淘汰，后续 selection 和多 seed 汇总只读取其正式 records，不读取训练恢复 shard。因此在不修改锁定代码和 controller 的前提下，扩展保留策略只作用于 I16 step-50 的训练恢复能力。

删除前对 I16 step-50 的 22 个文件和父目录恢复指针逐一记录路径、字节数与 SHA-256，共 23 条；inventory SHA-256 为 `1eaefb37659c661d26a5fd5646f4b91e5a6bb3be136bdf3a28d396fb4e04a0cb`。在 contract 与 full-benchmark verifier 再次返回 `status=verified`、23 条 inventory 全部反向校验成功后，精确删除四个 FSDP model shard、四个 optimizer shard、四个 extra-state shard、`data.pt` 和 `latest_checkpointed_iteration.txt`，共 14 个文件、`7,556,249,986` 字节；没有删除目录或使用通配删除。

删除后 14 个目标全部不存在，保留的 9 个 checkpoint 文件共 `159,221,447` 字节，包括 PEFT adapter、adapter config、HF/tokenizer 配置和 FSDP config；9 个文件逐哈希不变，contract、summary、records 和 paired comparison 哈希也不变。contract 与 full-benchmark verifier 再次返回 `status=verified`，根盘可用空间增至 `26,134,761,472` 字节。最终 step-50 恢复裁剪 manifest 位于 `analysis/checkpoint_retention/phase1_i16_uniform_r32_b64m16n8_step50_seed42_v3_step50_recovery_prune_manifest.json`，SHA-256 为 `f16823664ce0eca5eb41374c99f478410eea49ab71379619742365d0e09a3d6b`。I16 正式 adapter 仍可用于推理和复核，但 step-50 已不能本地续训；如需恢复完整训练状态，只能从头重跑 I16 seed 42 至 step 50。

controller 已于 `2026-10-03T07:33:32+08:00` 自动进入 I32 seed-42 训练。I32 是 `32 P2 + 0 random` 的 signal-only endpoint，仅用于确定剂量曲线和解释机制，不参与候选晋级；无论其单 seed 结果如何，当前符合晋级条件的 mixed candidate 仍只有 I8。I32 完成前继续保持代码 provenance 锁定，并只在 step 25、step 50、六基准完成或明确异常时检查。

#### 19.2.17 多 seed 阶段的容量不变量（2026-10-03）

I16 恢复 shard 裁剪后，根盘可用 `26,133,291,008` 字节；I0/I8/I16 的实测完整 checkpoint 均约 `7.715 GB`，其中可在正式验证后裁剪的 step-50 训练恢复文件约 `7.556 GB`，必须永久保留的 adapter 与推理配置约 `159 MB`。按该实测值且暂不计后续小规模日志与评测 records，I32 的容量轨迹为：

| 容量节点 | 预计可用字节 | 操作含义 |
|---|---:|---|
| I32 尚未保存 checkpoint | 26,133,291,008 | 当前状态 |
| I32 step 25 后 | 18,417,803,193 | 保留活动 run 的唯一恢复点 |
| I32 step 50 后 | 10,702,315,378 | 足够完成评测，但不足以让下一 run 再写两份 checkpoint |
| I32 正式评测验证后删除 step 25 | 18,417,803,193 | 只删除已验证 run 的中间恢复点 |
| 再裁剪 I32 step-50 恢复 shard | 25,974,053,179 | 保留 adapter、配置、contract、records、summary 和 paired comparison |

因此多 seed 阶段固定采用以下容量不变量，而不是临时观察磁盘后任意删除：

1. 任一新 50-step run 在写 checkpoint 前应至少有约 `17 GB` 可用空间，以容纳两份约 `7.715 GB` 的完整状态和日志/评测余量；
2. 活动 run 在 step-50 contract、完整六基准和 paired comparison 验证前，不删除它的 step-25 或 step-50 任一恢复文件；
3. 上一 run 正式验证后，先逐文件 inventory 并删除 step-25；若下一 run 的两份 checkpoint 仍无空间，再只裁剪上一 run 的 step-50 model/optimizer/extra/data 恢复文件，永久保留 adapter、HF/tokenizer 配置和正式统计 artifact；
4. 每次裁剪前后都运行 contract/full-benchmark verifier，记录 inventory、manifest、释放字节和不可恢复性，不删除目录，不使用宽泛通配目标；
5. 裁剪优先级固定为 signal-only I32、已完成且已有正式结果的后续 seed run、已淘汰方法，最后才考虑核心 seed-42 I0/I8；当前不需要提前裁剪 I0/I8 seed-42；
6. controller 在相邻 run 之间没有停顿，因此清理上一 run 的窗口是下一 run 启动后、首次 step-25 保存前；仍只在预定关键节点检查，不逐 step 轮询。

该策略能循环释放约 `15.27 GB/run`，使 seed-43/44 的 I0 与入选 mixed candidate 顺序训练可持续进行。它只改变已完成 checkpoint 的本地恢复能力，不改变训练、采样、评测或候选选择协议；若任何 verifier 不通过，则停止裁剪并保留现状。

#### 19.2.18 候选选择代码的传递依赖审计（2026-10-03）

在 I32 结果和 seed-42 selection 产生前，对最终统计链路做只读审计。`compare_paired_full_bench.py`、`select_phase1_seed42_candidate.py` 和 `aggregate_multiseed_full_bench.py` 的当前 SHA-256 分别为 `5764621d26742238f77a4e8d47d5c37b14ca80ae2b65e6ea4bd3865cf2041fc8`、`ed2b580b637d70e0f7fb25a7868c264c15643647132ef1764140c7d7d6355989` 和 `0ca4bb7fd01f59afd99cf991596626892ed18370b35d68fa74b76cb32a05460b`，均与现有 training contract 的 code provenance 一致。

语义审计确认：paired loader 拒绝重复 record key 和非布尔 `correct`，要求 baseline/candidate 的六 benchmark、sample key、problem、gold answer 和每题采样数精确一致；seed-42 selector 强制每份 comparison 有 7,248 条 records、10,000 次 bootstrap、bootstrap seed 42、六个固定 benchmark、`record_keys_exactly_matched=true` 和 `interpretation=single_seed_screen_only`，仅允许 I8/I16 晋级并以 I8 作为精确并列时的固定 tie-break；三 seed aggregator 强制 training seeds 精确为 `42/43/44`、每 seed 7,248 条、跨 seed snapshot/problem invariant 一致，并实现预注册的六项稳定泛化门槛。`tests/test_full_bench_eval.py` 全部 13 项通过。

审计同时发现一个 provenance 覆盖缺口：selector 直接导入的 `scripts/analysis/evaluate_single_seed_screen.py` 未列入 `phase1_training_contract.py` 的 `CODE_PROVENANCE_FILES`，因此 contract 聚合哈希没有覆盖该传递依赖。该文件当前 SHA-256 为 `eec1662ef6a4993740530678e681da04a51f063b068a1237d02745d04374182c`，mtime 为 `2026-10-01T00:03:26.817273437+08:00`，早于 v3 正式训练，且当前实现精确执行 macro delta `>=0.005` 与 `P(delta>0)>=0.90` 双门槛。

本轮不在 controller 运行期间修改 contract 或分析代码，而是用独立 audit `analysis/phase1_selection_dependency_audit.json` 锁定遗漏依赖与直接 importer；audit SHA-256 为 `ae6ccfc7b160ffeb413ed8a2396d938282356a4ba82e5d4c13fbc60ee7917f71`。`phase1_seed42_selection.json` 生成后必须重新计算两者哈希，只有与 audit 完全一致且 selection schema/阈值复核通过时才接受选择结果。当前 provenance-locked pipeline 完成后，任何新正式 run 都必须先把该依赖加入未来 contract 的 provenance 覆盖；论文中不得声称本轮原始 contract 已覆盖这一遗漏文件。

同时逐层审计 `run_phase1_signal_random_confirmation.sh -> run_phase1_signal_random_step50_fullbench.sh -> start_phase1_signal_random_uniform_r32_4gpu.sh -> start_full_gradient_uniform_r8_4gpu.sh -> run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh`。前三个 Phase-1 shell 的 SHA-256 分别为 `b6d4bdfc7af77019fbdfad8bec76187ba0254db6e24a2e5acd632ac5ed6aaa0f`、`e87909adbf9fcf985d4890b9dfd208aa53f700d1cba04e67b1a5979b291e0085` 和 `288c21a83028e92a60f17a2ee696ce4a989eaebd07db003f8dca181394e0617a`，下层两个 launcher 分别为 `6d814d184c8b3b30d9759dc81d12f1e312b6387468c3eed002305cb7f8ef865c` 和 `915edb150517297aca8616d4722ceb3510e1c5a2c92ff84edcc0171d04c0c309`，均与 contract provenance 一致，且 `bash -n` 通过。seeds 43/44 的 `TRAIN_SEED` 会同时进入 run 名、contract、data shuffle、PPO dataloader 和 rollout seed；评测 snapshot、采样 seed 42、温度、top-p、样本数和长度上限保持固定。训练目录非空时 launcher 拒绝覆盖，已有 summary 也必须通过 contract 与 full-benchmark verifier，不能仅凭文件存在跳过验证。

#### 19.2.19 seed-42 问题级异质性与格式机制诊断（2026-10-03）

在不读取 I32 中间结果的前提下，对 I0/I8/I16 已验证的三份 7,248-record 正式结果按 `(benchmark, problem_index, sample_index)` 精确配对，得到 902 个相同问题。该分析明确标记为 post-hoc mechanism/concentration diagnostic，不是候选选择指标，也不能替代多 seed 门槛。三份输入 records 哈希仍分别为 I0 `7ea31a89f04ab52e3044f079e3f191e402639b52667fa85cf7b88b79236fd9ad`、I8 `9e1c31b6f4aace7408362fbc612b1ad6f22be8291e7b73c89f270f72b97892b6` 和 I16 `5ec73279993a0df7e8a0d450b111c06928c63bae800300bb57867243c7b09281`。

| 对比 I0 | macro delta | 问题改善/不变/下降 | benchmark 等权改善比例 | benchmark 等权下降比例 | 样本正确净翻转 | 格式有效净翻转 | 格式有效条件正确率变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| I8 | +0.035999 | 222 / 567 / 113 | 29.64% | 17.54% | +307 | +357 | +2.226 pp |
| I16 | -0.016665 | 164 / 565 / 173 | 21.51% | 29.72% | -100 | -128 | -0.691 pp |

I8 的逐集问题改善/下降数为 AIME24 `12/7`、AIME25 `6/7`、AMC23 `24/6`、HMMT Feb `5/6`、MATH500 `138/50`、Minerva `37/37`。因此 I8 并非只靠极少数离群问题产生正均值：无论 raw count 还是 benchmark 等权比例，改善问题都明显多于下降问题；但 902 题中仍有 567 题不变，而且最大均值增益仍集中在 AMC23 和 MATH500，不能据此跳过多 seed。

剂量响应进一步支持非单调解释：121 个问题在 I8/I16 都改善，101 个只在 I8 改善而 I16 不改善，只有 43 个只在 I16 改善而 I8 不改善，其余 637 个两者都不改善。I8 并不是 I16 的弱化版噪声结果；增加 P2 slot 会丢失一批 I8 的问题级收益。

样本级机制上，I8 相对 I0 有 `745` 个错误到正确、`438` 个正确到错误，净正确翻转 `+307`；格式无效到有效为 `913`，反向为 `556`，净格式翻转 `+357`。格式改善解释了相当一部分收益，但不能解释全部：overall 格式有效率提高 `+4.9255` 个百分点后，格式有效样本内正确率仍提高 `+2.2262` 个百分点；该条件正确率在 AIME24、AMC23、HMMT Feb、MATH500、Minerva 为正，仅 AIME25 为 `-0.4883` 个百分点。完整诊断位于 `analysis/phase1_seed42_problem_heterogeneity_diagnostic.json`，SHA-256 为 `54670f79b06b1ccdf27df02bb142e4e8181b736aa9da1cfcf7de92449e8443f9`。这些结果只说明 I8 的单 seed 现象具有一定广度和双重机制，正式表述仍必须是“强单 seed 候选”。

#### 19.2.20 seed-42 训练轨迹与泛化错配诊断（2026-10-03）

对 I0/I8/I16 三份完整日志中的 50 个 optimizer step 按 step 精确配对，输入日志 SHA-256 分别为 `3de946e6aaec9662be29b0a77555ee2963168cf999b8abc646c99fdcadb7d96f`、`d5b8a0eecefd673cbe36a75915c447d66fc6821ec72a2aa23db32a1188285bb7` 和 `4beebd5e60c8df41d1e2be05e9cfa0f1a653fa1d8b30ffe16bdfe95c89c35015`。该分析同样是 post-hoc diagnostic，不用于候选选择或早停。

| 方法 | 训练 score 均值 | 训练 parse 均值 | entropy 均值 | grad norm 均值 | 最后 10 步 score | 六基准 Macro Avg@k |
|---|---:|---:|---:|---:|---:|---:|
| I0 | 0.363164 | 0.548555 | 0.890928 | 0.011917 | 0.358789 | 0.330162 |
| I8 | 0.376758 | 0.562227 | 0.868708 | 0.066184 | 0.381445 | 0.366161 |
| I16 | 0.373203 | 0.559414 | 0.927497 | 0.070362 | 0.385742 | 0.313498 |

I8 与 I16 的训练 score 和 parse 轨迹高度相似，Pearson 相关分别为 `0.8625` 和 `0.7995`；50 步均值只相差 `+0.003555` 和 `+0.002813`，但 I8 的正式六基准 macro 高出 `+0.052664`。I16 最后 10 步训练 score 反而比 I8 高 `0.004297`，却在正式评测中显著更差。两组相对 I0 的训练 score 均提高：I8 为 `+0.013594`，I16 为 `+0.010039`；只有 I8 的正式评测提高，说明训练 reward 无法区分有效的 signal 比例。

梯度和 entropy 同样不能作为性能代理。I16 的 entropy 在全部 50 步都高于 I8，grad norm 在 50 步中的 41 步高于 I8，但 I16 明确未过 paired screen。I0 的 grad norm 虽远小于两个 signal 初始化，评测仍高于 I16；因此较大 grad norm 只反映参数化/子空间几何下的梯度尺度，不代表更强优化或泛化。完整诊断位于 `analysis/phase1_seed42_training_dynamics_diagnostic.json`，SHA-256 为 `03a6005f79cb2074f04baa17218df80ecafc0ea937e7c910044649886c731863`。后续不能依据训练 score、parse、entropy、grad norm 或其末段趋势改写固定六基准与多 seed 选择规则。

#### 19.2.21 训练集与评测集题面重合敏感性（2026-10-03）

为判断 I8 的 seed-42 增益能否被训练题重合解释，对实际训练输入 `dapo-math-17k-boxed.parquet` 的 17,917 行与固定评测 snapshot 的 902 道唯一题做了 post-hoc 去污染审计。训练 parquet SHA-256 为 `8e3c9314db8b83c61ab62a3dc85e0a704dcc5f3a9d404d236943f719095ce82f`；I0/I8 records 哈希保持为 `7ea31a...fd9ad` 与 `9e1c31...892b6`。该审计只用于论文边界条件与敏感性分析，不回写 seed-42 selector，也不是新的晋级门槛。

Unicode NFKC、大小写和空白归一化后发现 9 道完全相同题；进一步只忽略 LaTeX 排版 token 后共发现 13 道完全相同题，其中 AMC23 为 7/40，MATH500 为 6/500，其余四集为 0。这 13 题的训练与评测答案全部一致。另采用词/数字 5-gram 规则，将 Jaccard `>=0.8` 或短文本 containment `>=0.9` 的训练题视为潜在近重复，共标记 29 题：AMC23 16 题、MATH500 13 题。该词法集合会故意包含题面模板相近但答案不同的样本，而且漏掉一条只在 LaTeX-layout 归一化后完全相同的 MATH500 index 175；因此最保守口径固定为 exact 与 lexical 集合的 30 题并集，不能把 29 题集合本身称为全部重合，也不能称为语义重复的精确估计。

| I8 vs I0 口径 | 保留题数 | 保留样本数 | macro delta | 95% CI | P(delta>0) |
|---|---:|---:|---:|---:|---:|
| 原始固定评测 | 902 | 7,248 | +0.035999 | [+0.020847, +0.051859] | 1.0000 |
| 剔除 13 道 LaTeX-layout exact overlap | 889 | 7,000 | +0.033268 | [+0.018440, +0.048607] | 1.0000 |
| 剔除 exact 与 lexical 的 30 题并集 | 872 | 6,680 | +0.036218 | [+0.020710, +0.052865] | 1.0000 |

step-50 的 `data.pt` 进一步证明 I0/I8 都精确消费 3,200 个 prompt，sampler generator 与对应训练 seed 一致；I0/I8 的 sampler state 完全相同，I16/I32 也使用相同的 seed-42 数据顺序。按运行时 PyTorch `2.8.0+cu128` 和 torchdata `0.11.0+cpu` 重放前 3,200 个索引后，13 道 exact overlap 中实际只消费了 1 道，即 MATH500 index 175，位于 step 30。30 题并集中实际消费 8 道。预先重放的 seeds 43/44 分别会消费 3/1 道 exact overlap，以及 6/3 道 30 题并集；每个 seed 内 I0 与 I8 的数据暴露仍严格相同。按当时尚未修订的原三-seed方案，原计划最终报告同时给出原始固定主指标与同一 30 题并集过滤的 supplementary sensitivity，且不能看到 seeds 43/44 后再改变过滤集合；当前活动执行已由 19.2.27 修订为 seeds 42/43，但同一冻结过滤集合和双视图报告要求保持不变。

I8/I16/I32 共用的 P2 还存在独立的 probe 数据接触路径，因此继续审计其正式 source `phase05_hybrid_shared_rollout_d64c32a16_r32_seed42_v3/probe_summary.json`，source SHA-256 为 `f1bc8c3fc23b56d4853de84756bf7e6090136e7087f7fe0cf044e0ee8b6ce1bb`。artifact 保存了 64 个 discovery、32 个 calibration、16 个 audit 的 DAPO `extra_info.index` UUID 和 112 份对应 rollout cache；112 个 UUID 全部唯一、三组互斥且都能回映到当前训练 parquet。逐题匹配结果是三个 split 对 13 题 exact、29 题 lexical 和 30 题并集的命中数均为 0。因此 P2 的方向发现与 held-out 评分没有直接接触已识别的评测重合题；该结论仍不能排除 base model 预训练污染或低于词法阈值的语义改写。

剔除 13 道精确重合题后，AMC23/MATH500 的 delta 仍分别为 `+0.093750/+0.061741`；剔除 30 题并集后仍为 `+0.111979/+0.061214`。因此已识别的训练题重合不能解释 I8 的单 seed 增益，而且 seed-42 方法间不存在数据暴露差异；但数据并非完全去污染，论文必须披露 13 道 corpus-level exact overlap 与 1 道 actually-consumed exact overlap，并把过滤结果作为 sensitivity 而非替换正式主指标。该词法审计也不能排除更深层语义改写；按原三-seed claim policy，seeds 43/44 曾是“稳定泛化”措辞的必要条件，但该要求已由 19.2.27 的两-seed固定设置复现范围 supersede，当前无论结果如何都不使用“跨训练 seed 稳定泛化”表述。完整 artifact 位于 `analysis/phase1_seed42_train_eval_overlap_sensitivity.json`，修订后 SHA-256 为 `65e39bc856e075240964e0aec4e20d73231686214861675a60e8790a444d5767`。

#### 19.2.22 seed-43/44 held-out 论文 claim gate（2026-10-03）

现有三 seed gate 是 Phase-1 的正式主门槛，但 I8 由 seed 42 的 I8/I16 screen 选出，最终三 seed aggregate 又包含 seed 42，因此该 aggregate 不能单独当作无选择偏差的 held-out 稳定性证明。为防止 seed-42 的 `+0.035999` 掩盖两个新 seed 的不稳定，在文件系统尚无任何 seed-43/44 结果的 `2026-10-03T10:29:48+08:00`，额外固定一个只约束论文措辞、不修改 controller 和原主门槛的 claim gate。

首先必须完整通过原三 seed 主门槛。随后只看从未参与选择的 seeds 43/44，原始固定评测必须同时满足：两 seed 的 macro delta 都严格大于 0、两 seed 均值 `>=+0.005`、六 benchmark 中至少四个的两 seed 平均 delta 为正、任何 benchmark 的两 seed平均 delta 不低于 `-0.02`。同样四项条件还必须在 19.2.21 已锁定的 30 题 `conservative_union` 过滤后再次满足；过滤集合 SHA-256 依赖固定为 `65e39bc856e075240964e0aec4e20d73231686214861675a60e8790a444d5767`，不得看到 held-out 结果后修改。

只有“原三 seed 主门槛 + held-out 原始 gate + held-out 过滤 gate”全部通过，论文才允许使用“跨训练 seed 稳定泛化”表述。若原主门槛通过、但任一 held-out gate 失败，只能表述为 Phase-1 pass 或 promising initialization effect；若原主门槛失败，则方法失败，不从 I32、单 benchmark 或过滤子集寻找补救性亮点。两 held-out seeds 仍不足以精确估计 training-seed population，因此这个加强 gate 是必要的 claim hygiene，不是充分的外部泛化证明；正式投稿仍需标准 LoRA 等论文基线。完整预注册位于 `analysis/phase1_heldout_claim_gate_preregistration.json`，SHA-256 为 `07502ef7ff60070977ae03db4b884ae232fa13e5b34d531a621ace405fdbb55a`。

执行能力审计确认，原三 seed 主门槛已有 `aggregate_multiseed_full_bench.py` 和边界测试，但两套 held-out claim gate 目前只有预注册阈值，没有专用 evaluator。为避免看到 seeds 43/44 结果后再决定平均口径或过滤行为，在尚无 held-out 评测结果时另行冻结 `analysis/phase1_final_gate_execution_spec.json`，SHA-256 为 `b8eb7eaaa72941fbd555e8ca7d8036b0fc88163a62bdf1088d1ae6a84f467fd8`。规范要求 baseline/candidate 每 seed 精确 7,248 条并跨 seed 保持 snapshot/invariant 一致；固定过滤集合必须从 SHA-256 为 `65e39b...5767` 的 overlap artifact 读取 30 个唯一 `(benchmark, problem_index)`，其中 AMC23 16 题、MATH500 14 题，过滤后每 seed 必须精确为 6,680 条、872 题和全部六集。两 seed 和 benchmark 均先按题内样本求均值，再做 benchmark 等权和 seed 等权；严格 `>0` 与非严格 `>=0.005/-0.02` 的边界、四集为正计数和最终三分支 claim policy 均已锁死。controller 活跃期间不修改分析代码；controller 结束后只能按该规范实现并通过列出的正反边界、哈希篡改、键不匹配和计数失败测试，不能根据实际 held-out 数值调整。

#### 19.2.23 Phase-1 通过后的投稿基线缺口（2026-10-03）

当前 I0 是最重要的同参数 random-orthogonal 对照，但不能替代标准 PEFT LoRA。现有 standard LoRA r8、random-B r8 和 original random-B 使用 rank 8/近似自适应 rank、dropout `0.05`、不同 probe 或不同训练链路，只能列为历史外部参照；仓库中没有 standard LoRA r32、alpha 64、dropout 0、三训练 seed 的协议匹配结果。在 I8 尚未通过 held-out gate 前不占用 GPU 补跑，以免为失败候选增加约 30 小时成本。

若且仅若 I8 同时通过 19.2.22 的原三 seed 主门槛、held-out 原始 gate 和 held-out 过滤 gate，投稿前必须补跑标准 PEFT LoRA r32 的 seeds 42/43/44：196 个 `all-linear` 模块、rank 32、alpha 64、scaling 2、dropout 0、50 steps、64 prompts x 8 responses，其余 optimizer、RL、数据与 rollout seed、checkpoint 和固定 7,248-record 六基准协议全部与 Phase 1 相同；同时记录 PEFT 版本和默认 initializer 的精确定义。I8 对 standard LoRA r32 必须按每 seed paired bootstrap 和三 seed hierarchical bootstrap 评估，并通过与 Phase 1 相同的主门槛及 held-out claim gate，才允许声称优于标准 LoRA。I0 对 standard LoRA r32 的三 seed comparison 则用于拆分“随机正交 A”本身与“8 个 P2 + 24 个随机方向”的增量。

此外需在同一 snapshot 上补一份 base model 的 7,248-record 评测，作为无 RL 的绝对参考；若算力允许，再用当前 dropout 0 协议补 standard LoRA r8 seed 42，展示参数/rank tradeoff，不能拿旧 dropout-0.05 结果冒充协议匹配基线。论文同时报告 trainable/stored adapter 参数、峰值显存、训练/评测墙钟以及 112-prompt P2 probe 开销。完整条件计划位于 `analysis/phase1_paper_baseline_requirements.json`，SHA-256 为 `daefd41f3ac580f7b491c6ff32a3dc00534088837e6525e1299913cec1e22ee1`。

#### 19.2.24 I32 seed-42 step-25 关键节点验证（2026-10-03）

按“不逐 step 轮询”的约定，下一次训练侧读取发生在 `2026-10-03T11:15:47+08:00`。controller PID `27424`、trainer PID `587978`、TaskRunner PID `589359` 均存活且仍位于 PGID/SID `27424`。`latest_checkpointed_iteration.txt` 已原子提交为 `25`，mtime 为 `11:14:43.365+08:00`；step-25 日志完整，checkpoint 保存仅耗时 `6.49` 秒。

checkpoint 精确包含预期的 22 个非空文件，无缺失或额外文件：四份 model shard 各 `1,814,714,347` 字节，四份 optimizer shard 各 `74,331,291` 字节，四份 extra-state shard 各 `15,141` 字节，并包含 `data.pt`、PEFT adapter、HF/tokenizer 配置和 FSDP config。payload 合计 `7,715,471,431` 字节，run 目录为 `7,715,491,913` 字节。`data.pt` 可读且记录 25 batches / 1,600 prompts；adapter config 为 rank 32、alpha 64、scaling 2、dropout 0，safetensors 可读并精确包含 392 个 A/B tensor，对应 196 个模块。

step 25 的 entropy `0.926910`、grad norm `0.062493`、score `0.277344`、parse `0.521484` 和全部其余记录指标均为有限值；精确错误扫描未发现 traceback、CUDA OOM、NCCL error/timeout、NaN/Inf、no-space、killed 或 segfault。保存后的根盘可用 `18,340,450,304` 字节，足够再写一份约 7.715 GB 的 step-50 checkpoint。I32 step 25 继续保留，在 step-50 contract 和完整六基准正式验证前不删除；下一次检查固定在 step 50，按最近八步均值约 524 秒估计窗口为 `14:50--15:15`，中间不轮询。完整验证 artifact 位于 `analysis/checkpoint_retention/phase1_i32_uniform_r32_b64m16n8_step50_seed42_v3_step25_validation.json`，SHA-256 为 `007815016babe230d31e53f2460c5acb49dbc9fcf6687dd9949683f6be3d4de6`。

#### 19.2.25 I32 完整结果、seed-42 选择与存储闭环（2026-10-03）

I32 seed-42 v3 于 `2026-10-03T14:48:27+08:00` 正常提交 step 50。checkpoint 精确包含预期 22 个文件，payload 为 `7,715,471,431` 字节；`data.pt` 记录 50 batches / 3,200 prompts，adapter 为 196 个模块、rank 32、alpha 64、scaling 2、dropout 0，392 个 A/B tensor 均可读。step-50 adapter SHA-256 为 `05fd90b6dd106587913c6016375ecf2f7ed3206e3b9959afccdfffec875923ac`，training contract SHA-256 为 `37640cf2274530c001aa08ff2b8d4ea0510f2a050e002382298d5cf5a53c5219`；contract verifier 通过。最后一步 entropy、grad norm、score 和 parse 分别为 `0.863285/0.066143/0.369141/0.570313`，均为有限值，完整日志未发现 traceback、CUDA OOM、NCCL error/timeout、NaN/Inf、no-space、killed 或 segfault。

六基准的四个 shard 均精确写出 1,812 条记录和明确结束标记，四份 manifest 可解析，合并后为 7,248 条无重复匹配记录、六个预定 benchmark。与此前评测相同，三个已完成 shard 的 Python 父进程被各自的 residual EngineCore 卡在退出阶段；在确认全部 shard 行数、manifest、`Wrote 1812 rows` 标记和错误扫描后，只向三个 EngineCore PID `806371/806388/806394` 发送 `SIGTERM`，未终止 shard 父进程、wrapper 或 controller。wrapper 随后自然完成 aggregate，独立 full-benchmark verifier 在存储处置前后均返回 `status=verified`。

I32 的 Macro Avg@k、Macro Pass@k 和 overall parse rate 分别为 `0.376479/0.628601/0.872103`。相对 I0 的 paired problem-bootstrap macro delta 为 `+0.046317`，95% CI `[+0.030682,+0.063484]`，`P(delta>0)=1.0`；逐集 delta 为 AIME24 `+0.051042`、AIME25 `+0.022917`、AMC23 `+0.114844`、HMMT Feb `+0.016667`、MATH500 `+0.066000`、Minerva `+0.006434`，即 seed 42 上六集均为正。comparison SHA-256 为 `3b25ec1ee7a51433347a88fc3089b30a41c0210e7a42fc6b1912ad7822d370b4`。

四组 seed-42 结果形成明显的非单调剂量响应：I0 为 `0.330162`，I8 为 `0.366161`（delta `+0.035999`），I16 为 `0.313498`（delta `-0.016665`），I32 为 `0.376479`（delta `+0.046317`）。因此不能把结果解释成“P2 signal slot 越多越好”或单一的平滑最优混合比例；I16 的失败与 I32 的全面正收益同时存在，更像是特定 signal/random basis 组合与有限步优化路径的交互。I32 仍只是预注册的 hard signal-only endpoint，明确不具备 mixed-candidate 选择资格，不能在看到它优于 I8 后改为追加 I32 seeds 43/44。

进一步的固定-record post-hoc 诊断表明，I32 相对 I0 有 239 道问题改善、559 道不变、104 道下降，benchmark 等权改善/下降比例为 `32.19%/15.02%`；I8 对应为 222/567/113 和 `29.64%/17.54%`。I32 的 seed-42 优势在问题数上确实略广，但不是 I8 收益的简单幅度放大：两者只有 163 道共同改善，59 道仅 I8 改善、76 道仅 I32 改善，正改善集合 Jaccard 为 `0.547`。样本级净正确翻转为 I32 `+373`、I8 `+307`，净格式有效翻转为 `+397/+357`；I32 的 overall 格式有效率和有效格式条件正确率相对 I0 分别提高 `+5.477/+2.960` 个百分点，而且后者在六集均为正。这支持“不同 basis 组合进入不同有限步路径”的描述，但不识别因果机制，也不改变 I8 的预注册选择。完整 artifact 为 `analysis/phase1_seed42_i32_endpoint_diagnostic.json`，SHA-256 `759bccccdf3c24eb5eba42bacaf31f2ac3755c12da2de8f754d1bd5cdf48c75d`；其结果明确禁止用作 I32 追加训练或稳定泛化证据。

selector 依赖在选择后重新固定：`evaluate_single_seed_screen.py` SHA-256 仍为 `eec1662ef6a4993740530678e681da04a51f063b068a1237d02745d04374182c`，`select_phase1_seed42_candidate.py` 仍为 `ed2b580b637d70e0f7fb25a7868c264c15643647132ef1764140c7d7d6355989`，与提前保存的 dependency audit 一致。I8 与 I16 中只有 I8 通过预注册 screen；最终 `phase1_seed42_selection.json` 选择 I8，SHA-256 为 `247fc05892648c41815252cecccb70c0152eddab940be642dc5d29fea65c27c8`。这仍只是“选择 I8 进入复验”，不是稳定泛化结论。controller 于 `2026-10-03T17:15:26+08:00` 自动启动 seed-43 I0；后续只在 step 25/50、完整六基准完成或明确故障时检查，不逐 step 轮询。

正式验证后完成两级不可恢复存储处置。首先，对 I32 `global_step_25` 的 22 个文件逐项记录字节数和 SHA-256 后删除整份 recovery checkpoint，释放约 7.71 GB；审计文件为 `analysis/checkpoint_retention/phase1_i32_uniform_r32_b64m16n8_step50_seed42_v3_step25_deletion.json`，SHA-256 `815146197df2864367be692d57b20bc0ea82f18fc28be5d1d3e1104acf9f14e5`。随后因剩余 16.86 GB 仍不足以安全容纳 seed-43 的两份约 7.72 GB checkpoint，逐项哈希并删除 I32 step-50 的四个 model shard、四个 optimizer shard、四个 extra-state shard和 `data.pt`，共 13 个文件、`7,556,249,984` payload bytes；审计文件为 `analysis/checkpoint_retention/phase1_i32_uniform_r32_b64m16n8_step50_seed42_v3_step50_pruning.json`，SHA-256 `c27956f8c50bb6133dee0ac1cf3ee9bf00ad7528bac88e204ec7a00dff33ad3e`。

裁剪后永久保留 I32 step-50 adapter、HF/tokenizer 配置、FSDP config、training contract、四份 shard/manifests/logs、7,248 条 merged records、JSON/CSV summary、paired comparison 和 selection；9 个 checkpoint 内保留文件逐项哈希复核通过，contract 与 full-benchmark verifier 也再次通过。根盘可用空间升至 `24,414,089,216` 字节（约 22.7 GiB），预计写入 seed-43 step 25/50 后仍保留约 9 GB 运行余量。代价是 I32 的 step-25 与 step-50 optimizer/FSDP recovery state 均不可恢复，不能从本地续训或重建完整 FSDP checkpoint；这不改变已经验证的 adapter 评测结果。

#### 19.2.26 I32 条件式 Phase-1b 确认计划（2026-10-03）

I32 在 seed 42 的 macro delta 为 `+0.046317`，高于 I8 的 `+0.035999`，且六集均为正；因此即使 I8 最终通过 seeds 43/44，审稿人仍会追问 24 个随机补空间方向是否必要。另一方面，I32 从一开始就被预注册为不能参与 Phase-1 mixed-candidate 选择，不能在观察到 seed-42 排名后把它改称原 Phase-1 winner。为同时保留统计纪律和论文机制可解释性，在尚无 seed-43/44 六基准评测结果的 `2026-10-03T18:50:04+08:00`，冻结一项不自动启动的条件式 Phase-1b 计划；artifact 为 `analysis/phase1b_i32_confirmation_preregistration.json`，SHA-256 `13def2ed32f78f8a4a311e02a74465a9d5babc4dac8c2e911d57fb04050ebd23`。

Phase-1b 最早只能在当前 controller 终止、I0/I8 三 seed 全部正式验证且主/held-out gates 已执行之后启动，并需另做容量与算力优先级决策。若执行，只新增 I32 seeds 43/44，复用已验证的 I32 seed 42 与 I0 seeds 42/43/44；固定使用同一 P2、196 模块 uniform-r32、alpha 64、dropout 0、50 steps 和同一 7,248-record 六基准协议，禁止重估 signal 或按 seed 旋转方向。I32 对 I0 必须独立通过与 I8 完全相同的三 seed 主 gate、held-out 原始 gate 和固定 30 题过滤 gate；即便通过，也只能称为看到 seed-42 endpoint 后预先规定、由两个新 seed 确认的 Phase-1b 结果，不能回溯性挽救原 I8 Phase-1。

若 I8 与 I32 都通过，还需对两者做 seeds 42/43/44 精确配对和分层 bootstrap，并重点用未参与 endpoint 排名的 seeds 43/44 判定方向。声称随机补空间优于 signal-only，必须让 `I8-I32` 在两个 held-out seed 均严格为正、均值至少 `+0.005`、至少四集均值为正、最差集不低于 `-0.02`，且原始与固定过滤评测都通过；反向声称 signal-only 优于混合也使用完全对称的 `I32-I8` gate。若两方向都未通过，只能报告两个稳定的 signal-informed 初始化，不能宣称随机补空间或纯 signal 更优。该计划不取代标准 PEFT LoRA r32：I32 是机制控制优先级，标准 LoRA 是论文外部基线优先级，实际启动顺序须在 Phase 1 完成后按算力决定，而不是按中间结果挑选。

#### 19.2.27 用户指示的两 seed 执行范围修订（2026-10-03）

为加快实验进度并先固定当前 `1.5B` 模型与数学 RL 任务上的结论，用户于 seed-43 I0 训练进行中、任何 seed-43 六基准结果尚不可见时，明确要求暂时只跑两个训练 seed。当前活动范围因此从原来的 seeds `42/43/44` 改为 `42/43`：seed 42 是已经用于 I8/I16 选择的 discovery/selection seed，seed 43 是唯一未参与选择的 held-out training seed；seed 44 的 I0/I8 均不启动。原三 seed gate、held-out 两 seed gate、最终 gate 执行规范和 I32 Phase-1b 计划全部保留为历史预注册与未来扩展，但当前不能被称为已执行、通过或失败。

修订后的 seed-43 主确认同时要求原始评测和固定 30 题过滤评测满足：macro delta 至少 `+0.005`、problem-cluster bootstrap `P(delta>0)>=0.90`、至少四个 benchmark delta 为正、最差 benchmark delta 不低于 `-0.02`。两 seed 描述性汇总还要求 seed 42/43 delta 均严格为正、均值至少 `+0.005`、至少四个 benchmark 的两 seed 均值为正、最差两 seed benchmark 均值不低于 `-0.02`。两 seed hierarchical bootstrap 仍报告，但只能描述当前两个轨迹的不确定性，不能冒充训练 seed population 的稳定估计。只有这些条件全部通过，才允许表述：“在该固定模型/任务设置中，I8 相对匹配 I0 的提升在一个 held-out training seed 上复现，并在两个已测试 seed 上均为正。”禁止使用“stable across training seeds”、广泛泛化、原三 seed gate 通过或优于标准 LoRA等表述。

完整修订位于 `analysis/phase1_two_seed_scope_amendment.json`，SHA-256 为 `92d44f339e0129c81c43fd691395c3d87399a17c7fbbe9b42f34149fcf2cc65d`。为避免当时的 controller 在 seed-43 comparison 后自动进入 seed 44，曾部署 transient systemd path guard `peft-phase1-stop-after-seed43.path`，监听 `analysis/I0_vs_I8_step50_seed43_paired.json` 的原子出现；触发时只向 PGID `27424` 发送 `SIGSTOP`，随后要求人工验证 seed-43 artifacts、确认无 seed-44 训练状态并终止 controller。guard 部署时为 `active/waiting`，触发文件和任何 seed-44 artifact 均不存在；这是历史运行时状态，当前无活动 controller，换服务器后的 guard 必须由 19.2.41 的入口重新建立。

#### 19.2.28 两 seed 最终 gate 执行规范冻结（2026-10-03）

在 seed-43 I0 仍处于训练阶段、任何 seed-43 六基准结果尚不可见的 `2026-10-03T22:18:26+08:00`，进一步冻结两 seed 范围的最终执行规范 `analysis/phase1_two_seed_final_gate_execution_spec.json`，SHA-256 为 `7c146c48bfb35759386e32eda2761b20a003fae04f430bceca57c548f462596d`。该规范不改变 19.2.27 的候选、阈值、排除集或允许表述，只关闭尚未明确的实现自由度：single-seed problem bootstrap 和 two-seed hierarchical bootstrap 都固定为 10,000 次、随机种子 42；后者每次先从 `[42,43]` 有放回抽两个训练 seed，再在所选 seed 的各 benchmark 内按问题有放回抽样，但区间与 `P(delta>0)` 只作为描述性不确定性，不参与 pass/fail。

原始 7,248-record 视图与固定排除 30 题后的 6,680-record 视图均需执行 seed-43 gate 和两 seed 描述性 gate，四个 gate 全部通过才允许固定模型/任务复现表述。输入必须在 method 内、seed 内和 seed 间精确匹配 record key 及题目不变量，并通过独立 full-benchmark verification artifact 的 records SHA-256 绑定；最终输出还需保存四个输入 records、依赖代码、范围修订、排除源与执行规范的哈希、完整计数、所有 Boolean check 和最终 claim policy。controller 终止前仍不修改分析代码；终止后须先实现规范中列出的阈值边界、哈希篡改、键/计数/过滤失败和 claim-policy 测试，再读取真实 gate 输出。

随后对已完成的 seed-42 I0/I8 输入重新执行当前冻结的 training-contract verifier 与 full-benchmark verifier，并补齐持久化 verification artifacts。I0 artifact 为 `analysis/verification/phase1_i0_uniform_r32_b64m16n8_step50_seed42_v3_postverify.json`，SHA-256 `0ecb6922f59e7487e6d1ee3bacfc1bcaf8164ad1a76d8fe7824b73ac83fd33d4`，绑定 records SHA-256 `7ea31a89f04ab52e3044f079e3f191e402639b52667fa85cf7b88b79236fd9ad`；I8 artifact 为 `analysis/verification/phase1_i8_uniform_r32_b64m16n8_step50_seed42_v3_postverify.json`，SHA-256 `46fdc7b0108e2decd26951904cfa59e331e2571fd7c8f955e8095dfb58205e2f`，绑定 records SHA-256 `9e1c31b6f4aace7408362fbc612b1ad6f22be8291e7b73c89f270f72b97892b6`。两者均确认 merged records 为 7,248、四个 shard 各 1,812、四份 manifest 存在，contract 与 full-benchmark 状态均为 `verified`；artifact 还保存 adapter、summary、contract、manifest 和 verifier 代码哈希。seed-43 I0/I8 必须生成同结构的独立验收记录后才能进入最终 gate。

#### 19.2.29 两 seed 投稿基线条件修订（2026-10-03）

原 `phase1_paper_baseline_requirements.json` 绑定 seeds 42/43/44 和原三 seed gate，不能在当前两 seed 活动范围中直接执行。于任何 seed-43 六基准结果和任何协议匹配 standard-LoRA-r32 结果可见前，新增条件修订 `analysis/phase1_two_seed_paper_baseline_amendment.json`，SHA-256 为 `02978fcdc0f6ce43682440582deb0ff427fdfeda189d457ad45cf31f13a3931b`。该修订不自动启动实验；只有 I8 的两 seed 固定模型/任务四 gate 全部通过、当前 controller 终止且分析器测试完成后，才允许为 seeds 42/43 启动协议匹配 standard LoRA r32 和同 snapshot base-model 评测。

匹配基线固定为 `all-linear` 196 模块、rank 32、alpha 64、scaling 2、dropout 0、A/B 均可训练、50 steps、64 prompts x 8 responses，并逐项复用 Phase 1 的 optimizer、GRPO、数据/rollout seed、checkpoint 和六基准协议。当前环境为 PEFT `0.19.1`；verl 标准 LoRA 分支未覆盖 `init_lora_weights`，因此使用默认 `True`：线性层 A 为 `kaiming_uniform_(a=sqrt(5))`、B 为全零、初始 delta weight 精确为零。PEFT config/layer 与三份 verl 实现源码哈希已经写入修订并逐项复核。旧 `start_standard_lora_r8_b64_50_4gpu.sh` 的 rank/alpha/dropout/checkpoint provenance 均不匹配，明确禁止冒充论文主基线。

I8 相对 standard LoRA 的原始与固定 30 题过滤比较，均需复用两 seed execution spec 的 seed-43 gate、两 seed 点估计 gate 与描述性层次 bootstrap；只有两视图全部通过，才允许在这个固定 `1.5B` 数学 RL 设置和两个已测试 seed 上声称优于协议匹配 standard LoRA。即便通过，仍禁止“跨训练 seed 稳定”或跨模型/任务广泛泛化表述。预计两份 standard-LoRA 训练加评测串行约 19--21 小时，base-model 评测约 2.3--2.9 小时；若当前 I8 gate 失败则不消耗这些算力。

#### 19.2.30 固定随机补空间下的稳定性边界（2026-10-03）

在 seed-43 I0 仍训练中、任何 seed-43 六基准结果尚不可见时，进一步审计了 Phase-1 中“seed 重复”的实际随机变量。四组 allocation 都只在训练前生成一次，`complement_seed` 固定为 `42`；I0 的 32 个随机正交方向和 I8 的 24 个随机补空间方向随后由所有训练 seed 原样复用。launcher 中 `TRAIN_SEED` 只改变训练数据 shuffle、PPO actor dataloader 和 on-policy rollout seed，不重新生成 `rank_map.json` 或 `subspaces.safetensors`。I0 固定 subspace SHA-256 为 `30e0a7eba1ff06b6798805c2feeb3eb21501c79ea53a71906889ea7051a2e276`，I8 为 `63842778dad6064beeae7c5de6a497e2bfd58d83ee93c2f5f567592045950135`；contract 也绑定这些相同字节。

因此，当前 seeds 42/43 检验的是“同一初始化抽样下，两条不同数据顺序与 rollout 随机轨迹的复现”，而不是随机补空间或初始化 seed 的鲁棒性。即便四个两 seed gate 全部通过，也只能使用 19.2.27--19.2.28 已冻结的固定模型/任务措辞；不得声称对 initialization seed 稳健、对 random-complement draw 稳健，或 24 个随机补方向具有普遍收益。若最终把 I8 作为论文方法，后续初始化鲁棒性必须用新 complement seed 重新生成成对的 I0/I8 allocation，并保持训练与评测协议匹配；单纯追加 training seed 不能覆盖这一维度。若后续独立确认无随机补空间的 I32，则该随机变量不存在，但仍须披露 I32 的 post-selection Phase-1b 地位和固定 P2 discovery source。

完整只读审计位于 `analysis/phase1_fixed_complement_seed_stability_audit.json`，SHA-256 为 `fd6867bda07fc41e366ad6e0feec76157f65e40a018d7577af81188a6be4e4fb`。该 artifact 明确 `active_scope_change=false`、`no_new_run_authorized=true`，不修改当前两 seed 范围；算力顺序仍是先完成当前 gate 和强制的 matched standard-LoRA 基线，再决定是否投入随机补空间鲁棒性实验。

#### 19.2.31 matched standard LoRA 初始化 seed 修正（2026-10-03）

19.2.29 的条件基线规范原本要求不增加独立 initializer seed，但进一步代码审计表明，这一要求不能保证公平。当前 FSDP standard-LoRA 分支在 `get_peft_model()` 前没有显式固定用于 PEFT A 初始化的 torch RNG；engine 配置虽有默认 `seed=42`，却仅在 `full_determinism=true` 时调用，而当前默认值为 `false`。后续 `sync_module_states=true` 只会同步同一次 run 的 rank，不能保证训练 seed 42/43 两个独立 run 得到相同的初始 A。因此若原样执行，standard LoRA 会同时改变训练轨迹和未记录的初始化抽样，而 I0/I8 只改变训练轨迹，比较不对称。

在任何 matched-standard-LoRA-r32 结果产生前，现将其初始化策略前瞻性修正为：新增独立且固定的 `standard_lora_init_seed=42`，训练 seeds 42/43 都在 scoped/forked RNG context 中用 PEFT `0.19.1` 默认 Kaiming-uniform 规则生成位级一致的 A，并验证全部 B 精确为零；构造完成后恢复外层 CPU/device RNG，不能为了初始化而启用全局 deterministic training。`DATA_SEED`、PPO dataloader seed 和 rollout seed 仍分别等于 training seed，因此 matched LoRA 与固定 allocation 的 I0/I8 都只比较 on-policy 训练轨迹变化。

未来实现必须在 optimizer 和首次 rollout 前按 canonical tensor order 保存初始 A/B hash，并要求两个 standard-LoRA contract 的初始状态哈希完全相同；缺失 init seed、跨 run hash 不同、B 非零、PEFT 版本或 initializer source hash 改变均 fail closed。测试还必须证明改 init seed 会改变至少一个 A、固定 init seed 时更改 training-seed metadata 不改变 A/B，并证明 scoped initializer 会恢复调用者 RNG。该修正只提高 matched baseline 的内部公平性，不构成 standard LoRA 或 I8 对 initialization seed 稳健的证据。

完整修正规范位于 `analysis/phase1_two_seed_standard_lora_init_seed_correction.json`，SHA-256 为 `2d66fb8756eae2f0bb984a968a137b01fcf46070c1ad3e4136aa73221b17d28a`。它只覆盖原 amendment 的 `standard_initializer.seed_policy`，明确 `active_scope_change=false`、`automatic_launch=false`、`no_new_run_authorized=true`；当前 controller 终止前不修改 PEFT/FSDP 实现，且仍只有 I8 两 seed gate 通过后才允许实现和启动 matched baseline。

#### 19.2.32 P2 probe 实测计算成本与投稿边界（2026-10-03）

I8 的 50-step 训练不能把 P2 当作免费初始化。正式 source `phase05_hybrid_shared_rollout_d64c32a16_r32_seed42_v3` 的日志于 `2026-09-30T21:39:04.159397308+08:00` 创建，`probe_summary.json` 于 `2026-10-01T02:33:45.102299834+08:00` 原子写入，同一台 4-GPU 机器上的实测墙钟为 `17,680.94` 秒，即 `4.911` 小时或约 `19.65` four-GPU-hours。13 个 probe step 的 `timing_s/step` 累计 `17,541.45` 秒，与文件时间只差约 `139.49` 秒；其中 full-gradient probe backward 累计 `15,215.21` 秒（`4.226` 小时），生成累计 `1,552.98` 秒（`0.431` 小时），因此该成本主要来自真实梯度计算，而非空闲或事后处理。

本次 probe 从 208 个 raw prompt group、1,664 条 rollout 中收集 64/32/16 个有效 discovery/calibration/audit prompt，共 112 个唯一 prompt 和 896 条被接受 prompt 对应的 rollout；P2 随后在同一模型/任务设置的 I8/I16/I32 和各 training seed 间复用。seed-42 I8 的 50 个训练 step 累计 `25,991.46` 秒（`7.220` 小时），所以当前 probe 相当于单次 I8 训练本体的 `68.0%` 墙钟；对两个 I8 training seed 摊销后，训练侧额外开销仍为 `34.0%`，即每 seed 平均 `2.456` 小时。probe 加两份 I8 训练、尚未计评测时合计约 `19.351` 小时，而两份单纯 50-step 训练为 `14.440` 小时。

投稿可以把 I8 与 matched standard LoRA 的 50-step 比较称为“相同训练 steps、相同训练协议”，但不能称为“相同总算力”或“更高计算效率”。必须分别报告 probe、训练和评测的 wall time/GPU-hours，并说明 probe 在同一 model/task 内只支付一次、跨新模型或新任务原则上需重新支付。上述 step-equivalent 只用于描述墙钟，不能把额外 standard-LoRA steps 当成与 probe 优化等价；本轮也不因此自动新增 compute-matched run。若将来没有独立的总算力匹配实验，则不得声称 I8 比 standard LoRA 更 compute-efficient、P2 免费或成本可忽略。

完整成本审计位于 `analysis/phase1_p2_probe_compute_cost_audit.json`，SHA-256 为 `7445c35faf0686dd345d760ad9cc8dc3b17313776deedcefaa70efb7a944fb33`；它绑定 source log、probe summary 和 seed-42 I8 training log 的哈希，并明确 `active_scope_change=false`、`automatic_launch=false`、`no_new_run_authorized=true`。

#### 19.2.33 seed-42 最终权重更新几何诊断（2026-10-03）

对四个已完成的 seed-42 rank-32 adapter 做只读 post-hoc 参数空间分析。按每层 `Delta W = 2 B A`，用 rank-32 Gram 恒等式计算而不物化完整矩阵。I0/I8/I16/I32 的全局 `||Delta W||_F` 分别为 `0.142385/0.138948/0.139673/0.140077`，只相差约 `2.5%`；对应 Macro Avg@k 却为 `0.330162/0.366161/0.313498/0.376479`。更新范数最大的 I0 并非最佳，I8/I32 的收益因此不能解释成“参数更新更大”。

训练日志中的平均 grad norm 分别为 `0.011917/0.066184/0.070362/0.070694`。signal-informed 三组虽比 I0 高约 `5.6--5.9` 倍，最终 `Delta W` 范数却没有更大，说明这个 grad norm 主要受 LoRA 因子参数化、A 子空间朝向与优化器几何影响，不能作为训练强度或泛化代理。最终 A 相对初始 A 的全局 Frobenius 漂移仅为 I0 `0.0245%`、I8 `0.0164%`、I16 `0.0143%`、I32 `0.0154%`；尽管 `lora_freeze_a=false`，50-step 轨迹实际上近似固定 A、主要学习 B。

更重要的是，四组在 196 个模块上的 `||Delta W_m||_F` 分布几乎相同：任意方法对的逐模块 Pearson 为 `0.9980--0.9995`，范数向量 cosine 均高于 `0.9994`；但把所有模块的完整 `Delta W` 拼接后，方法间方向 cosine 只有 `0.0145--0.0810`。因此区别不是“哪些层更新得更多”，而是相同层内沿什么方向更新。signal-informed 三组的平均 module stable rank 约 `11.77--12.39`，高于 I0 的 `5.66`，但 stable rank 最高的 I16 恰好评测最差，所以它同样不是性能代理。这些结果支持“初始子空间朝向通过 on-policy feedback 导向不同有限步路径”的描述，但不构成因果识别：参数 Frobenius 几何没有纳入激活协方差、base-weight 尺度或非线性 function-space 效应。

完整诊断位于 `analysis/phase1_seed42_weight_update_geometry_diagnostic.json`，SHA-256 为 `4d816f6f68ad36275430a8202c1823c838868349a0922efd877cd9495dd4c216`。它绑定四份 adapter、初始 subspace 和训练日志哈希，明确标记为单 seed post-hoc、不得进入候选选择或 gate，且不授权新增实验；如用于论文，controller 终止后还需把当前公式实现固化成可测试的独立分析脚本。

#### 19.2.34 Phase-1 初始子空间嵌套设计审计（2026-10-03）

进一步对固定的 I0/I8/I16/I32 `subspaces.safetensors` 做 CPU-only 只读审计。构造器对同一模块和四个 signal count 都重置同一个 stable seed，先生成完全相同的 `32 x width` 高斯矩阵；signal count 为 `s` 时，输出空间等于前 `s` 个有序 P2 signal direction 与高斯矩阵前 `32-s` 行共同张成的空间。因此对 `s<t`，两组在一般位置下共享前 `s` 个 signal generator 和前 `32-t` 个 random generator，理论公共维数为 `32-(t-s)`。这是一组 generator/span 层面的 signal-slot replacement，而不是四个独立随机子空间。

实际 196 个模块的 principal-angle 审计逐模块精确复现该结构。I0/I8、I0/I16、I0/I32 的公共维数分别固定为 `24/16/0`；I8/I16、I8/I32、I16/I32 分别固定为 `24/8/16`。所有应共享方向的最小 principal cosine 均不低于 `0.9999989`，而第一条非共享方向的全模块最大 cosine 仅为 `0.1751--0.3133`。归一化 projector overlap 分别为 `0.751197/0.504651/0.018386/0.751168/0.260378/0.504661`，与公共维数占 `32` 的比例及有限维随机交叠一致。四组最大正交误差低于 `9.54e-7`；I8/I16/I32 之间共享的前 8 或 16 个 signal rows 在全部 196 个模块上位级一致。

边界必须写准确：每组都会针对自己的 signal prefix 重新对同一批 random rows 做 residualization，所以保留下来的 random-complement **张成分量**共享，但其正交基行通常发生旋转。允许说“I8 相对 I0 在每个模块保留 24 维精确公共子空间并替换 8 维输入方向”，不能说“I8 只是逐行替换 I0 的 8 行、其余 24 行位级不变”。这一匹配设计使 seed-42 的非单调结果更具体地指向“可用输入方向不同”，而非来自互不相关的整套随机 draw；但它仍不能单独证明 8 个 signal slots 导致评测提升，也不提供 random-complement seed、训练 seed、模型或任务泛化证据。

结合 19.2.33，四组虽然初始空间存在上述大维度精确交集，最终完整 `Delta W` 方向 cosine 仍仅为 `0.0145--0.0810`，与“少量初始方向替换可通过 on-policy feedback 导向不同有限步路径”一致；这仍是设计解释加单 seed post-hoc 几何，而非因果识别。完整 artifact 位于 `analysis/phase1_initial_subspace_nesting_audit.json`，SHA-256 为 `34ba15d77a4c9b53f1c62a54f283dec18611ab55af0967aab193c8ce4e11505a`；该审计不修改当前两 seed gate、controller 或运行范围，也不授权新增实验。

#### 19.2.35 seed-43 I0 评测跨卡恢复边界（2026-10-03）

为应对当时约 6 小时的硬卡时限制，对正式四卡评测 wrapper 做只读恢复审计。每个 GPU shard 会先一次性执行本 shard 的全部 1,812 个 request，随后才写出 JSONL；因此正在生成但尚未写完的 shard 没有逐 request checkpoint，卡中断后必须整片重跑。另一方面，完整 shard 是独立确定的 request partition，具备固定 `seed+shard_index`、manifest 和 1,812 条 records，可以跨卡安全复用。

现有 `run_full_bench_vllm_4gpu.sh` 每次固定启动全部四个 shard，外层 Phase-1 wrapper 也只用最终 summary 判断是否跳过评测。因此若硬中断发生在 1--3 个 shard 已完成、summary 尚未出现时，直接重启 controller 会重跑全部四片；这不是结果正确性问题，但会浪费下一张卡的评测时间。为此新增独立、人工调用且不自动启动的 `ops/resume_seed43_i0_eval_missing_shards.sh`。初版 SHA-256 为 `ad9df154ecc82a577d118032f1a105374a099fc3c146efc2891e2033b5c6df18`；经 19.2.40 的半成品状态 fail-closed 收紧后，当前 SHA-256 为 `bce6bb415033ccaaa512b7e6b96b54f5e6a7ef2a9e908f8334c462c692495d80`。它只覆盖固定的 seed-43 I0 eval name，不修改正式 evaluator、wrapper、controller 或 provenance-covered 代码。

恢复器先钉住 benchmark snapshot、Phase-1 preparation、training-contract verifier、evaluator 和正式 full-benchmark verifier 哈希；随后必须由 finalized training contract 对当前 adapter 内容返回 `status=verified`、正确的 I0/seed-43/experiment identity 和非空 `adapter_sha256`，不能仅凭 adapter 路径相同就混用 shard。若 summary 已存在则只做正式验收。否则它拒绝与现存目标 eval 或残留 vLLM EngineCore 并发，逐行流式验证每个可复用 shard：精确 1,812 个唯一 request key、与 snapshot 的题目不变量一致、按全局 request 次序模 4 得到的 partition 完全一致、record/manifest 中 adapter、base model、采样参数、seed 和 shard index 全部匹配。已有非空但验证失败的 shard fail closed，不自动删除或覆盖；只对完全缺失的 shard 按显式 GPU 映射补跑，最后 aggregate 并调用冻结的 `verify_full_bench_run.py`。脚本支持 `--dry-run`，但在该审计时点的训练/评测仍运行时不会调用。

shell 语法和嵌入式 Python 编译均已验证；合成的 1,812-record shard 正向 fixture 通过，随后把 manifest temperature 从 `0.6` 篡改为 `0.7` 的负向 fixture 被正确拒绝。该恢复能力只减少硬中断后的重复计算，不改变任何 records、gate、评测随机性或论文结论；最终处置只走 aggregate-only 且没有补跑 shard，见 19.2.44。

#### 19.2.36 seed-43 I0 step-25 非破坏性留存清单工具（2026-10-03）

按当时容量估计，root 完成 I0 step-50 与评测后预计只剩约 8 GiB，不足以在下一张卡同时容纳 I8 的 step-25 和 step-50 两份约 7.72 GB recovery checkpoint；但训练未完成前绝不能提前删除 I0 step-25。为把历史 I0/I8/I16/I32 的手工留存流程收紧为可重复操作，新增纯 inventory 工具 `ops/inventory_seed43_i0_step25_after_postverify.sh`，修订后 SHA-256 为 `575f8003cf28b601e732bd2adab981f91c0ca740d33ae94d7e47b76425607e12`。该工具不包含任何删除命令，也不接受删除参数。

工具只在固定 seed-43 I0 postverify artifact 存在且 schema、训练 seed、候选、7,248/1,812 计数、training-contract 与 full-benchmark `verified` 状态全部匹配时继续；records、summary、step-50 adapter 和 contract 还必须分别指向该固定 train/eval name 的精确规范路径，不能由 postverify 改指任意同内容文件，随后才重新核对四者哈希。step-25 必须恰好包含预期的 22 个普通非 symlink 文件，文件集合有缺失、额外项或类型变化都 fail closed。通过后才逐文件流式计算绝对/相对路径、字节数和 SHA-256，原子写入 inventory 与 `inventory_ready_no_deletion_performed` manifest；manifest 显式固定 `deletion_performed=false` 和 `irreversible_action_authorized=false`。

在该节记录的历史时点，真实 postverify 尚不存在，运行该工具以状态 2 正确拒绝且没有产生清单。隔离合成测试中，标准 22 文件 fixture 成功生成 22 行 inventory，所有源文件仍存在；加入第 23 个 `unexpected.pt` 后正确拒绝，并确认没有写出 inventory/manifest、额外文件也未被修改；把 postverify records 改指另一个内容相同的路径也被正确拒绝。最终 postverify 与正式 inventory 已按此流程完成，见 19.2.44；没有删除 seed-43 I0 step-25 checkpoint。

#### 19.2.37 seed-43 postverify 证据结构补强（2026-10-03）

在该节记录的历史时点，seed-43 I0 summary 尚不存在、I0/I8 两个 postverify path watcher 均为 `active/waiting`，因此对自动验收脚本做结构审计。原脚本会重新运行 training-contract 与 full-benchmark verifier，并绑定 records、summary、adapter、contract 和自身哈希，但比 19.2.28 已保存的 seed-42 postverify artifact 少四份 shard manifest 哈希以及两份 verifier 代码哈希，不满足“同结构独立验收”的最强口径。seed-43 I0 的最终 summary 与增强后的 postverify 结果见 19.2.44。

因此只增强独立运维脚本 `ops/seed43_i0_postverify.sh`，不修改训练器、评测器、contract、gate 或任何运行参数。修订后 SHA-256 为 `c25685daa65bb66b5f5974857e3e3cc32e61a81359e90983cc1dcd6f542ffaa0`。脚本在确认四个 shard 各 1,812 行时，同时要求四份固定 manifest 均非空，把它们逐份加入同一次 SHA-256 集合与 `artifacts.manifests`；并将实际执行的 `phase1_training_contract.py` 与 `verify_full_bench_run.py` 哈希写入 `verifier_code`。最终原子提交前的 jq 自检要求 manifest 精确四份、所有 path 非空、六个新增哈希均为 64 位字符串。

两个 transient watcher 的 `ExecStart` 都直接引用该脚本路径，未固化旧脚本字节，修订后仍保持 `active/waiting`，summary 出现时会执行增强版本。`bash -n` 和新增 jq 表达式已通过；隔离完整 fixture 模拟 7,248 merged records、四个 1,812-record shard、四份 manifest 和两个 verifier JSON，成功产出 `status=verified`、四个 manifest hash 与两个 verifier hash。删除第 4 份 manifest 的负向 fixture 非零退出且没有写 audit。该补强只提高 provenance 完整性，不读取评测分数，也不改变结果或 gate。

#### 19.2.38 seed-43 I0 step-50 完成与正式评测交接（2026-10-04）

按预定低频节点，`peft-phase1-seed43-i0-step50-check.timer` 于 `2026-10-04T00:45:00+08:00` 触发并以 exit status 0 完成。训练已经在 `00:27` 左右达到 `50/50`，总墙钟 `7:10:07`、日志平均 `516.15 s/step`；`latest_checkpointed_iteration.txt` 精确为 `50`。step-50 checkpoint 包含预期 22 个文件，文件 payload 合计 `7,715,471,431` bytes，目录 apparent bytes 为 `7,715,487,815`。adapter 为 `147,770,464` bytes，SHA-256 `0037992fca30100e435c48f6e03f67f09031949324869c20524eb2dee96b7ec5`。

训练 contract 已从 `prepared` 转为 `complete`，独立 `phase1_training_contract.py verify` 返回 `status=verified`、I0、training seed 43 和固定 experiment name；contract SHA-256 为 `6470539ab870d2e0ef3c75f534186ce5f289839005fc3db3429e0df02311a673`。完整训练日志 SHA-256 为 `7d92e06bf54f18c3bcdad5d96c5d006055e71225a1ce33b9fb5b28d7e4ddfa73`。step 50 的 entropy/grad norm/score/parse 分别为 `0.866897/0.013536/0.416016/0.568359`，单步耗时 `512.77 s`、checkpoint 保存 `5.45 s`；fatal scan 没有运行时错误，grep 命中的 `NCCL_DEBUG` 和 `nccl_timeout` 只是启动配置文本。

同一 controller 已完成到正式六基准评测的交接。`00:45` 观察时四个固定 shard evaluator 和四个 vLLM EngineCore 已运行约 18 分钟，四份 manifest 均已原子写出并绑定哈希；协议保持 seed 42、temperature `0.6`、top-p `0.95`、max-new-tokens `32768`、四卡各一 shard 和固定六 benchmark。summary 尚不存在，符合生成仍在进行的预期；本次审计没有读取 shard progress、records、分数或任何评测结果，也没有执行 gate。root 尚余 `8,849,797,120` bytes，足够写出历史同协议约 0.36--0.39 GB 的完整评测目录。

完整只读 artifact 位于 `analysis/phase1_seed43_i0_step50_training_completion_audit.json`，SHA-256 为 `0fcfd63d19f116fbf1036a18f4a843a350c40cc2fcbef1290bc5617abf3ddb90`。在该审计时点，状态正式转为 `training_complete_evaluation_active`，下一次内容检查由 `03:35` timer 执行，期间不轮询评测；这不是当前状态，最终聚合和验收见 19.2.44。

#### 19.2.39 两 seed analyzer 实现边界准备审计（2026-10-04）

在不读取 seed-43 I0 评测分数的前提下，重新核对两-seed scope amendment、最终 gate execution spec、固定 30 题排除源和当前分析代码。两份冻结规范 SHA-256 仍分别为 `92d44f339e0129c81c43fd691395c3d87399a17c7fbbe9b42f34149fcf2cc65d` 与 `7c146c48bfb35759386e32eda2761b20a003fae04f430bceca57c548f462596d`；paired bootstrap、旧 multiseed analyzer、现有测试和排除源哈希也与 execution spec 逐项一致。

现有 `scripts/analysis/aggregate_multiseed_full_bench.py` 明确硬编码 seeds `42/43/44`、三个 seed bootstrap slot 和原三-seed gate，不能把 only seeds 42/43 强行传入后解释为当前最终 gate。execution spec 同时明确要求：controller 正式停止、seed-43 I0/I8 均完成独立 postverify、四份 records 固定后，才能实现专用两-seed analyzer 和完整边界测试，再读取数值输出。因此当前不修改该 analyzer。准备度审计位于 `analysis/phase1_two_seed_analyzer_readiness_audit.json`，SHA-256 为 `224b2c2ce9a241c97bc40f7897ce29bbd97183fdd573fd9cf62f28f66e0199c4`；状态为 `implementation_deferred_by_frozen_execution_spec`，不是实验结论。

#### 19.2.40 seed-43 I0 跨卡恢复状态机收紧（2026-10-04）

对 19.2.35 恢复器做第二次静态审计时发现，初版会把“shard records 缺失或为空、但 manifest 已存在”的半成品状态也列为可补跑；这比已记录的“只补跑完全缺失 shard”合同更宽。该工具当时未运行，因此只修改独立 ops 脚本：现在仅当 shard records 与 manifest **都不存在**时才标记为 missing；只存在一侧，或任一现有文件为空时，均以状态 2 fail closed，要求人工审计，绝不自动覆盖。

收紧后的 `ops/resume_seed43_i0_eval_missing_shards.sh` SHA-256 为 `bce6bb415033ccaaa512b7e6b96b54f5e6a7ef2a9e908f8334c462c692495d80`，`bash -n` 与 executable-mode 检查通过；环境没有 `shellcheck`，因此不声称该项通过。变更审计为 `analysis/phase1_seed43_i0_eval_recovery_tool_audit.json`，SHA-256 `a4cdce20b3bd358aeb181e604da01457f290f24534435a8e78ae12ed96d4be51`，其中嵌入的新工具哈希已与实际文件交叉验证。本次未执行 recovery、未启动 GPU 进程，也未读取或修改评测输出。

#### 19.2.41 下一张卡 seed-43 I8 受控启动入口（2026-10-04）

现有 seed-44 stop guard 与 I8 postverify watcher 都是 transient systemd unit，实例重启后不会保留；原 controller 又会在 seed-43 comparison 后继续进入 seed 44，因此下一张卡不能裸启动 controller。新增默认无运行时修改的 `ops/start_seed43_i8_next_card_guarded.sh`，支持两种模式：实例和 stopped controller 仍在时，`existing` 模式验证 PID/PGID、停止状态、工作目录及现有 guards；实例重启时，`replacement` 模式先把新 controller staging 为 stopped session leader，绑定其新 PGID 部署 I8 postverify watcher 与 seed-44 stop guard，全部验证后才允许继续。两种模式都只有显式 `--execute` 才会发送 `SIGCONT` 或创建 replacement controller。

入口强制验证 seed-42 selection 仍为 `advance_multiseed/I8`、seed-43 I0 postverify 及其全部绑定哈希、无 evaluator/EngineCore、无 seed-44 artifact、无已存在 comparison，并钉住 controller、method launcher、paired comparator、selector、postverify 和两-seed scope amendment 的代码/规范哈希。replacement 模式还要求 controller lock 空闲、没有任何 seed-43 I8 部分状态和 stale unit；新 controller 在 guard 部署失败时保持 stopped，不会形成无保护启动竞态。由于复用原 controller，它会确定性地重新验证已完成 run，并原子重写 seed-42 comparison/selection artifact，但已有 adapter/summary 会跳过训练与评测；这一行为已显式披露。所有运行时 mutation 之前还要求 `/root` 至少有 `16,500,000,000` bytes 可用空间，用于两份实测合计约 `15.431 GB` 的 I8 checkpoint 以及评测、日志和运行余量；不足时只打印实际/要求字节并拒绝，不自动删除或清理。

工具显式切换到仓库根目录，不依赖调用者 cwd；从 `/tmp` 调用无副作用 help 已通过。当前 SHA-256 为 `7e8e35193eca3c524ee57ff3162ffd72b875958e8e7bffc2025cd8cf6b7aca6c`；`bash -n`、格式检查和 postverify artifact 遍历 fixture 也均通过。容量审计时 `/root` 可用 `8,840,220,672` bytes，距门槛还差 `7,659,779,328` bytes；I0 step-25 实占 `7,715,557,376` bytes，按当时盘面即使只释放它也仅到 `16,555,778,048` bytes，比门槛多约 55.8 MB，尚未计评测最终落盘增长，因此不能预设它单独足够。静态审计位于 `analysis/phase1_seed43_i8_next_card_guarded_start_audit.json`，更新后 SHA-256 为 `214fc96011242b4939e9fb34e8fffa0a6d3d40f32a7930cb24933c2dbe471062`，状态为 `static_ready_runtime_release_pending_i0_postverify`。在该静态审计时点没有调用 `--execute`、没有恢复或创建 controller、没有部署新 unit，也没有执行任何删除；I0 正式 postverify 前该入口保持不可释放。I0 现已 postverify，但换服务器后的实时存储和运行时 preflight 仍须重新执行，见 19.2.47。

#### 19.2.42 COLING 证据就绪矩阵（2026-10-04）

为避免把当前固定设置复验与完整投稿支撑混为一谈，新增 `analysis/phase1_coling_evidence_readiness_matrix.json`，纳入 19.2.43 存储门槛后的 SHA-256 为 `8b48cbd307900ab33cae4d45f1a1a9513d92e08db217c053b2907b6f52533f66`。该 JSON 是当时冻结的准备度快照，不随随后产物自动改写。矩阵逐项区分十个证据轴：seed-42 I0/I8 固定协议结果已独立验证；seed-43 I0 训练完成而评测最后审计为运行中；seed-43 I8 尚未启动；两-seed gate 因输入不全且受冻结实现时序约束而不可计算；matched standard-LoRA-r32、同 snapshot base-model参考、独立 random-complement seed、跨任务、跨模型和完整方法间资源表均尚未闭合。另设 operational readiness，记录当时 seed-43 I8 只有 `8,840,220,672` bytes、低于 `16,500,000,000` bytes 启动门槛，状态为 `not_ready`，且存储 inventory 不授权删除。矩阵内全部 path/SHA 绑定已按实际字节重算通过，`submission_readiness_decision` 明确为 `not_ready`，且 `automatic_launch_authorized=false`。

该快照给出的优先顺序是：先完成并独立验收 seed-43 I0；下一张足够长的卡只运行 seed-43 I8；controller 停止后实现并测试冻结的两-seed analyzer；若且仅若四个 gate 全通过，再补 seeds 42/43 的协议匹配 standard LoRA r32 与同 snapshot base-model评测；之后优先补独立 random-complement draw，若要使用跨设置泛化措辞，还需第二任务和第二模型。其中 seed-43 I0 已由 19.2.44 闭合，但 submission readiness 仍为 `not_ready`，其余缺口不变。即便未来两-seed gate 通过，也只允许 19.2.27 中固定 `1.5B` 数学 RL 设置的 held-out-seed复现表述，不能提前写成训练 seed 稳定、初始化稳健、优于标准 LoRA或广泛泛化。

#### 19.2.43 seed-43 I8 非破坏性存储就绪清单（2026-10-04）

为解决 19.2.41 的容量硬门槛，对实验目录与可重建缓存只做目录级字节盘点，未读取 checkpoint/评测内容，也未删除或移动文件。当前 `/root` 可用 `8,840,220,672` bytes；I0 step-25 实占 `7,715,557,376` bytes；保守按历史最大 Phase-1 评测目录再预留 `402,681,856` bytes 后，仅处理 step-25 的投影可用空间为 `16,153,096,192` bytes，仍低于 `16.5 GB` 门槛 `346,903,808` bytes。

可重建缓存候选包括 uv `449,064,960` bytes、pip `51,445,760` bytes、vLLM compile cache `329,523,200` bytes、DeepSeek Hugging Face hub cache `1,258,328,064` bytes和 LiveCodeBench cache `4,486,094,848` bytes。step-25 加 uv/pip 后的保守投影只高于门槛 `153,606,912` bytes，过于脆弱；再加 vLLM compile cache 可高出 `483,130,112` bytes，但会付出下次重编译卡时；处理 HF 模型 cache 可高出 `911,424,256` bytes，但需先确认无其他 workflow 依赖；LiveCodeBench cache 可高出 `4,139,191,040` bytes，却会损害后续跨任务准备和增加重下载成本。正式 outputs `1.533 GB`、analysis `261.6 MB` 和 provenance logs 必须保留，不列为回收候选。

完整清单位于 `analysis/phase1_seed43_i8_storage_readiness_inventory.json`，SHA-256 为 `8bb0679409e9781ad115fc20034d56b5ec2cd81a88310021a2f93a7e2489475a`。五种投影均通过统一公式和 margin 算术断言；artifact 明确 `deletion_performed=false`、`irreversible_action_authorized=false`、`active_runtime_modified=false`。这是 I0 postverify 前的历史处置顺序；I0 postverify 和 22 文件 inventory 已由 19.2.44 完成，但 checkpoint 仍未删除，后续换服务器接续也不得把本节当成删除授权。

#### 19.2.44 seed-43 I0 离线聚合与正式验收（2026-10-04）

实例卡时结束后，seed-43 I0 的四个 shard 和四份 manifest 仍完整存在，但 summary、merged records 和 postverify 尚未生成。`ops/resume_seed43_i0_eval_missing_shards.sh` 正式运行时重新验证 training contract、adapter SHA-256 `0037992fca30100e435c48f6e03f67f09031949324869c20524eb2dee96b7ec5`，并确认四个 shard 均为精确的 1,812 条合法 JSON；由于 `missing_shards=0`，没有启动 evaluator、EngineCore 或任何 GPU 推理，只调用冻结 evaluator 的 `--aggregate_only` 路径合并现存输出。

最终 merged records 精确为 7,248 条，六个 benchmark 的记录数依次为 AIME24 960、AIME25 960、AMC23 1,280、HMMT Feb 960、MATH500 2,000、Minerva 1,088。独立 full-benchmark verifier 返回 `status=verified`，records SHA-256 为 `a5d8f67ae6461f6577e4891a9049b5d8f37e6f71b1ec5bf0f9152b4ac02224da`，summary SHA-256 为 `f2942bee41a19d31733032e437a184a8b3c5394bb5eef82b64e017df87e1b5de`。正式结果如下：

| benchmark | Avg@k | Pass@k | Parse rate | Hit max rate | Mean length |
|---|---:|---:|---:|---:|---:|
| AIME24 | 0.269792 | 0.666667 | 0.775000 | 0.072917 | 13551.2 |
| AIME25 | 0.190625 | 0.466667 | 0.816667 | 0.076042 | 13642.6 |
| AMC23 | 0.578125 | 0.950000 | 0.800781 | 0.026563 | 7706.6 |
| HMMT Feb | 0.104167 | 0.366667 | 0.886458 | 0.058333 | 14773.4 |
| MATH500 | 0.628000 | 0.780000 | 0.876500 | 0.014000 | 4263.1 |
| Minerva | 0.195772 | 0.319853 | 0.579963 | 0.009191 | 5697.4 |
| **Macro/overall** | **0.327747** | **0.591642** | **0.798565** | **0.037390** | **8951.1** |

seed-43 I0 的 Macro Avg@k 比 seed-42 I0 的 `0.330162` 低约 `0.002415`。这两个 I0 run 共享固定随机子空间，但训练数据顺序与 on-policy rollout seed 不同；接近的点估计为随机基线的两条轨迹提供了有用背景，但没有 candidate 侧的 seed-43 I8，就不能形成 paired improvement、held-out-seed复现或两-seed gate 结论。

随后 `ops/seed43_i0_postverify.sh I0` 再次验证 training contract、固定 snapshot 协议、7,248 条 merged records、四个 1,812 条 shard、四份 manifest 及所有绑定哈希，并原子写入 `analysis/verification/phase1_i0_uniform_r32_b64m16n8_step50_seed43_v3_postverify.json`；该文件 SHA-256 为 `a9cc92ca9971adec8ee2808f5fada3034d7247e3f6d793ca3e08e63b9910f9ff`，状态为 `verified`。之后仅生成 step-25 的 22 文件非破坏性 inventory：文件清单 SHA-256 为 `4f3b7a7a914c35a7d2e71efa56f10209c857365a0d909deeee2dd11ab69a6a82`，inventory manifest SHA-256 为 `42a90c4ab439936158886c33303e18f566a43eab994b6f6f69b97c4ef54078c1`；没有删除、移动或覆盖 checkpoint。

#### 19.2.45 返回自有服务器的初始迁移边界（2026-10-04）

本节记录把初始化二进制加入 Git 前的迁移边界，随后已由 19.2.46 窄范围修订。Git 仓库原本保存训练器、评测器、启动器、分析器、测试和本文，而 `.gitignore` 默认排除 `runs/`、checkpoint、records、日志、数据集和模型权重。当时仅克隆 GitHub 分支不足以继续正式 seed-43 I8；最小 I8 初始化集合包含 `phase1_training_preparation.json`、I8 的 `rank_map.json`、`allocation_summary.json` 和 `subspaces.safetensors`，其 SHA-256 分别为 `d60fc87775a5b1a0e1d27f922b6c4f2b100f0e8614faeb261afaf2c1c88dd386`、`2f20e70333a66f979d7630c83823304d9b9c196d65e86e74ce329200a87478f5`、`0c7ba14adca289fd1d88377d6c91afaab0f1b26914c03b088b0bddf78e55e131` 和 `63842778dad6064beeae7c5de6a497e2bfd58d83ee93c2f5f567592045950135`。正式 preflight 还会验证 Phase-0.5/0.6 来源 artifact，不能绕过校验。

训练和评测的外部依赖仍包括：DeepSeek-R1-Distill-Qwen-1.5B base model、训练 parquet（2,281,735 bytes，SHA-256 `8e3c9314db8b83c61ab62a3dc85e0a704dcc5f3a9d404d236943f719095ce82f`）以及固定 benchmark snapshot records（191,951,094 bytes，SHA-256 `3416ba93286b324a9663777fa472d0b568af5446319f27f876a63b59b3e863da`）。seed-43 I0 的 summary、四份 manifest、training contract、postverify 和 step-25 inventory 已作为小型文本证据进入 GitHub；若要在目标服务器执行 paired comparison 或重跑完整 verifier，还必须另行迁移 I0 merged/raw records 和 step-50 adapter。若要保留本地恢复能力，则额外迁移完整 checkpoint。大型运行产物与 raw records 应使用 `rsync`、对象存储或独立归档传输，并在目标服务器逐项复核本文记录的哈希。

#### 19.2.46 GitHub 最小 I8 初始化二进制修订（2026-10-04）

19.2.45 的默认大文件边界随后按用户的异地继续实验需求做了窄范围修订。GitHub 分支 `experiments/spar-lora-v0` 的提交 `12c316b2dbc239d8889d377ea4c8f7cd36e79480` 显式 force-track 三个正式 I8 preflight 所需、且单文件低于 GitHub 100 MB 硬限制的 safetensors；没有放宽仓库对其他 `*.safetensors` 的 ignore，也没有上传模型、训练数据、checkpoint、adapter、评测 records、rollout cache 或其他 candidate family。

| artifact | bytes | SHA-256 |
|---|---:|---|
| Phase-0.5 `candidates_P2.safetensors` | 65,162,736 | `a9e33ccadf14a0e6253ae22919fac86a4a812df6d4db27ee22a26ca852c4b9dc` |
| Phase-0.5 `atom_scores_P2.safetensors` | 309,232 | `a2fcd4416d2a740abd252a19cb0dc52f7ae2547e13a995c3a8b9befc6ce70564` |
| Phase-1 I8 `subspaces.safetensors` | 65,162,736 | `63842778dad6064beeae7c5de6a497e2bfd58d83ee93c2f5f567592045950135` |

提交前重新执行冻结的 `prepare_phase1_signal_random_artifacts.py --verify-method I8`，返回 `status=verified`：196 个模块、uniform-r32、8 个 signal directions、24 个随机补空间方向、signal-prefix 最大误差 0、最大正交误差 `9.5367431640625e-7`。因此把该分支克隆到相同的 `/root/peft-for-rl` 路径后，P2 来源张量、atom scores、I8 rank/allocation/preparation 文本及最终子空间已经自包含；其他绝对路径下的 base model、训练 parquet 和 benchmark snapshot 仍由目标环境提供。19.2.45 中关于这三个初始化二进制必须外部传输的陈述由本节取代，关于模型、数据、原始评测 records 和可恢复 checkpoint 的边界保持不变。

#### 19.2.47 当前接续状态与迁移清单（2026-10-04）

截至本文更新时间，seed-43 I0 已完成并 verified，step-25 非破坏性 inventory 已生成且 checkpoint 原文件仍保留；seed-43 I8 尚未启动。进程复核未发现 Phase-1 controller、full-benchmark evaluator 或 vLLM EngineCore 在运行。当前不存在需要从本机“续跑”的活动进程，回到自有服务器后的下一项 GPU 工作是单独启动 seed-43 I8，而不是重跑 I0、启动 seed 44 或直接计算尚不完整的两-seed gate。

GitHub 分支 `experiments/spar-lora-v0` 已包含：

- 训练、评测、验证、比较与受控启动代码，以及相关测试；
- Phase-0.5/0.6 与 Phase-1 的小型配置、summary、manifest 和审计 JSON；
- Phase-0.5 P2 candidate、P2 atom scores 和 Phase-1 I8 subspace 三份 safetensors；
- I8 的 preparation、rank map、allocation summary，以及本文档。

GitHub 明确不包含：

- base model、训练 parquet 和固定 benchmark snapshot records；
- seed-43 I0 adapter、checkpoint、merged/raw records 和 evaluator shard records；
- rollout cache、日志及除已列三份之外的 candidate/subspace 二进制；
- 任何 seed-43 I8 训练或评测结果，因为该 run 尚未发生。

目标服务器应优先克隆到 `/root/peft-for-rl`，因为冻结 JSON 与 contract 中保存了该绝对仓库路径；若必须使用其他路径，需要先审计所有 path-bound verifier，而不能直接绕过。恢复外部模型与数据后，先运行 I8 initialization verifier 和 guarded preflight，再按冻结的 seed-43 I8 协议训练/评测并生成独立 postverify。只有 I8 与 I0 的 seed-43 records 都完成哈希绑定后，才能实现和测试两-seed analyzer、生成 paired comparison 并执行四个冻结 gate。

另有离线迁移基线 `/root/peft-for-rl-phase1-0cca959.bundle`，大小 `123,948,750` bytes，SHA-256 为 `78d3bd743508543497410d1f4f60c3cd5c4bd79a5ad10a3b44ea4138c4abcc81`；`git bundle verify` 确认其包含分支到提交 `0cca95911fc36271480318718c9f4d634d8d4d0f` 的完整历史及三份最小初始化二进制。该 bundle 是网络不可用时的迁移基线，不包含本节之后的新文档提交；正常迁移仍以 GitHub 分支最新 HEAD 为准。

论文证据距离也据此分层：完成 seed-43 I8 和四个 gate，只能闭合当前固定 `1.5B` 数学 RL 设置中“一个 held-out training seed 上复现、两个已测试 seed delta 均为正”的最小结论；完整主实验支撑仍缺协议匹配 standard-LoRA-r32、同 snapshot base-model参考、独立 random-complement draw，以及用于跨设置措辞的第二任务和第二模型。当前不能写成跨训练 seed 稳定、初始化稳健、优于标准 LoRA或广泛泛化。

#### 19.2.48 seed-43 I8 异地交接包与单方法恢复入口（2026-10-04）

迁移审计进一步区分了公开 Git 内容与必须由用户控制渠道传输的运行态输入。GitHub 新增机器可读 handoff manifest、payload SHA-256 清单、中文恢复说明和 `ops/run_seed43_i8_migrated_guarded.sh`；这些文件位于冻结训练 code provenance 之外，没有修改训练器、评测器、contract verifier 或 full-benchmark verifier。公开仓库继续保留 I8 初始化空间，但不上传训练 parquet、benchmark records、I0 records 或 adapter：adapter 单文件超过普通 GitHub 100 MB 限制，且数据、题目和模型输出的公开再分发许可未确认。

离线交接包固定包含训练 parquet、冻结 benchmark snapshot、seed-43 I0 merged records，以及 I0 adapter 的 config 和 safetensors。base model 不进入包，可从上游恢复后按五个文件 SHA-256 验收；I0 raw shards、rollout cache、日志及 step-25/step-50 恢复 checkpoint 也不进入包，因为 I0 已完成，后续 paired comparison 只需要 verified merged records，I8 训练不依赖 I0 checkpoint。完整内容、大小、哈希和排除理由由 `runs/phase1-signal-random-v1/migration/phase1_seed43_i8_handoff_manifest.json` 固定。

最终本地交接包为 `/root/peft-for-rl-phase1-seed43-i8-handoff-20261004.tar.zst`，大小 `182,837,537` bytes，SHA-256 为 `60bfd3c6c4291b8b5d1811f04655638321d038903ba82287c5958285e043960e`。归档通过 `zstd -t`，成员白名单精确为上述五个文件，并已逐成员从压缩流重新计算 SHA-256 验收。

原 `start_seed43_i8_next_card_guarded.sh --mode replacement` 会重启完整 controller，在只有最小交接包的目标机器上可能因缺少 seed-42 历史 adapter 而重跑旧流水线，因此不能用于此次迁移。新入口默认 dry-run，钉住选择结果、两-seed amendment、冻结启动器/verifier、全部 payload、base model 和 I8 初始化哈希；同时拒绝 controller/evaluator/EngineCore、seed 44、任何 seed-43 I8 部分状态和不足 `16,500,000,000` bytes 的 `/root` 空间。只有显式 `--execute` 才单独运行 seed-43 I8 revision 3，随后生成 I8 postverify 并退出，不进入 seed 44。

冻结 contract 仍绑定 `/root/peft-for-rl`、`/data/peft-for-rl-runtime` 和既有 Python 环境路径。异地磁盘应通过 bind mount 映射到规范路径；普通 symlink 会被 `Path.resolve()` 展开。此次不冒险改写历史 JSON 或 provenance，任意 checkout relocation 留给逻辑 artifact ID、内容 SHA 和 relocation manifest 完整定义后的协议 v2。具体恢复命令见 `docs/phase1_seed43_i8_migration_zh.md`。

### 19.3 Phase 2：隔离 token selector

若 Phase 0 不能可靠预测训练结果，再直接训练四个 selector 对照，但仍保持 uniform-r32 和同一个 centered estimator：no-mask、random-mask、top-surprisal、advantage-aware stable-band。这里的 mask 仍只参与 probe discovery，不参与 270-step GRPO loss。

需要额外按以下维度报告 mask 组成：

```text
advantage sign and magnitude decile
correct / incorrect rollout
token position decile
sampled surprisal decile
distribution entropy decile
final-answer region / reasoning region
```

这能判断当前 mask 的收益究竟来自高梯度 token、答案尾部格式，还是某种正负优势不平衡。

### 19.4 Phase 3：高 rank 启动后的动态门控

只使用 Phase 1/2 选出的一个初始化，比较：

| 组 | 启动 rank | step 100 active budget | rank 策略 |
|---|---:|---:|---|
| R0 | 32 | r32 | 不裁剪上界 |
| R1 | 24 或等参数预算 | 固定 | 从小 rank 启动控制 |
| R2 | 32 | 与 R1 相同 | step 50 后软门控 |
| R3 | 32 | 与 R1 相同 | 同数量随机 gate，裁剪正则对照 |

R2 对 R1 回答“先高 rank 探索再裁剪是否优于直接小 rank”；R2 对 R3 回答 allocator 是否真的识别了有用分量；R0 对 R2 回答压缩是否损伤泛化。没有这三个对照，单独报告 adaptive rank 没有解释力。

### 19.5 训练、保存与停止规则

所有正式 run 从一开始配置 270 steps，并在 25/50/75/100/150/270 保存：

```text
model shards
optimizer shards
lr scheduler / scaler
RNG state
data position
rank gates and allocator EMA
probe artifact hash and exact rank map
```

评测节点固定为 step 50、100、150、270；不依据训练 reward 临时选择最好 checkpoint。允许在 step 50/100 根据预注册规则停止，例如 Macro Avg@k 明显低于 random-B baseline 且多项 benchmark 同向下降，但停止后不得删 checkpoint。训练日志新增 `Delta W` 和单步 update 的 stable/entropy rank、A 子空间旋转角、gate histogram、正负优势分层 entropy flow。

## 20. 更新后的优先判断

综合本机实验和近期研究，当前最值得押注的不是“更精确地对小批梯度做 top-k”，而是：

1. 用 cross-fit 或 between/within 分解找到跨 rollout 可重复的梯度信号；
2. 只让稳定信号占初始化的一部分，其余 rank 保持随机正交覆盖；
3. 从 r32 启动，待真实训练轨迹形成后用软 gate 收缩，避免小 rank 的早期路径锁定；
4. rank 分配依据 held-out gradient fidelity、稳定性和 function-space contribution，不依据训练 reward；
5. token selector 从“每条 response 的最高 surprisal”改为 advantage-aware 且排除极端尾部，并仅用于 discovery；
6. 全程保留 optimizer state，并把 nominal rank 和 effective rank 分开报告。

这套路线能够同时解释当前两个看似矛盾的结果：random-B 可能因为覆盖广、估计偏差低而稳健；centered covariance 可能因为覆盖 prompt 多样性而达到本机最好结果。下一轮的核心问题不是在两者中选一个，而是验证“可重复的 covariance 信号 + 随机补空间”能否稳定超过二者。
