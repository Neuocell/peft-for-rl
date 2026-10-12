# 下一轮实验计划：学习轨迹与外部有效性

更新时间：2026-10-06 UTC

Owner decision：不再运行 seed44，也不继续用训练预算收紧随机种子方差。现有 seed42/43 用于报告观察到的变异性，不把第三 seed 作为论文或实验推进的前置条件。

Matched standard LoRA seed42/43 已完成训练、六任务评测和 postverify，现已进入主表。下一轮按原定 P0 执行 seed43 Step 25 学习轨迹评测，不新增训练。

## 1. 当前运行与恢复边界

| 实验 | GPU | 当前状态 | 证据边界 |
| --- | --- | --- | --- |
| matched standard LoRA r32 seed42 | 完成并 verified；Macro `0.330053` | Step41 后中断；从完整 Step25 恢复并重算 Step26--41 | 最终结果披露 checkpoint resume；模型、优化器、RNG、scheduler 和 dataloader state 均恢复 |
| matched standard LoRA r32 seed43 | 完成并 verified；Macro `0.338761` | 原始无中断运行 | 与 seed42 共用固定 LoRA init seed42，但 training/data/rollout seed 为43 |

两组均为 7248 records/902 problems，且通过 contract/postverify；两-seed Macro 均值为 `0.334407`。Seed42 恢复记录见 `runs/matched-standard-lora-r32-v2/analysis/recovery/seed42_interruption_and_resume_20261005.json`。

## 2. 下一轮 P0：复用 step25 adapter 补学习曲线

不新增训练，直接评测本机已有的三个 seed43 step25 adapter。2026-10-06 已冻结 contract 和双通道执行队列：

| 方法 | step25 adapter | step50 对照 | 目的 |
| --- | --- | --- | --- |
| matched standard LoRA r32 | 已存在 | 已完成并 postverify | 建立普通 LoRA 的 25→50 step 轨迹 |
| I8 | 已存在 | 已完成并 postverify | 判断 mixed signal 初始化的早期优势是否持续 |
| I32 | 已存在 | 已完成并 postverify | 判断 signal-only 初始化的早期优势是否持续 |

三个 step25 点必须使用同一固定六任务 snapshot、eval seed42、temperature0.6、top-p0.95、max-new-tokens32768、7248 records/902 problems，并各自产生独立 postverify/adapter hash。GPU0--3 先评 standard LoRA、随后自动评 I32；GPU4--7 并行评 I8。不为缺失的 I0 step25 checkpoint 重训 I0。执行 contract 为 `runs/step25-trajectory-v1/analysis/step25_trajectory_evaluation_contract_20261006.json`。

主要分析不是只看 Macro，而是比较 25→50 的：六任务 delta、paired problem bootstrap、训练 reward 与 held-out accuracy 的偏离、response length/parse/boxed 行为变化。它能回答“信号初始化加速早期学习”与“信号初始化改变最终可达性能”这两个不同 claim。

## 3. 下一轮 P1：基于 matched LoRA 结果分叉

| matched LoRA 对比结果 | 下一步 |
| --- | --- |
| I8 在两个已观察 seed 上都优于 matched LoRA，且差距不只来自单一任务 | 当前观察满足前半项，但 seed43 仅 `+0.002934`；先完成 Step25 轨迹与逐任务统计，再进入外部有效性预检 |
| 优势只出现在一个 seed，或主要由单一任务驱动 | 不追加 seed；以 trajectory interaction/benchmark heterogeneity 为论文主线，补统计和行为分析 |
| matched LoRA 与 I8/I32 相当或更好 | 停止方法 superiority 路线；转为初始化与 on-policy trajectory 交互/负结果论文 |

I8 是第二模型复现的默认 signal 方法，因为它是原 selection winner 且保留 random complement；I32 作为当前模型内 ablation，不在外部模型上同时铺开。

## 4. 条件性新训练：第二模型 I0 vs I8

本机候选优先检查 `/home/wangls/models/Qwen3-4B`。正式训练前必须通过以下 fail-closed 预检：

1. tokenizer/chat template、boxed reward、vLLM rollout、FSDP checkpoint 和 PEFT all-linear target 兼容；
2. no-RL 固定小评测不存在明显性能地板；
3. 为 Qwen3-4B 重新生成 gradient-signal candidate/subspace，不复用 1.5B 模型的方向；
4. 冻结 I0/I8 allocation、初始化 hash、训练预算和独立 contract；
5. 4×48 GB 单组训练能在目标上下文长度下稳定运行，且两组预算一致。

通过后使用 8 卡并行：GPU0--3 跑 I0，GPU4--7 跑 I8。只比较同一第二模型内的 paired result，不把它和当前 1.5B 的绝对 Macro 混成排行榜。若 Qwen3-4B 出现性能地板或资源不稳定，则停止该候选，不降级到 0.6B 追求便宜的正结果。

## 5. 暂缓与取消

1. seed44 已取消；不创建 contract、不预留 GPU、不将其列为论文必需实验。
2. 同-adapter 跨 GPU 组复评只在出现直接设备异常证据时执行，不为随机种子差异常规加跑。
3. AdaLoRA 完整 run 暂缓；它不能解决当前方法 baseline 和外部有效性问题。
4. Adaptive-rank control 暂缓；只有论文主线明确包含参数压缩时，才补 uniform-r16 与 learned-map 的同预算比较。
5. 不为补齐 step25 曲线重训已缺失的 I0 或 seed42 checkpoint。

## 6. 非 GPU 并行工作

Step25 评测期间继续做三项整理：完成已有 seed42 I0/I8 records 的 formal gate；生成三方法 25→50 paired trajectory 分析；准备但不自动启动 Qwen3-4B fail-closed 预检。随机种子差异作为已观察限制报告，不再扩展为独立实验线。
