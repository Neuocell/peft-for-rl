# Step 25 进度与外部复现预检计划

更新时间：2026-10-06 05:10 UTC

本文记录 Step 25 轨迹评测的中途状态，并冻结后续决策顺序。它不授权 Qwen3-4B 的 probe 或正式训练。

## 1. 当前执行状态

| 方法 | Step 25 状态 | Macro Avg@k | Step 50 Macro | 25→50 delta |
| --- | --- | ---: | ---: | ---: |
| matched standard LoRA r32 seed43 | 2026-10-06 04:25 UTC 完成并 verified | `0.313818` | `0.338761` | `+0.024943` |
| I8 seed43 | 2026-10-06 04:19 UTC 完成并 verified | `0.328308` | `0.341696` | `+0.013388` |
| I32 seed43 | 2026-10-06 04:26 UTC 启动；约 15%--17% | pending | `0.333201` | pending |

当前 `peft-step25-trajectory-lane-a.service` 正在 GPU0--3 运行 I32。GPU4--7 已空闲，但在 I32 和统一轨迹分析完成前不启动新的正式训练。

## 2. 已完成的 paired 分析

所有比较使用完全匹配的 902 problems/7248 records，problem-cluster bootstrap 10,000 次，bootstrap seed42。

| 比较 | Macro delta | 95% CI | P(delta>0) | 解释 |
| --- | ---: | --- | ---: | --- |
| standard Step50 - Step25 | `+0.024943` | `[+0.008855,+0.042168]` | `0.9990` | 普通 LoRA 在后 25 步继续稳定提升 |
| I8 Step50 - Step25 | `+0.013388` | `[-0.001059,+0.027844]` | `0.9667` | 点估计为正，但 CI 跨 0 |
| I8 - standard at Step25 | `+0.014490` | `[-0.000852,+0.031424]` | `0.9669` | I8 有早期领先，证据处于边界 |
| I8 - standard at Step50 | `+0.002934` | `[-0.010343,+0.016290]` | `0.6667` | 后期领先明显收缩，不能称为稳定 superiority |

逐任务方向并不一致。Step25 的 I8-standard delta 在 AIME24/AIME25/AMC23/HMMT 为正，在 MATH500/Minerva 为负；其中 AIME25 明确为正、Minerva 明确为负。Step50 只有 AIME24/AMC23/MATH500 为正，AIME25 明确转负。当前证据更支持“信号初始化改变早期 trajectory，普通 LoRA 随训练追近，并伴随 benchmark-specific trade-off”，而不是统一提高最终可达性能。

## 3. I32 完成后的固定分析

1. 核对 I32 的 4 shards、7248 records、adapter/snapshot hash 和独立 postverify。
2. 计算 I32 Step25→50、I32-standard Step25/50、I8-I32 Step25/50 的逐任务 delta 和 10,000 次 paired bootstrap。
3. 对 standard、I8、I32 计算 difference-in-differences：`(candidate-standard)_Step50 - (candidate-standard)_Step25`，以 problem 为 cluster，并同时报告六任务等权 Macro 和逐任务区间。
4. 对齐训练日志中的 reward、response length、parse、boxed、hit-max 与 held-out accuracy，检查训练 reward 是否与 full-bench 方向分离。
5. 生成一份独立 trajectory report，再更新论文路线；不依据单个 benchmark 或单个 bootstrap 指标临时修改方法选择。

## 4. 当前论文决策

在 I32 结果未知时，主线暂定为机制/审慎经验研究：signal-informed initialization 提供早期 trajectory bias，但收益随 training seed、训练阶段和 benchmark 改变。I8 仍是外部模型的唯一 signal 方法候选，因为它是原 Phase-1 selection winner 且保留 random complement；不在第二模型同时铺开 I32，也不增加 seed44。

若目标是方法性能论文，第二模型 I0 vs I8 仍是必要证据，但它属于高成本、条件性扩展。若目标是机制/负结果论文，完成 I32 轨迹和现有 formal gate 后即可先写主结果，第二模型不是阻塞项。

## 5. Qwen3-4B 已完成的静态预检

候选模型为 `/home/wangls/models/Qwen3-4B`，权重总大小约 8.04 GB。

| 检查 | 状态 | 结果 |
| --- | --- | --- |
| Transformers/PEFT/vLLM 版本 | pass | Transformers `4.57.3`、PEFT `0.19.1`、vLLM `0.11.0` 均识别 Qwen3 |
| tokenizer/chat template | pass | thinking on/off 均可渲染；正式协议仍需冻结是否启用 thinking |
| PEFT all-linear meta construction | pass | 252 modules，suffix 为 q/k/v/o/gate/up/down，rank32 时 66,060,288 trainable params |
| FSDP wrap discovery | pass-static | 模型声明 `Qwen3DecoderLayer`；仓库按 `_no_split_modules` 动态发现，不依赖 Qwen2 类名 |
| boxed reward | pending-runtime | 需受控 completion 单测和实际 rollout smoke |
| vLLM LoRA rollout | pending-runtime | 需本地权重加载、LoRA attach、长输出和显存实测 |
| FSDP checkpoint/resume | pending-runtime | 需 4-GPU 2-step smoke，包括 adapter export 和 resume |
| no-RL floor | pending | 需预先冻结的小评测，不根据结果换 subset 或 thinking mode |
| Qwen-specific P2 probe | pending | 必须重新生成；禁止复用 1.5B 的 196-module directions |

## 6. 存储与资源约束

`/home` 当前只剩约 86 GB。现有 1.5B checkpoint 每个约 7.2 GB；按参数规模外推，Qwen3-4B 的两个方法各保留 Step25/50 很可能逼近或超过本地安全余量。

正式外部复现的大 checkpoint 根目录应放在 `/media/nfs_share/wangls/peft-for-rl-external-validity-v1`。该 NFS 当前约有 928 GB 可用且用户可写。模型和训练数据继续从本地只读加载，Ray 临时目录放 `/dev/shm`。启动前必须验证 NFS 的持续写入、`fsync`、原子重命名和至少 200 GB 的保留空间；验证失败时停止，不在 86 GB 本地盘上冒险启动双 run。

原 1.5B P2 probe 实测耗时 4.91 小时、约 19.65 four-GPU-hours，峰值主机内存估计 180 GiB。Qwen3-4B 的 probe 预计更慢且内存更高，必须先生成按 252 modules/4B hidden sizes 计算的新内存预算；不能沿用原脚本的 132/180 GiB 常量。

## 7. 外部复现的 fail-closed 顺序

1. `T0 trajectory closeout`：I32 verified，统一轨迹报告完成。
2. `Q0 asset freeze`：全量模型 shard、tokenizer、config、数据、代码和 package lock 哈希冻结。
3. `Q1 inference/reward smoke`：1-GPU base + LoRA attach、thinking 模式、boxed reward、max-length 行为通过。
4. `Q2 no-RL floor`：运行预先冻结的小评测；若 parse/accuracy 出现明显地板则停止该模型，不更换小模型追正结果。
5. `Q3 4-GPU train smoke`：标准 LoRA 2-step，验证峰值显存、FSDP、vLLM sync、checkpoint、adapter export 和 resume。
6. `Q4 fresh P2`：使用相同 discovery/calibration/audit 定义重新生成 Qwen P2，记录 probe wall/GPU-hours；若内存或 capture gate 失败则停止 I8。
7. `Q5 paired contract`：固定 I0/I8、training seed42、init/complement seed42、I8 signal fraction8、50 steps、相同 batch/rollout/eval；禁止根据 smoke 结果调 I8 fraction。
8. `Q6 paired execution`：GPU0--3 I0、GPU4--7 I8；只在 Qwen3-4B 内比较，不与 1.5B 绝对分数混排。

在 `Q0`--`Q4` 全部通过并生成新的机器可读 contract 前，`Q5/Q6` 不得启动。
