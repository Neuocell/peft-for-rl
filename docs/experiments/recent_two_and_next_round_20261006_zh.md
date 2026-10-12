# 最近两次实验汇总与下一轮执行单

更新时间：2026-10-06 UTC

## 1. 最近两次实验

最近完成的是完全同协议的 standard LoRA r32 training seed42/43。两组均为 DeepSeek-R1-Distill-Qwen-1.5B、196 个 all-linear module、rank32、alpha64、dropout0、50 optimizer steps，并使用相同的固定初始化，初始化 SHA-256 为 `039db13cae337b17c4e233341afa4252fa46c645d16943493cf0b82e55f167ad`。

| Run | 训练过程 | Macro Avg@k | Macro Pass@k | Parse rate | 验收 |
| --- | --- | ---: | ---: | ---: | --- |
| standard LoRA r32 seed42 | Step 41 后 worker 中断；从完整 Step 25 恢复并重算 Step 26--41，最终完成 Step 50 | `0.330053` | `0.567031` | `0.801600` | verified |
| standard LoRA r32 seed43 | Step 0--50 无中断 | `0.338761` | `0.613206` | `0.815259` | verified |

两 seed 的 Macro Avg@k 均值为 `0.334407`。每组均包含固定六任务、902 problems、7248 records，并通过独立 postverify。

| Benchmark | Seed42 | Seed43 | Seed43 - Seed42 |
| --- | ---: | ---: | ---: |
| AIME24 | `0.246875` | `0.242708` | `-0.004167` |
| AIME25 | `0.197917` | `0.248958` | `+0.051042` |
| AMC23 | `0.610156` | `0.599219` | `-0.010938` |
| HMMT Feb | `0.092708` | `0.101042` | `+0.008333` |
| MATH500 | `0.647000` | `0.632000` | `-0.015000` |
| Minerva | `0.185662` | `0.208640` | `+0.022978` |

Seed42 的结果可用，但必须随结果披露恢复边界：Ray worker 在 Step 42 计算期间死亡，最后完整步为 Step 41；恢复使用 Step 25 的 model、optimizer、extra state、RNG、scheduler 和 dataloader state，Step 26--41 被重算。恢复记录位于 `runs/matched-standard-lora-r32-v2/analysis/recovery/seed42_interruption_and_resume_20261005.json`。

## 2. 与 Phase-1 的对照

| 方法 | Seed42 | Seed43 | 两-seed mean |
| --- | ---: | ---: | ---: |
| I0 random baseline | `0.330162` | `0.327747` | `0.328955` |
| I8 signal+random | `0.366161` | `0.341696` | `0.353928` |
| I32 signal-only | `0.376479` | `0.333201` | `0.354840` |
| matched standard LoRA r32 | `0.330053` | `0.338761` | `0.334407` |

I8 相对 standard LoRA 在 seed42/43 分别为 `+0.036108` 和 `+0.002934`，两-seed mean 为 `+0.019521`。I32 分别为 `+0.046426` 和 `-0.005560`，两-seed mean 为 `+0.020433`。

当前最稳妥的结论是：I8 在两个已观察 training seed 上都高于同协议 standard LoRA，但 seed43 的优势很小；I32 在 seed43 发生反转，显示更强的 seed sensitivity。只有两个 training seed，且 I8/I0 固定复用了同一个 random-complement seed42，因此这些结果不能支持广义 seed stability、初始化随机性鲁棒性或跨模型/任务泛化。

## 3. 下一轮：Step 25 学习轨迹

本轮不新增训练，只评测本机已有的三个 seed43 Step 25 adapter：standard LoRA、I8、I32。冻结协议与 Step 50 相同：六任务 snapshot SHA-256 `3416ba...63da`、eval seed42、temperature0.6、top-p0.95、max-new-tokens32768、902 problems、7248 records。

执行队列：

| Lane | GPU | 顺序 | Systemd service |
| --- | --- | --- | --- |
| A | 0--3 | standard LoRA Step 25，然后 I32 Step 25 | `peft-step25-trajectory-lane-a.service` |
| B | 4--7 | I8 Step 25 | `peft-step25-trajectory-lane-b.service` |

冻结 contract：`runs/step25-trajectory-v1/analysis/step25_trajectory_evaluation_contract_20261006.json`。每个任务必须在启动前核对 base、snapshot、adapter hash 和 LoRA config，拒绝覆盖已有输出；完成后必须有 4 个 shard、7248 条唯一 records、正确的 manifest 字段及独立 postverify。

完成后三组主要分析为：各 benchmark 的 Step 25→50 delta；同 problem paired bootstrap；训练 reward 与 held-out accuracy 的偏离；response length、parse、boxed 和 hit-max 行为变化。该分析回答“signal initialization 是否加速早期学习”与“是否改变 Step 50 终点”，不回答外部有效性。

## 4. 下一决策门

Step 25 三点完成后，先形成轨迹分析再决定第二模型预检。外部有效性候选仍是 `/home/wangls/models/Qwen3-4B` 上的 I0 vs I8 配对复现，但必须先通过 tokenizer/chat template、boxed reward、vLLM、FSDP/PEFT、no-RL 小评测和 4×48 GB 稳定性预检，并为新模型重新生成 gradient-signal subspace。未通过则停止该候选，不用更小模型追求正结果。
