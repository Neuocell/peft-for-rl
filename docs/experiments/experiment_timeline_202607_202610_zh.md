# 实验时间线：2026-07 至 2026-10

更新时间：2026-10-05 UTC

本文把分散在 `docs/`、`runs/analysis/`、Phase-0 和 Phase-1 目录中的实验按时间重排。日期精确到已有产物能支持的粒度；缺少明确启动时间的旧实验只标月份或阶段，不反推虚假的具体日期。

## 1. 设备和协议边界

| 设备域 | 主要时期与实验 | 已知边界 |
| --- | --- | --- |
| Tina/旧本机历史目录 | 7--9 月 stable LoRA/OFT、BISO/GeoRA/BOET/Skew、RLPO、AdaLoRA 等 | 许多 run 使用不同训练步数、dropout 或旧 benchmark 数据；119-run inventory 中大多为 B/C/D |
| 9 月本机，4 x NVIDIA L40S | SPAR、random-B、full-gradient、token-mask covariance 和早期 Phase-0 设计 | 旧总档案记录的硬件；不能与后来的 A6000 本机视为同一设备域 |
| 原服务器 `/root/peft-for-rl` | Phase-0.5/0.6、Phase-1 seed42 screen、seed43 I0、I8 rev3 | 旧 manifest 缺 GPU UUID、driver 和完整 package lock；部分产物迁移后本机只有 summary/postverify |
| 当前本机 `/home/wangls/peft-for-rl`，8 x RTX A6000 | 2026-10-04/05 的 seed43 I8 rev4、I32 v1、证据盘点 | I8 使用 GPU 0--3，I32 使用 GPU 4--7；driver 590.48.01 |

同一设备上的分数也可能因 training seed 和 on-policy trajectory 不同而变化；不同设备上的分数更不能在没有 matched control 时直接解释为硬件效应。当前 I0 seed42/43 只差 `-0.002416`，而 I8/I32 分别下降 `-0.024465/-0.043278`，更符合方法与 training-seed 交互，而不是统一设备偏移，但旧服务器环境记录不足以严格排除设备差异。

## 2. 2026-07 至 08：stable baseline 与旧方法族

### Stable LoRA/OFT

- 最早的 stable LoRA 文档记录 base `0.315514`，step100/200/300 为 `0.331179/0.340859/0.346207`。该 base 是 summary-only 的旧口径，现已被严格审计 base `0.324348` 替代。
- 旧 inventory 还找到一条具有 records、但训练协议不匹配 Phase-1 的 LoRA 轨迹：step100--300 约 `0.389839` 到 `0.409672`。它使用 100--300 steps、dropout 0.05，且没有固定跨 seed 初始化，属于 B-grade 历史上下文。
- OFT step100--600 的历史宏平均依次为 `0.323712/0.337397/0.359758/0.378069/0.374849/0.366732`，属于 C-grade 描述性结果。

### 旧方法族盘点

截至 2026-10-04，统一盘点覆盖 119 个 full-bench 目录：active repo 23、legacy Tina 96；A=3、A-reference=1、B=23、C=72、D=20；45 个有 records，74 个只有 summary。

这一阶段还包含 BISO、GeoRA、BOET、Skew、SPO 等分支。历史代表点包括 BISO step200 `0.410365`、BOET step200 `0.394733`、GeoRA 约 step500 `0.385174`、Skew step100 `0.386569`。这些值只能说明旧探索达到过的量级，不能和 Phase-1 的 50-step frozen protocol 排名。

## 3. 2026-09 上旬：RLPO、AdaLoRA 与 adaptive-rank

### RLPO-init 与 CHR-RLPO

- RLPO-init r32 完成 270 steps。旧 full-bench 的 HMMT 输入损坏，原六任务 `0.378684` 无效；用 HMMT-only 修复后的描述性六任务 Macro 为 `0.39865`。
- D1 activation-rank 离线诊断得到 mean rank `16.78`、范围 8--28、约为 r32 的 `50.09%` 参数/FLOPs；calibration/validation rank Spearman `0.9871`。它主要改善最差模块的 retained energy，而不是全局平均。
- CHR-RLPO 使用 D1 map 做静态异构 rank，从 base 重新训练。step200 Macro `0.380842`；完成到 step270 后为 `0.393234`。训练在 step106 后遇到 vLLM wake-up OOM，并从 step100 在没有 optimizer state 的情况下恢复，因此轨迹不是严格连续实验。
- 这些结果说明 RLPO 初始化和静态压缩值得保留，但没有同 step、同初始化的 fixed-rank control，不能把 CHR 的变化单独归因于 heterogeneous rank。

### AdaLoRA

- 自研 adaptive-LoRA 原型在 gate、FSDP、vLLM 权重格式和显存上多次失败，没有 full-bench。
- 唯一 full-bench 来自有已知实现缺陷的早期 official AdaLoRA step150：HMMT 输入损坏，Minerva 缺 36 条；五个有效任务宏平均 `0.3720`。
- 后续 official-fixed run 又使用 `orth_reg_weight=0.5`，正交项约 `2.6`，远大于常见 PG loss `0.01--0.15`。当前代码默认已降至 `1e-3`，但没有用最终实现完成六任务实验。
- 结论：AdaLoRA 当前是工程诊断分支，不存在可引用的有效“方法分数”。

## 4. 2026-09 中旬：梯度/子空间系列

### 本机 probe 系列第一批

统一历史表中主要结果为：

| 方法 | Step | Macro Avg@k | 状态边界 |
| --- | ---: | ---: | --- |
| token-mask centered covariance adaptive eq-r28 | 100 | `0.382131` | 历史描述；与 uncentered 的 mask 和 optimizer continuity 不同 |
| full-gradient mean / Gain-LCB uniform-r32 | 150 | `0.375749` | 缺同 step/rank 对照 |
| centered covariance adaptive eq-r28 | 50 | `0.375451` | 历史描述 |
| uncentered covariance adaptive eq-r28 | 100 | `0.369779` | 与 centered 非严格单变量消融 |
| original random-B near-r32 | 50 | `0.367304` | rank map 几乎退化为 r32 |
| full-gradient mean r8 | 50 | `0.358565` | 历史描述 |
| random-B uniform-r8 | 50 | `0.358323` | 比同档 standard LoRA r8 高，但非当前 Phase-1 |
| SPAR adaptive eq-r8 | 50 | `0.348220` | allocator 证据不足 |
| windowed consensus r8 | 50 | `0.348194` | 历史描述 |
| standard LoRA r8 | 50 / 100 | `0.323111 / 0.315783` | 只属于这一批旧协议 |

这一阶段最稳定的共同观察是：训练 reward 上升不保证 full-bench 上升；初始化 span 会影响早期训练；但“找到宽 candidate span”和“正确分配 rank/atom”是两个问题。

### GradTop、Stable-SNR、Dominant 与压缩

- GradTop v1 的 Macro 从 step50 `0.391198` 上升到 step150 `0.404574`，之后 step200/250/270 为 `0.401638/0.399401/0.399411`。训练 reward 后期仍上升，但准确率已平台化，主要变化转为 parse 提高和输出缩短。
- Stable-SNR v2 的 rank 分配出现 attention/MLP 预算塌缩，未启动正式训练；v3 修复类型预算后，独立 validation rank32 capture 仍只有 `21.59%`，fixed-A gate 被拒绝。trainable-A pilot 只跑 50 steps，未做 full-bench。
- Dominant-atoms eq-r8 step50 Macro `0.373654`。其训练 reward 有竞争力，但事后 allocator 验证没有显示可靠跨模块排序。
- GradTop step100 后做 activation-rank compact，mean rank `18.29`，step150 Macro `0.397868`；相对未压缩 step150 约 `-0.67 pp`，95% CI 跨 0。它支持“可压缩”，不支持“压缩提高准确率”。
- GradTop step100→270 chord，lambda 1.25 的 Macro `0.403345`。相对 step270 准确率 `+0.39 pp`，95% CI `[-0.63,+1.42] pp`；format `+0.93 pp` 且输出约短 433 tokens。它是行为控制 Pareto 点，不是准确率改进证据。
- TinyLoRA 只完成 8 steps、无 checkpoint/full-bench；与 dominant main run 的 batch reward 基本相同，但端到端更慢。它只是诊断。

## 5. 2026-09-30 至 10-01：Phase-0、0.5、0.6

- Phase-0 完成 P0/P1/P2/P3 probe 诊断。P3 没有同时改善 held-out capture 和稳定性，判定 no-go；P2 成为后续唯一已验证 source。
- Phase-0.5 测试 P2 加 P3 hybrid directions，H2/H4/H8/H16 均未通过 calibration admissibility，判定 no-go。
- Phase-0.6 crossfit replay 的 candidate/basis/atom-score invariants 大面积失败，诊断为 `crossfit_estimator_failure`。C candidates 永久禁止进入训练。
- 明确回退到只使用已审计 P2 的 I0/I8/I16/I32，随后冻结 Phase-1 allocation、contract 和评测协议。

## 6. 2026-10-01 至 10-05：Phase-1

### 方法和 screen

四组均为 uniform-r32、alpha64、dropout0、196 modules、50 optimizer steps：I0=`0 signal + 32 random`，I8=`8+24`，I16=`16+16`，I32=`32+0`。I32 是 signal-only endpoint，不参加 I8/I16 mixed-candidate selection。

- Seed42 screen 在原服务器完成：I0 `0.330162`、I8 `0.366161`、I16 `0.313498`、I32 `0.376479`。
- I8 在合规 mixed candidates 中胜出。I32 分数更高，但只能进入单独 Phase-1b。
- 原三-seed I8 confirmation 后来显式修订为 seeds 42/43；这只能支撑固定 setting 的 held-out seed replication，不能支撑广义 seed stability。

### Seed43、迁移和本机完成

- 原服务器 seed43 I0 完成并 verified：`0.327747`。
- 原服务器 I8 rev3 在 23/50 中断且无 checkpoint，不是实验结果。
- 2026-10-04 本机完成 119-run evidence audit，并验证 no-RL base `0.3243481413` 可严格复用。
- 本机 GPU 0--3 从冻结 base/allocation 独立启动 I8 rev4，2026-10-05 01:15 UTC 完成训练、7248-record full-bench 和 postverify：`0.341696`。
- 本机 GPU 4--7 并行完成 I32 seed43 v1，2026-10-05 01:43 UTC 完成并 verified：`0.333201`。它只属于 Phase-1b endpoint。

### 当前快照

| 方法 | Seed42 | Seed43 | 两-seed简单均值 | 当前解释 |
| --- | ---: | ---: | ---: | --- |
| I0 | `0.330162` | `0.327747` | `0.328955` | matched random baseline，跨两 seed 接近 |
| I8 | `0.366161` | `0.341696` | `0.353928` | 两 seed 均高于 I0，但 seed43 效应缩小 |
| I16 | `0.313498` | 未运行 | n/a | seed42 negative screen |
| I32 | `0.376479` | `0.333201` | `0.354840` | signal-only endpoint，seed sensitivity 更强 |

Seed43 paired bootstrap：I8-I0 `+0.013949`，95% CI `[0.001000,0.027188]`；I32-I0 `+0.005454`，CI `[-0.005884,0.016747]`；I8-I32 `+0.008494`，CI `[-0.004310,0.022078]`。

I8 seed43 只有 3/6 benchmark delta 为正，已无法满足预注册的至少 4/6 条件。因此 fixed-setting replication gate 不可能全条件通过；formal gate 仍需取回 seed42 I0/I8 records 才能完整执行 filtered view 和最终分类。
