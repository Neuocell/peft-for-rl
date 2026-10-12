# 方法与结果登记表

更新时间：2026-10-06 UTC

本表按协议族登记结果，不构造跨协议排行榜。`Macro` 默认指六任务等权 Macro Avg@k；修复拼接或任务不全会单独标注。

## 1. 当前证据口径

| 等级 | 判定 | Claim boundary |
| --- | --- | --- |
| A | 冻结 Phase-1 协议 + contract/postverify | 可用于当前固定模型/任务/step 的主要比较 |
| A-reference | 严格审计 no-RL base | 只能作绝对位置，不是 paired training-seed baseline |
| B | records 和评测口径可复用，但训练不匹配 | 可作历史量级，不可支持 Phase-1 superiority |
| C | summary 完整但 records/协议不足 | 只作描述性历史 |
| D | 不完整、修复拼接、已知损坏、screen 或不兼容 | 诊断、失败记录和方法设计依据 |

设备、step、dropout、数据顺序、初始化 seed、benchmark snapshot 和 parser 任一不同，都可能阻止因果比较。

## 2. Phase-1 冻结协议与结果

共同协议：DeepSeek-R1-Distill-Qwen-1.5B、GRPO、196 all-linear modules、uniform rank32、alpha64、dropout0、50 steps、64 prompts/step、8 responses/prompt、LR `1e-6`、PPO mini-batch16/epoch1。评测为固定六任务、902 problems、7248 records、eval seed42、temperature0.6、top-p0.95、max-new-tokens32768。

| 方法 | Signal + random | Seed42 | Seed43 | 两-seed mean | 证据/角色 |
| --- | --- | ---: | ---: | ---: | --- |
| Base, no RL | n/a | `0.324348` | n/a | n/a | A-reference；严格复用 |
| Matched standard LoRA r32 | n/a | `0.330053` | `0.338761` | `0.334407` | A matched external baseline；seed42 有已披露的 Step25 resume |
| I0 | 0 + 32 | `0.330162` | `0.327747` | `0.328955` | A matched baseline |
| I8 | 8 + 24 | `0.366161` | `0.341696` | `0.353928` | A；原 mixed winner |
| I16 | 16 + 16 | `0.313498` | 未运行 | n/a | seed42 summary-only negative screen |
| I32 | 32 + 0 | `0.376479` | `0.333201` | `0.354840` | 单独 Phase-1b endpoint |

| 比较 | Seed42 delta | Seed43 delta | Seed43 paired 95% CI | 当前结论 |
| --- | ---: | ---: | --- | --- |
| I8 - I0 | `+0.035999` | `+0.013949` | `[0.001000,0.027188]` | 两 seed 点估计为正，但 seed43 仅 3/6 tasks 为正，预注册复现条件不满足 |
| I8 - matched standard LoRA | `+0.036108` | `+0.002934` | 未计算 | 两 seed 点估计为正，但 held-out seed 差距很小；不能称为 seed-stable superiority |
| I16 - I0 | `-0.016665` | n/a | n/a | 不晋级 |
| I32 - I0 | `+0.046317` | `+0.005454` | `[-0.005884,0.016747]` | seed43 不确定；不能补救 I8 gate |
| I32 - matched standard LoRA | `+0.046426` | `-0.005560` | 未计算 | 跨 seed 反转；两-seed mean delta `+0.020433` 只作描述 |
| I8 - I32 | `-0.010318` | `+0.008494` | `[-0.004310,0.022078]` | 排序跨 seed 反转，不能判断 mixed 或 signal-only 更优 |

允许的表述：I8 在两个已观察 training seeds 上都高于 matched standard LoRA，且两-seed mean delta 为 `+0.019521`；但 seed43 差距仅 `+0.002934`，幅度对 seed 敏感。禁止的表述：seed-stable、普遍 robust、random complement 明确优于 signal-only、跨模型/任务泛化。

## 3. Stable LoRA、OFT 与旧方法族

### 3.1 最早 stable 文档轨迹（C-grade）

| 方法 | Step | Macro | 说明 |
| --- | ---: | ---: | --- |
| Base old | 0 | `0.315514` | summary-only 旧口径；不再作为当前 base |
| Stable LoRA | 100 | `0.331179` | 历史 baseline |
| Stable LoRA | 200 | `0.340859` | 历史 baseline |
| Stable LoRA | 300 | `0.346207` | 历史 baseline |

### 3.2 有 records 的旧 LoRA 轨迹（B-grade）

| Step | 100 | 200 | 220 | 240 | 260 | 280 | 300 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Macro | `.389839` | `.402729` | `.409959` | `.399797` | `.406434` | `.405869` | `.409672` |

这些 run 使用 100--300 steps、dropout0.05，且没有 Phase-1 的 fixed-init cross-seed contract。分数高不代表 I-method 失败，也不能替代现已完成的 50-step matched standard LoRA seeds42/43。

### 3.3 OFT（C-grade）

| Step | 100 | 200 | 300 | 400 | 500 | 600 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Macro | `.323712` | `.337397` | `.359758` | `.378069` | `.374849` | `.366732` |

### 3.4 其他旧方法族代表点

| 方法族 | 代表点 | 等级/边界 |
| --- | ---: | --- |
| BISO | step200 `.410365` | B-grade 代表；非 Phase-1 matched |
| BOET | step200 `.394733` | C-grade |
| GeoRA | 约 step500 `.385174` | C-grade；另有子空间动力学诊断 |
| Skew | step100 `.386569` | C-grade |
| SPO 与其他 adapters | 见 119-run inventory | 不用单点做当前 leaderboard |

## 4. September 梯度与子空间方法

### 4.1 第一批 probe/full-gradient 结果

| 方法 | Step | Macro | 证据边界 |
| --- | ---: | ---: | --- |
| token-mask centered covariance adaptive eq-r28 | 100 | `.382131` | centered/uncentered 非严格消融 |
| full-gradient mean / Gain-LCB uniform-r32 | 150 | `.375749` | 缺同 step/rank control |
| centered covariance adaptive eq-r28 | 50 | `.375451` | 历史描述 |
| uncentered covariance adaptive eq-r28 | 100 | `.369779` | mask/optimizer continuity 混杂 |
| original random-B near-r32 | 50 | `.367304` | 192/196 modules 为 r32，不证明 adaptive rank |
| full-gradient mean uniform-r8 | 50 | `.358565` | 历史描述 |
| random-B uniform-r8 | 50 | `.358323` | 只和同批 r8 baseline 有限比较 |
| uncentered covariance adaptive eq-r28 | 50 | `.350770` | 历史描述 |
| SPAR positive-probe eq-r8 | 50 | `.348220` | rank allocator 未独立验证 |
| windowed consensus uniform-r8 | 50 | `.348194` | 历史描述 |
| full-gradient mean uniform-r8 | 100 | `.340163` | reward 与 full-bench 不同步 |
| standard LoRA r8 | 50 / 100 | `.323111 / .315783` | 只属于本批协议 |

可支持：初始化方向在旧 r8 协议中有明显相关性；centered covariance 是值得后续做干净消融的候选。不可支持：token mask 的独立贡献、adaptive rank 优势、centered 相对 uncentered 的单变量因果结论。

### 4.2 RLPO / CHR-RLPO（历史描述）

| 对象 | Step/状态 | Macro/指标 | 证据边界 |
| --- | --- | ---: | --- |
| RLPO-init r32 | 270 complete | corrected `.39865` | 原 HMMT 损坏，靠 HMMT-only repair；与 LoRA step200 不同步数 |
| D1 activation-rank | offline | mean rank16.78；50.09% r32；Spearman `.9871` | teacher-forced energy oracle，不是 reward oracle |
| CHR-RLPO | 200 | `.380842` | OOM 后无 optimizer-state resume |
| CHR-RLPO | 270 | `.393234` | 约半参数接近 r32，但缺 matched fixed-r16/r32 control |

### 4.3 GradTop、Dominant、Compact 和 Chord

| 方法 | Step | Macro | 解释 |
| --- | ---: | ---: | --- |
| GradTop | 50 | `.391198` | 快速早期学习 |
| GradTop | 100 | `.403326` | 接近峰值 |
| GradTop | 150 | `.404574` | 该轨迹宏平均峰值 |
| GradTop | 200 / 250 / 270 | `.401638 / .399401 / .399411` | 后期 reward 上升而准确率平台/回落 |
| Dominant atoms eq-r8 | 50 | `.373654` | 低秩能训练；allocator 未通过反事实检验 |
| Compact GradTop mean-r18.29 | 150 | `.397868` | 保留60.77%参数，约损失0.67 pp；CI跨0 |
| Chord lambda1.25 | post-hoc | `.403345` | 准确率相对step270不确定；format/长度改善可靠 |
| TinyLoRA | 8 training steps | 无 full-bench | batch reward近似持平，当前实现更慢；诊断而非结果 |

这些 September full-bench 在当前 Phase-1 inventory 中因 snapshot/协议不匹配只能作历史描述，不应与 A-grade Phase-1 放在同一无协议排行中。

### 4.4 Stable-SNR 分支

| 版本 | 完成内容 | 未完成/判定 |
| --- | --- | --- |
| v2 | probe、rank map、回看诊断 | attention/MLP budget collapse；拒绝训练 |
| v3 fixed-A | discovery/calibration/validation probe | rank32 held-out capture低；fixed-A gate拒绝 |
| v3 trainable-A | 50-step pilot | 无 full-bench，不能判断 held-out 性能 |

## 5. AdaLoRA 登记

| 分支 | 状态 | 可用性 |
| --- | --- | --- |
| 自研 adaptive-LoRA prototypes | 多次在 gate、checkpoint、vLLM/FSDP 或内存处中止 | 无 full-bench，不可引用 |
| 早期 official step150 | 五有效任务 Macro `.3720`；HMMT 无效；Minerva 少36条 | D-grade 工程诊断 |
| official-fixed / isolated | 最远到 step123/23 等；`orth_reg_weight=0.5` 主导 PG | 不能解释方法性能 |
| 当前最终代码 | 默认 `orth_reg_weight=1e-3`，同步/allocator 路径已修复 | 尚无从 step0 完成的六任务 run |

正确写法是“最终 AdaLoRA 实现尚未完成有效实验”，而不是“AdaLoRA 得分为某值”。

## 6. Phase-0/0.5/0.6 与失败记录

| 对象 | 状态 | 后果 |
| --- | --- | --- |
| Phase-0 P0/P1/P2/P3 | 完成；P3 no-go | P2 为后续 validated source |
| Phase-0.5 H2/H4/H8/H16 | no-go | hybrid 不进入训练 |
| Phase-0.6 crossfit replay | invalid，`crossfit_estimator_failure` | C candidates 永久禁止训练 |
| I8 seed43 rev3 | 23/50 中断，无 checkpoint | 不是结果；rev4 独立重跑 |
| Phase-1 formal two-seed gate | 未执行完整 | 缺本机 seed42 I0/I8 records；已知 3/6 positive 使全条件 pass 不可能 |

失败记录本身是重要证据：它们约束哪些方向不能在论文中事后包装成成功方法。

## 7. 当前最可靠的跨实验观察

1. 找到有用 candidate span 与正确排序/分配 atom 是不同问题；GradTop 宽 span 有效，但 prefix 排序接近随机。
2. 在多条初始化中，短期训练主要由 B/left side 重组，A/right span 的功能漂移很小；这支持 fixed-A 消融，但不能替代该实验。
3. On-policy reward、held-out accuracy、parse 和长度会分离，不能只用训练 reward 选方法。
4. Phase-1 中 I0 跨 seed 稳定，而 I8/I32 更敏感；现有证据更支持“signal-informed initialization 与 trajectory seed 强交互”，不支持广义鲁棒性。

## 8. 证据入口

- Phase-1 current evidence：[experiment_evidence_index_20261005.json](../../runs/phase1-signal-random-v1/analysis/experiment_evidence_index_20261005.json)
- Historical inventory：[local_fullbench_evidence_inventory_20261004.json](../../runs/phase1-signal-random-v1/analysis/local_fullbench_evidence_inventory_20261004.json)
- September combined report：[rlpo_oracle_gradient_probe_experiment_summary_20260921/report.md](../../runs/analysis/rlpo_oracle_gradient_probe_experiment_summary_20260921/report.md)
- RLPO retrospective：[rlpo_experiment_retrospective_20260914/report.md](../../runs/analysis/rlpo_experiment_retrospective_20260914/report.md)
- AdaLoRA retrospective：[adalora_experiment_retrospective_20260914/report.md](../../runs/analysis/adalora_experiment_retrospective_20260914/report.md)
- TinyLoRA diagnostic：[tinylora-aligned-main-8step-20260920/report.md](../../runs/analysis/tinylora-aligned-main-8step-20260920/report.md)
