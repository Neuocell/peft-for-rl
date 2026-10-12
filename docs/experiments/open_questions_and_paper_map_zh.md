# 开放问题、后续实验与论文映射

更新时间：2026-10-06 UTC

本文不自动授权新训练。它把当前已经回答的问题、最低成本证据闭环、可选实验和论文路线分开。

## 1. 已回答与仍未回答

| 问题 | 当前答案 | 还缺什么 |
| --- | --- | --- |
| No-RL base 能否复用？ | 能；严格审计 Macro `.324348` | 无需重跑 |
| Signal-informed init 是否在 seed42 有效？ | I8/I32 相对 I0 明显为正 | 只属于 selection seed |
| I8 是否在 held-out seed43 复现？ | 宏平均仍为正，但缩小且仅3/6 tasks为正 | formal filtered gate；其全条件 pass 已不可能 |
| I8 和 I32 谁更好？ | 两-seed mean 几乎相同且排序反转 | 不再用新 seed 强行排序；作为未决差异报告 |
| 两 seed 差距是否由设备导致？ | I0 接近，代码/allocation hash一致，设备不是首要解释 | 同-adapter跨卡组复评或取回seed42 adapter本机复评 |
| 是否优于普通 LoRA？ | I8 在 seed42/43 分别高 `0.036108`/`0.002934`；I32 在 seed43 反转 | Step25 轨迹、paired bootstrap，以及外部模型复现；当前不能称为稳定 superiority |
| Adaptive rank 是否有效？ | 压缩可能，但现有 allocator 因果证据不足 | 同预算 fixed-rank 与 learned-map control |
| AdaLoRA 表现如何？ | 未知；旧结果实现或评测无效 | 最终代码从step0完整六任务 run |
| 是否跨模型/任务泛化？ | 没有证据 | 至少一个外部模型或任务，另行预注册 |

## 2. 先补证据，不先训练

### P0：取回 seed42 I0/I8 records 并执行 formal gate

期望 SHA-256：I0 `7ea31a89f04ab52e3044f079e3f191e402639b52667fa85cf7b88b79236fd9ad`；I8 `9e1c31b6f4aace7408362fbc612b1ad6f22be8291e7b73c89f270f72b97892b6`。

按冻结 `phase1_two_seed_final_gate_execution_spec.json` 生成 fixed 30-problem overlap filter、two-seed descriptive hierarchical bootstrap 和最终 decision。已知 seed43 unfiltered positive-count 为 3/6，所以结果只能落在“promising but not fixed”或“failed confirmation”，不能变成全 gate pass。

### P0.5：只在需要排除设备解释时做复评

优先级从高到低：

1. 取回 seed42 adapters，在本机相同评测栈复评；
2. 用同一 seed43 adapter 在 GPU 0--3 和 4--7 各评一次；
3. 只有发现同 adapter 异常差异时，才考虑重训设备复现。

复评可检验 evaluator/卡组差异，不能完整复原原服务器训练时的 binary/hardware 条件。

## 3. 后续训练决策表

| 目标 | 最小新增实验 | 能支持什么 | 仍不能支持什么 |
| --- | --- | --- | --- |
| 方法性能论文最低 baseline | matched LoRA seeds42/43（已完成并 verified） | 相同50-step预算下是否优于普通LoRA | 跨模型/任务泛化 |
| 区分早期加速与最终性能 | matched LoRA/I8/I32 的已有 seed43 step25 adapter 统一评测 | 25→50 学习曲线与行为变化 | 跨模型泛化 |
| 外部有效性 | 第二模型 I0 vs I8 paired run | 方法是否离开当前1.5B模型仍有效 | 广泛跨任务/规模泛化 |
| Adaptive-rank claim | uniform-r16、D1/learned-map，同初始化同step | heterogeneous map 是否胜过同预算 fixed rank | 训练前可部署性，除非 map 也不使用 hindsight |
| 修复 AdaLoRA 证据 | 最终实现完整 run + 六任务 | AdaLoRA 在正确实现下的结果 | 多seed稳定性 |
| Robustness/generalization | 第二模型 I0 vs I8 paired run | 当前方法的外部模型迁移 | 广泛跨任务/规模泛化；成本明显上升，需新预注册 |

Owner 于 2026-10-05 决定不再运行 seed44。随机种子差异作为限制报告，不继续消耗训练预算收紧方差；后续资源转向 matched baseline、已有 checkpoint 学习曲线和条件性外部模型复现。

## 4. 实验分支状态

| 分支 | 状态 | 建议 |
| --- | --- | --- |
| Stable LoRA/OFT 与旧 BISO/GeoRA/BOET/Skew | 已完成的历史上下文 | 冻结，不为追分重跑；论文附录列协议 |
| RLPO-init | 完成、结果经HMMT修复 | 保留为历史初始化基线；若用于主表需 matched step/snapshot |
| CHR-RLPO/D1 | 完成但有 hindsight 与 resume 混杂 | 只有要写压缩线时才补 fixed-rank controls |
| AdaLoRA | 实现修复完成，实验无效/未完成 | 不进入现有主结果；是否重跑取决于论文路线 |
| GradTop | 完整历史轨迹，当前 snapshot 不兼容 | 保留机制与行为分析；不要冒充 Phase-1 matched baseline |
| Stable-SNR v2/v3 | fixed-A gate no-go；pilot无fullbench | 暂停，除非重新定义 held-out marginal objective |
| Dominant-atoms | 完成50-step；allocator反事实失败 | 作为低rank/allocator负结果 |
| Compact/Chord | 完成 | 可作压缩和行为控制附录 |
| TinyLoRA | 8-step diagnostic | 不继续，除非独立提出参数压缩问题 |
| Phase-0 P3/hybrid/crossfit | no-go/invalid | 永久冻结，不事后复活 C candidates |
| Phase-1 I0/I8/I16/I32 | 当前主证据 | 闭合 formal gate；不再追加 seed；先完成 Step25 轨迹，再做第二模型 fail-closed 预检 |

## 5. 三条论文路线

### 路线 A：保守的 fixed-setting empirical study

核心结论：基于 RL gradient signal 的初始化能在 selection seed 上产生强收益，但 held-out training seed 的收益缩小，且未通过预注册 benchmark-coverage 条件；on-policy RL 的初始化结论需要 seed-aware confirmation。

当前证据足以写完整、诚实的结果章节。建议结构：动机；P2/I-method；预注册协议；seed42 screen；seed43 confirmation；trajectory/benchmark heterogeneity；设备与证据边界；限制。

主表放 Base、I0、I8、I16、I32；明确 I8 由 seed42 选择、I32 是事后单独预注册的 endpoint、two-seed scope 是 amendment。历史方法只在背景/附录按协议族列出。

### 路线 B：方法性能论文

Matched standard LoRA seeds42/43 已补齐；最低还需至少一个跨模型或跨任务复现。建议主表为 Base、matched LoRA、I0、I8、I32，I16 放 ablation；seed42/43 的差异直接作为限制，不再要求第三 training seed。

在补齐之前不能用旧 B-grade LoRA 的高分或旧 r8 LoRA 的低分代替 matched baseline，也不应称为 SOTA 或 robust method。

### 路线 C：机制/负结果论文

核心结论：candidate span 可以促进早期学习，但 on-policy trajectory 放大 initialization × training-seed 交互；宽 span 发现、atom 排序、rank 分配是三个不同问题；训练 reward、held-out accuracy 和生成行为会分离。

可联合使用 Phase-1 seed reversal、GradTop 平台期、Stable-SNR gate failure、Dominant allocator failure、Compact/Chord 的压缩与行为控制结果。下一步补统一 trajectory/update geometry 分析，不再补第三 seed；该路线对 matched LoRA 的依赖弱于方法性能路线。

## 6. 推荐决策顺序

1. 取回 seed42 I0/I8 records，执行并归档 formal gate。
2. 选择论文主线：方法性能，或 seed-sensitive mechanism/负结果。
3. 当前先评测现有 Step25 adapter 并统一 trajectory 分析；方法性能线随后对第二模型做 I0 vs I8 fail-closed 预检与配对复现。
4. 仅当复评发现同-adapter设备异常，才投入完整设备复现实验。
5. AdaLoRA、adaptive rank 和跨领域代码/agent任务都应视为独立扩展，不与 Phase-1 主线同时无限展开。

更细的 Phase-1 执行要求见 [Phase-1 后续实验与论文结构](../../runs/phase1-signal-random-v1/analysis/next_experiments_and_paper_structure_20261005.md)。
