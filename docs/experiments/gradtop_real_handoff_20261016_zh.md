# 当前 GradTop-r32 与 REAL 对照：实验及研究交接

**后续统一状态（2026-10-18）：** 见[REAL / GradTop近期汇总](real_gradtop_summary_20261018_zh.md)及[已提交证据](evidence/real_gradtop_20261018/README.md)。本文是至150阶段的历史交接，后文“未启动”“尚未修复”等判断均有当时时点；最新GradTop270、新REAL-P256至210和实际同步验收以新总表为准。

更新时间：2026-10-16，Asia/Shanghai；至150的续训及评测于09:57完成。仓库：`/root/peft-for-rl`。本文件仅以 `rlvr_v1_gradtop_r32_a64_lr1em5_b64m16n8_seed42_20261015` 为GradTop研究对象，整理其当前实现、10/50/100/150轨迹、同协议REAL/LoRA对照和下一步研究问题。

实际配置与源码审计已确认：step51/101两次恢复后的首次rollout走了仅加载base权重、未注册当前LoRA adapter的代码路径；对这两个历史批次没有运行时engine tensor/hash实测。Actor/model、optimizer、RNG及数据状态恢复验收属实，但不能把这条采样轨迹视作严格连续恢复。详见第4节。

**15:37执行更新：** 用户已授权超过17:00，从150续到200并完整评测；该续训已于15:35启动，修复后的首次rollout已通过实际四GPU的源权重、注册adapter、活动槽位及请求ID验收。此验收不追溯消除51/101对原轨迹的影响。最新进度、细粒度新增结果及修复边界见[150→200执行记录](gradtop_r32_resume200_syncfix_20261016_zh.md)；本文下方研究分析以至150的已有证据为基础。

## 1. 用户目标、已经确定的取舍

后续执行更新（2026-10-16 22:14）：200训练及六榜评测已完成，Avg.47.9857%、Macro42.0933%，相对150整体未显示稳定提升；用户继续授权200→250，保持同一GradTop配置和probe，重新验收首轮GPU adapter同步。最新运行入口及250−200/150比较安排见[200→250执行记录](gradtop_r32_resume250_20261016_zh.md)。

目标是在当前仓库协议下，以 **rank32** 为主，判断 GradTop 的低成本探测及其 RL 梯度聚合方式能否形成有说服力的方法贡献，并明确与 REAL 的关系。

- 用户明确不满意把 sign、mask、fixed-A、归一化、rank 微调叠成一个“新方法”。先解释什么方向值得保留，再由可证伪发现导出完整方法。
- 保留当前 GradTop 的 batch 梯度聚合作为研究起点。**本次 GradTop 没有额外的 prompt 随机符号**；GRPO advantage 的正负与额外 Rademacher sign 是两回事。
- 增量 SVD 是 GradTop 与本地 REAL Nyström 实现的实质差别之一，但既不自动构成论文贡献，也不自动等于在线适应策略漂移。
- GradTop只使用上方唯一experiment的配置、artifact、训练轨迹及评测；后文所有“GradTop”都指这一版本。
- 当前这一版已按授权从完整step50续到100/150并分别完成评测，复用原probe及方法配置。150→200的后续授权、运行时验收及自动评测另见上方执行记录。
- 新会话的首要工作是审计、判断研究价值并形成具体可证伪方案。研究诊断和新增候选的执行范围应另行明确，不把本文中的计划当成已完成结果。

当前最值得保留的问题是：**有限 rank 的 RL 适配应当保留逐题 reward-gradient 能量，还是保留能形成有效共享更新、并预测后续策略改善的方向？二者的差别来自采样波动、真实题间异质性，还是 on-policy 策略变化？**

## 2. 先读这些证据

| 阅读顺序 | 入口 | 作用 |
| --- | --- | --- |
| 1 | [完成状态](../../runs/ops/gradtop_r32_continue_step150_20261015/status.json)、[续训结果表](../../runs/ops/gradtop_r32_continue_step150_20261015/results.md)及[机器可读结果](../../runs/ops/gradtop_r32_continue_step150_20261015/results.json) | 100/150均已完成；最新结果的直接来源 |
| 1a | [10:36结果补充审计](../../runs/ops/gradtop_r32_continue_review_20261016/evidence_review.md)及[150−100配对结果](../../runs/ops/gradtop_r32_continue_review_20261016/paired_step150_vs_step100.json) | 新端点逐条指标重聚合及新增轨迹区间；不等于采样恢复语义审计 |
| 1b | [训练与恢复审计](../../runs/ops/gradtop_r32_continue_review_20261016/training_audit.md)及[机器可读记录](../../runs/ops/gradtop_r32_continue_review_20261016/training_audit.json) | 1–150训练日志、checkpoint状态与恢复首轮base-only代码路径的证据及边界 |
| 1c | [step150后是否继续的总判断](../../runs/ops/gradtop_r32_continue_review_20261016/README.md) | 当前收益、同步缺陷、探索续训与论文干净轨迹的区别；建议尚未执行 |
| 2 | [RLVR_PROTOCOL.md](../../RLVR_PROTOCOL.md) | 新主实验的统一规则和step计数 |
| 3 | [GradTop当前协议定义](gradtop_r32_clean_baseline_20261015_zh.md) | 当前实际运行版本与probe细节 |
| 4 | [REAL实现与论文对照](../real_reward_gradient_probe_zh.md) | 本地REAL目标、筛选、Nyström、fixed-A与论文偏差 |
| 5 | [GradTop首轮结果](../../runs/ops/gradtop_r32_clean_20261015/results.md)及[00:50证据审计快照](../../runs/ops/gradtop_real_handoff_20261016/evidence_audit.md) | 首轮来源、当时九组结果与成本；该快照不包含后来完成的GradTop100/150 |
| 6 | [共享更新研究路线](gradtop_update_geometry_research_20261015_zh.md) | H1/H2/H3、双重复rollout、采样噪声/异质性/漂移诊断 |
| 7 | [本次方法审计](../../runs/ops/gradtop_real_handoff_20261016/methods_audit.md)及[研究审阅](../../runs/ops/gradtop_real_handoff_20261016/research_handoff.md) | 代码、论文与研究边界的独立核对 |

证据冲突时，以对应实验的冻结plan/contract、artifact、原始records与验证结果为准。00:50审计及工作区文件均为当时快照，保留原样；100/150新结果以原续训目录的results、paired文件及checkpoint/evaluation验证为准。仓库存在大量既有未提交改动，不能仅凭git HEAD复现当前方法；见[工作区快照](../../runs/ops/gradtop_real_handoff_20261016/workspace_snapshot.json)，运行plan的逐文件SHA256才描述实际冻结代码。

## 3. 当前协议及比较的边界

| 项目 | 当前主实验设置 |
| --- | --- |
| 模型 / GPU / precision | DeepSeek-R1-Distill-Qwen-1.5B / 4×L40S / bf16 |
| 数据 | 原17,917行boxed数学数据，共用本机local_asset_binding及seed42 sampler |
| Adapter | 196个all-linear模块；uniform rank32、alpha64、dropout0 |
| LR | 1e-5，constant，warmup0，weight decay0 |
| 正式训练 | DAPO-style GRPO；64 prompts/trainer step；K=8；PPO minibatch16；1 epoch |
| 长度 | prompt1024、response8192 |
| PPO clipping | low0.20 / high0.28 / C10；无额外KL、entropy或overlong shaping |
| 更新计数 | 每trainer step=4次逻辑Adam更新；step50/100/150=200/400/600次 |
| 累计训练采样 | step50/100/150分别为3,200/6,400/9,600 prompts；probe另计 |
| 评测 | AIME24、AIME25、AMC23、HMMT Feb各32次；MATH500、Minerva各4次；共902题、7,248 responses |
| 评测采样 | 同一benchmark snapshot；seed42、temperature0.6、top-p0.95、max_new_tokens32768 |
| 保存 | 每10步adapter snapshot；每50步及计划终点完整恢复点；保留最近2份完整点 |

**当前正式训练的`token-mean`是每个microbatch内部取token mean，再按response样本数比例累积并跨DP平均。它一般不等于整个逻辑minibatch的全局token mean。** 新GradTop显式probe使用全局token分母，REAL probe又使用每response长度归一化，三者不能直接视为同一个梯度目标。

LoRA/GradTop正式训练A+B，共36,929,536个可训练参数；REAL固定A、训练B，共20,643,840个。它们是同rank、同训练step的端到端方法比较，**不是等可训练参数比较，也不是只变一个初始化的消融**。B-only少约44.10%可训练参数，不代表整卡显存同比下降。

正式run通过`local_asset_binding`登记共同本机资产；后续实验继承这套登记及父协议SHA。实际GradTop协议：[gradtop_r32_clean_20261015.json](../../configs/experiments/gradtop_r32_clean_20261015.json)。

## 4. 当前实验身份与状态

**2026-10-16 09:57:04（北京时间）最终状态为`complete`，已完成评测为[10,50,100,150]，`auto_resume=false`。** Step100/150完整checkpoint分别通过400/600次逻辑Adam更新、四rank RNG/scheduler及6,400/9,600题dataloader进度验收；两次新评测均为902题、7,248条回答且四shard与整体结果验证通过。来源：[step100验收](../../runs/ops/gradtop_r32_continue_step150_20261015/checkpoint_step100_verification.json)、[step150验收](../../runs/ops/gradtop_r32_continue_step150_20261015/checkpoint_step150_verification.json)。status中的`latest_logged_step=149`是最后一次训练轮询留下的字段，不是训练终点；完成状态与完整step150验收共同确认已到150。

过程时点保留：2026-10-16 00:55时最新完成step87、状态为`train_to100`，当时只有10/50评测。00:50证据审计描述的是该早期时点，不是当前完成状态。

**续训采样同步审计：已核实代码路径缺陷。** 当前`grad_subspace`进入`_is_lora`路径，实际配置为`lora_adapter_path=None`、`load_format=dummy`、`val_before_train=False`。Checkpoint restore加载actor状态，但没有同步rollout；rollout初始化只启动engine并sleep。第一次generate的wakeup中，`base_sync_done=False`，`collect_lora_params(False)`排除`lora_`参数，`vllm.update_weights(False)`只加载base；此时没有已注册adapter ID，生成请求的`lora_request=None`。该次同步末尾才置`base_sync_done=True`，第二次wakeup才执行`add_lora`。

因此，按本次实际配置和已核实代码路径，两段恢复后的首轮采样（step51/101）使用base-only路径，而非恢复后的当前adapter策略；日志中的首轮异常与该路径对应。**证据边界是配置、源码和日志对应，尚未对运行时engine中的tensor/hash进行直接实测。** Actor/model、optimizer、RNG与数据恢复状态仍已通过验收，但不能将它们升级为连续策略采样等价证明。端点六榜分数仍是实际adapter的有效评测；训练轨迹与同协议基线比较必须披露这两次首轮采样偏差，不能当作已完全闭合的严格恢复实验。

完整链路见[训练与恢复审计](../../runs/ops/gradtop_r32_continue_review_20261016/training_audit.md)。部分下游`fsdp_utils`/vLLM源码不在原32项哈希清单内，审计记录了当前SHA256与早于本run的mtime；这进一步限定结论为当前实现与实际配置的代码推导，而非对原engine权重的直接测量。首次rollout的old log-prob仍由已恢复actor重算，因此生成策略与该actor不一致；偏差对后续更新和最终分数的累计影响尚未量化。

下次任何续训前，必须修复首轮adapter同步，并验收恢复后的第一次生成已经加载、注册并实际使用对应checkpoint的adapter。此项修复与验收尚未执行；本文不启动200步，也不把当前结果自动解释为下一段必然继续改善。

修复后从150继续，只能保证后续恢复按修复后的路径执行，不能追溯消除step51/101的采样已经进入optimizer更新的影响。这种续训可用于探索；用于论文的干净训练轨迹需要修复后重跑，同时审计LoRA/REAL对照的恢复路径。当前仅作判断，不启动新训练。

训练审计也未见持续上升的末段reward：101–120与131–150窗口均值为51.06%→50.01%，entropy为0.8408→0.7860。窗口题目不同，且前一窗口含异常step101；这些仅是平台迹象，不能据此证明过拟合或预测200步结果。

| 简称 | 确切身份 / 来源 | 已完成范围 | 应如何使用 |
| --- | --- | --- | --- |
| 当前LoRA | `rlvr_v1_lora_r32_a64_lr1em5_b64m16n8_seed42` | 50/100/150 | 当前协议对照 |
| REAL64 | `rlvr_v1_real_r32_a64_lr1em5_b64m16n8_seed42` | 64-step probe，正式训练/评测50 | “64”指probe采样step，不是rank |
| REAL32 | `rlvr_v1_real_probe32_r32_a64_lr1em5_b64m16n8_seed42` | 32-step统计前缀artifact，正式训练/评测50/100/150 | 正式rank仍32；非独立重采的一次32-step probe |
| GradTop | `rlvr_v1_gradtop_r32_a64_lr1em5_b64m16n8_seed42_20261015` | Probe、训练至150、评测10/50/100/150全部完成 | 本文唯一GradTop版本；完整恢复点为100/150 |
| 共享更新、逐题能量与策略漂移诊断 | 第7节研究设计 | 尚未执行 | 研究假说不计为实验结果 |

REAL32的100/150最终完成证据在[real_probe32_early_step100](../../runs/ops/real_probe32_early_step100/status.json)。此前[real_probe32_continue_step150](../../runs/ops/real_probe32_continue_step150/status.json)是早期控制器，不能只读它的旧状态就判断REAL仍未完成。

当前GradTop首轮和续训使用同一experiment，但有独立控制器目录：[首轮](../../runs/ops/gradtop_r32_clean_20261015/plan.json)、[续训](../../runs/ops/gradtop_r32_continue_step150_20261015/plan.json)。续训顺序固定为50→100→eval100→150→eval150，不根据100的分数临时只报告最好结果。

## 5. 已完成的当前协议结果

原00:50审计覆盖九组结果，检查了既有verification、shard completion、summary、manifest及算术。现追加当前GradTop100/150，共十一组：两次新评测的`verification.json`均为`verified`、`shards_completion_verification.json`均为`all_shards_verified`；数值与配对区间直接取自续训`results.json/.md`及原paired文件。本次文档更新未重跑生成、判分或bootstrap，不把旧审计快照扩称为对新结果的重新独立审计。

[10:36补充审计](../../runs/ops/gradtop_r32_continue_review_20261016/evidence_review.md)另已重聚合100/150各7,248条原始记录的正确性、parse、长度、hit-max及按题Pass@k，核对8个manifest，并独立计算150−100题目配对区间；没有重新推理或判分。该审计与原runtime文件分开保存。

共同配置：DeepSeek-R1-Distill-Qwen-1.5B；rank32/alpha64/LR1e-5/seed42。Step 为 trainer step，逻辑 optimizer updates=4×Step。REAL64/REAL32 后缀指 probe rollout 步数。评测 temperature0.6/top_p0.95/max_new_tokens32768；902题、7248条回答；四个小任务 k32，MATH500/Minerva k4。

Avg. 按各任务 response 数加权；任务单元格为 Avg@k / Pass@k (%)，一位小数。

| Method | Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA | 50 | 44.5 | 31.5 / 73.3 | 23.8 / 56.7 | 73.8 / 97.5 | 11.0 / 50.0 | 70.0 / 78.8 | 22.3 / 34.9 |
| LoRA | 100 | 44.5 | 27.7 / 76.7 | 23.1 / 50.0 | 74.2 / 95.0 | 9.9 / 40.0 | 72.2 / 80.2 | 22.9 / 34.6 |
| LoRA | 150 | 44.9 | 30.5 / 73.3 | 22.8 / 56.7 | 74.1 / 100.0 | 10.2 / 36.7 | 72.5 / 80.8 | 22.7 / 36.8 |
| REAL64 | 50 | 46.2 | 30.6 / 70.0 | 22.7 / 50.0 | 75.4 / 95.0 | 10.0 / 26.7 | 73.2 / 79.4 | 28.3 / 40.4 |
| REAL32 | 50 | 46.4 | 31.7 / 76.7 | 25.0 / 56.7 | 74.8 / 95.0 | 10.7 / 30.0 | 74.0 / 80.4 | 25.8 / 38.6 |
| REAL32 | 100 | 47.8 | 35.1 / 73.3 | 26.2 / 53.3 | 76.2 / 95.0 | 10.1 / 40.0 | 73.5 / 79.4 | 30.9 / 43.4 |
| REAL32 | 150 | 45.5 | 28.3 / 73.3 | 25.8 / 56.7 | 75.8 / 95.0 | 10.7 / 30.0 | 72.7 / 80.2 | 23.1 / 35.3 |
| GradTop | 10 | 43.3 | 29.8 / 76.7 | 22.3 / 56.7 | 70.8 / 92.5 | 9.4 / 33.3 | 70.1 / 79.4 | 22.4 / 35.7 |
| GradTop | 50 | 47.2 | 30.8 / 70.0 | 25.3 / 56.7 | 76.0 / 95.0 | 10.5 / 40.0 | 72.8 / 78.8 | 32.5 / 43.4 |
| GradTop | 100 | 47.4 | 29.9 / 70.0 | 23.9 / 50.0 | 78.0 / 97.5 | 9.9 / 33.3 | 73.6 / 79.8 | 32.8 / 43.4 |
| GradTop | 150 | 48.4 | 33.0 / 73.3 | 25.6 / 56.7 | 78.0 / 100.0 | 10.0 / 40.0 | 74.1 / 80.4 | 33.6 / 45.2 |

Macro 是六任务等权平均；Parse/Hit max/Mean length 按全部回答汇总。除 tokens 外均为%。

| Method | Step | Macro Avg@k | Macro Pass@k | Parse | Hit max | Mean length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA | 50 | 38.7224 | 65.2044 | 88.51 | 2.11 | 7749.7 |
| LoRA | 100 | 38.3390 | 62.7376 | 91.64 | 0.92 | 6441.1 |
| LoRA | 150 | 38.8141 | 64.0386 | 91.45 | 0.76 | 5684.6 |
| REAL64 | 50 | 40.0388 | 60.2513 | 92.37 | 0.23 | 5088.3 |
| REAL32 | 50 | 40.3361 | 62.8894 | 92.18 | 0.97 | 5338.1 |
| REAL32 | 100 | 42.0021 | 64.0748 | 96.32 | 1.67 | 5952.6 |
| REAL32 | 150 | 39.4078 | 61.7490 | 91.98 | 3.08 | 5399.9 |
| GradTop | 10 | 37.4610 | 62.3714 | 87.57 | 1.92 | 7162.2 |
| GradTop | 50 | 41.3282 | 63.9748 | 97.82 | 0.33 | 5002.3 |
| GradTop | 100 | 41.3425 | 62.3359 | 97.13 | 0.80 | 5394.2 |
| GradTop | 150 | 42.3971 | 65.9368 | 99.03 | 0.87 | 5486.8 |

已存 paired CI：题目为 cluster、10,000 bootstrap、seed42；差值是 candidate−baseline 的 Macro Avg@k（百分点）。这是题目重采样区间，不覆盖训练 seed 波动，也不是 response-weighted Avg. 的 CI。

| Candidate | Baseline | Macro Δ(pp) | 95% CI(pp) | Source |
| --- | --- | ---: | --- | --- |
| GradTop step50 | LoRA step50 | +2.6057 | [+1.4295, +3.8130] | [paired_vs_lora_step50.json](/root/peft-for-rl/runs/ops/gradtop_r32_clean_20261015/paired_vs_lora_step50.json) |
| GradTop step50 | REAL32 step50 | +0.9920 | [-0.1483, +2.1325] | [paired_vs_real32_step50.json](/root/peft-for-rl/runs/ops/gradtop_r32_clean_20261015/paired_vs_real32_step50.json) |
| GradTop step50 | REAL64 step50 | +1.2894 | [+0.2312, +2.3687] | [paired_vs_real64_step50.json](/root/peft-for-rl/runs/ops/gradtop_r32_clean_20261015/paired_vs_real64_step50.json) |
| GradTop step100 | LoRA step100 | +3.0035 | [+1.7492, +4.3414] | [paired_step100_vs_lora.json](../../runs/ops/gradtop_r32_continue_step150_20261015/paired_step100_vs_lora.json) |
| GradTop step100 | REAL32 step100 | -0.6596 | [-2.1011, +0.8246] | [paired_step100_vs_real32.json](../../runs/ops/gradtop_r32_continue_step150_20261015/paired_step100_vs_real32.json) |
| GradTop step100 | GradTop step50 | +0.0144 | [-1.0081, +1.0496] | [paired_step100_vs_own_step50.json](../../runs/ops/gradtop_r32_continue_step150_20261015/paired_step100_vs_own_step50.json) |
| GradTop step150 | LoRA step150 | +3.5830 | [+2.2411, +4.9986] | [paired_step150_vs_lora.json](../../runs/ops/gradtop_r32_continue_step150_20261015/paired_step150_vs_lora.json) |
| GradTop step150 | REAL32 step150 | +2.9892 | [+1.6527, +4.4124] | [paired_step150_vs_real32.json](../../runs/ops/gradtop_r32_continue_step150_20261015/paired_step150_vs_real32.json) |
| GradTop step150 | GradTop step50 | +1.0689 | [+0.0105, +2.1266] | [paired_step150_vs_own_step50.json](../../runs/ops/gradtop_r32_continue_step150_20261015/paired_step150_vs_own_step50.json) |
| GradTop step150 | GradTop step100 | +1.0545 | [-0.2529, +2.3554] | [补充paired_step150_vs_step100.json](../../runs/ops/gradtop_r32_continue_review_20261016/paired_step150_vs_step100.json) |
| REAL32 step50 | LoRA step50 | +1.6137 | [+0.2854, +2.9320] | [paired_comparison.json](/root/peft-for-rl/runs/ops/real_probe32_r32_lr1e5_seed42_after_real64_eval/paired_comparison.json) |
| REAL64 step50 | LoRA step50 | +1.3164 | [+0.1646, +2.4712] | [paired_comparison.json](/root/peft-for-rl/runs/ops/real_r32_lr1e5_seed42_after_eval/paired_comparison.json) |
| LoRA step100 | LoRA step50 | -0.3834 | [-1.5500, +0.7458] | [step100_vs_step50.json](/root/peft-for-rl/runs/ops/rlvr_protocol/rlvr_v1_lora_r32_a64_lr1em5_b64m16n8_seed42.complete_results/step100_vs_step50.json) |
| LoRA step150 | LoRA step50 | +0.0917 | [-1.1236, +1.2568] | [step150_vs_step50.json](/root/peft-for-rl/runs/ops/rlvr_protocol/rlvr_v1_lora_r32_a64_lr1em5_b64m16n8_seed42.complete_results/step150_vs_step50.json) |

自身50/100比较是同一实际训练轨迹的延长预算比较，不是同step方法比较。GradTop150相对100的Macro为+1.0545 pp、weighted Avg.为+0.9244 pp（原始未舍入值）；新增Macro配对区间[-0.2529,+2.3554] pp跨0，六个单任务区间也全部跨0。不能据点估计判定稳定提升，也不能用这个Macro区间替代weighted Avg.区间。原00:50审计未发现REAL32 step100/150对同step LoRA的已保存paired CI，本文不填补该缺项。

| Method | Probe rollout steps | Observations/accepted groups | Attempted prompts | Sampled responses | Probe wall | Formal step50 wall |
| --- | ---: | --- | ---: | ---: | --- | --- |
| GradTop | 4 | 16 minibatches / 256 observed groups | 256 | 2048 | 2250.33s (0.625h) | 21512.50s |
| REAL32 | 32 | 735 accepted groups | 2048 | 16384 | 未独立计时（缓存前缀重放） | 19922.43s |
| REAL64 | 64 | 1499 accepted groups | 4096 | 32768 | 40085.41s (11.135h) | 20612.55s |

GradTop probe 为37.51分钟，REAL64 为11.135小时，所选预算下实测总墙钟比17.81×；REAL64尝试prompt数是GradTop的16×，REAL32是8×。因此这支持“低预算probe方案”的成本优势，不能直接称为等样本算法加速17.81×。REAL32 artifact 来自 REAL64 step032 的精确累计缓存，没有独立采集墙钟成本。REAL只训练B（20,643,840参数）；GradTop/LoRA训练A+B（36,929,536参数），不是等可训练参数对照。


结果解释：当前GradTop从50到100的Macro几乎不变，150升至42.3971%；相对50增加1.0689 pp，题目配对区间下界仅+0.0105 pp，应保留其接近零的边界。相对同step REAL32，GradTop在100为−0.6596 pp、区间跨0，在150为+2.9892 pp、区间全正；相对LoRA100/150的区间也均全正。这支持当前单seed、固定endpoint的观察差异，仍需披露第4节恢复采样边界，不支持所有预算或跨训练seed的普遍优势。150相对100的六任务Avg@k点估计五升一平，但总体和逐任务区间均跨0；Macro Pass@k由62.3359%升至65.9368%，parse由97.13%升至99.03%，平均长度由5394.2升至5486.8。是否继续200不能从这一个上升区间自动决定，需结合追加预算所回答的问题另行判断。

补充记录显示，总正确回答3438→3505、可解析回答7040→7178，已解析子集正确率48.8352%→48.8298%近乎不变；但AIME24/25解析率下降而正确数增加，Minerva则解析明显增加。它既不能证明提升全由格式造成，也不足以证明推理能力稳定增强；这是观测分解，不是格式的因果消融。

截至00:50的九组summary、verification、records与artifact路径及小文件SHA见[当时的机器可读审计](../../runs/ops/gradtop_real_handoff_20261016/evidence_audit.json)。新100/150的直接来源为[续训results.json](../../runs/ops/gradtop_r32_continue_step150_20261015/results.json)、[eval100登记](../../runs/ops/gradtop_r32_continue_step150_20261015/evaluation_step100_job.json)和[eval150登记](../../runs/ops/gradtop_r32_continue_step150_20261015/evaluation_step150_job.json)，旧审计快照不改写。


## 6. 当前方法的确切定义与REAL的区别

### 6.1 当前GradTop

对一个线性权重的逻辑batch梯度G，用零函数probe `A_probe=0, B_probe=Ψ`，其中Gaussian Ψ的元素方差为1/width。读取

`S = grad(A_probe)^T / scaling = G^T Ψ`。

当前每个observation包含16个完整prompt组、每题K8，保留GRPO advantage和PPO loss分支，先聚合成G，再平方形成能量。probe不做optimizer更新或梯度裁剪。

跨observation累计的是因子拼接的二阶能量。当前每4个observation进行一次薄QR及小矩阵SVD：

`F_k = truncate_64([F_(k-1), S_(4k-3), S_(4k-2), S_(4k-1), S_(4k)])`。

压缩保留`UΣ`以携带能量，最终A取前32个单位输入方向的转置，B=0。它**不是**把全部梯度相加后做一次SVD；batch内部的相反贡献可能抵消，不同observation的相反梯度不会因因子拼接直接抵消。保留原始幅值，没有单位范数重权、显式指数遗忘或额外随机sign。

当前注册参数：16 observations=4个采样step=256 prompts，width8、capacity64、uniform32。信号秩不足时实现允许确定性QR补齐并披露；本次实际artifact的补齐方向数为0。16 observations不等于16次物理backward，microbatch会拆分。

### 6.2 本地REAL的统计目标和实现

每题K8个二元正确性reward，至少两正两负且response非空才进入统计；用population standard deviation计算advantage，每条response的log概率先除其有效长度，再按K平均，得到完整题组梯度G_p。

目标是未中心化逐题二阶矩 `C_prompt = E_p[G_p^T G_p]`。输出投影每题刷新，输入投影Ω固定：`H_p=Ψ_p^T G_p`，累计`Y=mean H_p^T(H_p Ω)`，对称化`M=Ω^T Y`，由`F=Y M^(†/2)`的top-r左奇异向量取P。当前r32时输入/输出宽度均128，正式A=P^T固定，B从0训练。

本地实现对完整题组收集后才平方；DP ranks处理同组，逐response backward再累计，尚不是不同DP rank独立处理不同prompt的吞吐优化实现。REAL也不收集dense base dW，也是流式估计。FSDP冻结参数仍可能占用零梯度buffer，估计器缓冲小不等于整卡总梯度存储同比减少。

### 6.3 Batch聚合、逐题能量和随机sign的依据

在等权、独立同分布题组梯度的理想参照下，令`μ=E[G_p]`、`Σ=E[(G_p−μ)^T(G_p−μ)]`，则

`C_batch(m) = E[Ḡ_m^T Ḡ_m] = μ^Tμ + Σ/m`，而`C_prompt = μ^Tμ + Σ`。

Batch聚合直接对应保留batch梯度更新；逐题能量更重视题间变化。交叉项不应一概视作污染，题间差异也不应一概视作噪声。实际GRPO归一化、clipping与训练权重使μ不能直接称为原始期望回报的无偏梯度。

额外sign构造`Z=(1/√m)Σ_p ξ_p G_p`，只保证条件期望`E_ξ[Z^T Z]=(1/m)Σ_p G_p^T G_p`。这给出逐题目标的随机估计依据，**没有证明该目标比batch目标好，也没有证明有限sign/投影预算下的子空间无偏或更好**。目前它仅适合机制对照。

## 7. 后续研究的具体入口和否定条件

完整方案已经写在[共享更新研究路线](gradtop_update_geometry_research_20261015_zh.md)，本轮不重新发明一组局部开关。最小诊断的角色是决定下一方法值得解决哪种问题：

| 假说 | 必须观察到的现象 | 反证 / 不推进条件 |
| --- | --- | --- |
| H1：逐题能量放大有限rollout噪声 | batch空间在独立重复和audit上更好覆盖共享/后续更新；逐题新增能量主要不可重复 | 逐题空间同样稳定且更好覆盖未来更新 |
| H2：被抵消成分含真实题间能力方向 | 被削弱方向在独立重复、题型及后续策略阶段仍有用 | 只有当前样本能量，没有独立效用证据 |
| H3：固定初始化主要受on-policy漂移限制 | 跨checkpoint变化超过同checkpoint重复噪声，旧空间损失真实更新覆盖 | 仅主角度变了，或变化不超过采样噪声 |

最小D0：base policy上预先固定128题，64 discovery / 64 audit；每题两组独立K8回答，共2,048 responses。两个重复有不同可追踪RNG子流，不合并成K16；多卡复制同一组不算独立重复。选层0/13/27的q_proj和down_proj共6个模块；两组独立输出投影各宽32，用于拟合/audit并交换角色，最终rank仍32。

同cache下比较`m∈{1,4,16,64}`，主比较预定m16。先分别控制等题权重、当前probe的全局token权重、正式actor的实际microbatch/DP权重，再比较聚合；正式执行权重不能仅由完整题组H和token总数重建。用独立重复的交叉矩估计可复现成分、差分估计rollout噪声，避免同一份噪声既拟合又打分。有限样本交叉矩可能不正定，不能把其截负后当成无偏PSD真值。

D0通过后才在已有LoRA step50做D1，仍为同128题的独立双重复；step100作为事先保留的条件扩展。D0+D1需4,096条回答，三点需6,144条。完整输入侧H每个checkpoint约1.92GiB，另计rollout cache及真实训练权重分支。当前研究收集/分析管线**尚未实现和运行**；这不是免费利用现有probe文件就能得到的诊断。

现有每10步snapshot只能计算区间累计有效权重位移，不能恢复每次Adam更新。若要讨论真正optimizer几何，需要独立登记真实A/B更新与Adam状态观测。有效权重增量为`ΔW=s(BΔA+ΔBA+ΔBΔA)`；对dense G做虚拟Adam不是LoRA因子Adam的等价物。

只有机制和正式训练效用方向一致，才推进完整新方法：例如估计可重复、可预测后续改善的更新空间，或者在证明漂移确实构成瓶颈后研究有触发准则、函数保持和optimizer迁移的在线子空间调整。若几何指标不能预测真实训练收益，应否定该故事。投影/采样误差压过效应时只允许预声明的一次128→256扩样，不据结果继续堆超参。

## 8. 与REAL的关系、创新判断和未闭合证据

论文源文件是[1683_Reward_Gradient_Aligned_L.pdf](/root/1683_Reward_Gradient_Aligned_L.pdf)，标题《Reward-Gradient Aligned Low-Rank Adaptation for Post-Training》。PDF首页标为ICLR 2027匿名在审；本机资料不能证明其最终接收状态、首个公开日期或作者代码发布日期。本地REAL是按论文实现，当前未核实有作者官方repo可供逐行对照。不能把用户“最近看到”当作该工作的首次出现时间。

相似方向可以做研究并发布，但是否形成独立方法贡献，取决于实际技术差异、证据及对相关工作的明确引用；换名、换一个分解器或多几个点的单seed分数不够。当前只能确认本地实现差别，尚未完成公开文献的新颖性检索，也未形成可声明独立优先权的证据链。

| 可能主张 | 当前支持程度 | 还缺什么 |
| --- | --- | --- |
| GradTop在本地实现中probe更便宜 | 有实际日志与采样预算证据 | 同目标、同估计质量、充分优化REAL后的端到端成本；不能把本地串行开销概括成REAL必然代价 |
| 增量SVD与Nyström不同 | 代码已确认 | 固定统计目标和sketch后的精度/成本对照，一次性SVD、重放顺序及capacity影响 |
| 首次流式 / 首次不算dense gradient | 不支持 | REAL和已有方法已经具备这些特征 |
| Batch目标更适合RL | 可研究，未证实 | 同loss/cache/冻结/预算下的诊断，真实更新与泛化证据，多训练seed |
| 在线子空间跟踪适应策略漂移 | 仅后续假说 | 先证实H3；当前增量SVD只发生在不更新policy的初始化probe中 |
| 长度偏好是新的RL-specific贡献 | 不能直接主张 | REAL已有probe-only长度偏好分支；需完整相关工作对照 |

建议新会话首先给出：当前最强可支持结论、最危险混杂、是否存在值得继续的一个主问题，以及最小证伪实验。确认主线后，再补同预算LR范围、多训练seed、rank16/64和跨模型；不同rank下alpha仍64，scaling并不恒定。现有单seed置信区间不能代替这些证据。

## 9. 代码、运行入口与交接约束

| 路径 | 查看内容 |
| --- | --- |
| [peft_gradient_subspace.py](../../verl/utils/peft_gradient_subspace.py) | random-B初始化、GradientProbeCollector、增量QR/SVD、rank导出 |
| [gradtop_objectives.py](../../verl/utils/gradtop_objectives.py) | 当前token_mean目标和全局有效token分母 |
| [peft_real.py](../../verl/utils/peft_real.py) | REAL contrast系数、逐题收集、Nyström及artifact |
| [dp_actor.py](../../verl/workers/actor/dp_actor.py) | backward执行、probe与正式训练归约、optimizer更新 |
| [ray_trainer.py](../../verl/trainer/ppo/ray_trainer.py) | 完整题组、筛选、数据顺序、checkpoint恢复 |
| [fsdp_workers.py](../../verl/workers/fsdp_workers.py)及[real_probe_mode.py](../../verl/utils/real_probe_mode.py) | 模型接入、冻结方式、确定性probe |
| [agent_loop.py](../../verl/experimental/agent_loop/agent_loop.py) | 按global step/样本/response派生采样seed |
| [run_rlvr_protocol_1p5b_4gpu.sh](../../scripts/local/run_rlvr_protocol_1p5b_4gpu.sh)及[verify_rlvr_protocol.py](../../scripts/analysis/verify_rlvr_protocol.py) | 统一训练、资产验证及contract |
| [run_gradtop_candidates_4gpu.sh](../../scripts/local/run_gradtop_candidates_4gpu.sh) | 当前probe的启动入口；必须显式METHOD=gradtop，不依赖默认METHOD |
| [run_gradtop_clean_after_real.py](../../scripts/local/run_gradtop_clean_after_real.py) | 已完成probe→train50→eval50→eval10 |
| [continue_gradtop_r32.py](../../scripts/local/continue_gradtop_r32.py) | 已完成50→100→eval100→150→eval150的控制器 |
| [real_probe_cache.py](../../scripts/local/real_probe_cache.py) | REAL32统计前缀来源，非全新32-step采集 |

完成的续训plan SHA256：`a98a83cb423040b55122855c5622491b437bab6820e6b03d688d94db65f3bdb6`。00:50审计时32个固定源码/协议文件全部与plan一致；这是当时的哈希核验记录。后续新想法先写独立设计/工作副本，不能为审计重启已完成的controller或重复启动同一experiment。

完成与恢复入口：

- [status.json](../../runs/ops/gradtop_r32_continue_step150_20261015/status.json)、[controller.log](../../runs/ops/gradtop_r32_continue_step150_20261015/controller.log)：最终target_step150、complete；保留的PID与旧轮询字段不是进程仍运行的证据。
- [train_to100.log](../../runs/ops/gradtop_r32_continue_step150_20261015/train_to100.log)、[train_to150.log](../../runs/ops/gradtop_r32_continue_step150_20261015/train_to150.log)：两段已完成训练与恢复的日志。
- [续训文档](gradtop_r32_continue_step150_20261015_zh.md)、[恢复验收](../../runs/ops/gradtop_r32_continue_step150_20261015/live_resume_verification.json)、[CPU检查](../../runs/ops/gradtop_r32_continue_step150_20261015/cpu_checks.log)，后者61项通过。

完整恢复必须包含四rank model/optimizer/RNG/scheduler及data.pt；adapter snapshot只用于评测。保存150时原retention2已删除50/actor完整状态，控制器恢复了备份的50评测adapter；50不能作为完整恢复点，100/150已有完整恢复验收。LoRA100当前也仅剩adapter，已有评测仍可验证；不要从它启动完整续训。

本次续训使用按实测checkpoint/snapshot大小和retention核算的空间门槛，具体记录见续训文档。如以后新增诊断或追加训练，需先明确范围、检查剩余空间并登记预算。任何下一段续训还必须先修复并验收首次rollout的adapter同步；当前没有200步计划。

只读查看命令：

```bash
cd /root/peft-for-rl
cat runs/ops/gradtop_r32_continue_step150_20261015/status.json
tail -n 30 runs/ops/gradtop_r32_continue_step150_20261015/controller.log
cat runs/ops/gradtop_r32_continue_step150_20261015/results.md
```

## 10. 可直接粘贴到新会话的任务

```text
请以 /root/peft-for-rl 为工作区，先阅读
docs/experiments/gradtop_real_handoff_20261016_zh.md，
并读取其中指向的原续训status/results；100/150已于2026-10-16 09:57完成，
00:50的audit文件是早期快照，新结果以原续训目录和新补充审计为准。

我要审查当前GradTop-r32与REAL对照是否存在值得发展成论文的方法主线。
请以当前rank32协议为主，核对代码、冻结plan/contract和原始结果。
GradTop只分析experiment rlvr_v1_gradtop_r32_a64_lr1em5_b64m16n8_seed42_20261015，
仅纳入其当前版本的配置、结果与同协议REAL/LoRA对照。

重点检查：
1. 当前batch聚合+random-B+增量SVD与REAL逐题能量+Nyström的真正目标差异。
2. probe成本优势有多少来自预算/实现，有多少可能在同目标同质量下保留。
3. 当前结果能否支持共享更新、rollout噪声、题间异质性或策略漂移中的一个机制。
4. 结合原PDF及公开相关工作，哪些内容已有、哪些差异有方法价值、哪些仍不能声称创新。
5. 给出一个优先研究问题和最小可证伪实验，明确控制变量、预算、成功/停止条件。

不要把sign、mask、fixed-A或更换分解器拼成默认主线；当前GradTop没有额外随机sign。
不要把单seed题目bootstrap区间写成跨训练seed显著性。
当前授权的GradTop续训已完成；本轮先审计和设计，并评估继续到200的必要性，
不把这个问题当成已授权的新训练计划。不要改原运行源码或重复启动已完成controller。
已确认恢复后首次rollout走base-only代码路径，尚未做运行时engine tensor/hash实测；
下一段续训前必须另行修复并验收首轮adapter同步，不能直接继续200。
```
