# 实验文档总入口

更新时间：2026-10-18（Asia/Shanghai）

这里是 `/root/peft-for-rl/docs` 中实验材料的统一入口。旧文档仍保留原始技术细节和当时的判断，但不再单独承担“当前实验状态”的职责。

## 先读什么

**当前 REAL / GradTop 统一入口（2026-10-18 15:50:48 快照）：** [近期实验、成功运行版本及精确配置](real_gradtop_summary_20261018_zh.md)、[版本登记JSON](../../configs/experiments/real_gradtop_versions_20261018.json)、[实现问题与修复说明](../../audits/real_gradtop_implementation_audit_20261018_zh.md)、[随GitHub提交的结果与证据](evidence/real_gradtop_20261018/README.md)。GradTop270已完成；新REAL-P256已连续训练到210，210六榜Macro35.8401%，200评测已中断、150队列已取消；15:51早期checkpoint CPU审计已补入。下方10月17日及更早内容保留为当时的进度记录，其中“REAL新跑尚未启动”等表述不代表当前状态。

2026-10-17 12:42 诊断全部完成：5个策略、5120回答、全部反传及跨阶段统计已验收。[完整诊断表](../../runs/ops/gradtop_real_gradient_diagnostics_20261017/results.md)、[解释与证据边界（第7节）](gradtop_real_longrun270_probe_budget_20261017_zh.md)。当前结果显示REAL初始化空间在受检策略上的逐题及共享能量覆盖均高于GradTop空间；REAL100→150时两个空间覆盖都下降，GradTop150→270覆盖上升但六榜无同步收益。增量SVD同流对照误差小。REAL等预算新长跑仍未启动。

2026-10-17 11:30 诊断进度：已完成GradTop100，累计1/5，队列恢复后进入REAL100。首点CPU汇总的PEFT键名前缀问题已修复，原始采样/反传完整并复核后复用。[逐点诊断结果](../../runs/ops/gradtop_real_gradient_diagnostics_20261017/results.md)、[动态状态](../../runs/ops/gradtop_real_gradient_diagnostics_20261017/status.json)。

2026-10-17 11:06 诊断已启动：[当前GradTop/REAL固定策略梯度诊断（第6节）](gradtop_real_longrun270_probe_budget_20261017_zh.md)。按用户要求优先执行诊断：GradTop100/150/270及REAL100/150，64题×2组K8/策略，共5120回答，6个预先选定模块；正式归约、等题归约、全局token归约、REAL原目标敏感性及增量SVD对照。CPU恒等式和小批次GPU加载/反传检查通过，11:06进入GradTop100正式采样。[动态状态](../../runs/ops/gradtop_real_gradient_diagnostics_20261017/status.json)。REAL新长跑尚未启动。

2026-10-17 10:53 完成更新：[当前GradTop270步及REAL后续设计](gradtop_real_longrun270_probe_budget_20261017_zh.md)。250段04:09完成，270训练及六榜评测07:09全部完成，251首轮四GPU实际adapter验收通过。[完整结果表](../../runs/ops/gradtop_r32_resume270_syncverified_20261017/results.md)：270 Macro41.6845%，250为41.6034%，150为42.3971%；270与250/150差值的配对区间均跨零。270解析率95.50%，低于150的99.03%，需跟进行为退化。REAL新跑及梯度机制诊断均尚未启动。

2026-10-17 00:23 最新安排：[当前GradTop结束于270及REAL等probe预算长跑设计](gradtop_real_longrun270_probe_budget_20261017_zh.md)。当前250训练已到221步；270队列已启动，等待250训练/评测完成后绑定实际源权重，重新验收251首轮adapter同步，再续到270并完整评测。新文档核对256对2048的probe prompt预算、REAL100→150下降与恢复混杂，以及GradTop A子空间变化很小的实际证据；REAL新跑仍为方案，尚未启动。

2026-10-16 22:14 最新续训：[当前GradTop从200续到250并重新核验同步](gradtop_r32_resume250_20261016_zh.md)。用户要求继续观察长跑，保持同一方法及probe；从完整200恢复，22:16已用新的源SHA/token/回执通过四GPU实际adapter及首次请求验收。到250后自动六榜评测并比较250−200、250−150。上一段200已完整完成，Avg.47.9857%、Macro42.0933%，详见其结果表。

2026-10-16 15:35 新续训：[当前GradTop验证首轮同步后150→200](gradtop_r32_resume200_syncfix_20261016_zh.md)。用户已允许超过17:00并要求200完整评测；原probe和科学配置不变，加入checkpoint→actor→vLLM注册及GPU槽位→首次请求的运行时验收。15:37实际四GPU验收通过，16份回执齐全，全部首次请求均携带正确adapter。当前细粒度评测已完成140/130/120/110并停队，GPU交给这段续训。

2026-10-16 细粒度评测：[当前GradTop从150向前每10步补评](gradtop_r32_fine_eval_20261016_zh.md)。复用150/100/50/10，新增队列140→130→120→110→90→80→70→60→40→30→20；优先利用当天17:00前卡时，动态进度与逐点更新表见文档入口。本任务仅评测当前已保存快照。

2026-10-16 研究交接：[当前GradTop-r32与REAL对照交接](gradtop_real_handoff_20261016_zh.md)。当前 `rlvr_v1_gradtop_r32_a64_lr1em5_b64m16n8_seed42_20261015` 已于09:57完成100/150训练及六榜评测，step150 Avg.为48.4%、Macro为42.3971%；交接汇总10/50/100/150轨迹、同协议REAL/LoRA及配对区间，并标明00:50审计仅为当时快照。step51/101恢复首轮base-only历史仍需保留；后续150→200已应用同步修复并通过实际GPU验收，见上方最新执行入口。

2026-10-15 已完成：[GradTop-r32当前协议对照](gradtop_r32_clean_baseline_20261015_zh.md)，batch聚合、width8/capacity64增量SVD、16次observation、uniform rank32、LR1e-5；该适配版含probe归约/分组变化，非历史原实现严格复现。[完整结果表](../../runs/ops/gradtop_r32_clean_20261015/results.md)给出step10/50轨迹和三条step50对照，主点Macro Avg@k为41.3282%。按用户要求新增[step50→100→150续训计划](gradtop_r32_continue_step150_20261015_zh.md)，复用原probe和完整恢复点，两个新端点分别进行六榜评测及同step LoRA/REAL32比较。[共享更新方向研究路线](gradtop_update_geometry_research_20261015_zh.md)给出区分rollout噪声、题型异质性与策略漂移的后续探测设计。Prompt-sign不再是默认主线。

2026-10-14 新设计：[GradTop 吸收 REAL 的下一轮实验设计](gradtop_real_next_round_design_20261014_zh.md)，核对最新 LoRA/REAL 原始结果，给出低成本 Prompt-sign、fixed-A、严格 REAL-sign 桥接与 reward 条件方差排序的数学定义、对照矩阵和成本口径；新增方法尚未实施。

2026-10-14 新实现：[GradTop 备选实验方案审查与实现](gradtop_candidates_20261014_zh.md)，包含 Prompt-sign、Masked-GradTop、三项机制对照、固定预算启动入口及本地资产预检状态；尚无本轮正式训练结果。

1. [NoRA + DoRA v1 实验记录](nora_dora_v1_experiment_report_20261010_zh.md)：记录当前运行中的 OFT 消融、公式、参数量、完整训练配置和预注册评测解释。
2. [NoRA + DoRA + OFT v1 实验记录](nora_doft_v1_experiment_report_20261010_zh.md)：记录方法公式、实现细节、参数量、完整训练/评测配置、配对统计、证据边界和 `no-go` 决策。
3. [本机 PEFT for RL 实验研究叙事](local_experiment_research_narrative_20261008_zh.md)：按研究问题串联固定 PEFT、AdaLoRA、RLPO、oracle rank、梯度探测、Phase-1、Step-25 和 Qwen3-4B，并解释每轮设置为什么由上一轮引出。
4. [本机历史与 8 x RTX A6000 后续完整评测结果对照](local_a6000_full_benchmark_comparison_2026_10_zh.md)：按实验逻辑分组列出 AdaLoRA、RLPO/CHR、Stable-SNR/Dominant、GradTop 及裁剪/外推，并继续记录迁移后的 Step-25/50、training seed 和 Qwen3-4B 结果。
5. [原 4 x L40S 完整评测结果对照](local_l40s_full_benchmark_comparison_2026_10_zh.md)：只统计原 L40S 主机完成的 18 个六任务结果。
6. [实验时间线](experiment_timeline_202607_202610_zh.md)：按时间说明做过什么、在哪台机器上做、结果是否完整。
7. [方法与结果登记表](method_and_results_registry_zh.md)：按方法列出配置、关键指标、证据等级和可支持的结论。
8. [最近两次实验汇总与下一轮执行单](recent_two_and_next_round_20261006_zh.md)：matched standard LoRA seed42/43 的结果、恢复边界和正在执行的 Step25 队列。
9. [Step 25 进度与外部复现预检计划](step25_progress_and_external_preflight_plan_20261006_zh.md)：当前 trajectory 统计、Qwen3-4B 静态预检、存储预算和 fail-closed gate。
10. [开放问题、后续实验与论文映射](open_questions_and_paper_map_zh.md)：区分必须补的证据、可选训练和三条论文路线。
11. [下一轮实验计划](next_round_experiment_plan_20261005_zh.md)：Step25 学习曲线与条件性跨模型复现。
12. [机器可读文档索引](experiment_docs_index_20261005.json)：登记原 `docs/*.md` 的类型、状态和替代关系。

当前 Phase-1 的更细粒度材料仍在：

- [Phase-0/1 时间线](../../runs/phase1-signal-random-v1/analysis/experiment_timeline_20261005.md)
- [Phase-1 方法与结果矩阵](../../runs/phase1-signal-random-v1/analysis/method_and_results_matrix_20261005.md)
- [Phase-1 后续实验与论文结构](../../runs/phase1-signal-random-v1/analysis/next_experiments_and_paper_structure_20261005.md)
- [Phase-1 证据索引](../../runs/phase1-signal-random-v1/analysis/experiment_evidence_index_20261005.json)
- [119 个历史 full-bench 盘点](../../runs/phase1-signal-random-v1/analysis/local_fullbench_evidence_inventory_20261004.json)

## 权威性顺序

出现数字或状态冲突时，按以下顺序解释：

1. 原始 records、summary、training contract、postverify 和冻结 preregistration/spec；
2. `runs/**/analysis` 中针对对应实验的审计或 retrospective；
3. 本目录的统一时间线和结果登记表；
4. 原 `docs/*.md` 中的历史总结、设计说明和启动说明。

本目录是便于阅读的综合层，不修改原始证据。尤其不能用历史文档顶部的“当前”“最好结果”覆盖后来完成的 Phase-1 结果。

## 证据等级

| 等级 | 定义 | 可以怎么用 |
| --- | --- | --- |
| A | 冻结 Phase-1 协议，具备 contract 和独立 postverify | 当前固定设置中的主要证据 |
| A-reference | 严格审计过的 no-RL base | 绝对参考；不是 training-seed paired control |
| B | 有兼容的 record-level full-bench，但训练协议不匹配 Phase-1 | 历史上下文、量级参考 |
| C | 完整的描述性历史结果，但 records 不可复用或协议不同 | 历史趋势，不能作匹配比较 |
| D | screen、修复拼接、不完整、不兼容或已知损坏 | 诊断和失败记录，不能作主结果 |

等级约束的是结论，不是简单按分数排序。不同 step、dropout、初始化、benchmark snapshot 或训练设备的结果不应放进无协议列的 leaderboard。

## 原有文档分类

| 文档 | 类型 | 当前用途 |
| --- | --- | --- |
| [results.md](../results.md) | 导航入口 | 指向本目录；不再维护独立 leaderboard |
| [local_gradient_subspace_experiments_2026_09_zh.md](../local_gradient_subspace_experiments_2026_09_zh.md) | 历史总档案 | 2026-09 梯度/子空间与 Phase-0/1 追加记录；顶部状态已过期 |
| [phase1_seed43_i8_migration_zh.md](../phase1_seed43_i8_migration_zh.md) | 迁移档案 | rev3/handoff 计划与验收；rev4 已在本机完成 |
| [experiment_config.md](../experiment_config.md)、[quickstart_baselines_zh.md](../quickstart_baselines_zh.md) | 历史 baseline 配方 | Stable LoRA/OFT 复现背景，不是冻结 Phase-1 协议 |
| [verl_rlpo_init_zh.md](../verl_rlpo_init_zh.md) | 实现说明 | RLPO 接入；完成结果以本登记表和 retrospective 为准 |
| [verl_adalora_code_walkthrough_zh.md](../verl_adalora_code_walkthrough_zh.md) | 实现说明 | AdaLoRA 工程路径；现有实验不能作为有效方法结果 |
| [full_gradient_rl_probe_v1_zh.md](../full_gradient_rl_probe_v1_zh.md)、[spar_lora_v0_static_zh.md](../spar_lora_v0_static_zh.md)、[windowed_adam_consensus_probe_zh.md](../windowed_adam_consensus_probe_zh.md) | 设计说明 | 方法定义；是否实际完成须查登记表 |
| [verl_rl_gradient_subspace_init_zh.md](../verl_rl_gradient_subspace_init_zh.md)、[verl_rl_gradient_subspace_stable_snr_v2_zh.md](../verl_rl_gradient_subspace_stable_snr_v2_zh.md)、[verl_rl_dominant_gradient_atom_probe_zh.md](../verl_rl_dominant_gradient_atom_probe_zh.md) | 设计/实现说明 | 对应 September probe 系列，不单独代表结果 |
| [rl_adaptive_rank_minimal_experiment_zh.md](../rl_adaptive_rank_minimal_experiment_zh.md) | 设计与诊断 | adaptive-rank 研究假设和离线证据 |
| [environment_train.md](../environment_train.md)、[environment_eval.md](../environment_eval.md)、[local_runtime_zh.md](../local_runtime_zh.md) | 环境/运行说明 | 复现和机器环境，不是结果证据 |
| [verl_grpo_training_pipeline_zh.md](../verl_grpo_training_pipeline_zh.md) | 实现说明 | GRPO/RLPO 训练链路 |
| [rl_code_agent_compute_survey_zh.md](../rl_code_agent_compute_survey_zh.md) | 计划/调研 | 跨领域实验计算预算，尚不是实验结果 |

完整逐文件状态以 JSON 索引为准。
