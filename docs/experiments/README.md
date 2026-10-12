# 实验文档总入口

更新时间：2026-10-10 UTC

这里是 `/home/wangls/peft-for-rl/docs` 中实验材料的统一入口。旧文档仍保留原始技术细节和当时的判断，但不再单独承担“当前实验状态”的职责。

## 先读什么

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
