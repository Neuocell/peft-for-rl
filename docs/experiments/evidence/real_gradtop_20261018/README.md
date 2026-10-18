# REAL / GradTop 归档证据（2026-10-18）

采集时间：2026-10-18 15:50:48，另补入15:51完成的早期checkpoint CPU审计，Asia/Shanghai。对应[实验总表](../../real_gradtop_summary_20261018_zh.md)、[方法版本JSON](../../../../configs/experiments/real_gradtop_versions_20261018.json)及[实现审计](../../../../audits/real_gradtop_implementation_audit_20261018_zh.md)。

## 可直接读取的结果

- [results.md](results.md)：19个完整六榜端点，含任务Avg@k/Pass@k和行为指标。
- [results.json](results.json)：完整精度结果、配对区间和REAL-P256当时状态。
- [artifact_fingerprints.json](artifact_fingerprints.json)：四个成功运行版本的三个初始化文件SHA256及大小。
- [manifest.json](manifest.json)：逐字节复制文件的原始路径、归档路径、SHA256及大小。
- [validation.json](validation.json)：本次发布验证的实际结果和范围。
- [actual_protocol_verification.json](actual_protocol_verification.json)：发布版本的真实校验器重新检查四份冻结协议、本机资产及初始化产物；不启动GPU任务。

`evaluations/<method>/step_*/` 保存每点原summary、verification、shard completion及四份manifest。LoRA150的历史shard状态名为 `all_four_shards_verified_before_exit_cleanup`，其他点为 `all_shards_verified`；两者均核对4个shard、7248回答。本次从summary重算加权/宏平均并核对原表，没有重跑生成、判分、raw-record聚合或bootstrap。

## 方法与执行来源

`probe/<version>/` 保存四份初始化的完整summary和rank_map。`protocols/` 保存两份旧REAL原始冻结协议，与仓库 `configs/experiments/real{32,64}_r32_frozen_*.json` 逐字节相同。

REAL-P256原运行目录中的 `source_snapshot/` 同样归档，保存49份实际冻结的源码及配置，便于与发布分支后来的集成改动区分。

`sources/runs/ops/` 保留原执行层级，重点入口：

| 目录 | 主要证据 |
| --- | --- |
| [gradtop_r32_continue_review_20261016](sources/runs/ops/gradtop_r32_continue_review_20261016) | 恢复源码链、日志和结果补审 |
| [gradtop_research_review_20261016](sources/runs/ops/gradtop_research_review_20261016) | 原实现审计、评分/格式敏感性及CPU代数检查 |
| [gradtop_r32_resume270_syncverified_20261017](sources/runs/ops/gradtop_r32_resume270_syncverified_20261017) | 最终270轨迹、完整checkpoint验收、251首轮实际GPU同步回执及控制器 |
| [gradtop_r32_resume200_syncfix_20261016](sources/runs/ops/gradtop_r32_resume200_syncfix_20261016)、[resume250](sources/runs/ops/gradtop_r32_resume250_syncverified_20261016) | 151/201首轮同步验收和冻结plan |
| [real_p256_r32_longrun270_20261017](sources/runs/ops/real_p256_r32_longrun270_20261017) | 新probe、原270计划、零B及第一step后非零B的实际四GPU验收 |
| [real_p256_stop210_eval_20261018](sources/runs/ops/real_p256_stop210_eval_20261018) | 后续停210计划、210权重/配置核验、完整210评分和用户停止200/取消150的状态快照 |
| [gradtop_real_gradient_diagnostics_20261017](sources/runs/ops/gradtop_real_gradient_diagnostics_20261017) | 5120回答诊断、五个policy的analysis、跨阶段比较和完成验收 |
| [real_p256_early_checkpoint_audit_20261018](sources/runs/ops/real_p256_early_checkpoint_audit_20261018) | 全22个snapshot、probe数值秩及早期训练指标的CPU审计；后续评测仅为建议，未排队 |
| [gradtop_epoch_boundary_audit_20261017](sources/runs/ops/gradtop_epoch_boundary_audit_20261017) | 实际sampler的epoch边界CPU重放 |
| [rlvr_protocol](sources/runs/ops/rlvr_protocol) | 原评测控制器及checkpoint验收辅助源码 |

原报告中的绝对路径、PID、状态、旧“待执行”判断和链接保持当时时点；以最新总表解释状态变化，不把归档副本作为实时状态接口。plan中的 `code_sha256` 描述实际运行源码，不能自动用当前发布commit替代。控制器保留了本机路径、依赖和资源绑定，用于审计；迁移执行前需要逐项重绑定，不能直接在新机器运行归档controller。

归档的 `.gitattributes` 对原始副本关闭换行转换，保留CSV的CRLF及历史源码空白，确保manifest字节哈希可复核；源码副本不执行新代码的空白清理。

## 大文件与复查边界

基座/adapter/subspaces权重、数据集、原始回答、梯度、optimizer、训练日志继续保存在本机runtime中，未加入Git。文件指纹和原报告使已归档结论可追溯；仅克隆仓库不能重新计算全部原始回答分数或恢复训练。重新运行需另备这些资产并核验指纹。

本归档保留旧GradTop/REAL32的51/101恢复首轮采样问题；后续修复回执不替代历史运行时权重证据。配对区间仅覆盖题目重采样，未覆盖训练seed。REAL-P256快照时仅210六榜完成，200中断、150取消，没有排新的评测队列。
