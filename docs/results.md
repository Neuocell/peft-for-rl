# 实验结果入口

2026-10-20 当前数学评测入口：[两版评测协议](evaluation_protocols_zh.md)、[本机实验双协议重报告](experiments/local_results_dual_protocol_20261020_zh.md)。重报告保留历史分数，并另列 PeRL 发布版 judge 的复评分；下文 Phase-1 数字属于原判分口径，不与新表混用。

更新时间：2026-10-06 UTC

`docs/` 中的实验结果现已统一整理到 [实验文档总入口](experiments/README.md)。请从以下三份文档阅读：

1. [2026-07 至 2026-10 实验时间线](experiments/experiment_timeline_202607_202610_zh.md)
2. [方法与结果登记表](experiments/method_and_results_registry_zh.md)
3. [最近两次实验汇总与下一轮执行单](experiments/recent_two_and_next_round_20261006_zh.md)
4. [开放问题、后续实验与论文映射](experiments/open_questions_and_paper_map_zh.md)

旧版 `results.md` 中的 `base=0.315514` 是 summary-only 历史口径，不再作为当前 base。严格审计并可复用的 no-RL base 为 `0.3243481413`。

当前 Phase-1 主结果为：

| 方法 | Seed42 | Seed43 | 两-seed简单均值 | 证据角色 |
| --- | ---: | ---: | ---: | --- |
| Matched standard LoRA r32 | `0.330053` | `0.338761` | `0.334407` | matched external baseline |
| I0 | `0.330162` | `0.327747` | `0.328955` | matched random baseline |
| I8 | `0.366161` | `0.341696` | `0.353928` | mixed candidate |
| I16 | `0.313498` | 未运行 | n/a | seed42 negative screen |
| I32 | `0.376479` | `0.333201` | `0.354840` | separate signal-only endpoint |

I8 相对 matched standard LoRA 的两-seed mean delta 为 `+0.019521`，但 seed43 仅为 `+0.002934`；I32 在 seed43 则低 `0.005560`。这些值不能和旧 LoRA/OFT/RLPO/GradTop 等不同 step、dropout、snapshot 或设备协议的结果直接组成 leaderboard。所有历史数值、失败记录和 claim boundary 已在登记表中展开。
