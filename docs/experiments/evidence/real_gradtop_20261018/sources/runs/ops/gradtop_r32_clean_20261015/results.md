# GradTop-r32 本轮改动版：正式评测结果

日期：2026-10-15（Asia/Shanghai）。全部训练与两次评测已完成；队列完成时间为19:53。每个结果均为通过验证的完整7,248条记录。

共同设置：DeepSeek-R1-Distill-Qwen-1.5B，rank32，alpha64，dropout0，LR=1e-5，training/eval seed42。Step为trainer step，optimizer updates=4×Step。REAL64/REAL32中的64/32指probe采样步数，不是正式rank。

沿用最近REAL结果表的格式：Avg.为全部7,248条回答的正确率（按各任务response数加权）；任务单元格为Avg@k / Pass@k（%），保留一位小数。六任务等权Macro指标另表列出。

| 方法 | Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA | 50 | 44.5 | 31.5 / 73.3 | 23.8 / 56.7 | 73.8 / 97.5 | 11.0 / 50.0 | 70.0 / 78.8 | 22.3 / 34.9 |
| REAL64 | 50 | 46.2 | 30.6 / 70.0 | 22.7 / 50.0 | 75.4 / 95.0 | 10.0 / 26.7 | 73.2 / 79.4 | 28.3 / 40.4 |
| REAL32 | 50 | 46.4 | 31.7 / 76.7 | 25.0 / 56.7 | 74.8 / 95.0 | 10.7 / 30.0 | 74.0 / 80.4 | 25.8 / 38.6 |
| GradTop（本轮改动版） | 10 | 43.3 | 29.8 / 76.7 | 22.3 / 56.7 | 70.8 / 92.5 | 9.4 / 33.3 | 70.1 / 79.4 | 22.4 / 35.7 |
| GradTop（本轮改动版） | 50 | **47.2** | 30.8 / 70.0 | 25.3 / 56.7 | 76.0 / 95.0 | 10.5 / 40.0 | 72.8 / 78.8 | 32.5 / 43.4 |

GradTop step10是同一轨迹的预声明观察点；主比较固定为step50，不用step10与其他方法step50作matched差值，也不按结果选择主checkpoint。

以下Macro Avg@k与Macro Pass@k均为六任务等权平均；Parse、Hit max按全部回答汇总。除Mean len（tokens）外，单位均为%。

| 方法 | Step | Macro Avg@k | Macro Pass@k | Parse | Hit max | Mean len |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRA | 50 | 38.7224 | 65.20 | 88.51 | 2.11 | 7749.7 |
| REAL64 | 50 | 40.0388 | 60.25 | 92.37 | 0.23 | 5088.3 |
| REAL32 | 50 | 40.3361 | 62.89 | 92.18 | 0.97 | 5338.1 |
| GradTop（本轮改动版） | 10 | 37.4610 | 62.37 | 87.57 | 1.92 | 7162.2 |
| GradTop（本轮改动版） | 50 | **41.3282** | 63.97 | 97.82 | 0.33 | 5002.3 |

Step50配对比较：题目为cluster，10,000次bootstrap，seed42；差值为本轮GradTop减对照，单位为百分点，指标为六任务等权Macro Avg@k。

| 对照 | Macro Avg@k差值 | 95% CI | 区间是否包含0 |
| --- | ---: | --- | --- |
| LoRA | +2.6057 | [+1.4295, +3.8130] | 否 |
| REAL64 | +1.2894 | [+0.2312, +2.3687] | 否 |
| REAL32 | +0.9920 | [-0.1483, +2.1325] | 是 |

在本次固定训练seed下，GradTop相对LoRA和REAL64的Macro差值区间均高于0；相对REAL32的点估计更高，但95% CI跨0。区间只反映题目重采样的不确定性，不覆盖training-seed波动。

相对REAL32，Minerva提高6.71个百分点，MATH500下降1.20个百分点，AIME24和HMMT也略降；不能称为所有任务一致改善。GradTop的Macro Pass@k仍低于LoRA，需与平均正确率和解析率一起看。

这条GradTop保留random-B、batch梯度聚合与增量SVD核心，但采用新probe归约/分组、16次observations和uniform rank32，是本轮改动版，不是历史原实现的严格复现。REAL固定A只训练B，GradTop训练A/B，属于同rank/同正式更新预算的方法比较，非等可训练参数对照。

实测GradTop probe耗时2250.33秒（37.51分钟），正式50步训练耗时21512.50秒（5.98小时）；不含等待和评测。REAL32由缓存前缀导出，不能据此给出与其独立采集的实测加速比。

数据来源：

- LoRA step50：[summary](../../../runs/outputs/full_bench_eval/rlvr_v1_lora_r32_a64_lr1em5_b64m16n8_seed42_step50_fullbench_32768_seed42/summary/rlvr_v1_lora_r32_a64_lr1em5_b64m16n8_seed42_step50_fullbench_32768_seed42.json)；同目录verification.json已通过。
- REAL64 step50：[summary](../../../runs/outputs/full_bench_eval/rlvr_v1_real_r32_a64_lr1em5_b64m16n8_seed42_step50_fullbench_32768_seed42/summary/rlvr_v1_real_r32_a64_lr1em5_b64m16n8_seed42_step50_fullbench_32768_seed42.json)；同目录verification.json已通过。
- REAL32 step50：[summary](../../../runs/outputs/full_bench_eval/rlvr_v1_real_probe32_r32_a64_lr1em5_b64m16n8_seed42_step50_fullbench_32768_seed42/summary/rlvr_v1_real_probe32_r32_a64_lr1em5_b64m16n8_seed42_step50_fullbench_32768_seed42.json)；同目录verification.json已通过。
- GradTop（本轮改动版） step10：[summary](../../../runs/outputs/full_bench_eval/rlvr_v1_gradtop_r32_a64_lr1em5_b64m16n8_seed42_20261015_step10_fullbench_32768_seed42/summary/rlvr_v1_gradtop_r32_a64_lr1em5_b64m16n8_seed42_20261015_step10_fullbench_32768_seed42.json)；同目录verification.json已通过。
- GradTop（本轮改动版） step50：[summary](../../../runs/outputs/full_bench_eval/rlvr_v1_gradtop_r32_a64_lr1em5_b64m16n8_seed42_20261015_step50_fullbench_32768_seed42/summary/rlvr_v1_gradtop_r32_a64_lr1em5_b64m16n8_seed42_20261015_step50_fullbench_32768_seed42.json)；同目录verification.json已通过。
- 对lora的[配对报告](paired_vs_lora_step50.json)。
- 对real64的[配对报告](paired_vs_real64_step50.json)。
- 对real32的[配对报告](paired_vs_real32_step50.json)。
