审计时间：2026-10-16T10:36:47.277782+08:00。Existing verified status reread, 8 manifests match; all 2x7248 raw correctness/parse/length/hitmax records freshly reaggregated and matched to summary; problem Pass@k recomputed; no grading/inference repeated. Paired150-vs100 freshly computed with existing compare_paired_records; other paired reports reused.

新增150−100配对结果：Macro Avg@k Δ **+1.0545 pp**，95%题目bootstrap CI **[-0.2529, +2.3554] pp**，10000次、seed42，902题/7248回答精确键匹配。此区间包含0；六个单任务区间也全部包含0。

| Task | Avg100 | Avg150 | Δ(pp) | Parse100 | Parse150 | Correct/parsed100 | Correct/parsed150 | Mean len100 | Mean len150 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| aime24 | 29.8958 | 33.0208 | +3.1250 | 98.5417 | 98.0208 | 30.3383 | 33.6876 | 7969.9 | 8326.3 |
| aime25 | 23.8542 | 25.6250 | +1.7708 | 99.2708 | 99.1667 | 24.0294 | 25.8403 | 7411.2 | 7462.0 |
| amc23 | 78.0469 | 78.0469 | +0.0000 | 99.5312 | 99.6094 | 78.4144 | 78.3529 | 4780.6 | 4675.6 |
| hmmt_feb | 9.8958 | 10.0000 | +0.1042 | 98.9583 | 98.7500 | 10.0000 | 10.1266 | 7401.0 | 7944.2 |
| math500 | 73.5500 | 74.0500 | +0.5000 | 99.9500 | 99.8000 | 73.5868 | 74.1984 | 3103.4 | 3150.4 |
| minerva | 32.8125 | 33.6397 | +0.8272 | 84.3750 | 97.9779 | 38.8889 | 34.3340 | 4504.3 | 4319.7 |
| ALL | 47.4338 | 48.3582 | +0.9244 | 97.1302 | 99.0342 | 48.8352 | 48.8298 | 5394.2 | 5486.8 |

parse定义已对全部原始记录核验：`format_valid == (extracted_answer is not None)`，且所有correct回答都parsed。它是答案能否提取的指标，不是对推理格式质量的完整评价。

| Task | Correct Δ | Parsed Δ | Both parsed N | Both parsed correct Δ | Newly parsed N / correct150 | Lost parse N / correct100 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| aime24 | +30 | -5 | 930 | +26 | 11 / 4 | 16 / 0 |
| aime25 | +17 | -1 | 945 | +16 | 7 / 2 | 8 / 1 |
| amc23 | +0 | +1 | 1269 | -3 | 6 / 3 | 5 / 0 |
| hmmt_feb | +1 | -2 | 938 | +2 | 10 / 1 | 12 / 2 |
| math500 | +10 | -3 | 1995 | +11 | 1 / 0 | 4 / 1 |
| minerva | +9 | +148 | 907 | -10 | 159 / 20 | 11 / 1 |
| ALL | +67 | +138 | 6984 | +42 | 194 / 30 | 56 / 5 |

整体正确回答 3438→3505（+67），可解析回答 7040→7178（+138）。Correct/parsed 48.83523%→48.82976%，近乎不变。两次都可解析的 6984 个回答键上，正确数净变 +42；新可解析键贡献 30 个正确，失去解析键损失 5 个正确。

这些是观测分解，不能把新增可解析回答中的正确数直接解释为“纯格式因果收益”。两checkpoint生成文本不同，按parse条件筛选也改变样本组成。AIME24/25解析率下降但正确数分别增加30/17；Minerva解析增加148、正确仅增加9，可解析子集的正确率下降。整体不支持断言提升全由格式引起，也不足以断言推理能力稳定提高。

回答平均长度仅增长 92.6 tokens（1.72%）；Hit max从 0.800% 到 0.869%。长度有增长，但未见大幅膨胀。

| Step | Reference | Existing Macro Δ(pp) | 95%CI(pp) |
| --- | --- | ---: | --- |
| 100 | lora | +3.0035 | [+1.7492, +4.3414] |
| 100 | real32 | -0.6596 | [-2.1011, +0.8246] |
| 100 | own_step50 | +0.0144 | [-1.0081, +1.0496] |
| 150 | lora | +3.5830 | [+2.2411, +4.9986] |
| 150 | real32 | +2.9892 | [+1.6527, +4.4124] |
| 150 | own_step50 | +1.0689 | [+0.0105, +2.1266] |

同step150相对LoRA/REAL32具有当前单seed下正向题目bootstrap区间；这是checkpoint150比较优势的证据，不能自动推出继续训练会继续改善。100→150整体增益区间仍跨0；150→50下界约+0.0105pp，属于接近0的边界证据。

本次新增配对：[JSON](/root/peft-for-rl/runs/ops/gradtop_r32_continue_review_20261016/paired_step150_vs_step100.json) / [Markdown](/root/peft-for-rl/runs/ops/gradtop_r32_continue_review_20261016/paired_step150_vs_step100.md)。完整逐条复核来源与SHA：[evidence_review.json](/root/peft-for-rl/runs/ops/gradtop_r32_continue_review_20261016/evidence_review.json)。
