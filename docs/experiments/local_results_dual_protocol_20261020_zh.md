# 本机保存实验的双协议结果报告

生成时间（UTC）：2026-10-20T08:13:38.093517+00:00

当前口径为 `perl_release_judge_v1`；历史本地重放版本为 `legacy_local_fullbench_v1`。无版本标签的旧文件按“原保存判分”报告，不声称已逐条重放为 v1。定义和复现入口见 [双协议说明](../evaluation_protocols_zh.md)。

本报告覆盖扫描根目录中可读取的历史评测，包含本机生成和迁入的记录；不把“本机可读”解释为全部在本机训练。重新判分保留每条原始生成、prompt、gold 和采样参数。

已完成 **90** 组回答的 PeRL 重判，其中 **84** 组为完整六任务 @32/@4；其余为 screen 或单任务，单列。

同一 judge 不代表训练配置、题目快照或提示词相同。表格按轨迹分组，不跨组排序；lr、rank、seed、syncfix、resume 和预算差异仍由 checkpoint 名称及原实验记录约束。

Avg. 按回答数加权；任务单元格为 Avg@k / Pass@k（%）。六任务标准题数为 30/30/40/30/500/272，总回答 7248。

独立逐条核验通过：**90** 组、**610,561** 个回答。原始字段、源文件 SHA256、source_judgment 和 gold 映射全部保留/核准；Avg@k、Pass@k 与总体加权 Avg. 独立复算一致。[核验回执](../../runs/eval_protocol_reaudit_20261020/verification.json)。

仓库发布版包含本报告和精确值 CSV；表内 records、summary 与核验回执链接指向原机器运行产物，并未随源码发布。另一台机器应对自己的原始回答运行复评，不能从本机表格推算新分数。

## 当前 PeRL 口径：完整六任务

### Base model (historical reference)

`base_fullbench_32768_rerun_records_20260819`

历史证据等级（step no-RL）：**A-reference**；原允许用途：absolute no-RL reference。重判不升级该等级。

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| — | 40.4 | 21.9 / 43.3 | 20.0 / 46.7 | 60.2 / 95.0 | 14.5 / 33.3 | 73.0 / 89.2 | 14.2 / 27.6 |

证据：

- step —（no-RL）：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/base_fullbench_32768_rerun_records_20260819.json)；checkpoint：`base_fullbench_32768_rerun_records_20260819`。

### BiSO (historical)

`dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_fullbench_32768`

历史证据等级（step 100, 180, 200, 220, 240, 300）：**B**；原允许用途：historical sensitivity or method context。重判不升级该等级。

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 46.8 | 30.1 / 60.0 | 22.9 / 50.0 | 71.6 / 97.5 | 14.1 / 36.7 | 81.7 / 91.6 | 18.2 / 31.6 |
| 180 | 48.5 | 30.4 / 50.0 | 22.7 / 53.3 | 74.2 / 95.0 | 14.7 / 33.3 | 83.7 / 92.2 | 22.2 / 34.2 |
| 200 | 49.4 | 35.2 / 60.0 | 25.8 / 50.0 | 72.7 / 97.5 | 13.8 / 36.7 | 83.2 / 91.8 | 24.9 / 36.4 |
| 220 | 48.3 | 32.5 / 53.3 | 20.5 / 46.7 | 70.6 / 95.0 | 14.6 / 36.7 | 83.7 / 92.8 | 25.1 / 35.7 |
| 240 | 49.4 | 30.7 / 53.3 | 25.3 / 53.3 | 73.0 / 95.0 | 16.2 / 50.0 | 83.4 / 91.6 | 26.3 / 38.6 |
| 300 | 49.3 | 31.0 / 50.0 | 23.3 / 50.0 | 72.4 / 95.0 | 15.3 / 36.7 | 83.8 / 92.6 | 27.8 / 38.2 |

证据：

- step 100：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step100_fullbench_32768.json)；checkpoint：`dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step100_fullbench_32768`。
- step 180：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step180_fullbench_32768.json)；checkpoint：`dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step180_fullbench_32768`。
- step 200：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step200_fullbench_32768.json)；checkpoint：`dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step200_fullbench_32768`。
- step 220：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step220_fullbench_32768.json)；checkpoint：`dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step220_fullbench_32768`。
- step 240：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step240_fullbench_32768.json)；checkpoint：`dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step240_fullbench_32768`。
- step 300：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step300_fullbench_32768.json)；checkpoint：`dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step300_fullbench_32768`。

### BiSO (historical)

`paper_clean_biso_r16match_270_fullbench_32768`

历史证据等级（step 200, 240, 260, 270）：**B**；原允许用途：historical sensitivity or method context。重判不升级该等级。

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 200 | 46.7 | 24.3 / 50.0 | 22.6 / 56.7 | 74.9 / 95.0 | 13.4 / 33.3 | 81.8 / 91.2 | 19.2 / 30.5 |
| 240 | 48.3 | 34.4 / 60.0 | 23.9 / 56.7 | 73.1 / 95.0 | 14.5 / 33.3 | 82.0 / 91.6 | 20.6 / 31.2 |
| 260 | 47.5 | 27.0 / 50.0 | 24.6 / 46.7 | 71.3 / 92.5 | 14.2 / 43.3 | 82.6 / 92.4 | 22.5 / 33.8 |
| 270 | 48.5 | 29.2 / 56.7 | 24.5 / 53.3 | 74.4 / 97.5 | 14.9 / 33.3 | 83.5 / 92.4 | 21.4 / 33.1 |

证据：

- step 200：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/paper_clean_biso_r16match_270_step200_fullbench_32768.json)；checkpoint：`paper_clean_biso_r16match_270_step200_fullbench_32768`。
- step 240：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/paper_clean_biso_r16match_270_step240_fullbench_32768.json)；checkpoint：`paper_clean_biso_r16match_270_step240_fullbench_32768`。
- step 260：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/paper_clean_biso_r16match_270_step260_fullbench_32768.json)；checkpoint：`paper_clean_biso_r16match_270_step260_fullbench_32768`。
- step 270：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/paper_clean_biso_r16match_270_step270_fullbench_32768.json)；checkpoint：`paper_clean_biso_r16match_270_step270_fullbench_32768`。

### CHR/RLPO (historical)

`chr_rlpo_v0_d1act95_r8to28_mean16p78_fullbench_32768_v1`

历史证据等级（step 200, 270）：**D**；原允许用途：incompatible record-level result。重判不升级该等级。

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 200 | 46.5 | 29.4 / 60.0 | 24.0 / 53.3 | 71.4 / 95.0 | 15.2 / 40.0 | 80.0 / 90.0 | 18.1 / 30.1 |
| 270 | 48.1 | 32.1 / 66.7 | 25.3 / 56.7 | 71.8 / 95.0 | 15.4 / 36.7 | 82.5 / 92.2 | 20.0 / 29.8 |

证据：

- step 200：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/chr_rlpo_v0_d1act95_r8to28_mean16p78_step200_fullbench_32768_v1.json)；checkpoint：`chr_rlpo_v0_d1act95_r8to28_mean16p78_step200_fullbench_32768_v1`。
- step 270：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/chr_rlpo_v0_d1act95_r8to28_mean16p78_step270_fullbench_32768_v1.json)；checkpoint：`chr_rlpo_v0_d1act95_r8to28_mean16p78_step270_fullbench_32768_v1`。

### DoRA syncfix1

`rlvr_v1_dora_r32_a64_lr1em5_b64m16n8_seed42_syncfix1_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 80 | 48.6 | 33.1 / 60.0 | 22.5 / 43.3 | 73.6 / 97.5 | 14.4 / 36.7 | 83.0 / 91.8 | 23.1 / 33.8 |
| 90 | 48.7 | 26.5 / 53.3 | 23.5 / 50.0 | 75.6 / 97.5 | 13.1 / 36.7 | 84.3 / 92.2 | 24.5 / 36.8 |
| 100 | 48.5 | 33.1 / 53.3 | 22.6 / 40.0 | 73.8 / 95.0 | 12.2 / 43.3 | 82.8 / 92.4 | 24.3 / 37.9 |
| 130 | 47.6 | 25.0 / 43.3 | 23.0 / 40.0 | 72.3 / 92.5 | 11.8 / 26.7 | 83.1 / 91.6 | 26.7 / 37.5 |
| 160 | 48.6 | 26.4 / 46.7 | 25.5 / 43.3 | 71.8 / 92.5 | 13.0 / 30.0 | 84.0 / 93.0 | 27.4 / 37.9 |

证据：

- step 80：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/rlvr_v1_dora_r32_a64_lr1em5_b64m16n8_seed42_syncfix1_step80_fullbench_32768_seed42.json)；checkpoint：`rlvr_v1_dora_r32_a64_lr1em5_b64m16n8_seed42_syncfix1_step80_fullbench_32768_seed42`。
- step 90：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/rlvr_v1_dora_r32_a64_lr1em5_b64m16n8_seed42_syncfix1_step90_fullbench_32768_seed42.json)；checkpoint：`rlvr_v1_dora_r32_a64_lr1em5_b64m16n8_seed42_syncfix1_step90_fullbench_32768_seed42`。
- step 100：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/rlvr_v1_dora_r32_a64_lr1em5_b64m16n8_seed42_syncfix1_step100_fullbench_32768_seed42.json)；checkpoint：`rlvr_v1_dora_r32_a64_lr1em5_b64m16n8_seed42_syncfix1_step100_fullbench_32768_seed42`。
- step 130：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/rlvr_v1_dora_r32_a64_lr1em5_b64m16n8_seed42_syncfix1_step130_fullbench_32768_seed42.json)；checkpoint：`rlvr_v1_dora_r32_a64_lr1em5_b64m16n8_seed42_syncfix1_step130_fullbench_32768_seed42`。
- step 160：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/rlvr_v1_dora_r32_a64_lr1em5_b64m16n8_seed42_syncfix1_step160_fullbench_32768_seed42.json)；checkpoint：`rlvr_v1_dora_r32_a64_lr1em5_b64m16n8_seed42_syncfix1_step160_fullbench_32768_seed42`。

### DoRA v1

`rlvr_v1_dora_r32_a64_lr1em6_b64m16n8_seed42_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 43.1 | 29.3 / 60.0 | 21.0 / 53.3 | 64.6 / 95.0 | 14.1 / 33.3 | 75.5 / 89.2 | 15.3 / 27.2 |

证据：

- step 50：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/rlvr_v1_dora_r32_a64_lr1em6_b64m16n8_seed42_step50_fullbench_32768_seed42.json)；checkpoint：`rlvr_v1_dora_r32_a64_lr1em6_b64m16n8_seed42_step50_fullbench_32768_seed42`。

### Full parameter

`full_parameter_1p5b_lr1e6_b64m16n8_seed42_v2_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 45.5 | 27.6 / 53.3 | 22.4 / 53.3 | 69.4 / 97.5 | 15.4 / 43.3 | 79.5 / 90.6 | 17.6 / 28.3 |
| 100 | 48.0 | 30.0 / 53.3 | 21.4 / 56.7 | 74.1 / 97.5 | 13.5 / 36.7 | 83.7 / 92.0 | 21.8 / 33.5 |
| 150 | 49.2 | 31.4 / 56.7 | 25.1 / 50.0 | 77.8 / 95.0 | 13.4 / 33.3 | 83.5 / 92.0 | 21.2 / 32.0 |
| 200 | 49.1 | 29.9 / 53.3 | 22.5 / 36.7 | 77.9 / 95.0 | 13.4 / 40.0 | 83.7 / 93.0 | 23.3 / 33.1 |

证据：

- step 50：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/full_parameter_1p5b_lr1e6_b64m16n8_step50_seed42_v2_step50_fullbench_32768_seed42.json)；checkpoint：`full_parameter_1p5b_lr1e6_b64m16n8_step50_seed42_v2_step50_fullbench_32768_seed42`。
- step 100：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/full_parameter_1p5b_lr1e6_b64m16n8_step50_seed42_v2_step100_fullbench_32768_seed42.json)；checkpoint：`full_parameter_1p5b_lr1e6_b64m16n8_step50_seed42_v2_step100_fullbench_32768_seed42`。
- step 150：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/full_parameter_1p5b_lr1e6_b64m16n8_step50_seed42_v2_step150_fullbench_32768_seed42.json)；checkpoint：`full_parameter_1p5b_lr1e6_b64m16n8_step50_seed42_v2_step150_fullbench_32768_seed42`。
- step 200：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/full_parameter_1p5b_lr1e6_b64m16n8_step50_seed42_v2_step200_fullbench_32768_seed42.json)；checkpoint：`full_parameter_1p5b_lr1e6_b64m16n8_step50_seed42_v2_step200_fullbench_32768_seed42`。

### GeoRA (historical)

`paper_clean_dapo_heldout637_seed42_geora_b64m16n8_lr5e-6_wd0_fullbench_32768`

历史证据等级（step 100, 160, 180, 200）：**B**；原允许用途：historical sensitivity or method context。重判不升级该等级。

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 45.4 | 25.9 / 46.7 | 16.0 / 33.3 | 68.5 / 95.0 | 10.4 / 26.7 | 81.5 / 91.6 | 26.0 / 38.2 |
| 160 | 43.8 | 18.9 / 36.7 | 18.5 / 43.3 | 68.7 / 97.5 | 8.6 / 30.0 | 78.6 / 90.6 | 25.9 / 39.7 |
| 180 | 42.9 | 18.3 / 36.7 | 16.0 / 36.7 | 67.1 / 92.5 | 9.0 / 33.3 | 77.9 / 89.8 | 25.4 / 37.9 |
| 200 | 41.9 | 17.4 / 33.3 | 17.1 / 33.3 | 63.4 / 92.5 | 9.7 / 26.7 | 76.0 / 88.8 | 26.1 / 39.7 |

证据：

- step 100：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/paper_clean_dapo_heldout637_seed42_geora_b64m16n8_lr5e-6_wd0_step100_fullbench_32768.json)；checkpoint：`paper_clean_dapo_heldout637_seed42_geora_b64m16n8_lr5e-6_wd0_step100_fullbench_32768`。
- step 160：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/paper_clean_dapo_heldout637_seed42_geora_b64m16n8_lr5e-6_wd0_step160_fullbench_32768.json)；checkpoint：`paper_clean_dapo_heldout637_seed42_geora_b64m16n8_lr5e-6_wd0_step160_fullbench_32768`。
- step 180：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/paper_clean_dapo_heldout637_seed42_geora_b64m16n8_lr5e-6_wd0_step180_fullbench_32768.json)；checkpoint：`paper_clean_dapo_heldout637_seed42_geora_b64m16n8_lr5e-6_wd0_step180_fullbench_32768`。
- step 200：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/paper_clean_dapo_heldout637_seed42_geora_b64m16n8_lr5e-6_wd0_step200_fullbench_32768.json)；checkpoint：`paper_clean_dapo_heldout637_seed42_geora_b64m16n8_lr5e-6_wd0_step200_fullbench_32768`。

### GeoRA (historical)

`paper_clean_dapo_heldout637_seed42_geora_r16a16_b64m16n8_lr2e-6_wd1e-2_clean270_20260830_rerun_fullbench_32768`

历史证据等级（step 160）：**D**；原允许用途：incompatible record-level result。重判不升级该等级。

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 160 | 46.1 | 35.0 / 60.0 | 20.7 / 46.7 | 73.8 / 95.0 | 0.0 / 0.0 | 83.0 / 91.8 | 18.8 / 30.9 |

证据：

- step 160：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/paper_clean_dapo_heldout637_seed42_geora_r16a16_b64m16n8_lr2e-6_wd1e-2_clean270_20260830_rerun_step160_fullbench_32768.json)；checkpoint：`paper_clean_dapo_heldout637_seed42_geora_r16a16_b64m16n8_lr2e-6_wd1e-2_clean270_20260830_rerun_step160_fullbench_32768`。

### GradTop adaptive rank / lr1e-6 (historical)

`gradtop_probe12_r8to32_mean31p63_fullbench_32768_v1`

历史证据等级（step 50, 100, 150, 200, 250, 270）：**D**；原允许用途：incompatible record-level result。重判不升级该等级。

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 47.7 | 31.6 / 70.0 | 24.3 / 53.3 | 73.8 / 97.5 | 14.7 / 36.7 | 82.2 / 92.0 | 17.8 / 30.1 |
| 100 | 48.9 | 31.1 / 73.3 | 25.5 / 50.0 | 74.8 / 97.5 | 15.3 / 40.0 | 83.0 / 92.0 | 21.9 / 34.9 |
| 150 | 49.5 | 32.9 / 73.3 | 23.9 / 53.3 | 76.8 / 97.5 | 13.6 / 36.7 | 84.0 / 92.2 | 23.1 / 34.6 |
| 200 | 48.7 | 31.5 / 70.0 | 24.0 / 53.3 | 74.5 / 97.5 | 13.1 / 33.3 | 83.3 / 92.4 | 23.3 / 36.0 |
| 250 | 48.7 | 28.7 / 73.3 | 23.1 / 50.0 | 75.5 / 95.0 | 15.6 / 43.3 | 83.3 / 92.0 | 23.1 / 32.7 |
| 270 | 48.7 | 27.6 / 76.7 | 24.2 / 60.0 | 75.4 / 95.0 | 14.5 / 40.0 | 83.7 / 92.0 | 23.8 / 33.5 |

证据：

- step 50：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/gradtop_probe12_r8to32_mean31p63_step50_fullbench_32768_v1.json)；checkpoint：`gradtop_probe12_r8to32_mean31p63_step50_fullbench_32768_v1`。
- step 100：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/gradtop_probe12_r8to32_mean31p63_step100_fullbench_32768_v1.json)；checkpoint：`gradtop_probe12_r8to32_mean31p63_step100_fullbench_32768_v1`。
- step 150：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/gradtop_probe12_r8to32_mean31p63_step150_fullbench_32768_v1.json)；checkpoint：`gradtop_probe12_r8to32_mean31p63_step150_fullbench_32768_v1`。
- step 200：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/gradtop_probe12_r8to32_mean31p63_step200_fullbench_32768_v1.json)；checkpoint：`gradtop_probe12_r8to32_mean31p63_step200_fullbench_32768_v1`。
- step 250：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/gradtop_probe12_r8to32_mean31p63_step250_fullbench_32768_v1.json)；checkpoint：`gradtop_probe12_r8to32_mean31p63_step250_fullbench_32768_v1`。
- step 270：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/gradtop_probe12_r8to32_mean31p63_step270_fullbench_32768_v1.json)；checkpoint：`gradtop_probe12_r8to32_mean31p63_step270_fullbench_32768_v1`。

### GradTop compaction (historical)

`gradtop_act90_compact_r8to28_mean18p29_fullbench_32768_v1`

历史证据等级（step 150）：**D**；原允许用途：incompatible record-level result。重判不升级该等级。

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 150 | 48.7 | 29.7 / 73.3 | 23.6 / 56.7 | 74.7 / 97.5 | 14.4 / 36.7 | 84.2 / 92.6 | 22.0 / 32.0 |

证据：

- step 150：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/gradtop_step100_act90_compact_r8to28_mean18p29_step150_fullbench_32768_v1.json)；checkpoint：`gradtop_step100_act90_compact_r8to28_mean18p29_step150_fullbench_32768_v1`。

### GradTop extrapolation (historical)

`gradtop_270_chord_lambda1p25_fullbench_32768_v1`

历史证据等级（step 100）：**D**；原允许用途：incompatible record-level result。重判不升级该等级。

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 49.4 | 31.8 / 70.0 | 23.8 / 53.3 | 75.7 / 97.5 | 15.3 / 50.0 | 84.0 / 92.8 | 23.5 / 35.3 |

证据：

- step 100：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/gradtop_step100_270_chord_lambda1p25_fullbench_32768_v1.json)；checkpoint：`gradtop_step100_270_chord_lambda1p25_fullbench_32768_v1`。

### GradTop-S2

`gradtop_s2_op8_r32_p4_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 49.1 | 32.5 / 73.3 | 24.2 / 46.7 | 77.2 / 95.0 | 13.5 / 33.3 | 83.4 / 92.4 | 21.0 / 34.2 |
| 100 | 51.1 | 33.5 / 76.7 | 27.4 / 60.0 | 77.0 / 97.5 | 14.1 / 36.7 | 85.2 / 93.0 | 26.8 / 38.6 |
| 150 | 51.3 | 33.0 / 76.7 | 25.0 / 60.0 | 77.3 / 95.0 | 15.3 / 40.0 | 85.2 / 93.0 | 29.6 / 41.5 |
| 200 | 52.2 | 33.9 / 73.3 | 29.1 / 53.3 | 79.7 / 95.0 | 16.0 / 43.3 | 85.6 / 93.0 | 26.9 / 40.4 |

证据：

- step 50：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/gradtop_s2_op8_r32_p4_step50_fullbench_32768_seed42.json)；checkpoint：`gradtop_s2_op8_r32_p4_step50_fullbench_32768_seed42`。
- step 100：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/gradtop_s2_op8_r32_p4_step100_fullbench_32768_seed42.json)；checkpoint：`gradtop_s2_op8_r32_p4_step100_fullbench_32768_seed42`。
- step 150：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/gradtop_s2_op8_r32_p4_step150_fullbench_32768_seed42.json)；checkpoint：`gradtop_s2_op8_r32_p4_step150_fullbench_32768_seed42`。
- step 200：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/gradtop_s2_op8_r32_p4_step200_fullbench_32768_seed42.json)；checkpoint：`gradtop_s2_op8_r32_p4_step200_fullbench_32768_seed42`。

### GradTop-S3

`gradtop_s3_w16_r32_p4_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 48.7 | 27.7 / 73.3 | 23.9 / 56.7 | 76.6 / 95.0 | 12.8 / 40.0 | 84.0 / 92.0 | 23.3 / 34.6 |
| 100 | 49.3 | 27.4 / 73.3 | 24.9 / 50.0 | 76.4 / 95.0 | 14.7 / 40.0 | 83.5 / 90.8 | 26.0 / 38.2 |
| 150 | 50.6 | 32.3 / 73.3 | 27.2 / 53.3 | 76.6 / 95.0 | 14.5 / 33.3 | 84.1 / 91.8 | 27.0 / 39.7 |
| 200 | 50.7 | 28.7 / 66.7 | 24.8 / 53.3 | 78.8 / 95.0 | 17.0 / 40.0 | 83.9 / 92.0 | 28.5 / 39.7 |

证据：

- step 50：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/gradtop_s3_w16_r32_p4_step50_fullbench_32768_seed42.json)；checkpoint：`gradtop_s3_w16_r32_p4_step50_fullbench_32768_seed42`。
- step 100：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/gradtop_s3_w16_r32_p4_step100_fullbench_32768_seed42.json)；checkpoint：`gradtop_s3_w16_r32_p4_step100_fullbench_32768_seed42`。
- step 150：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/gradtop_s3_w16_r32_p4_step150_fullbench_32768_seed42.json)；checkpoint：`gradtop_s3_w16_r32_p4_step150_fullbench_32768_seed42`。
- step 200：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/gradtop_s3_w16_r32_p4_step200_fullbench_32768_seed42.json)；checkpoint：`gradtop_s3_w16_r32_p4_step200_fullbench_32768_seed42`。

### LoRA (historical)

`dapo_boxed_stable_lora_b64_lr5e6_actual_v1_fullbench_32768`

历史证据等级（step 100, 200, 220, 240, 260, 280, 300）：**B**；原允许用途：historical sensitivity or method context。重判不升级该等级。

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 47.5 | 30.8 / 73.3 | 24.0 / 56.7 | 72.3 / 100.0 | 14.8 / 36.7 | 82.2 / 91.2 | 18.8 / 31.2 |
| 200 | 49.1 | 32.4 / 80.0 | 23.1 / 50.0 | 74.9 / 95.0 | 14.8 / 50.0 | 83.8 / 91.6 | 22.9 / 35.3 |
| 220 | 49.8 | 32.5 / 73.3 | 24.6 / 50.0 | 76.6 / 95.0 | 14.7 / 40.0 | 84.1 / 92.0 | 23.3 / 32.7 |
| 240 | 49.0 | 29.4 / 70.0 | 24.0 / 56.7 | 74.1 / 97.5 | 15.4 / 43.3 | 83.4 / 91.0 | 25.5 / 36.8 |
| 260 | 49.6 | 30.7 / 76.7 | 24.2 / 60.0 | 74.1 / 95.0 | 14.9 / 36.7 | 84.4 / 93.0 | 26.7 / 39.0 |
| 280 | 49.4 | 32.2 / 70.0 | 24.2 / 60.0 | 74.9 / 97.5 | 13.1 / 36.7 | 84.2 / 92.8 | 24.6 / 36.8 |
| 300 | 50.0 | 32.4 / 76.7 | 24.2 / 60.0 | 74.8 / 95.0 | 14.1 / 36.7 | 84.9 / 92.4 | 26.9 / 39.0 |

证据：

- step 100：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step100_fullbench_32768.json)；checkpoint：`dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step100_fullbench_32768`。
- step 200：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step200_fullbench_32768.json)；checkpoint：`dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step200_fullbench_32768`。
- step 220：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step220_fullbench_32768.json)；checkpoint：`dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step220_fullbench_32768`。
- step 240：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step240_fullbench_32768.json)；checkpoint：`dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step240_fullbench_32768`。
- step 260：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step260_fullbench_32768.json)；checkpoint：`dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step260_fullbench_32768`。
- step 280：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step280_fullbench_32768.json)；checkpoint：`dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step280_fullbench_32768`。
- step 300：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step300_fullbench_32768.json)；checkpoint：`dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step300_fullbench_32768`。

### LoRA (historical)

`dapo_boxed_stable_lora_b64_v1_fullbench_32768`

历史证据等级（step 200）：**B**；原允许用途：historical sensitivity or method context。重判不升级该等级。

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 200 | 45.5 | 25.5 / 63.3 | 24.4 / 53.3 | 70.9 / 95.0 | 14.1 / 33.3 | 79.8 / 90.4 | 16.8 / 27.9 |

证据：

- step 200：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/dapo_boxed_stable_lora_b64_v1_step200_fullbench_32768.json)；checkpoint：`dapo_boxed_stable_lora_b64_v1_step200_fullbench_32768`。

### LoRA (historical)

`paper_clean_dapo_heldout637_seed42_lora_r16a32_b64m16n8_lr1e-6_wd0_fullbench_32768`

历史证据等级（step 270）：**B**；原允许用途：historical sensitivity or method context。重判不升级该等级。

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 270 | 46.6 | 35.3 / 76.7 | 24.8 / 60.0 | 68.7 / 95.0 | 14.9 / 40.0 | 79.7 / 90.6 | 16.9 / 28.3 |

证据：

- step 270：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/paper_clean_dapo_heldout637_seed42_lora_r16a32_b64m16n8_lr1e-6_wd0_step270_fullbench_32768.json)；checkpoint：`paper_clean_dapo_heldout637_seed42_lora_r16a32_b64m16n8_lr1e-6_wd0_step270_fullbench_32768`。

### Matched standard LoRA

`matched_standard_lora_r32a64_b64m16n8_seed43_v2_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 25 | 38.8 | 20.8 / 53.3 | 17.3 / 46.7 | 56.4 / 95.0 | 12.7 / 40.0 | 71.7 / 89.4 | 15.3 / 30.1 |
| 50 | 41.1 | 24.3 / 66.7 | 24.9 / 53.3 | 59.9 / 97.5 | 14.4 / 43.3 | 71.3 / 87.2 | 16.1 / 27.9 |

证据：

- step 25：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/matched_standard_lora_r32a64_b64m16n8_step25_seed43_v2_fullbench_32768_seed42.json)；checkpoint：`matched_standard_lora_r32a64_b64m16n8_step25_seed43_v2_fullbench_32768_seed42`。
- step 50：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/matched_standard_lora_r32a64_b64m16n8_step50_seed43_v2_fullbench_32768_seed42.json)；checkpoint：`matched_standard_lora_r32a64_b64m16n8_step50_seed43_v2_fullbench_32768_seed42`。

### Matched standard LoRA

`matched_standard_lora_r32a64_b64m16n8_seed42_v2_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 40.5 | 24.7 / 50.0 | 19.7 / 60.0 | 61.0 / 97.5 | 13.3 / 33.3 | 72.5 / 88.6 | 13.8 / 25.0 |

证据：

- step 50：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/matched_standard_lora_r32a64_b64m16n8_step50_seed42_v2_fullbench_32768_seed42.json)；checkpoint：`matched_standard_lora_r32a64_b64m16n8_step50_seed42_v2_fullbench_32768_seed42`。

### NoRA refv2 / syncfix2

`rlvr_v1_nora_init_r32_a64_lr1em5_b64m16n8_seed42_syncfix2_refv2_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 50.1 | 31.4 / 73.3 | 25.2 / 50.0 | 75.9 / 95.0 | 15.7 / 40.0 | 82.8 / 90.8 | 28.4 / 41.9 |
| 100 | 47.7 | 25.4 / 73.3 | 25.2 / 46.7 | 73.7 / 95.0 | 13.1 / 30.0 | 80.7 / 90.6 | 26.7 / 37.9 |

证据：

- step 50：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/rlvr_v1_nora_init_r32_a64_lr1em5_b64m16n8_seed42_syncfix2_refv2_step50_fullbench_32768_seed42.json)；checkpoint：`rlvr_v1_nora_init_r32_a64_lr1em5_b64m16n8_seed42_syncfix2_refv2_step50_fullbench_32768_seed42`。
- step 100：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/rlvr_v1_nora_init_r32_a64_lr1em5_b64m16n8_seed42_syncfix2_refv2_step100_fullbench_32768_seed42.json)；checkpoint：`rlvr_v1_nora_init_r32_a64_lr1em5_b64m16n8_seed42_syncfix2_refv2_step100_fullbench_32768_seed42`。

### NoRA v1 (historical)

`nora_init_r32a64_lr1e5_b64m16n8_seed42_v1_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 49.5 | 29.2 / 76.7 | 23.4 / 56.7 | 75.5 / 95.0 | 13.9 / 36.7 | 84.2 / 90.8 | 27.4 / 40.8 |
| 100 | 48.1 | 26.8 / 73.3 | 23.4 / 50.0 | 73.3 / 97.5 | 13.5 / 33.3 | 83.7 / 92.4 | 24.0 / 35.3 |

证据：

- step 50：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/nora_init_r32a64_lr1e5_b64m16n8_step50_seed42_v1_fullbench_32768_seed42.json)；checkpoint：`nora_init_r32a64_lr1e5_b64m16n8_step50_seed42_v1_fullbench_32768_seed42`。
- step 100：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/nora_init_r32a64_lr1e5_b64m16n8_step100_seed42_v1_fullbench_32768_seed42.json)；checkpoint：`nora_init_r32a64_lr1e5_b64m16n8_step100_seed42_v1_fullbench_32768_seed42`。

### NoRA+DoFT (historical)

`nora_doft_r8b32_mag_init_lr1e6_b64m16n8_50_4gpu_boxed_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 40.6 | 21.7 / 50.0 | 19.2 / 53.3 | 60.9 / 97.5 | 13.8 / 30.0 | 73.8 / 90.4 | 15.2 / 26.5 |

证据：

- step 50：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/nora_doft_r8b32_mag_init_lr1e6_b64m16n8_50_4gpu_boxed_step50_fullbench_32768_seed42.json)；checkpoint：`nora_doft_r8b32_mag_init_lr1e6_b64m16n8_50_4gpu_boxed_step50_fullbench_32768_seed42`。

### NoRA+DoRA continuous (historical)

`rlvr_v1_nora_dora_continuous_r32_a64_lr1em6_b64m16n8_seed42_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 48.4 | 30.9 / 53.3 | 20.5 / 43.3 | 73.0 / 97.5 | 14.6 / 36.7 | 83.5 / 92.2 | 24.7 / 34.6 |
| 60 | 48.2 | 28.2 / 56.7 | 22.5 / 53.3 | 73.2 / 97.5 | 12.2 / 36.7 | 84.0 / 91.8 | 25.3 / 36.0 |
| 70 | 49.7 | 31.0 / 53.3 | 23.0 / 40.0 | 76.3 / 97.5 | 13.5 / 30.0 | 83.5 / 93.0 | 28.1 / 40.8 |
| 80 | 45.0 | 24.5 / 40.0 | 21.0 / 40.0 | 64.7 / 92.5 | 10.6 / 26.7 | 79.7 / 91.0 | 28.0 / 41.5 |
| 100 | 44.0 | 22.3 / 40.0 | 22.4 / 40.0 | 65.2 / 92.5 | 11.4 / 30.0 | 76.9 / 89.8 | 25.8 / 38.2 |

证据：

- step 50：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/rlvr_v1_nora_dora_continuous_r32_a64_lr1em6_b64m16n8_seed42_step50_fullbench_32768_seed42.json)；checkpoint：`rlvr_v1_nora_dora_continuous_r32_a64_lr1em6_b64m16n8_seed42_step50_fullbench_32768_seed42`。
- step 60：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/rlvr_v1_nora_dora_continuous_r32_a64_lr1em6_b64m16n8_seed42_step60_fullbench_32768_seed42.json)；checkpoint：`rlvr_v1_nora_dora_continuous_r32_a64_lr1em6_b64m16n8_seed42_step60_fullbench_32768_seed42`。
- step 70：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/rlvr_v1_nora_dora_continuous_r32_a64_lr1em6_b64m16n8_seed42_step70_fullbench_32768_seed42.json)；checkpoint：`rlvr_v1_nora_dora_continuous_r32_a64_lr1em6_b64m16n8_seed42_step70_fullbench_32768_seed42`。
- step 80：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/rlvr_v1_nora_dora_continuous_r32_a64_lr1em6_b64m16n8_seed42_step80_fullbench_32768_seed42.json)；checkpoint：`rlvr_v1_nora_dora_continuous_r32_a64_lr1em6_b64m16n8_seed42_step80_fullbench_32768_seed42`。
- step 100：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/rlvr_v1_nora_dora_continuous_r32_a64_lr1em6_b64m16n8_seed42_step100_fullbench_32768_seed42.json)；checkpoint：`rlvr_v1_nora_dora_continuous_r32_a64_lr1em6_b64m16n8_seed42_step100_fullbench_32768_seed42`。

### NoRA+DoRA syncfix1

`rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_syncfix1_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 200 | 50.1 | 36.1 / 56.7 | 26.5 / 53.3 | 74.1 / 97.5 | 13.9 / 40.0 | 83.8 / 92.0 | 25.3 / 37.9 |
| 250 | 50.2 | 27.5 / 50.0 | 26.5 / 43.3 | 80.0 / 95.0 | 14.1 / 36.7 | 84.5 / 92.4 | 24.8 / 38.6 |
| 270 | 51.5 | 37.1 / 60.0 | 27.4 / 43.3 | 76.2 / 95.0 | 15.8 / 43.3 | 85.4 / 92.2 | 25.9 / 36.4 |

证据：

- step 200：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_syncfix1_step200_fullbench_32768_seed42.json)；checkpoint：`rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_syncfix1_step200_fullbench_32768_seed42`。
- step 250：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_syncfix1_step250_fullbench_32768_seed42.json)；checkpoint：`rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_syncfix1_step250_fullbench_32768_seed42`。
- step 270：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_syncfix1_step270_fullbench_32768_seed42.json)；checkpoint：`rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_syncfix1_step270_fullbench_32768_seed42`。

### NoRA+DoRA syncfix1 / budget extension

`rlvr_ext_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_from270_to540_v1_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 300 | 50.6 | 31.5 / 53.3 | 27.1 / 53.3 | 77.3 / 95.0 | 15.2 / 30.0 | 84.8 / 92.0 | 25.2 / 37.5 |

证据：

- step 300：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/rlvr_ext_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_from270_to540_v1_step300_fullbench_32768_seed42.json)；checkpoint：`rlvr_ext_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_from270_to540_v1_step300_fullbench_32768_seed42`。

### NoRA+DoRA v1 (historical)

`rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 49.3 | 28.4 / 56.7 | 24.4 / 53.3 | 74.1 / 97.5 | 16.0 / 43.3 | 83.0 / 92.4 | 27.7 / 41.5 |
| 70 | 49.0 | 29.2 / 56.7 | 26.7 / 56.7 | 73.8 / 100.0 | 13.1 / 33.3 | 82.5 / 92.2 | 26.8 / 39.7 |
| 80 | 50.1 | 35.6 / 53.3 | 25.3 / 43.3 | 74.8 / 97.5 | 13.3 / 40.0 | 82.9 / 92.2 | 27.7 / 41.2 |
| 90 | 47.6 | 26.1 / 46.7 | 22.0 / 43.3 | 73.4 / 95.0 | 11.8 / 36.7 | 81.8 / 92.0 | 27.6 / 38.6 |
| 100 | 47.2 | 27.0 / 50.0 | 24.5 / 36.7 | 69.1 / 92.5 | 12.7 / 30.0 | 80.5 / 90.6 | 28.2 / 40.8 |

证据：

- step 50：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_step50_fullbench_32768_seed42.json)；checkpoint：`rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_step50_fullbench_32768_seed42`。
- step 70：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_step70_fullbench_32768_seed42.json)；checkpoint：`rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_step70_fullbench_32768_seed42`。
- step 80：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_step80_fullbench_32768_seed42.json)；checkpoint：`rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_step80_fullbench_32768_seed42`。
- step 90：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_step90_fullbench_32768_seed42.json)；checkpoint：`rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_step90_fullbench_32768_seed42`。
- step 100：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_step100_fullbench_32768_seed42.json)；checkpoint：`rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_step100_fullbench_32768_seed42`。

### Other historical

`dominant_atoms_v1_trainableA_pbudget8_fullbench_32768_v1`

历史证据等级（step 50）：**D**；原允许用途：incompatible record-level result。重判不升级该等级。

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 46.2 | 30.2 / 70.0 | 22.9 / 53.3 | 68.7 / 95.0 | 15.1 / 40.0 | 80.8 / 91.4 | 17.8 / 29.0 |

证据：

- step 50：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/dominant_atoms_v1_trainableA_pbudget8_step50_fullbench_32768_v1.json)；checkpoint：`dominant_atoms_v1_trainableA_pbudget8_step50_fullbench_32768_v1`。

### Other historical

`full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 44.0 | 31.1 / 73.3 | 21.2 / 46.7 | 65.2 / 92.5 | 14.8 / 33.3 | 77.0 / 91.6 | 15.6 / 25.7 |

证据：

- step 50：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42.json)；checkpoint：`full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42`。

### Phase-1 (historical protocol)

`phase1_i32_uniform_r32_b64m16n8_seed43_v1_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 25 | 41.2 | 29.7 / 60.0 | 18.4 / 56.7 | 60.1 / 95.0 | 14.8 / 46.7 | 72.6 / 88.8 | 14.5 / 27.2 |
| 50 | 41.0 | 23.6 / 66.7 | 19.9 / 50.0 | 61.0 / 97.5 | 13.9 / 40.0 | 74.3 / 89.0 | 14.1 / 26.5 |

证据：

- step 25：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/phase1_i32_uniform_r32_b64m16n8_step25_seed43_v1_fullbench_32768_seed42.json)；checkpoint：`phase1_i32_uniform_r32_b64m16n8_step25_seed43_v1_fullbench_32768_seed42`。
- step 50：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/phase1_i32_uniform_r32_b64m16n8_step50_seed43_v1_fullbench_32768_seed42.json)；checkpoint：`phase1_i32_uniform_r32_b64m16n8_step50_seed43_v1_fullbench_32768_seed42`。

### Phase-1 (historical protocol)

`phase1_i8_uniform_r32_b64m16n8_seed43_v4_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 25 | 40.1 | 25.5 / 63.3 | 22.3 / 53.3 | 57.6 / 92.5 | 13.3 / 40.0 | 71.8 / 87.6 | 13.6 / 25.0 |
| 50 | 42.0 | 27.3 / 66.7 | 18.9 / 43.3 | 63.0 / 95.0 | 13.4 / 40.0 | 75.9 / 91.4 | 13.8 / 25.0 |

证据：

- step 25：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/phase1_i8_uniform_r32_b64m16n8_step25_seed43_v4_fullbench_32768_seed42.json)；checkpoint：`phase1_i8_uniform_r32_b64m16n8_step25_seed43_v4_fullbench_32768_seed42`。
- step 50：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/phase1_i8_uniform_r32_b64m16n8_step50_seed43_v4_fullbench_32768_seed42.json)；checkpoint：`phase1_i8_uniform_r32_b64m16n8_step50_seed43_v4_fullbench_32768_seed42`。

### Phase-1 (historical protocol)

`phase1_i0_uniform_r32_b64m16n8_seed43_v3_fullbench_32768_seed42`

历史证据等级（step 50）：**A**；原允许用途：frozen Phase-1 primary evidence。重判不升级该等级。

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 40.1 | 27.0 / 66.7 | 19.1 / 46.7 | 57.8 / 95.0 | 15.2 / 36.7 | 71.4 / 89.2 | 14.0 / 25.0 |

证据：

- step 50：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/phase1_i0_uniform_r32_b64m16n8_step50_seed43_v3_fullbench_32768_seed42.json)；checkpoint：`phase1_i0_uniform_r32_b64m16n8_step50_seed43_v3_fullbench_32768_seed42`。

### RLPO (historical)

`rlpo_init_r32a64_b64m16n8_fullbench_32768_v1`

历史证据等级（step 270）：**D**；原允许用途：known-damaged aggregate。重判不升级该等级。

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 270 | 46.3 | 30.6 / 80.0 | 23.8 / 53.3 | 73.8 / 95.0 | 0.0 / 0.0 | 83.5 / 92.2 | 20.1 / 31.2 |

证据：

- step 270：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/rlpo_init_r32a64_b64m16n8_step270_fullbench_32768_v1.json)；checkpoint：`rlpo_init_r32a64_b64m16n8_step270_fullbench_32768_v1`。

### SPO/POETBS (historical)

`paper_clean_spo_poetbs_b16d1_fullbench_32768`

历史证据等级（step 120）：**D**；原允许用途：incompatible record-level result。重判不升级该等级。

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 120 | 39.9 | 24.5 / 50.0 | 20.6 / 56.7 | 63.3 / 100.0 | 0.0 / 0.0 | 74.6 / 90.4 | 14.2 / 27.2 |

证据：

- step 120：[记录/汇总](../../runs/eval_protocol_reaudit_20261020/perl_release_judge/summary/paper_clean_spo_poetbs_b16d1_step120_fullbench_32768.json)；checkpoint：`paper_clean_spo_poetbs_b16d1_step120_fullbench_32768`。

## 历史保存口径与 v1 重放：完整六任务

### Base model (historical reference)

`base_fullbench_32768_rerun_records_20260819`

历史证据等级（step no-RL）：**A-reference**；原允许用途：absolute no-RL reference。重判不升级该等级。

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| — | 38.0 | 21.9 / 43.3 | 20.0 / 46.7 | 60.2 / 95.0 | 9.6 / 33.3 | 64.3 / 78.0 | 18.6 / 32.0 |

证据：

- step —（no-RL）：[记录/汇总](../../../Tina_orthres_run/outputs/full_bench_eval/base_fullbench_32768_rerun_records_20260819/records/base_fullbench_32768_rerun_records_20260819.jsonl)；checkpoint：`base_fullbench_32768_rerun_records_20260819`。

### BiSO (historical)

`dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_fullbench_32768`

历史证据等级（step 100, 180, 200, 220, 240, 300）：**B**；原允许用途：historical sensitivity or method context。重判不升级该等级。

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 44.1 | 30.1 / 60.0 | 22.9 / 50.0 | 71.6 / 97.5 | 10.0 / 33.3 | 71.4 / 79.8 | 22.8 / 37.1 |
| 180 | 45.7 | 29.6 / 50.0 | 22.7 / 53.3 | 74.1 / 95.0 | 10.6 / 36.7 | 72.8 / 78.8 | 27.8 / 39.0 |
| 200 | 46.9 | 35.2 / 60.0 | 25.8 / 50.0 | 72.7 / 97.5 | 9.6 / 30.0 | 73.2 / 79.8 | 29.7 / 41.5 |
| 220 | 45.4 | 32.5 / 53.3 | 20.5 / 46.7 | 70.5 / 95.0 | 9.7 / 26.7 | 73.0 / 80.0 | 29.5 / 41.2 |
| 240 | 46.4 | 30.7 / 53.3 | 25.3 / 53.3 | 73.5 / 95.0 | 11.8 / 46.7 | 72.4 / 79.0 | 29.7 / 42.6 |
| 300 | 46.1 | 31.0 / 50.0 | 23.3 / 50.0 | 72.3 / 95.0 | 10.1 / 30.0 | 72.6 / 80.2 | 31.8 / 42.6 |

证据：

- step 100：[记录/汇总](../../../Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step100_fullbench_32768/records/dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step100_fullbench_32768.jsonl)；checkpoint：`dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step100_fullbench_32768`。
- step 180：[记录/汇总](../../../Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step180_fullbench_32768/records/dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step180_fullbench_32768.jsonl)；checkpoint：`dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step180_fullbench_32768`。
- step 200：[记录/汇总](../../../Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step200_fullbench_32768/records/dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step200_fullbench_32768.jsonl)；checkpoint：`dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step200_fullbench_32768`。
- step 220：[记录/汇总](../../../Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step220_fullbench_32768/records/dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step220_fullbench_32768.jsonl)；checkpoint：`dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step220_fullbench_32768`。
- step 240：[记录/汇总](../../../Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step240_fullbench_32768/records/dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step240_fullbench_32768.jsonl)；checkpoint：`dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step240_fullbench_32768`。
- step 300：[记录/汇总](../../../Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step300_fullbench_32768/records/dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step300_fullbench_32768.jsonl)；checkpoint：`dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step300_fullbench_32768`。

### BiSO (historical)

`paper_clean_biso_r16match_270_fullbench_32768`

历史证据等级（step 200, 240, 260, 270）：**B**；原允许用途：historical sensitivity or method context。重判不升级该等级。

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 200 | 44.0 | 24.3 / 50.0 | 22.6 / 56.7 | 74.9 / 95.0 | 8.8 / 30.0 | 71.4 / 79.2 | 24.5 / 37.1 |
| 240 | 45.5 | 34.4 / 60.0 | 23.9 / 56.7 | 73.1 / 95.0 | 10.1 / 26.7 | 71.7 / 79.4 | 25.4 / 36.8 |
| 260 | 44.8 | 27.0 / 50.0 | 24.6 / 46.7 | 71.2 / 92.5 | 10.3 / 40.0 | 72.1 / 80.2 | 27.7 / 39.0 |
| 270 | 45.6 | 29.2 / 56.7 | 24.5 / 53.3 | 74.4 / 97.5 | 10.6 / 33.3 | 72.5 / 79.6 | 26.3 / 37.9 |

证据：

- step 200：[记录/汇总](../../../Tina_orthres_run/outputs/full_bench_eval/paper_clean_biso_r16match_270_step200_fullbench_32768/records/paper_clean_biso_r16match_270_step200_fullbench_32768.jsonl)；checkpoint：`paper_clean_biso_r16match_270_step200_fullbench_32768`。
- step 240：[记录/汇总](../../../Tina_orthres_run/outputs/full_bench_eval/paper_clean_biso_r16match_270_step240_fullbench_32768/records/paper_clean_biso_r16match_270_step240_fullbench_32768.jsonl)；checkpoint：`paper_clean_biso_r16match_270_step240_fullbench_32768`。
- step 260：[记录/汇总](../../../Tina_orthres_run/outputs/full_bench_eval/paper_clean_biso_r16match_270_step260_fullbench_32768/records/paper_clean_biso_r16match_270_step260_fullbench_32768.jsonl)；checkpoint：`paper_clean_biso_r16match_270_step260_fullbench_32768`。
- step 270：[记录/汇总](../../../Tina_orthres_run/outputs/full_bench_eval/paper_clean_biso_r16match_270_step270_fullbench_32768/records/paper_clean_biso_r16match_270_step270_fullbench_32768.jsonl)；checkpoint：`paper_clean_biso_r16match_270_step270_fullbench_32768`。

### CHR/RLPO (historical)

`chr_rlpo_v0_d1act95_r8to28_mean16p78_fullbench_32768_v1`

历史证据等级（step 200, 270）：**D**；原允许用途：incompatible record-level result。重判不升级该等级。

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 200 | 43.8 | 29.4 / 60.0 | 24.1 / 53.3 | 71.4 / 95.0 | 10.8 / 33.3 | 69.8 / 78.2 | 23.0 / 36.4 |
| 270 | 45.2 | 32.1 / 66.7 | 25.3 / 56.7 | 71.8 / 95.0 | 10.6 / 23.3 | 72.0 / 79.6 | 24.2 / 34.9 |

证据：

- step 200：[记录/汇总](../../runs/outputs/full_bench_eval/chr_rlpo_v0_d1act95_r8to28_mean16p78_step200_fullbench_32768_v1/records/chr_rlpo_v0_d1act95_r8to28_mean16p78_step200_fullbench_32768_v1.jsonl)；checkpoint：`chr_rlpo_v0_d1act95_r8to28_mean16p78_step200_fullbench_32768_v1`。
- step 270：[记录/汇总](../../runs/outputs/full_bench_eval/chr_rlpo_v0_d1act95_r8to28_mean16p78_step270_fullbench_32768_v1/records/chr_rlpo_v0_d1act95_r8to28_mean16p78_step270_fullbench_32768_v1.jsonl)；checkpoint：`chr_rlpo_v0_d1act95_r8to28_mean16p78_step270_fullbench_32768_v1`。

### DoRA syncfix1

`rlvr_v1_dora_r32_a64_lr1em5_b64m16n8_seed42_syncfix1_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 80 | 45.9 | 33.1 / 60.0 | 22.5 / 43.3 | 73.6 / 97.5 | 9.0 / 30.0 | 72.8 / 80.2 | 28.2 / 40.8 |
| 90 | 45.6 | 26.5 / 53.3 | 23.5 / 50.0 | 75.6 / 97.5 | 8.1 / 30.0 | 73.2 / 80.2 | 29.0 / 44.1 |
| 100 | 45.8 | 33.1 / 53.3 | 22.6 / 40.0 | 73.8 / 95.0 | 8.4 / 40.0 | 72.7 / 80.4 | 27.9 / 42.6 |
| 130 | 44.9 | 25.0 / 43.3 | 23.0 / 40.0 | 72.3 / 92.5 | 8.3 / 23.3 | 72.5 / 79.2 | 30.8 / 43.0 |
| 160 | 45.7 | 26.4 / 46.7 | 25.5 / 43.3 | 71.8 / 92.5 | 9.8 / 26.7 | 72.5 / 80.0 | 32.1 / 44.1 |

证据：

- step 80：[记录/汇总](../../runs/dora-r32-lr1e5-syncfix1/outputs/full_bench_eval/rlvr_v1_dora_r32_a64_lr1em5_b64m16n8_seed42_syncfix1_step80_fullbench_32768_seed42/records/rlvr_v1_dora_r32_a64_lr1em5_b64m16n8_seed42_syncfix1_step80_fullbench_32768_seed42.jsonl)；checkpoint：`rlvr_v1_dora_r32_a64_lr1em5_b64m16n8_seed42_syncfix1_step80_fullbench_32768_seed42`。
- step 90：[记录/汇总](../../runs/dora-r32-lr1e5-syncfix1/outputs/full_bench_eval/rlvr_v1_dora_r32_a64_lr1em5_b64m16n8_seed42_syncfix1_step90_fullbench_32768_seed42/records/rlvr_v1_dora_r32_a64_lr1em5_b64m16n8_seed42_syncfix1_step90_fullbench_32768_seed42.jsonl)；checkpoint：`rlvr_v1_dora_r32_a64_lr1em5_b64m16n8_seed42_syncfix1_step90_fullbench_32768_seed42`。
- step 100：[记录/汇总](../../runs/dora-r32-lr1e5-syncfix1/outputs/full_bench_eval/rlvr_v1_dora_r32_a64_lr1em5_b64m16n8_seed42_syncfix1_step100_fullbench_32768_seed42/records/rlvr_v1_dora_r32_a64_lr1em5_b64m16n8_seed42_syncfix1_step100_fullbench_32768_seed42.jsonl)；checkpoint：`rlvr_v1_dora_r32_a64_lr1em5_b64m16n8_seed42_syncfix1_step100_fullbench_32768_seed42`。
- step 130：[记录/汇总](../../runs/dora-r32-lr1e5-syncfix1/outputs/full_bench_eval/rlvr_v1_dora_r32_a64_lr1em5_b64m16n8_seed42_syncfix1_step130_fullbench_32768_seed42/records/rlvr_v1_dora_r32_a64_lr1em5_b64m16n8_seed42_syncfix1_step130_fullbench_32768_seed42.jsonl)；checkpoint：`rlvr_v1_dora_r32_a64_lr1em5_b64m16n8_seed42_syncfix1_step130_fullbench_32768_seed42`。
- step 160：[记录/汇总](../../runs/dora-r32-lr1e5-syncfix1/outputs/full_bench_eval/rlvr_v1_dora_r32_a64_lr1em5_b64m16n8_seed42_syncfix1_step160_fullbench_32768_seed42/records/rlvr_v1_dora_r32_a64_lr1em5_b64m16n8_seed42_syncfix1_step160_fullbench_32768_seed42.jsonl)；checkpoint：`rlvr_v1_dora_r32_a64_lr1em5_b64m16n8_seed42_syncfix1_step160_fullbench_32768_seed42`。

### DoRA v1

`rlvr_v1_dora_r32_a64_lr1em6_b64m16n8_seed42_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 40.6 | 29.3 / 60.0 | 21.0 / 53.3 | 64.8 / 95.0 | 9.0 / 30.0 | 66.4 / 78.4 | 19.9 / 33.5 |

证据：

- step 50：[记录/汇总](../../runs/dora-r32-lr1e6-v1/outputs/full_bench_eval/rlvr_v1_dora_r32_a64_lr1em6_b64m16n8_seed42_step50_fullbench_32768_seed42/records/rlvr_v1_dora_r32_a64_lr1em6_b64m16n8_seed42_step50_fullbench_32768_seed42.jsonl)；checkpoint：`rlvr_v1_dora_r32_a64_lr1em6_b64m16n8_seed42_step50_fullbench_32768_seed42`。

### Full parameter

`full_parameter_1p5b_lr1e6_b64m16n8_seed42_v2_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 42.7 | 27.6 / 53.3 | 22.4 / 53.3 | 69.3 / 97.5 | 10.5 / 43.3 | 69.5 / 78.0 | 21.7 / 33.5 |
| 100 | 45.3 | 30.0 / 53.3 | 21.4 / 56.7 | 74.1 / 97.5 | 9.3 / 30.0 | 73.2 / 80.2 | 26.5 / 40.1 |
| 150 | 46.7 | 31.4 / 56.7 | 25.1 / 50.0 | 77.8 / 95.0 | 9.8 / 33.3 | 72.9 / 79.2 | 26.9 / 38.6 |
| 200 | 46.3 | 29.9 / 53.3 | 22.5 / 36.7 | 77.9 / 95.0 | 10.2 / 33.3 | 72.8 / 80.8 | 28.0 / 38.2 |

证据：

- step 50：[记录/汇总](../../runs/full-parameter-1p5b-unified-lr1e6-v2/outputs/full_bench_eval/full_parameter_1p5b_lr1e6_b64m16n8_step50_seed42_v2_step50_fullbench_32768_seed42/records/full_parameter_1p5b_lr1e6_b64m16n8_step50_seed42_v2_step50_fullbench_32768_seed42.jsonl)；checkpoint：`full_parameter_1p5b_lr1e6_b64m16n8_step50_seed42_v2_step50_fullbench_32768_seed42`。
- step 100：[记录/汇总](../../runs/full-parameter-1p5b-unified-lr1e6-v2/outputs/full_bench_eval/full_parameter_1p5b_lr1e6_b64m16n8_step50_seed42_v2_step100_fullbench_32768_seed42/records/full_parameter_1p5b_lr1e6_b64m16n8_step50_seed42_v2_step100_fullbench_32768_seed42.jsonl)；checkpoint：`full_parameter_1p5b_lr1e6_b64m16n8_step50_seed42_v2_step100_fullbench_32768_seed42`。
- step 150：[记录/汇总](../../runs/full-parameter-1p5b-unified-lr1e6-v2/outputs/full_bench_eval/full_parameter_1p5b_lr1e6_b64m16n8_step50_seed42_v2_step150_fullbench_32768_seed42/records/full_parameter_1p5b_lr1e6_b64m16n8_step50_seed42_v2_step150_fullbench_32768_seed42.jsonl)；checkpoint：`full_parameter_1p5b_lr1e6_b64m16n8_step50_seed42_v2_step150_fullbench_32768_seed42`。
- step 200：[记录/汇总](../../runs/full-parameter-1p5b-unified-lr1e6-v2/outputs/full_bench_eval/full_parameter_1p5b_lr1e6_b64m16n8_step50_seed42_v2_step200_fullbench_32768_seed42/records/full_parameter_1p5b_lr1e6_b64m16n8_step50_seed42_v2_step200_fullbench_32768_seed42.jsonl)；checkpoint：`full_parameter_1p5b_lr1e6_b64m16n8_step50_seed42_v2_step200_fullbench_32768_seed42`。

### GeoRA (historical)

`paper_clean_dapo_heldout637_seed42_geora_b64m16n8_lr5e-6_wd0_fullbench_32768`

历史证据等级（step 100, 160, 180, 200）：**B**；原允许用途：historical sensitivity or method context。重判不升级该等级。

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 42.8 | 25.9 / 46.7 | 16.0 / 33.3 | 68.5 / 95.0 | 7.4 / 23.3 | 72.0 / 80.4 | 28.7 / 41.2 |
| 160 | 41.6 | 18.9 / 36.7 | 18.5 / 43.3 | 68.7 / 97.5 | 6.1 / 23.3 | 69.7 / 80.4 | 29.6 / 45.2 |
| 180 | 41.0 | 18.3 / 36.7 | 16.0 / 36.7 | 67.1 / 92.5 | 8.4 / 36.7 | 69.2 / 79.2 | 29.0 / 41.5 |
| 200 | 39.7 | 17.4 / 33.3 | 17.1 / 33.3 | 63.4 / 92.5 | 8.6 / 26.7 | 67.8 / 78.8 | 27.5 / 42.3 |

证据：

- step 100：[记录/汇总](../../../Tina_orthres_run/outputs/full_bench_eval/paper_clean_dapo_heldout637_seed42_geora_b64m16n8_lr5e-6_wd0_step100_fullbench_32768/records/paper_clean_dapo_heldout637_seed42_geora_b64m16n8_lr5e-6_wd0_step100_fullbench_32768.jsonl)；checkpoint：`paper_clean_dapo_heldout637_seed42_geora_b64m16n8_lr5e-6_wd0_step100_fullbench_32768`。
- step 160：[记录/汇总](../../../Tina_orthres_run/outputs/full_bench_eval/paper_clean_dapo_heldout637_seed42_geora_b64m16n8_lr5e-6_wd0_step160_fullbench_32768/records/paper_clean_dapo_heldout637_seed42_geora_b64m16n8_lr5e-6_wd0_step160_fullbench_32768.jsonl)；checkpoint：`paper_clean_dapo_heldout637_seed42_geora_b64m16n8_lr5e-6_wd0_step160_fullbench_32768`。
- step 180：[记录/汇总](../../../Tina_orthres_run/outputs/full_bench_eval/paper_clean_dapo_heldout637_seed42_geora_b64m16n8_lr5e-6_wd0_step180_fullbench_32768/records/paper_clean_dapo_heldout637_seed42_geora_b64m16n8_lr5e-6_wd0_step180_fullbench_32768.jsonl)；checkpoint：`paper_clean_dapo_heldout637_seed42_geora_b64m16n8_lr5e-6_wd0_step180_fullbench_32768`。
- step 200：[记录/汇总](../../../Tina_orthres_run/outputs/full_bench_eval/paper_clean_dapo_heldout637_seed42_geora_b64m16n8_lr5e-6_wd0_step200_fullbench_32768/records/paper_clean_dapo_heldout637_seed42_geora_b64m16n8_lr5e-6_wd0_step200_fullbench_32768.jsonl)；checkpoint：`paper_clean_dapo_heldout637_seed42_geora_b64m16n8_lr5e-6_wd0_step200_fullbench_32768`。

### GeoRA (historical)

`paper_clean_dapo_heldout637_seed42_geora_r16a16_b64m16n8_lr2e-6_wd1e-2_clean270_20260830_rerun_fullbench_32768`

历史证据等级（step 160）：**D**；原允许用途：incompatible record-level result。重判不升级该等级。

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 160 | 44.1 | 35.0 / 60.0 | 20.7 / 46.7 | 73.8 / 95.0 | 0.0 / 0.0 | 71.9 / 79.6 | 25.6 / 37.5 |

证据：

- step 160：[记录/汇总](../../../Tina_orthres_run/outputs/full_bench_eval/paper_clean_dapo_heldout637_seed42_geora_r16a16_b64m16n8_lr2e-6_wd1e-2_clean270_20260830_rerun_step160_fullbench_32768/records/paper_clean_dapo_heldout637_seed42_geora_r16a16_b64m16n8_lr2e-6_wd1e-2_clean270_20260830_rerun_step160_fullbench_32768.jsonl)；checkpoint：`paper_clean_dapo_heldout637_seed42_geora_r16a16_b64m16n8_lr2e-6_wd1e-2_clean270_20260830_rerun_step160_fullbench_32768`。

### GradTop adaptive rank / lr1e-6 (historical)

`gradtop_probe12_r8to32_mean31p63_fullbench_32768_v1`

历史证据等级（step 50, 100, 150, 200, 250, 270）：**D**；原允许用途：incompatible record-level result。重判不升级该等级。

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 45.1 | 31.6 / 70.0 | 24.3 / 53.3 | 73.5 / 97.5 | 10.2 / 33.3 | 72.0 / 80.0 | 23.2 / 36.4 |
| 100 | 46.2 | 31.0 / 73.3 | 25.5 / 50.0 | 74.8 / 97.5 | 10.8 / 43.3 | 72.5 / 79.4 | 27.2 / 39.0 |
| 150 | 46.5 | 32.9 / 73.3 | 23.9 / 53.3 | 76.8 / 97.5 | 9.1 / 33.3 | 73.0 / 80.2 | 27.1 / 38.6 |
| 200 | 46.2 | 31.5 / 70.0 | 24.0 / 53.3 | 74.5 / 97.5 | 9.2 / 33.3 | 73.1 / 80.8 | 28.8 / 41.5 |
| 250 | 46.0 | 28.6 / 73.3 | 23.1 / 50.0 | 75.5 / 95.0 | 11.2 / 40.0 | 72.8 / 79.4 | 28.4 / 40.8 |
| 270 | 46.1 | 27.6 / 76.7 | 24.2 / 60.0 | 75.4 / 95.0 | 10.6 / 36.7 | 73.0 / 80.0 | 28.9 / 40.8 |

证据：

- step 50：[记录/汇总](../../runs/outputs/full_bench_eval/gradtop_probe12_r8to32_mean31p63_step50_fullbench_32768_v1/records/gradtop_probe12_r8to32_mean31p63_step50_fullbench_32768_v1.jsonl)；checkpoint：`gradtop_probe12_r8to32_mean31p63_step50_fullbench_32768_v1`。
- step 100：[记录/汇总](../../runs/outputs/full_bench_eval/gradtop_probe12_r8to32_mean31p63_step100_fullbench_32768_v1/records/gradtop_probe12_r8to32_mean31p63_step100_fullbench_32768_v1.jsonl)；checkpoint：`gradtop_probe12_r8to32_mean31p63_step100_fullbench_32768_v1`。
- step 150：[记录/汇总](../../runs/outputs/full_bench_eval/gradtop_probe12_r8to32_mean31p63_step150_fullbench_32768_v1/records/gradtop_probe12_r8to32_mean31p63_step150_fullbench_32768_v1.jsonl)；checkpoint：`gradtop_probe12_r8to32_mean31p63_step150_fullbench_32768_v1`。
- step 200：[记录/汇总](../../runs/outputs/full_bench_eval/gradtop_probe12_r8to32_mean31p63_step200_fullbench_32768_v1/records/gradtop_probe12_r8to32_mean31p63_step200_fullbench_32768_v1.jsonl)；checkpoint：`gradtop_probe12_r8to32_mean31p63_step200_fullbench_32768_v1`。
- step 250：[记录/汇总](../../runs/outputs/full_bench_eval/gradtop_probe12_r8to32_mean31p63_step250_fullbench_32768_v1/records/gradtop_probe12_r8to32_mean31p63_step250_fullbench_32768_v1.jsonl)；checkpoint：`gradtop_probe12_r8to32_mean31p63_step250_fullbench_32768_v1`。
- step 270：[记录/汇总](../../runs/outputs/full_bench_eval/gradtop_probe12_r8to32_mean31p63_step270_fullbench_32768_v1/records/gradtop_probe12_r8to32_mean31p63_step270_fullbench_32768_v1.jsonl)；checkpoint：`gradtop_probe12_r8to32_mean31p63_step270_fullbench_32768_v1`。

### GradTop compaction (historical)

`gradtop_act90_compact_r8to28_mean18p29_fullbench_32768_v1`

历史证据等级（step 150）：**D**；原允许用途：incompatible record-level result。重判不升级该等级。

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 150 | 46.0 | 29.7 / 73.3 | 23.6 / 56.7 | 74.8 / 97.5 | 10.0 / 33.3 | 73.6 / 79.8 | 27.0 / 37.5 |

证据：

- step 150：[记录/汇总](../../runs/outputs/full_bench_eval/gradtop_step100_act90_compact_r8to28_mean18p29_step150_fullbench_32768_v1/records/gradtop_step100_act90_compact_r8to28_mean18p29_step150_fullbench_32768_v1.jsonl)；checkpoint：`gradtop_step100_act90_compact_r8to28_mean18p29_step150_fullbench_32768_v1`。

### GradTop extrapolation (historical)

`gradtop_270_chord_lambda1p25_fullbench_32768_v1`

历史证据等级（step 100）：**D**；原允许用途：incompatible record-level result。重判不升级该等级。

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 46.4 | 31.8 / 70.0 | 23.8 / 53.3 | 75.8 / 97.5 | 10.6 / 46.7 | 73.2 / 80.4 | 26.9 / 39.7 |

证据：

- step 100：[记录/汇总](../../runs/outputs/full_bench_eval/gradtop_step100_270_chord_lambda1p25_fullbench_32768_v1/records/gradtop_step100_270_chord_lambda1p25_fullbench_32768_v1.jsonl)；checkpoint：`gradtop_step100_270_chord_lambda1p25_fullbench_32768_v1`。

### GradTop-S2

`gradtop_s2_op8_r32_p4_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 46.3 | 32.5 / 73.3 | 24.2 / 46.7 | 77.2 / 95.0 | 9.8 / 33.3 | 72.2 / 79.8 | 26.4 / 40.1 |
| 100 | 48.2 | 33.5 / 76.7 | 27.4 / 60.0 | 77.0 / 97.5 | 10.1 / 33.3 | 74.2 / 80.6 | 31.5 / 43.4 |
| 150 | 48.3 | 33.0 / 76.7 | 25.0 / 60.0 | 77.3 / 95.0 | 11.0 / 43.3 | 73.8 / 79.8 | 34.6 / 46.0 |
| 200 | 49.2 | 33.8 / 73.3 | 29.1 / 53.3 | 79.7 / 95.0 | 11.2 / 40.0 | 74.2 / 80.2 | 32.4 / 45.6 |

证据：

- step 50：[记录/汇总](../../../peft-for-rl-gradtop-s2-20261018/runs/gradtop_s2_op8_r32_p4/evaluation/records/gradtop_s2_op8_r32_p4_step50_fullbench_32768_seed42.jsonl)；checkpoint：`gradtop_s2_op8_r32_p4_step50_fullbench_32768_seed42`。
- step 100：[记录/汇总](../../../peft-for-rl-gradtop-s2-20261018/runs/gradtop_s2_op8_r32_p4/evaluation_step100/records/gradtop_s2_op8_r32_p4_step100_fullbench_32768_seed42.jsonl)；checkpoint：`gradtop_s2_op8_r32_p4_step100_fullbench_32768_seed42`。
- step 150：[记录/汇总](../../../peft-for-rl-gradtop-s2-20261018/runs/gradtop_s2_op8_r32_p4/evaluation_step150/records/gradtop_s2_op8_r32_p4_step150_fullbench_32768_seed42.jsonl)；checkpoint：`gradtop_s2_op8_r32_p4_step150_fullbench_32768_seed42`。
- step 200：[记录/汇总](../../../peft-for-rl-gradtop-s2-20261018/runs/gradtop_s2_op8_r32_p4/evaluation_step200/records/gradtop_s2_op8_r32_p4_step200_fullbench_32768_seed42.jsonl)；checkpoint：`gradtop_s2_op8_r32_p4_step200_fullbench_32768_seed42`。

### GradTop-S3

`gradtop_s3_w16_r32_p4_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 45.8 | 27.7 / 73.3 | 23.9 / 56.7 | 76.6 / 95.0 | 8.4 / 33.3 | 73.1 / 79.8 | 27.9 / 40.1 |
| 100 | 46.6 | 27.4 / 73.3 | 24.9 / 50.0 | 76.4 / 95.0 | 10.5 / 40.0 | 73.0 / 79.0 | 31.2 / 44.9 |
| 150 | 47.9 | 32.3 / 73.3 | 27.2 / 53.3 | 76.6 / 95.0 | 9.6 / 30.0 | 73.8 / 80.0 | 32.4 / 46.3 |
| 200 | 47.9 | 28.7 / 66.7 | 24.8 / 53.3 | 78.8 / 95.0 | 12.0 / 36.7 | 73.4 / 80.0 | 33.7 / 46.0 |

证据：

- step 50：[记录/汇总](../../../peft-for-rl-gradtop-s3-20261018/runs/gradtop_s3_w16_r32_p4/evaluation/records/gradtop_s3_w16_r32_p4_step50_fullbench_32768_seed42.jsonl)；checkpoint：`gradtop_s3_w16_r32_p4_step50_fullbench_32768_seed42`。
- step 100：[记录/汇总](../../../peft-for-rl-gradtop-s3-20261018/runs/gradtop_s3_w16_r32_p4/evaluation_step100/records/gradtop_s3_w16_r32_p4_step100_fullbench_32768_seed42.jsonl)；checkpoint：`gradtop_s3_w16_r32_p4_step100_fullbench_32768_seed42`。
- step 150：[记录/汇总](../../../peft-for-rl-gradtop-s3-20261018/runs/gradtop_s3_w16_r32_p4/evaluation_step150/records/gradtop_s3_w16_r32_p4_step150_fullbench_32768_seed42.jsonl)；checkpoint：`gradtop_s3_w16_r32_p4_step150_fullbench_32768_seed42`。
- step 200：[记录/汇总](../../../peft-for-rl-gradtop-s3-20261018/runs/gradtop_s3_w16_r32_p4/evaluation_step200/records/gradtop_s3_w16_r32_p4_step200_fullbench_32768_seed42.jsonl)；checkpoint：`gradtop_s3_w16_r32_p4_step200_fullbench_32768_seed42`。

### LoRA (historical)

`dapo_boxed_stable_lora_b64_lr5e6_actual_v1_fullbench_32768`

历史证据等级（step 100, 200, 220, 240, 260, 280, 300）：**B**；原允许用途：historical sensitivity or method context。重判不升级该等级。

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 44.9 | 30.8 / 73.3 | 24.1 / 56.7 | 72.3 / 100.0 | 10.2 / 36.7 | 71.4 / 79.0 | 25.2 / 37.5 |
| 200 | 46.2 | 32.4 / 80.0 | 23.1 / 50.0 | 74.9 / 95.0 | 10.7 / 46.7 | 72.8 / 79.6 | 27.7 / 41.9 |
| 220 | 47.0 | 32.5 / 73.3 | 24.6 / 50.0 | 76.6 / 95.0 | 10.4 / 40.0 | 73.2 / 79.8 | 28.6 / 39.7 |
| 240 | 46.0 | 29.4 / 70.0 | 24.0 / 56.7 | 74.1 / 97.5 | 10.5 / 43.3 | 72.5 / 78.8 | 29.4 / 41.2 |
| 260 | 46.6 | 30.7 / 76.7 | 24.2 / 60.0 | 74.1 / 95.0 | 10.8 / 36.7 | 73.2 / 80.2 | 30.8 / 43.8 |
| 280 | 46.6 | 32.2 / 70.0 | 24.2 / 60.0 | 74.9 / 97.5 | 9.6 / 36.7 | 73.2 / 80.2 | 29.4 / 41.9 |
| 300 | 46.9 | 32.3 / 76.7 | 24.2 / 60.0 | 74.8 / 95.0 | 9.6 / 36.7 | 73.3 / 79.6 | 31.6 / 43.4 |

证据：

- step 100：[记录/汇总](../../../Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step100_fullbench_32768/records/dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step100_fullbench_32768.jsonl)；checkpoint：`dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step100_fullbench_32768`。
- step 200：[记录/汇总](../../../Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step200_fullbench_32768/records/dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step200_fullbench_32768.jsonl)；checkpoint：`dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step200_fullbench_32768`。
- step 220：[记录/汇总](../../../Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step220_fullbench_32768/records/dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step220_fullbench_32768.jsonl)；checkpoint：`dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step220_fullbench_32768`。
- step 240：[记录/汇总](../../../Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step240_fullbench_32768/records/dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step240_fullbench_32768.jsonl)；checkpoint：`dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step240_fullbench_32768`。
- step 260：[记录/汇总](../../../Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step260_fullbench_32768/records/dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step260_fullbench_32768.jsonl)；checkpoint：`dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step260_fullbench_32768`。
- step 280：[记录/汇总](../../../Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step280_fullbench_32768/records/dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step280_fullbench_32768.jsonl)；checkpoint：`dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step280_fullbench_32768`。
- step 300：[记录/汇总](../../../Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step300_fullbench_32768/records/dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step300_fullbench_32768.jsonl)；checkpoint：`dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step300_fullbench_32768`。

### LoRA (historical)

`dapo_boxed_stable_lora_b64_v1_fullbench_32768`

历史证据等级（step 200）：**B**；原允许用途：historical sensitivity or method context。重判不升级该等级。

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 200 | 43.0 | 25.5 / 63.3 | 24.4 / 53.3 | 70.9 / 95.0 | 9.6 / 26.7 | 70.2 / 78.8 | 21.7 / 34.6 |

证据：

- step 200：[记录/汇总](../../../Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_lora_b64_v1_step200_fullbench_32768/records/dapo_boxed_stable_lora_b64_v1_step200_fullbench_32768.jsonl)；checkpoint：`dapo_boxed_stable_lora_b64_v1_step200_fullbench_32768`。

### LoRA (historical)

`paper_clean_dapo_heldout637_seed42_lora_r16a32_b64m16n8_lr1e-6_wd0_fullbench_32768`

历史证据等级（step 270）：**B**；原允许用途：historical sensitivity or method context。重判不升级该等级。

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 270 | 43.9 | 35.3 / 76.7 | 24.9 / 60.0 | 68.7 / 95.0 | 10.0 / 36.7 | 69.5 / 78.6 | 21.9 / 32.4 |

证据：

- step 270：[记录/汇总](../../../Tina_orthres_run/outputs/full_bench_eval/paper_clean_dapo_heldout637_seed42_lora_r16a32_b64m16n8_lr1e-6_wd0_step270_fullbench_32768/records/paper_clean_dapo_heldout637_seed42_lora_r16a32_b64m16n8_lr1e-6_wd0_step270_fullbench_32768.jsonl)；checkpoint：`paper_clean_dapo_heldout637_seed42_lora_r16a32_b64m16n8_lr1e-6_wd0_step270_fullbench_32768`。

### Matched standard LoRA

`matched_standard_lora_r32a64_b64m16n8_seed43_v2_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 25 | 36.9 | 20.2 / 53.3 | 17.3 / 46.7 | 56.6 / 95.0 | 9.3 / 36.7 | 63.4 / 78.4 | 21.5 / 34.2 |
| 50 | 39.0 | 24.3 / 66.7 | 24.9 / 53.3 | 59.9 / 97.5 | 10.1 / 40.0 | 63.2 / 76.6 | 20.9 / 33.8 |

证据：

- step 25：[记录/汇总](../../runs/matched-standard-lora-r32-v2/outputs/full_bench_eval/matched_standard_lora_r32a64_b64m16n8_step25_seed43_v2_fullbench_32768_seed42/records/matched_standard_lora_r32a64_b64m16n8_step25_seed43_v2_fullbench_32768_seed42.jsonl)；checkpoint：`matched_standard_lora_r32a64_b64m16n8_step25_seed43_v2_fullbench_32768_seed42`。
- step 50：[记录/汇总](../../runs/matched-standard-lora-r32-v2/outputs/full_bench_eval/matched_standard_lora_r32a64_b64m16n8_step50_seed43_v2_fullbench_32768_seed42/records/matched_standard_lora_r32a64_b64m16n8_step50_seed43_v2_fullbench_32768_seed42.jsonl)；checkpoint：`matched_standard_lora_r32a64_b64m16n8_step50_seed43_v2_fullbench_32768_seed42`。

### Matched standard LoRA

`matched_standard_lora_r32a64_b64m16n8_seed42_v2_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 38.5 | 24.7 / 50.0 | 19.8 / 60.0 | 61.0 / 97.5 | 9.3 / 23.3 | 64.7 / 77.4 | 18.6 / 32.0 |

证据：

- step 50：[记录/汇总](../../runs/matched-standard-lora-r32-v2/outputs/full_bench_eval/matched_standard_lora_r32a64_b64m16n8_step50_seed42_v2_fullbench_32768_seed42/records/matched_standard_lora_r32a64_b64m16n8_step50_seed42_v2_fullbench_32768_seed42.jsonl)；checkpoint：`matched_standard_lora_r32a64_b64m16n8_step50_seed42_v2_fullbench_32768_seed42`。

### NoRA refv2 / syncfix2

`rlvr_v1_nora_init_r32_a64_lr1em5_b64m16n8_seed42_syncfix2_refv2_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 47.1 | 31.4 / 73.3 | 25.2 / 50.0 | 75.9 / 95.0 | 11.0 / 36.7 | 72.4 / 79.2 | 31.7 / 45.6 |
| 100 | 45.7 | 25.4 / 73.3 | 25.2 / 46.7 | 73.7 / 95.0 | 10.0 / 40.0 | 72.1 / 79.6 | 31.5 / 42.6 |

证据：

- step 50：[记录/汇总](../../runs/nora-init-r32-lr1e5-syncfix2-refv2/early-checkpoint-evaluation/outputs/full_bench_eval/rlvr_v1_nora_init_r32_a64_lr1em5_b64m16n8_seed42_syncfix2_refv2_step50_fullbench_32768_seed42/records/rlvr_v1_nora_init_r32_a64_lr1em5_b64m16n8_seed42_syncfix2_refv2_step50_fullbench_32768_seed42.jsonl)；checkpoint：`rlvr_v1_nora_init_r32_a64_lr1em5_b64m16n8_seed42_syncfix2_refv2_step50_fullbench_32768_seed42`。
- step 100：[记录/汇总](../../runs/nora-init-r32-lr1e5-syncfix2-refv2/early-checkpoint-evaluation/outputs/full_bench_eval/rlvr_v1_nora_init_r32_a64_lr1em5_b64m16n8_seed42_syncfix2_refv2_step100_fullbench_32768_seed42/records/rlvr_v1_nora_init_r32_a64_lr1em5_b64m16n8_seed42_syncfix2_refv2_step100_fullbench_32768_seed42.jsonl)；checkpoint：`rlvr_v1_nora_init_r32_a64_lr1em5_b64m16n8_seed42_syncfix2_refv2_step100_fullbench_32768_seed42`。

### NoRA v1 (historical)

`nora_init_r32a64_lr1e5_b64m16n8_seed42_v1_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 46.4 | 29.2 / 76.7 | 23.4 / 56.7 | 75.5 / 95.0 | 9.9 / 43.3 | 72.9 / 78.8 | 31.1 / 44.1 |
| 100 | 45.4 | 26.8 / 73.3 | 23.4 / 50.0 | 73.3 / 97.5 | 9.4 / 33.3 | 73.2 / 80.4 | 29.0 / 41.2 |

证据：

- step 50：[记录/汇总](../../runs/nora-r32-lr1e5-probe-v1/outputs/full_bench_eval/nora_init_r32a64_lr1e5_b64m16n8_step50_seed42_v1_fullbench_32768_seed42/records/nora_init_r32a64_lr1e5_b64m16n8_step50_seed42_v1_fullbench_32768_seed42.jsonl)；checkpoint：`nora_init_r32a64_lr1e5_b64m16n8_step50_seed42_v1_fullbench_32768_seed42`。
- step 100：[记录/汇总](../../runs/nora-r32-lr1e5-probe-v1/outputs/full_bench_eval/nora_init_r32a64_lr1e5_b64m16n8_step100_seed42_v1_fullbench_32768_seed42/records/nora_init_r32a64_lr1e5_b64m16n8_step100_seed42_v1_fullbench_32768_seed42.jsonl)；checkpoint：`nora_init_r32a64_lr1e5_b64m16n8_step100_seed42_v1_fullbench_32768_seed42`。

### NoRA+DoFT (historical)

`nora_doft_r8b32_mag_init_lr1e6_b64m16n8_50_4gpu_boxed_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 38.3 | 21.7 / 50.0 | 19.2 / 53.3 | 60.9 / 97.5 | 8.8 / 23.3 | 65.0 / 79.0 | 20.3 / 33.8 |

证据：

- step 50：[记录/汇总](../../runs/nora-doft-v1/outputs/full_bench_eval/nora_doft_r8b32_mag_init_lr1e6_b64m16n8_50_4gpu_boxed_step50_fullbench_32768_seed42/records/nora_doft_r8b32_mag_init_lr1e6_b64m16n8_50_4gpu_boxed_step50_fullbench_32768_seed42.jsonl)；checkpoint：`nora_doft_r8b32_mag_init_lr1e6_b64m16n8_50_4gpu_boxed_step50_fullbench_32768_seed42`。

### NoRA+DoRA continuous (historical)

`rlvr_v1_nora_dora_continuous_r32_a64_lr1em6_b64m16n8_seed42_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 45.7 | 30.9 / 53.3 | 20.5 / 43.3 | 73.0 / 97.5 | 11.5 / 36.7 | 72.4 / 80.2 | 29.9 / 40.4 |
| 60 | 45.7 | 28.2 / 56.7 | 22.5 / 53.3 | 73.2 / 97.5 | 9.0 / 33.3 | 73.5 / 79.8 | 30.5 / 42.3 |
| 70 | 46.8 | 31.0 / 53.3 | 23.0 / 40.0 | 76.3 / 97.5 | 9.8 / 26.7 | 72.9 / 80.6 | 32.0 / 46.7 |
| 80 | 42.6 | 24.5 / 40.0 | 21.0 / 40.0 | 65.2 / 92.5 | 8.8 / 20.0 | 69.0 / 79.2 | 32.7 / 47.1 |
| 100 | 41.9 | 22.3 / 40.0 | 22.4 / 40.0 | 65.2 / 92.5 | 9.7 / 30.0 | 66.9 / 78.0 | 31.3 / 45.6 |

证据：

- step 50：[记录/汇总](../../runs/nora-dora-continuous-r32-lr1e6-v1/outputs/full_bench_eval/rlvr_v1_nora_dora_continuous_r32_a64_lr1em6_b64m16n8_seed42_step50_fullbench_32768_seed42/records/rlvr_v1_nora_dora_continuous_r32_a64_lr1em6_b64m16n8_seed42_step50_fullbench_32768_seed42.jsonl)；checkpoint：`rlvr_v1_nora_dora_continuous_r32_a64_lr1em6_b64m16n8_seed42_step50_fullbench_32768_seed42`。
- step 60：[记录/汇总](../../runs/nora-dora-continuous-r32-lr1e6-v1/outputs/full_bench_eval/rlvr_v1_nora_dora_continuous_r32_a64_lr1em6_b64m16n8_seed42_step60_fullbench_32768_seed42/records/rlvr_v1_nora_dora_continuous_r32_a64_lr1em6_b64m16n8_seed42_step60_fullbench_32768_seed42.jsonl)；checkpoint：`rlvr_v1_nora_dora_continuous_r32_a64_lr1em6_b64m16n8_seed42_step60_fullbench_32768_seed42`。
- step 70：[记录/汇总](../../runs/nora-dora-continuous-r32-lr1e6-v1/outputs/full_bench_eval/rlvr_v1_nora_dora_continuous_r32_a64_lr1em6_b64m16n8_seed42_step70_fullbench_32768_seed42/records/rlvr_v1_nora_dora_continuous_r32_a64_lr1em6_b64m16n8_seed42_step70_fullbench_32768_seed42.jsonl)；checkpoint：`rlvr_v1_nora_dora_continuous_r32_a64_lr1em6_b64m16n8_seed42_step70_fullbench_32768_seed42`。
- step 80：[记录/汇总](../../runs/nora-dora-continuous-r32-lr1e6-v1/outputs/full_bench_eval/rlvr_v1_nora_dora_continuous_r32_a64_lr1em6_b64m16n8_seed42_step80_fullbench_32768_seed42/records/rlvr_v1_nora_dora_continuous_r32_a64_lr1em6_b64m16n8_seed42_step80_fullbench_32768_seed42.jsonl)；checkpoint：`rlvr_v1_nora_dora_continuous_r32_a64_lr1em6_b64m16n8_seed42_step80_fullbench_32768_seed42`。
- step 100：[记录/汇总](../../runs/nora-dora-continuous-r32-lr1e6-v1/outputs/full_bench_eval/rlvr_v1_nora_dora_continuous_r32_a64_lr1em6_b64m16n8_seed42_step100_fullbench_32768_seed42/records/rlvr_v1_nora_dora_continuous_r32_a64_lr1em6_b64m16n8_seed42_step100_fullbench_32768_seed42.jsonl)；checkpoint：`rlvr_v1_nora_dora_continuous_r32_a64_lr1em6_b64m16n8_seed42_step100_fullbench_32768_seed42`。

### NoRA+DoRA syncfix1

`rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_syncfix1_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 200 | 47.2 | 36.1 / 56.7 | 26.5 / 53.3 | 74.1 / 97.5 | 9.2 / 40.0 | 73.0 / 79.2 | 29.8 / 43.4 |
| 250 | 47.4 | 27.5 / 50.0 | 26.5 / 43.3 | 80.0 / 95.0 | 10.9 / 33.3 | 73.6 / 80.0 | 29.4 / 42.6 |
| 270 | 48.6 | 37.1 / 60.0 | 27.4 / 43.3 | 76.2 / 95.0 | 11.7 / 43.3 | 74.2 / 79.8 | 30.7 / 42.3 |

证据：

- step 200：[记录/汇总](../../runs/nora-dora-r32-lr1e6-syncfix1/late-checkpoint-evaluation/outputs/full_bench_eval/rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_syncfix1_step200_fullbench_32768_seed42/records/rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_syncfix1_step200_fullbench_32768_seed42.jsonl)；checkpoint：`rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_syncfix1_step200_fullbench_32768_seed42`。
- step 250：[记录/汇总](../../runs/nora-dora-r32-lr1e6-syncfix1/late-checkpoint-evaluation/outputs/full_bench_eval/rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_syncfix1_step250_fullbench_32768_seed42/records/rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_syncfix1_step250_fullbench_32768_seed42.jsonl)；checkpoint：`rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_syncfix1_step250_fullbench_32768_seed42`。
- step 270：[记录/汇总](../../runs/nora-dora-r32-lr1e6-syncfix1/step270-evaluation/outputs/full_bench_eval/rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_syncfix1_step270_fullbench_32768_seed42/records/rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_syncfix1_step270_fullbench_32768_seed42.jsonl)；checkpoint：`rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_syncfix1_step270_fullbench_32768_seed42`。

### NoRA+DoRA syncfix1 / budget extension

`rlvr_ext_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_from270_to540_v1_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 300 | 47.7 | 31.5 / 53.3 | 27.1 / 53.3 | 77.3 / 95.0 | 11.5 / 30.0 | 74.0 / 80.0 | 29.3 / 41.5 |

证据：

- step 300：[记录/汇总](../../runs/nora-dora-r32-lr1e6-syncfix1-extend540/step300-evaluation/outputs/full_bench_eval/rlvr_ext_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_from270_to540_v1_step300_fullbench_32768_seed42/records/rlvr_ext_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_from270_to540_v1_step300_fullbench_32768_seed42.jsonl)；checkpoint：`rlvr_ext_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_from270_to540_v1_step300_fullbench_32768_seed42`。

### NoRA+DoRA v1 (historical)

`rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 46.4 | 28.4 / 56.7 | 24.4 / 53.3 | 74.1 / 97.5 | 11.5 / 40.0 | 72.0 / 79.8 | 32.8 / 46.7 |
| 70 | 46.3 | 29.2 / 56.7 | 26.7 / 56.7 | 73.8 / 100.0 | 9.8 / 33.3 | 72.1 / 79.8 | 31.2 / 44.1 |
| 80 | 47.3 | 34.8 / 53.3 | 25.3 / 43.3 | 74.8 / 97.5 | 10.0 / 40.0 | 72.6 / 80.0 | 31.6 / 45.2 |
| 90 | 44.9 | 26.1 / 46.7 | 22.0 / 43.3 | 73.4 / 95.0 | 7.9 / 36.7 | 71.5 / 80.0 | 32.0 / 43.4 |
| 100 | 44.3 | 27.0 / 50.0 | 24.5 / 36.7 | 69.1 / 92.5 | 8.6 / 23.3 | 69.8 / 78.6 | 32.7 / 45.2 |

证据：

- step 50：[记录/汇总](../../runs/nora-dora-r32-lr1e6-v1/outputs/full_bench_eval/rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_step50_fullbench_32768_seed42/records/rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_step50_fullbench_32768_seed42.jsonl)；checkpoint：`rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_step50_fullbench_32768_seed42`。
- step 70：[记录/汇总](../../runs/nora-dora-r32-lr1e6-v1/outputs/full_bench_eval/rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_step70_fullbench_32768_seed42/records/rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_step70_fullbench_32768_seed42.jsonl)；checkpoint：`rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_step70_fullbench_32768_seed42`。
- step 80：[记录/汇总](../../runs/nora-dora-r32-lr1e6-v1/outputs/full_bench_eval/rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_step80_fullbench_32768_seed42/records/rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_step80_fullbench_32768_seed42.jsonl)；checkpoint：`rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_step80_fullbench_32768_seed42`。
- step 90：[记录/汇总](../../runs/nora-dora-r32-lr1e6-v1/outputs/full_bench_eval/rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_step90_fullbench_32768_seed42/records/rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_step90_fullbench_32768_seed42.jsonl)；checkpoint：`rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_step90_fullbench_32768_seed42`。
- step 100：[记录/汇总](../../runs/nora-dora-r32-lr1e6-v1/outputs/full_bench_eval/rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_step100_fullbench_32768_seed42/records/rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_step100_fullbench_32768_seed42.jsonl)；checkpoint：`rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_step100_fullbench_32768_seed42`。

### Other historical

`dominant_atoms_v1_trainableA_pbudget8_fullbench_32768_v1`

历史证据等级（step 50）：**D**；原允许用途：incompatible record-level result。重判不升级该等级。

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 43.2 | 30.2 / 70.0 | 22.9 / 53.3 | 68.7 / 95.0 | 10.1 / 40.0 | 70.6 / 79.2 | 21.7 / 33.1 |

证据：

- step 50：[记录/汇总](../../runs/outputs/full_bench_eval/dominant_atoms_v1_trainableA_pbudget8_step50_fullbench_32768_v1/records/dominant_atoms_v1_trainableA_pbudget8_step50_fullbench_32768_v1.jsonl)；checkpoint：`dominant_atoms_v1_trainableA_pbudget8_step50_fullbench_32768_v1`。

### Other historical

`full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 41.3 | 31.1 / 73.3 | 21.2 / 46.7 | 65.2 / 92.5 | 10.1 / 36.7 | 67.0 / 79.8 | 20.4 / 32.7 |

证据：

- step 50：[记录/汇总](../../runs/phase1-signal-random-v1/imported-runtime/outputs/full_bench_eval/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42/records/full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42.jsonl)；checkpoint：`full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42`。

### Phase-1 (historical protocol)

`phase1_i32_uniform_r32_b64m16n8_seed43_v1_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 25 | 38.8 | 29.7 / 60.0 | 18.4 / 56.7 | 60.0 / 95.0 | 9.8 / 40.0 | 63.4 / 77.0 | 19.9 / 35.3 |
| 50 | 38.9 | 23.5 / 66.7 | 19.9 / 50.0 | 61.2 / 97.5 | 10.2 / 36.7 | 65.2 / 78.2 | 19.9 / 32.7 |

证据：

- step 25：[记录/汇总](../../runs/phase1-signal-random-v1/outputs/full_bench_eval/phase1_i32_uniform_r32_b64m16n8_step25_seed43_v1_fullbench_32768_seed42/records/phase1_i32_uniform_r32_b64m16n8_step25_seed43_v1_fullbench_32768_seed42.jsonl)；checkpoint：`phase1_i32_uniform_r32_b64m16n8_step25_seed43_v1_fullbench_32768_seed42`。
- step 50：[记录/汇总](../../runs/phase1-signal-random-v1/outputs/full_bench_eval/phase1_i32_uniform_r32_b64m16n8_step50_seed43_v1_fullbench_32768_seed42/records/phase1_i32_uniform_r32_b64m16n8_step50_seed43_v1_fullbench_32768_seed42.jsonl)；checkpoint：`phase1_i32_uniform_r32_b64m16n8_step50_seed43_v1_fullbench_32768_seed42`。

### Phase-1 (historical protocol)

`phase1_i8_uniform_r32_b64m16n8_seed43_v4_fullbench_32768_seed42`

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 25 | 38.0 | 25.5 / 63.3 | 22.3 / 53.3 | 57.7 / 92.5 | 9.8 / 36.7 | 63.2 / 76.8 | 18.5 / 32.4 |
| 50 | 39.9 | 27.3 / 66.7 | 18.9 / 43.3 | 63.2 / 95.0 | 9.4 / 43.3 | 66.9 / 79.4 | 19.4 / 30.9 |

证据：

- step 25：[记录/汇总](../../runs/phase1-signal-random-v1/outputs/full_bench_eval/phase1_i8_uniform_r32_b64m16n8_step25_seed43_v4_fullbench_32768_seed42/records/phase1_i8_uniform_r32_b64m16n8_step25_seed43_v4_fullbench_32768_seed42.jsonl)；checkpoint：`phase1_i8_uniform_r32_b64m16n8_step25_seed43_v4_fullbench_32768_seed42`。
- step 50：[记录/汇总](../../runs/phase1-signal-random-v1/outputs/full_bench_eval/phase1_i8_uniform_r32_b64m16n8_step50_seed43_v4_fullbench_32768_seed42/records/phase1_i8_uniform_r32_b64m16n8_step50_seed43_v4_fullbench_32768_seed42.jsonl)；checkpoint：`phase1_i8_uniform_r32_b64m16n8_step50_seed43_v4_fullbench_32768_seed42`。

### Phase-1 (historical protocol)

`phase1_i0_uniform_r32_b64m16n8_seed43_v3_fullbench_32768_seed42`

历史证据等级（step 50）：**A**；原允许用途：frozen Phase-1 primary evidence。重判不升级该等级。

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 | 38.0 | 27.0 / 66.7 | 19.1 / 46.7 | 57.8 / 95.0 | 10.4 / 36.7 | 62.8 / 78.0 | 19.6 / 32.0 |

证据：

- step 50：[记录/汇总](../../runs/phase1-signal-random-v1/outputs/full_bench_eval/phase1_i0_uniform_r32_b64m16n8_step50_seed43_v3_fullbench_32768_seed42/records/phase1_i0_uniform_r32_b64m16n8_step50_seed43_v3_fullbench_32768_seed42.jsonl)；checkpoint：`phase1_i0_uniform_r32_b64m16n8_step50_seed43_v3_fullbench_32768_seed42`。

### RLPO (historical)

`rlpo_init_r32a64_b64m16n8_fullbench_32768_v1`

历史证据等级（step 270）：**D**；原允许用途：known-damaged aggregate。重判不升级该等级。

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 270 | 44.3 | 30.6 / 80.0 | 23.8 / 53.3 | 73.8 / 95.0 | 0.0 / 0.0 | 73.4 / 80.4 | 25.7 / 37.5 |

证据：

- step 270：[记录/汇总](../../runs/outputs/full_bench_eval/rlpo_init_r32a64_b64m16n8_step270_fullbench_32768_v1/records/rlpo_init_r32a64_b64m16n8_step270_fullbench_32768_v1.jsonl)；checkpoint：`rlpo_init_r32a64_b64m16n8_step270_fullbench_32768_v1`。

### SPO/POETBS (historical)

`paper_clean_spo_poetbs_b16d1_fullbench_32768`

历史证据等级（step 120）：**D**；原允许用途：incompatible record-level result。重判不升级该等级。

| Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 120 | 38.2 | 24.5 / 50.0 | 20.6 / 56.7 | 63.3 / 100.0 | 0.0 / 0.0 | 65.6 / 78.8 | 19.4 / 34.2 |

证据：

- step 120：[记录/汇总](../../../Tina_orthres_run/outputs/full_bench_eval/paper_clean_spo_poetbs_b16d1_step120_fullbench_32768/records/paper_clean_spo_poetbs_b16d1_step120_fullbench_32768.jsonl)；checkpoint：`paper_clean_spo_poetbs_b16d1_step120_fullbench_32768`。

## 非完整六任务的独立结果

以下 Avg. 只针对实际任务/子集计算，不能与六任务总体 Avg. 比较。

| Checkpoint | 回答数 | 当前 Avg. | 原保存 Avg. | 任务与 k |
| --- | ---: | ---: | ---: | --- |
| `smoke_paper_clean_biso_step200` | 1 | 100.0 | 100.0 | aime24@1(1题) |
| `gradtop_step100_270_chord_lambda0_screen48x4` | 192 | 41.1 | 37.0 | aime24@4(8题), aime25@4(8题), amc23@4(8题), hmmt_feb@4(8题), math500@4(8题), minerva@4(8题) |
| `gradtop_step100_270_chord_lambda0p5_screen48x4` | 192 | 47.4 | 43.8 | aime24@4(8题), aime25@4(8题), amc23@4(8题), hmmt_feb@4(8题), math500@4(8题), minerva@4(8题) |
| `gradtop_step100_270_chord_lambda1_screen48x4` | 192 | 48.4 | 44.8 | aime24@4(8题), aime25@4(8题), amc23@4(8题), hmmt_feb@4(8题), math500@4(8题), minerva@4(8题) |
| `gradtop_step100_270_chord_lambda1p25_screen48x4` | 192 | 55.7 | 52.1 | aime24@4(8题), aime25@4(8题), amc23@4(8题), hmmt_feb@4(8题), math500@4(8题), minerva@4(8题) |
| `rlpo_init_r32a64_b64m16n8_step270_hmmt_feb_32768_fixed_v1` | 960 | 16.6 | 12.0 | hmmt_feb@32(30题) |

## 新旧口径变化与来源

分数变化来自同一回答的判分规则变化，不是重新训练或重新生成带来的提升。`原保存 judge` 为 v2 的条目不冒充历史 v1；若存在经过哈希匹配的 v1 重放，另列其 Avg.。

| Checkpoint | 原保存 judge | 原保存 Avg. | v1重放 Avg. | PeRL Avg. | 错→对 | 对→错 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `base_fullbench_32768_rerun_records_20260819` | legacy_unversioned | 38.0 | — | 40.4 | 263 | 91 |
| `dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step100_fullbench_32768` | legacy_unversioned | 44.1 | — | 46.8 | 292 | 97 |
| `dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step180_fullbench_32768` | legacy_unversioned | 45.7 | — | 48.5 | 307 | 102 |
| `dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step200_fullbench_32768` | legacy_unversioned | 46.9 | — | 49.4 | 289 | 102 |
| `paper_clean_biso_r16match_270_step200_fullbench_32768` | legacy_unversioned | 44.0 | — | 46.7 | 290 | 94 |
| `smoke_paper_clean_biso_step200` | legacy_unversioned | 100.0 | — | 100.0 | 0 | 0 |
| `dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step220_fullbench_32768` | legacy_unversioned | 45.4 | — | 48.3 | 311 | 99 |
| `dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step240_fullbench_32768` | legacy_unversioned | 46.4 | — | 49.4 | 316 | 97 |
| `paper_clean_biso_r16match_270_step240_fullbench_32768` | legacy_unversioned | 45.5 | — | 48.3 | 294 | 97 |
| `paper_clean_biso_r16match_270_step260_fullbench_32768` | legacy_unversioned | 44.8 | — | 47.5 | 302 | 110 |
| `paper_clean_biso_r16match_270_step270_fullbench_32768` | legacy_unversioned | 45.6 | — | 48.5 | 304 | 96 |
| `dapo_boxed_stable_biso_b64_nmask_lr1e6_v1_step300_fullbench_32768` | legacy_unversioned | 46.1 | — | 49.3 | 323 | 93 |
| `chr_rlpo_v0_d1act95_r8to28_mean16p78_step200_fullbench_32768_v1` | legacy_unversioned | 43.8 | — | 46.5 | 292 | 102 |
| `chr_rlpo_v0_d1act95_r8to28_mean16p78_step270_fullbench_32768_v1` | legacy_unversioned | 45.2 | — | 48.1 | 304 | 92 |
| `rlvr_v1_dora_r32_a64_lr1em5_b64m16n8_seed42_syncfix1_step80_fullbench_32768_seed42` | legacy_unversioned | 45.9 | — | 48.6 | 315 | 114 |
| `rlvr_v1_dora_r32_a64_lr1em5_b64m16n8_seed42_syncfix1_step90_fullbench_32768_seed42` | legacy_unversioned | 45.6 | — | 48.7 | 325 | 103 |
| `rlvr_v1_dora_r32_a64_lr1em5_b64m16n8_seed42_syncfix1_step100_fullbench_32768_seed42` | legacy_unversioned | 45.8 | — | 48.5 | 304 | 105 |
| `rlvr_v1_dora_r32_a64_lr1em5_b64m16n8_seed42_syncfix1_step130_fullbench_32768_seed42` | legacy_unversioned | 44.9 | — | 47.6 | 301 | 102 |
| `rlvr_v1_dora_r32_a64_lr1em5_b64m16n8_seed42_syncfix1_step160_fullbench_32768_seed42` | legacy_unversioned | 45.7 | — | 48.6 | 303 | 95 |
| `rlvr_v1_dora_r32_a64_lr1em6_b64m16n8_seed42_step50_fullbench_32768_seed42` | legacy_unversioned | 40.6 | — | 43.1 | 271 | 92 |
| `full_parameter_1p5b_lr1e6_b64m16n8_step50_seed42_v2_step50_fullbench_32768_seed42` | legacy_unversioned | 42.7 | — | 45.5 | 298 | 94 |
| `full_parameter_1p5b_lr1e6_b64m16n8_step50_seed42_v2_step100_fullbench_32768_seed42` | legacy_unversioned | 45.3 | — | 48.0 | 288 | 88 |
| `full_parameter_1p5b_lr1e6_b64m16n8_step50_seed42_v2_step150_fullbench_32768_seed42` | legacy_unversioned | 46.7 | — | 49.2 | 292 | 106 |
| `full_parameter_1p5b_lr1e6_b64m16n8_step50_seed42_v2_step200_fullbench_32768_seed42` | legacy_unversioned | 46.3 | — | 49.1 | 298 | 101 |
| `paper_clean_dapo_heldout637_seed42_geora_b64m16n8_lr5e-6_wd0_step100_fullbench_32768` | legacy_unversioned | 42.8 | — | 45.4 | 286 | 95 |
| `paper_clean_dapo_heldout637_seed42_geora_b64m16n8_lr5e-6_wd0_step160_fullbench_32768` | legacy_unversioned | 41.6 | — | 43.8 | 256 | 93 |
| `paper_clean_dapo_heldout637_seed42_geora_r16a16_b64m16n8_lr2e-6_wd1e-2_clean270_20260830_rerun_step160_fullbench_32768` | legacy_unversioned | 44.1 | — | 46.1 | 262 | 114 |
| `paper_clean_dapo_heldout637_seed42_geora_b64m16n8_lr5e-6_wd0_step180_fullbench_32768` | legacy_unversioned | 41.0 | — | 42.9 | 246 | 108 |
| `paper_clean_dapo_heldout637_seed42_geora_b64m16n8_lr5e-6_wd0_step200_fullbench_32768` | legacy_unversioned | 39.7 | — | 41.9 | 242 | 82 |
| `gradtop_probe12_r8to32_mean31p63_step50_fullbench_32768_v1` | legacy_unversioned | 45.1 | — | 47.7 | 294 | 102 |
| `gradtop_probe12_r8to32_mean31p63_step100_fullbench_32768_v1` | legacy_unversioned | 46.2 | — | 48.9 | 298 | 103 |
| `gradtop_probe12_r8to32_mean31p63_step150_fullbench_32768_v1` | legacy_unversioned | 46.5 | — | 49.5 | 310 | 91 |
| `gradtop_probe12_r8to32_mean31p63_step200_fullbench_32768_v1` | legacy_unversioned | 46.2 | — | 48.7 | 293 | 110 |
| `gradtop_probe12_r8to32_mean31p63_step250_fullbench_32768_v1` | legacy_unversioned | 46.0 | — | 48.7 | 302 | 105 |
| `gradtop_probe12_r8to32_mean31p63_step270_fullbench_32768_v1` | legacy_unversioned | 46.1 | — | 48.7 | 301 | 106 |
| `gradtop_step100_act90_compact_r8to28_mean18p29_step150_fullbench_32768_v1` | legacy_unversioned | 46.0 | — | 48.7 | 302 | 104 |
| `gradtop_step100_270_chord_lambda0_screen48x4` | legacy_unversioned | 37.0 | — | 41.1 | 8 | 0 |
| `gradtop_step100_270_chord_lambda0p5_screen48x4` | legacy_unversioned | 43.8 | — | 47.4 | 7 | 0 |
| `gradtop_step100_270_chord_lambda1_screen48x4` | legacy_unversioned | 44.8 | — | 48.4 | 7 | 0 |
| `gradtop_step100_270_chord_lambda1p25_fullbench_32768_v1` | legacy_unversioned | 46.4 | — | 49.4 | 322 | 99 |
| `gradtop_step100_270_chord_lambda1p25_screen48x4` | legacy_unversioned | 52.1 | — | 55.7 | 8 | 1 |
| `gradtop_s2_op8_r32_p4_step50_fullbench_32768_seed42` | legacy_unversioned | 46.3 | — | 49.1 | 301 | 102 |
| `gradtop_s2_op8_r32_p4_step100_fullbench_32768_seed42` | legacy_unversioned | 48.2 | — | 51.1 | 303 | 95 |
| `gradtop_s2_op8_r32_p4_step150_fullbench_32768_seed42` | fullbench_math_v2 | 53.9 | 48.3 | 51.3 | 2 | 190 |
| `gradtop_s2_op8_r32_p4_step200_fullbench_32768_seed42` | fullbench_math_v2 | 55.3 | 49.2 | 52.2 | 3 | 225 |
| `gradtop_s3_w16_r32_p4_step50_fullbench_32768_seed42` | legacy_unversioned | 45.8 | — | 48.7 | 302 | 94 |
| `gradtop_s3_w16_r32_p4_step100_fullbench_32768_seed42` | legacy_unversioned | 46.6 | — | 49.3 | 305 | 112 |
| `gradtop_s3_w16_r32_p4_step150_fullbench_32768_seed42` | legacy_unversioned | 47.9 | — | 50.6 | 300 | 105 |
| `gradtop_s3_w16_r32_p4_step200_fullbench_32768_seed42` | legacy_unversioned | 47.9 | — | 50.7 | 310 | 109 |
| `dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step100_fullbench_32768` | legacy_unversioned | 44.9 | — | 47.5 | 304 | 114 |
| `dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step200_fullbench_32768` | legacy_unversioned | 46.2 | — | 49.1 | 314 | 108 |
| `dapo_boxed_stable_lora_b64_v1_step200_fullbench_32768` | legacy_unversioned | 43.0 | — | 45.5 | 272 | 90 |
| `dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step220_fullbench_32768` | legacy_unversioned | 47.0 | — | 49.8 | 307 | 106 |
| `dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step240_fullbench_32768` | legacy_unversioned | 46.0 | — | 49.0 | 314 | 94 |
| `dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step260_fullbench_32768` | legacy_unversioned | 46.6 | — | 49.6 | 316 | 97 |
| `paper_clean_dapo_heldout637_seed42_lora_r16a32_b64m16n8_lr1e-6_wd0_step270_fullbench_32768` | legacy_unversioned | 43.9 | — | 46.6 | 289 | 95 |
| `dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step280_fullbench_32768` | legacy_unversioned | 46.6 | — | 49.4 | 301 | 100 |
| `dapo_boxed_stable_lora_b64_lr5e6_actual_v1_step300_fullbench_32768` | legacy_unversioned | 46.9 | — | 50.0 | 324 | 99 |
| `matched_standard_lora_r32a64_b64m16n8_step25_seed43_v2_fullbench_32768_seed42` | legacy_unversioned | 36.9 | — | 38.8 | 242 | 108 |
| `matched_standard_lora_r32a64_b64m16n8_step50_seed42_v2_fullbench_32768_seed42` | legacy_unversioned | 38.5 | — | 40.5 | 241 | 99 |
| `matched_standard_lora_r32a64_b64m16n8_step50_seed43_v2_fullbench_32768_seed42` | legacy_unversioned | 39.0 | — | 41.1 | 254 | 103 |
| `rlvr_v1_nora_init_r32_a64_lr1em5_b64m16n8_seed42_syncfix2_refv2_step50_fullbench_32768_seed42` | legacy_unversioned | 47.1 | — | 50.1 | 313 | 95 |
| `rlvr_v1_nora_init_r32_a64_lr1em5_b64m16n8_seed42_syncfix2_refv2_step100_fullbench_32768_seed42` | legacy_unversioned | 45.7 | — | 47.7 | 260 | 110 |
| `nora_init_r32a64_lr1e5_b64m16n8_step50_seed42_v1_fullbench_32768_seed42` | legacy_unversioned | 46.4 | — | 49.5 | 322 | 98 |
| `nora_init_r32a64_lr1e5_b64m16n8_step100_seed42_v1_fullbench_32768_seed42` | legacy_unversioned | 45.4 | — | 48.1 | 299 | 105 |
| `nora_doft_r8b32_mag_init_lr1e6_b64m16n8_50_4gpu_boxed_step50_fullbench_32768_seed42` | legacy_unversioned | 38.3 | — | 40.6 | 267 | 99 |
| `rlvr_v1_nora_dora_continuous_r32_a64_lr1em6_b64m16n8_seed42_step50_fullbench_32768_seed42` | legacy_unversioned | 45.7 | — | 48.4 | 314 | 118 |
| `rlvr_v1_nora_dora_continuous_r32_a64_lr1em6_b64m16n8_seed42_step60_fullbench_32768_seed42` | legacy_unversioned | 45.7 | — | 48.2 | 286 | 102 |
| `rlvr_v1_nora_dora_continuous_r32_a64_lr1em6_b64m16n8_seed42_step70_fullbench_32768_seed42` | legacy_unversioned | 46.8 | — | 49.7 | 306 | 98 |
| `rlvr_v1_nora_dora_continuous_r32_a64_lr1em6_b64m16n8_seed42_step80_fullbench_32768_seed42` | legacy_unversioned | 42.6 | — | 45.0 | 272 | 98 |
| `rlvr_v1_nora_dora_continuous_r32_a64_lr1em6_b64m16n8_seed42_step100_fullbench_32768_seed42` | legacy_unversioned | 41.9 | — | 44.0 | 258 | 102 |
| `rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_syncfix1_step200_fullbench_32768_seed42` | legacy_unversioned | 47.2 | — | 50.1 | 318 | 106 |
| `rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_syncfix1_step250_fullbench_32768_seed42` | legacy_unversioned | 47.4 | — | 50.2 | 300 | 100 |
| `rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_syncfix1_step270_fullbench_32768_seed42` | legacy_unversioned | 48.6 | — | 51.5 | 317 | 104 |
| `rlvr_ext_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_from270_to540_v1_step300_fullbench_32768_seed42` | legacy_unversioned | 47.7 | — | 50.6 | 305 | 98 |
| `rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_step50_fullbench_32768_seed42` | legacy_unversioned | 46.4 | — | 49.3 | 313 | 105 |
| `rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_step70_fullbench_32768_seed42` | legacy_unversioned | 46.3 | — | 49.0 | 296 | 103 |
| `rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_step80_fullbench_32768_seed42` | legacy_unversioned | 47.3 | — | 50.1 | 298 | 95 |
| `rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_step90_fullbench_32768_seed42` | legacy_unversioned | 44.9 | — | 47.6 | 299 | 103 |
| `rlvr_v1_nora_dora_r32_a64_lr1em6_b64m16n8_seed42_step100_fullbench_32768_seed42` | legacy_unversioned | 44.3 | — | 47.2 | 294 | 88 |
| `dominant_atoms_v1_trainableA_pbudget8_step50_fullbench_32768_v1` | legacy_unversioned | 43.2 | — | 46.2 | 289 | 78 |
| `full_gradient_mean_gain_lcb_uniform_r8_b64m16n8_step50_fullbench_32768_seed42` | legacy_unversioned | 41.3 | — | 44.0 | 281 | 87 |
| `phase1_i32_uniform_r32_b64m16n8_step25_seed43_v1_fullbench_32768_seed42` | legacy_unversioned | 38.8 | — | 41.2 | 268 | 94 |
| `phase1_i8_uniform_r32_b64m16n8_step25_seed43_v4_fullbench_32768_seed42` | legacy_unversioned | 38.0 | — | 40.1 | 243 | 92 |
| `phase1_i0_uniform_r32_b64m16n8_step50_seed43_v3_fullbench_32768_seed42` | legacy_unversioned | 38.0 | — | 40.1 | 259 | 103 |
| `phase1_i32_uniform_r32_b64m16n8_step50_seed43_v1_fullbench_32768_seed42` | legacy_unversioned | 38.9 | — | 41.0 | 258 | 106 |
| `phase1_i8_uniform_r32_b64m16n8_step50_seed43_v4_fullbench_32768_seed42` | legacy_unversioned | 39.9 | — | 42.0 | 263 | 107 |
| `rlpo_init_r32a64_b64m16n8_step270_fullbench_32768_v1` | legacy_unversioned | 44.3 | — | 46.3 | 245 | 102 |
| `rlpo_init_r32a64_b64m16n8_step270_hmmt_feb_32768_fixed_v1` | legacy_unversioned | 12.0 | — | 16.6 | 45 | 1 |
| `paper_clean_spo_poetbs_b16d1_step120_fullbench_32768` | legacy_unversioned | 38.2 | — | 39.9 | 220 | 98 |

## 无法按当前口径重报告的历史汇总

下列 checkpoint 有原汇总而没有通过本次核验的原始回答。保留旧报告及其数字，当前口径记为“无法复评”，不由旧分数推算新分数。部分条目属于其他机器迁入的归档。

| Checkpoint | 原回答数 | 当前口径 | 原汇总位置 |
| --- | ---: | --- | --- |
| `phase1_i0_uniform_r32_b64m16n8_step50_seed42_v3_fullbench_32768_seed42` | 7248 | 无法复评 | `peft-for-rl/runs/phase1-signal-random-v1/outputs/full_bench_eval/phase1_i0_uniform_r32_b64m16n8_step50_seed42_v3_fullbench_32768_seed42/summary/phase1_i0_uniform_r32_b64m16n8_step50_seed42_v3_fullbench_32768_seed42.json` |
| `phase1_i16_uniform_r32_b64m16n8_step50_seed42_v3_fullbench_32768_seed42` | 7248 | 无法复评 | `peft-for-rl/runs/phase1-signal-random-v1/outputs/full_bench_eval/phase1_i16_uniform_r32_b64m16n8_step50_seed42_v3_fullbench_32768_seed42/summary/phase1_i16_uniform_r32_b64m16n8_step50_seed42_v3_fullbench_32768_seed42.json` |
| `phase1_i32_uniform_r32_b64m16n8_step50_seed42_v3_fullbench_32768_seed42` | 7248 | 无法复评 | `peft-for-rl/runs/phase1-signal-random-v1/outputs/full_bench_eval/phase1_i32_uniform_r32_b64m16n8_step50_seed42_v3_fullbench_32768_seed42/summary/phase1_i32_uniform_r32_b64m16n8_step50_seed42_v3_fullbench_32768_seed42.json` |
| `phase1_i8_uniform_r32_b64m16n8_step50_seed42_v3_fullbench_32768_seed42` | 7248 | 无法复评 | `peft-for-rl/runs/phase1-signal-random-v1/outputs/full_bench_eval/phase1_i8_uniform_r32_b64m16n8_step50_seed42_v3_fullbench_32768_seed42/summary/phase1_i8_uniform_r32_b64m16n8_step50_seed42_v3_fullbench_32768_seed42.json` |
| `gradtop_original_random_b_r8to32_step50_fullbench_32768_seed42_repro_v1` | 7248 | 无法复评 | `peft-for-rl/runs/random-b-original-repro-v1/outputs/full_bench_eval/gradtop_original_random_b_r8to32_step50_fullbench_32768_seed42_repro_v1/summary/gradtop_original_random_b_r8to32_step50_fullbench_32768_seed42_repro_v1.json` |
| `base_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/base_fullbench_32768/summary/base_fullbench_32768.json` |
| `base_fullbench_8192` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/base_fullbench_8192/summary/base_fullbench_8192.json` |
| `dapo_boxed_stable_biso_b32a16_gate_lr05_v1_step100_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b32a16_gate_lr05_v1_step100_fullbench_32768/summary/dapo_boxed_stable_biso_b32a16_gate_lr05_v1_step100_fullbench_32768.json` |
| `dapo_boxed_stable_biso_b32a16_gate_lr05_v1_step160_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b32a16_gate_lr05_v1_step160_fullbench_32768/summary/dapo_boxed_stable_biso_b32a16_gate_lr05_v1_step160_fullbench_32768.json` |
| `dapo_boxed_stable_biso_b32a16_gate_lr05_v1_step180_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b32a16_gate_lr05_v1_step180_fullbench_32768/summary/dapo_boxed_stable_biso_b32a16_gate_lr05_v1_step180_fullbench_32768.json` |
| `dapo_boxed_stable_biso_b32a16_gate_lr05_v1_step200_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b32a16_gate_lr05_v1_step200_fullbench_32768/summary/dapo_boxed_stable_biso_b32a16_gate_lr05_v1_step200_fullbench_32768.json` |
| `dapo_boxed_stable_biso_b32a16_gate_lr05_v1_step220_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b32a16_gate_lr05_v1_step220_fullbench_32768/summary/dapo_boxed_stable_biso_b32a16_gate_lr05_v1_step220_fullbench_32768.json` |
| `dapo_boxed_stable_biso_b32a16_gate_lr05_v1_step240_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b32a16_gate_lr05_v1_step240_fullbench_32768/summary/dapo_boxed_stable_biso_b32a16_gate_lr05_v1_step240_fullbench_32768.json` |
| `dapo_boxed_stable_biso_b32a16_gate_lr05_v1_step260_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b32a16_gate_lr05_v1_step260_fullbench_32768/summary/dapo_boxed_stable_biso_b32a16_gate_lr05_v1_step260_fullbench_32768.json` |
| `dapo_boxed_stable_biso_b32a16_gate_lr05_v1_step280_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b32a16_gate_lr05_v1_step280_fullbench_32768/summary/dapo_boxed_stable_biso_b32a16_gate_lr05_v1_step280_fullbench_32768.json` |
| `dapo_boxed_stable_biso_b32a16_gate_lr05_v1_step300_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b32a16_gate_lr05_v1_step300_fullbench_32768/summary/dapo_boxed_stable_biso_b32a16_gate_lr05_v1_step300_fullbench_32768.json` |
| `dapo_boxed_stable_biso_b32a16_rawl2_3e6_fix1_v1_step100_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b32a16_rawl2_3e6_fix1_v1_step100_fullbench_32768/summary/dapo_boxed_stable_biso_b32a16_rawl2_3e6_fix1_v1_step100_fullbench_32768.json` |
| `dapo_boxed_stable_biso_b32a16_rawl2_3e6_fix1_v1_step160_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b32a16_rawl2_3e6_fix1_v1_step160_fullbench_32768/summary/dapo_boxed_stable_biso_b32a16_rawl2_3e6_fix1_v1_step160_fullbench_32768.json` |
| `dapo_boxed_stable_biso_b32a16_rawl2_3e6_fix1_v1_step180_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b32a16_rawl2_3e6_fix1_v1_step180_fullbench_32768/summary/dapo_boxed_stable_biso_b32a16_rawl2_3e6_fix1_v1_step180_fullbench_32768.json` |
| `dapo_boxed_stable_biso_b32a16_rawl2_3e6_fix1_v1_step200_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b32a16_rawl2_3e6_fix1_v1_step200_fullbench_32768/summary/dapo_boxed_stable_biso_b32a16_rawl2_3e6_fix1_v1_step200_fullbench_32768.json` |
| `dapo_boxed_stable_biso_b32a16_rawl2_3e6_fix1_v1_step220_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b32a16_rawl2_3e6_fix1_v1_step220_fullbench_32768/summary/dapo_boxed_stable_biso_b32a16_rawl2_3e6_fix1_v1_step220_fullbench_32768.json` |
| `dapo_boxed_stable_biso_b32a16_rawl2_3e6_fix1_v1_step240_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b32a16_rawl2_3e6_fix1_v1_step240_fullbench_32768/summary/dapo_boxed_stable_biso_b32a16_rawl2_3e6_fix1_v1_step240_fullbench_32768.json` |
| `dapo_boxed_stable_biso_b32a16_rawl2_3e6_fix1_v1_step260_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b32a16_rawl2_3e6_fix1_v1_step260_fullbench_32768/summary/dapo_boxed_stable_biso_b32a16_rawl2_3e6_fix1_v1_step260_fullbench_32768.json` |
| `dapo_boxed_stable_biso_b32a16_rawl2_3e6_fix1_v1_step280_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b32a16_rawl2_3e6_fix1_v1_step280_fullbench_32768/summary/dapo_boxed_stable_biso_b32a16_rawl2_3e6_fix1_v1_step280_fullbench_32768.json` |
| `dapo_boxed_stable_biso_b32a16_rawl2_3e6_fix1_v1_step300_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b32a16_rawl2_3e6_fix1_v1_step300_fullbench_32768/summary/dapo_boxed_stable_biso_b32a16_rawl2_3e6_fix1_v1_step300_fullbench_32768.json` |
| `dapo_boxed_stable_biso_b32a16_v1_step100_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b32a16_v1_step100_fullbench_32768/summary/dapo_boxed_stable_biso_b32a16_v1_step100_fullbench_32768.json` |
| `dapo_boxed_stable_biso_b32a16_v1_step120_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b32a16_v1_step120_fullbench_32768/summary/dapo_boxed_stable_biso_b32a16_v1_step120_fullbench_32768.json` |
| `dapo_boxed_stable_biso_b32a16_v1_step140_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b32a16_v1_step140_fullbench_32768/summary/dapo_boxed_stable_biso_b32a16_v1_step140_fullbench_32768.json` |
| `dapo_boxed_stable_biso_b32a16_v1_step160_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b32a16_v1_step160_fullbench_32768/summary/dapo_boxed_stable_biso_b32a16_v1_step160_fullbench_32768.json` |
| `dapo_boxed_stable_biso_b32a16_v1_step180_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b32a16_v1_step180_fullbench_32768/summary/dapo_boxed_stable_biso_b32a16_v1_step180_fullbench_32768.json` |
| `dapo_boxed_stable_biso_b32a16_v1_step200_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b32a16_v1_step200_fullbench_32768/summary/dapo_boxed_stable_biso_b32a16_v1_step200_fullbench_32768.json` |
| `dapo_boxed_stable_biso_b32a16_v1_step220_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b32a16_v1_step220_fullbench_32768/summary/dapo_boxed_stable_biso_b32a16_v1_step220_fullbench_32768.json` |
| `dapo_boxed_stable_biso_b32a16_v1_step240_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b32a16_v1_step240_fullbench_32768/summary/dapo_boxed_stable_biso_b32a16_v1_step240_fullbench_32768.json` |
| `dapo_boxed_stable_biso_b32a16_v1_step260_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b32a16_v1_step260_fullbench_32768/summary/dapo_boxed_stable_biso_b32a16_v1_step260_fullbench_32768.json` |
| `dapo_boxed_stable_biso_b32a16_v1_step280_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b32a16_v1_step280_fullbench_32768/summary/dapo_boxed_stable_biso_b32a16_v1_step280_fullbench_32768.json` |
| `dapo_boxed_stable_biso_b32a16_v1_step300_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_b32a16_v1_step300_fullbench_32768/summary/dapo_boxed_stable_biso_b32a16_v1_step300_fullbench_32768.json` |
| `dapo_boxed_stable_biso_geora_mask0.3_v1_step100_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_geora_mask0.3_v1_step100_fullbench_32768/summary/dapo_boxed_stable_biso_geora_mask0.3_v1_step100_fullbench_32768.json` |
| `dapo_boxed_stable_biso_geora_mask0.3_v1_step300_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_geora_mask0.3_v1_step300_fullbench_32768/summary/dapo_boxed_stable_biso_geora_mask0.3_v1_step300_fullbench_32768.json` |
| `dapo_boxed_stable_biso_geora_mask0.3_v1_step400_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_geora_mask0.3_v1_step400_fullbench_32768/summary/dapo_boxed_stable_biso_geora_mask0.3_v1_step400_fullbench_32768.json` |
| `dapo_boxed_stable_biso_geora_mask0.3_v1_step500_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_geora_mask0.3_v1_step500_fullbench_32768/summary/dapo_boxed_stable_biso_geora_mask0.3_v1_step500_fullbench_32768.json` |
| `dapo_boxed_stable_biso_geora_mask0.3_v1_step600_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_biso_geora_mask0.3_v1_step600_fullbench_32768/summary/dapo_boxed_stable_biso_geora_mask0.3_v1_step600_fullbench_32768.json` |
| `dapo_boxed_stable_boet_r16a32_v1_step100_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_boet_r16a32_v1_step100_fullbench_32768/summary/dapo_boxed_stable_boet_r16a32_v1_step100_fullbench_32768.json` |
| `dapo_boxed_stable_boet_r16a32_v1_step200_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_boet_r16a32_v1_step200_fullbench_32768/summary/dapo_boxed_stable_boet_r16a32_v1_step200_fullbench_32768.json` |
| `dapo_boxed_stable_boet_r16a32_v1_step300_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_boet_r16a32_v1_step300_fullbench_32768/summary/dapo_boxed_stable_boet_r16a32_v1_step300_fullbench_32768.json` |
| `dapo_boxed_stable_geora_r16a32rho02_v1_step100_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_geora_r16a32rho02_v1_step100_fullbench_32768/summary/dapo_boxed_stable_geora_r16a32rho02_v1_step100_fullbench_32768.json` |
| `dapo_boxed_stable_geora_r16a32rho02_v1_step200_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_geora_r16a32rho02_v1_step200_fullbench_32768/summary/dapo_boxed_stable_geora_r16a32rho02_v1_step200_fullbench_32768.json` |
| `dapo_boxed_stable_geora_r16a32rho02_v1_step300_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_geora_r16a32rho02_v1_step300_fullbench_32768/summary/dapo_boxed_stable_geora_r16a32rho02_v1_step300_fullbench_32768.json` |
| `dapo_boxed_stable_geora_r16a32rho02_v1_step400_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_geora_r16a32rho02_v1_step400_fullbench_32768/summary/dapo_boxed_stable_geora_r16a32rho02_v1_step400_fullbench_32768.json` |
| `dapo_boxed_stable_geora_r16a32rho02_v1_step500_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_geora_r16a32rho02_v1_step500_fullbench_32768/summary/dapo_boxed_stable_geora_r16a32rho02_v1_step500_fullbench_32768.json` |
| `dapo_boxed_stable_lora_b64_v1_step100_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_lora_b64_v1_step100_fullbench_32768/summary/dapo_boxed_stable_lora_b64_v1_step100_fullbench_32768.json` |
| `dapo_boxed_stable_lora_lr5e6_v1_step300_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_lora_lr5e6_v1_step300_fullbench_32768/summary/dapo_boxed_stable_lora_lr5e6_v1_step300_fullbench_32768.json` |
| `dapo_boxed_stable_oft_v1_step100_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_oft_v1_step100_fullbench_32768/summary/dapo_boxed_stable_oft_v1_step100_fullbench_32768.json` |
| `dapo_boxed_stable_oft_v1_step200_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_oft_v1_step200_fullbench_32768/summary/dapo_boxed_stable_oft_v1_step200_fullbench_32768.json` |
| `dapo_boxed_stable_oft_v1_step300_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_oft_v1_step300_fullbench_32768/summary/dapo_boxed_stable_oft_v1_step300_fullbench_32768.json` |
| `dapo_boxed_stable_oft_v1_step400_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_oft_v1_step400_fullbench_32768/summary/dapo_boxed_stable_oft_v1_step400_fullbench_32768.json` |
| `dapo_boxed_stable_oft_v1_step500_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_oft_v1_step500_fullbench_32768/summary/dapo_boxed_stable_oft_v1_step500_fullbench_32768.json` |
| `dapo_boxed_stable_oft_v1_step600_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_oft_v1_step600_fullbench_32768/summary/dapo_boxed_stable_oft_v1_step600_fullbench_32768.json` |
| `dapo_boxed_stable_skew_r32a64_v1_step100_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_skew_r32a64_v1_step100_fullbench_32768/summary/dapo_boxed_stable_skew_r32a64_v1_step100_fullbench_32768.json` |
| `dapo_boxed_stable_skew_r32a64_v1_step200_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_skew_r32a64_v1_step200_fullbench_32768/summary/dapo_boxed_stable_skew_r32a64_v1_step200_fullbench_32768.json` |
| `dapo_boxed_stable_skew_r32a64_v1_step300_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_skew_r32a64_v1_step300_fullbench_32768/summary/dapo_boxed_stable_skew_r32a64_v1_step300_fullbench_32768.json` |
| `dapo_boxed_stable_v1_step100_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_v1_step100_fullbench_32768/summary/dapo_boxed_stable_v1_step100_fullbench_32768.json` |
| `dapo_boxed_stable_v1_step200_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_v1_step200_fullbench_32768/summary/dapo_boxed_stable_v1_step200_fullbench_32768.json` |
| `dapo_boxed_stable_v1_step300_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_v1_step300_fullbench_32768/summary/dapo_boxed_stable_v1_step300_fullbench_32768.json` |
| `dapo_boxed_stable_v1_step400_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_v1_step400_fullbench_32768/summary/dapo_boxed_stable_v1_step400_fullbench_32768.json` |
| `dapo_boxed_stable_v1_step500_base_spectrum_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_v1_step500_base_spectrum_fullbench_32768/summary/dapo_boxed_stable_v1_step500_base_spectrum_fullbench_32768.json` |
| `dapo_boxed_stable_v1_step500_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_v1_step500_fullbench_32768/summary/dapo_boxed_stable_v1_step500_fullbench_32768.json` |
| `dapo_boxed_stable_v1_step600_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_v1_step600_fullbench_32768/summary/dapo_boxed_stable_v1_step600_fullbench_32768.json` |
| `dapo_boxed_stable_v1_step700_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_v1_step700_fullbench_32768/summary/dapo_boxed_stable_v1_step700_fullbench_32768.json` |
| `dapo_boxed_stable_v1_step800_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_boxed_stable_v1_step800_fullbench_32768/summary/dapo_boxed_stable_v1_step800_fullbench_32768.json` |
| `dapo_math17k_lora_step100_fullbench_8192` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_math17k_lora_step100_fullbench_8192/summary/dapo_math17k_lora_step100_fullbench_8192.json` |
| `dapo_math17k_lora_step200_fullbench_8192` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_math17k_lora_step200_fullbench_8192/summary/dapo_math17k_lora_step200_fullbench_8192.json` |
| `dapo_perl_lora_b8_step100_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_perl_lora_b8_step100_fullbench_32768/summary/dapo_perl_lora_b8_step100_fullbench_32768.json` |
| `dapo_perl_lora_b8_step200_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_perl_lora_b8_step200_fullbench_32768/summary/dapo_perl_lora_b8_step200_fullbench_32768.json` |
| `dapo_perl_lora_b8_step300_fullbench_32768` | 7248 | 无法复评 | `Tina_orthres_run/outputs/full_bench_eval/dapo_perl_lora_b8_step300_fullbench_32768/summary/dapo_perl_lora_b8_step300_fullbench_32768.json` |

## 排除与待完成记录

| 来源 | 状态 | 原因 |
| --- | --- | --- |
| `Tina_orthres_run/outputs/heldout_biso_b64_nmask_rollout4_20260818/step100/records/biso_b64_nmask_lr1e6_step100_heldout_rollout4.jsonl` | excluded | 训练 heldout 验证 schema 缺少 benchmark 字段；保留原验证结果，不转换为六任务评测 |
| `Tina_orthres_run/outputs/heldout_biso_b64_nmask_rollout4_20260818/step120/records/biso_b64_nmask_lr1e6_step120_heldout_rollout4.jsonl` | excluded | 训练 heldout 验证 schema 缺少 benchmark 字段；保留原验证结果，不转换为六任务评测 |
| `Tina_orthres_run/outputs/heldout_biso_b64_nmask_rollout4_20260818/step140/records/biso_b64_nmask_lr1e6_step140_heldout_rollout4.jsonl` | excluded | 训练 heldout 验证 schema 缺少 benchmark 字段；保留原验证结果，不转换为六任务评测 |
| `Tina_orthres_run/outputs/heldout_biso_b64_nmask_rollout4_20260818/step160/records/biso_b64_nmask_lr1e6_step160_heldout_rollout4.jsonl` | excluded | 训练 heldout 验证 schema 缺少 benchmark 字段；保留原验证结果，不转换为六任务评测 |
| `Tina_orthres_run/outputs/heldout_biso_b64_nmask_rollout4_20260818/step180/records/biso_b64_nmask_lr1e6_step180_heldout_rollout4.jsonl` | excluded | 训练 heldout 验证 schema 缺少 benchmark 字段；保留原验证结果，不转换为六任务评测 |
| `Tina_orthres_run/outputs/heldout_biso_b64_nmask_rollout4_20260818/step200/records/biso_b64_nmask_lr1e6_step200_heldout_rollout4.jsonl` | excluded | 训练 heldout 验证 schema 缺少 benchmark 字段；保留原验证结果，不转换为六任务评测 |
| `Tina_orthres_run/outputs/heldout_biso_b64_nmask_rollout4_20260818/step220/records/biso_b64_nmask_lr1e6_step220_heldout_rollout4.jsonl` | excluded | 训练 heldout 验证 schema 缺少 benchmark 字段；保留原验证结果，不转换为六任务评测 |
| `Tina_orthres_run/outputs/heldout_biso_b64_nmask_rollout4_20260818/step240/records/biso_b64_nmask_lr1e6_step240_heldout_rollout4.jsonl` | excluded | 训练 heldout 验证 schema 缺少 benchmark 字段；保留原验证结果，不转换为六任务评测 |
| `Tina_orthres_run/outputs/heldout_biso_b64_nmask_rollout4_20260818/step260/records/biso_b64_nmask_lr1e6_step260_heldout_rollout4.jsonl` | excluded | 训练 heldout 验证 schema 缺少 benchmark 字段；保留原验证结果，不转换为六任务评测 |
| `Tina_orthres_run/outputs/heldout_biso_b64_nmask_rollout4_20260818/step280/records/biso_b64_nmask_lr1e6_step280_heldout_rollout4.jsonl` | excluded | 训练 heldout 验证 schema 缺少 benchmark 字段；保留原验证结果，不转换为六任务评测 |

复现命令、源码版本及协议边界见双协议说明。机器可读清单、逐任务精确值、原文件 SHA256 和题目/prompt 指纹保存在同目录结果 CSV 与运行目录 inventory.json。
