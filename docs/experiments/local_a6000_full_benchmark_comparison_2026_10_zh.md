# 本机历史与 8 x RTX A6000 后续完整评测结果对照

更新时间：2026-10-09（UTC）

本文承接 [原 4 x L40S 完整评测结果对照](local_l40s_full_benchmark_comparison_2026_10_zh.md)，但范围不再局限于迁移后新跑的结果。前半部分补齐当前本机历史目录中从 AdaLoRA、RLPO-init、D1/CHR-RLPO、Stable-SNR、Dominant-atoms 到 GradTop 及其裁剪/外推分支的实验；后半部分继续记录从 L40S 迁移到 `dmac-105` 后完成的 Phase-1、Step-25 和 Qwen3-4B 实验。当前机器为 8 x NVIDIA RTX A6000，driver `590.48.01`。

完整六任务评测统一使用 AIME24、AIME25、AMC23、HMMT Feb、MATH500、Minerva，共 902 道题、7,248 次生成；temperature 0.6、top-p 0.95、max new tokens 32768、eval seed 42。`Macro Avg@k` 和 `Pass@k` 先在任务内汇总，再对六任务等权平均；`Parse`、`Hit max` 和 `Mean len` 按全部生成汇总。除 `Mean len` 外均为 0--1 比例。

本文按实验问题分表，不构造跨协议总排行榜。不同 step、初始化、rank 预算、training seed、设备或模型的数字只在各自逻辑组内解释；同一 checkpoint 可以重复出现在不同对照中。表中只对每个逻辑组内最高的可比 `Macro Avg@k` 加粗。AdaLoRA 的损坏/缺失任务和 RLPO-init 的修复拼接会显式标注，不把不完整数据包装成原生 7,248-record 结果。

第 1--5 节历史主线除特别说明外，使用 DeepSeek-R1-Distill-Qwen-1.5B、196 个 `all-linear` 模块、LoRA dropout 0.05、learning rate `1e-6`、weight decay 0、constant scheduler、无 warmup、prompt batch 64、PPO mini-batch 16、每 prompt 8 个 rollout、PPO epoch 1、最大 prompt/response 长度 1024/8192、data/dataloader seed 42、计划训练 270 步并每 50 步保存。异构 rank 分支令每模块 `alpha_m=2r_m`，保持 LoRA scaling 为 2。第 6 节之后的迁移实验采用各节明示的冻结 Phase-1 协议，例如 dropout 为 0，不能和历史主线当作单变量对照。

## 1. AdaLoRA：工程诊断结果，不作为正式算法结论

唯一做过大规模评测的是早期 `init_r=32 -> target_r=8` 的 step-150 checkpoint。该训练同时存在四类已知问题：rank schedule 按 trainer step 配置却按每个 mini-batch optimizer step 执行，实际快 4 倍；官方正交 loss 没有进入 PPO loss；actor 与 vLLM 曾存在 `E/ranknum` 缩放不等价；评测的 HMMT 输入为空且 Minerva 少 36 条。后续 official-fixed 版本修复了接入链路，但曾使用 `orth_reg_weight=0.5`，正交项约 `2.6`、远大于常见 `0.01--0.15` 的 PG loss，最终 `1e-3` 配置没有形成一轮完整训练和评测。

因此下表只比较五个可用任务，综合 Parse、Hit max 和 Mean len 也只聚合这五项的 6,252 条有效记录。LoRA step-200 仅用于定位异常，不是同 step、同训练链路的严格 control。

| 方法 | Step | Macro Avg@k | Pass@k | Parse | Hit max | Mean len | AIME24 | AIME25 | AMC23 | HMMT | MATH500 | Minerva |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 早期 AdaLoRA r32->r8 | 150 | `0.371959 (5-task)` | `0.578239 (5-task)` | `0.778791` | `0.038548` | `8341.0` | `0.267708` | `0.175000` | `0.588281` | `invalid` | `0.631500` | `0.197304 (1052/1088)` |
| 历史 plain LoRA r32 参考 | 200 | **`0.425349 (5-task)`** | `0.650051 (5-task)` | `0.870547` | `0.033079` | `8155.8` | `0.255208` | `0.243750` | `0.709375` | `n/a in 5-task compare` | `0.701500` | `0.216912` |

AdaLoRA step-150 的五任务 Macro 低 `0.053390`，五个任务的 Parse 都比该 LoRA 参考低；但训练步数和实现状态不匹配，不能把差异归因于 AdaLoRA 算法本身。评测时 196 个模块的有效秩均值为 `9.878`、范围 `0--25`，其中 3 个模块为零秩，说明非均匀裁剪确实发生过，但没有形成可信的六任务效果闭环。该结果证据等级为 D。

## 2. RLPO-init 与 D1/CHR-RLPO：固定主子空间到静态异构 rank

`RLPO-init` 对每个 base linear weight 做 SVD，以 top-32 右奇异向量初始化 A，B 为零；A/B 均可训练，不加正交 loss，也不做在线裁剪。`D1` 随后在训练完成的 RLPO adapter 上按 activation-weighted output energy 反推各模块需要的 prefix rank，得到 `8--28`、均值 `16.78` 的 hindsight rank map。`CHR-RLPO` 使用这张 map 从 base 重新训练，`alpha_m=2r_m`，参数/FLOP 约为 r32 的 `50.09%`。

| 方法 | Step | Macro Avg@k | Pass@k | Parse | Hit max | Mean len | AIME24 | AIME25 | AMC23 | HMMT | MATH500 | Minerva |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RLPO-init r32，HMMT 修复拼接 | 270 | **`0.398649`** | `0.643722` | `0.927428` | `0.018764` | `7582.5` | `0.306250` | `0.237500` | `0.737500` | `0.119792` | `0.733500` | `0.257353` |
| CHR-RLPO D1，mean r16.78 | 200 | `0.380842` | `0.593773` | `0.875828` | `0.028008` | `8372.6` | `0.293750` | `0.240625` | `0.714063` | `0.108333` | `0.698500` | `0.229779` |
| CHR-RLPO D1，mean r16.78 | 270 | `0.393234` | `0.593655` | `0.901490` | `0.021799` | `7918.9` | `0.320833` | `0.253125` | `0.717969` | `0.106250` | `0.719500` | `0.241728` |

RLPO-init 原 full-bench 的 HMMT schema 损坏，表中用独立修复补评的 960 条 HMMT 替换原坏数据，再按相同口径重算六任务指标；它是可信的 corrected/stitched 结果，但不是单目录原生 summary。原错误 Macro `0.378684` 不应继续引用。

CHR 用约一半 adapter 参数在 step-270 距 RLPO-init `-0.005415`，支持“静态异构低预算可以保留大部分性能”，但不能证明异构 map 优于 matched fixed-r16。原因是 D1 使用训练后信息，且 CHR 在 step106 后从无 optimizer state 的 step100 checkpoint 恢复，step200/270 不是严格无中断轨迹。

## 3. Stable-SNR 与 Dominant-atoms：低预算探测分支

Stable-SNR v2/v3 试图通过多窗口信号/噪声分解稳定 RL 梯度，但独立 validation 中，v3 即使使用 rank-32 candidate span 也只捕获 `21.59%` 的未来 sketch energy。v2 没有启动正式训练；v3 只有 trainable-A 50-step pilot，没有 full-bench，因此不进入结果表。

Dominant-atoms 改为寻找多个局部窗口反复出现的 top-2 强方向，分配等价 uniform-r8 的参数预算，得到 mean rank `10.67`、range `4--16`。它与 GradTop step-50 使用相同六任务评测协议，可用于观察极低秩预算的代价。

| 方法 | Step | Macro Avg@k | Pass@k | Parse | Hit max | Mean len | AIME24 | AIME25 | AMC23 | HMMT | MATH500 | Minerva |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dominant-atoms，eq-r8 / mean r10.67 | 50 | `0.373654` | `0.617703` | `0.872930` | `0.032561` | `8689.8` | `0.302083` | `0.229167` | `0.686719` | `0.101042` | `0.706000` | `0.216912` |
| GradTop v1，mean r31.63 | 50 | **`0.391198`** | `0.617606` | `0.897351` | `0.032009` | `8563.1` | `0.315625` | `0.242708` | `0.735156` | `0.102083` | `0.720000` | `0.231618` |

Dominant-atoms 的训练 reward 很有竞争力，但 full-bench 低 `0.017545`，六个任务均低于 GradTop。事后用真实 step-50 `DeltaW` 检验，异构 map 只保留 `47.67%` 能量，低于 uniform-r8 的 `54.49%` 和 greedy oracle 的 `60.39%`。这证明低 rank 可以训练，却不证明当前 allocator 找对了跨模块预算分配。

## 4. GradTop v1：连续 checkpoint 轨迹

GradTop 用 random-B 对 full-weight gradient 做右侧 sketch，累计 12 个 observation 后构造初始化 span。其 mean rank 为 `31.63`，196 个模块中 192 个仍是 r32，所以它不是成功的 adaptive-rank allocator，而是宽 candidate-span 初始化实验。

| 方法 | Step | Macro Avg@k | Pass@k | Parse | Hit max | Mean len | AIME24 | AIME25 | AMC23 | HMMT | MATH500 | Minerva |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GradTop v1，mean r31.63 | 50 | `0.391198` | `0.617606` | `0.897351` | `0.032009` | `8563.1` | `0.315625` | `0.242708` | `0.735156` | `0.102083` | `0.720000` | `0.231618` |
| GradTop v1，mean r31.63 | 100 | `0.403326` | `0.637562` | `0.928808` | `0.021385` | `7698.5` | `0.310417` | `0.255208` | `0.748438` | `0.108333` | `0.725500` | `0.272059` |
| GradTop v1，mean r31.63 | 150 | **`0.404574`** | `0.627172` | `0.947296` | `0.014901` | `6902.8` | `0.329167` | `0.238542` | `0.767969` | `0.090625` | `0.730000` | `0.271140` |
| GradTop v1，mean r31.63 | 200 | `0.401638` | `0.627518` | `0.945226` | `0.008002` | `6168.7` | `0.314583` | `0.239583` | `0.745313` | `0.091667` | `0.731000` | `0.287684` |
| GradTop v1，mean r31.63 | 250 | `0.399401` | `0.630904` | `0.943985` | `0.007864` | `5934.2` | `0.286458` | `0.231250` | `0.754688` | `0.112500` | `0.727500` | `0.284007` |
| GradTop v1，mean r31.63 | 270 | `0.399411` | `0.648570` | `0.947848` | `0.009244` | `5860.4` | `0.276042` | `0.241667` | `0.753906` | `0.106250` | `0.730000` | `0.288603` |

Macro 在 step100--150 达峰，随后轻微回落；但 step150--270 的 Pass、Parse、hit-max 和输出长度继续改善。训练 reward 后期继续上升，主要反映行为与格式变化，而不是 held-out accuracy 持续增长。step50 到 270 的 `DeltaW` top-8 right-subspace overlap 为 `89.14%`，但相邻非重叠更新 chord cosine 只有约 `0.04--0.08`：累积输入子空间稳定，不等于逐段梯度沿同一直线。

## 5. GradTop 后处理：物理裁剪与线性 chord

### 5.1 Step-100 裁剪后续训到 Step-150

Compact 从 GradTop step100 出发，用 activation-weighted 90% energy 阈值物理裁剪到 mean rank `18.29`、range `8--28`，保留原 GradTop `60.77%` 的 adapter 参数，再继续训练到 step150。

| 方法 | Step | Macro Avg@k | Pass@k | Parse | Hit max | Mean len | AIME24 | AIME25 | AMC23 | HMMT | MATH500 | Minerva |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GradTop full，mean r31.63 | 150 | **`0.404574`** | `0.627172` | `0.947296` | `0.014901` | `6902.8` | `0.329167` | `0.238542` | `0.767969` | `0.090625` | `0.730000` | `0.271140` |
| GradTop Compact，mean r18.29 | 150 | `0.397868` | `0.630222` | `0.942743` | `0.015315` | `7057.0` | `0.296875` | `0.236458` | `0.747656` | `0.100000` | `0.736000` | `0.270221` |

Compact 的 Macro 低 `0.006705`；paired 95% CI 跨 0。结果支持“可删约 39% adapter 参数并保持接近性能”，不支持裁剪会提高性能。

### 5.2 Step-100 到 Step-270 的 post-hoc chord

Chord 构造 `DeltaW = DeltaW100 + 1.25 * (DeltaW270 - DeltaW100)`，不是普通 step100 checkpoint。它用于检测累计权重轨迹能否外推。

| 方法 | Step | Macro Avg@k | Pass@k | Parse | Hit max | Mean len | AIME24 | AIME25 | AMC23 | HMMT | MATH500 | Minerva |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GradTop 原 checkpoint | 270 | `0.399411` | `0.648570` | `0.947848` | `0.009244` | `5860.4` | `0.276042` | `0.241667` | `0.753906` | `0.106250` | `0.730000` | `0.288603` |
| GradTop chord，lambda=1.25 | post-hoc 100->270 | **`0.403345`** | `0.646010` | `0.955850` | `0.006209` | `5483.5` | `0.317708` | `0.237500` | `0.757813` | `0.106250` | `0.731500` | `0.269301` |

Chord 的 Macro 点估计高 `0.003934`，但 95% CI 跨 0；更稳定的变化是 Parse 增加约 `0.0080`、hit-max 降低约 `0.0030`、平均输出缩短 `376.9` token。因此它是行为控制结果，不是准确率提升证据。

## 6. DeepSeek 1.5B：Seed43 的 Step-25 早期轨迹

这组比较同一模型、training seed43、rank32、alpha64、dropout0、all-linear 和相同 25-step 预算下的三种初始化。Standard LoRA 使用 PEFT 默认的 A Kaiming、B zero；I8 使用 8 个 P2 signal atom 加 24 个随机正交补方向；I32 使用 32 个 signal atom。三组 B 都是 zero-init，初始模型函数不变。

| 方法 | Step | Macro Avg@k | Pass@k | Parse | Hit max | Mean len | AIME24 | AIME25 | AMC23 | HMMT | MATH500 | Minerva |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Matched standard LoRA r32，seed43 | 25 | `0.313818` | `0.573763` | `0.794288` | `0.037942` | `9206.9` | `0.202083` | `0.172917` | `0.565625` | `0.092708` | `0.634500` | `0.215074` |
| I8：8 signal + 24 random，seed43 | 25 | `0.328308` | `0.591644` | `0.803118` | `0.031871` | `8838.9` | `0.255208` | `0.222917` | `0.576563` | `0.097917` | `0.632500` | `0.184743` |
| I32：32 signal + 0 random，seed43 | 25 | **`0.335436`** | `0.606601` | `0.808637` | `0.035458` | `8943.0` | `0.296875` | `0.184375` | `0.600000` | `0.097917` | `0.634000` | `0.199449` |

在 Step-25，I8 和 I32 相对 standard LoRA 分别高 `+0.014490` 和 `+0.021618`。I32 的早期 Macro 最高，但优势并不覆盖所有任务：standard LoRA 在 Minerva 最好，I8 在 AIME25 最好，I32 的主要优势来自 AIME24 和 AMC23。

I8-standard 的 problem-paired bootstrap delta 为 `+0.014490`，95% CI `[-0.000852,+0.031424]`，仍处于边界状态。这个结果支持早期 trajectory bias，不足以证明稳定的整体能力提升。

## 7. DeepSeek 1.5B：Seed43 的 Step-50 主对照

这组保持与 Step-25 相同的方法、training seed 和训练协议，只比较训练到 50 steps 后的结果。

| 方法 | Step | Macro Avg@k | Pass@k | Parse | Hit max | Mean len | AIME24 | AIME25 | AMC23 | HMMT | MATH500 | Minerva |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Matched standard LoRA r32，seed43 | 50 | `0.338761` | `0.613206` | `0.815259` | `0.034630` | `8940.4` | `0.242708` | `0.248958` | `0.599219` | `0.101042` | `0.632000` | `0.208640` |
| I8：8 signal + 24 random，seed43 | 50 | **`0.341696`** | `0.597693` | `0.836369` | `0.031733` | `8471.1` | `0.272917` | `0.188542` | `0.632031` | `0.093750` | `0.669000` | `0.193934` |
| I32：32 signal + 0 random，seed43 | 50 | `0.333201` | `0.602923` | `0.824779` | `0.026904` | `8353.1` | `0.235417` | `0.198958` | `0.611719` | `0.102083` | `0.652500` | `0.198529` |

I8 相对 standard LoRA 只剩 `+0.002934`，paired 95% CI `[-0.010343,+0.016290]`；I32 则低 `-0.005560`。Step-50 排序为 `I8 > standard LoRA > I32`，与 Step-25 的 `I32 > I8 > standard LoRA` 不同。

Macro 也不能替代其他行为指标。Standard LoRA 的 Pass@k 最高，并在 AIME25、Minerva 上最好；I8 的 Parse、AIME24、AMC23 和 MATH500 最好；I32 输出最短、hit-max 最低，并在 HMMT 上略高。初始化方法改变的不只是平均准确率，也改变了任务侧重和生成行为。

## 8. DeepSeek 1.5B：Step-25 到 Step-50 的方法内轨迹

这组不做不同方法的静态排名，而是检查每种初始化在后 25 steps 中发生了什么。上面两组完整结果在这里按方法重新配对。

### 8.1 Matched standard LoRA

| 方法 | Step | Macro Avg@k | Pass@k | Parse | Hit max | Mean len | AIME24 | AIME25 | AMC23 | HMMT | MATH500 | Minerva |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Matched standard LoRA r32，seed43 | 25 | `0.313818` | `0.573763` | `0.794288` | `0.037942` | `9206.9` | `0.202083` | `0.172917` | `0.565625` | `0.092708` | `0.634500` | `0.215074` |
| Matched standard LoRA r32，seed43 | 50 | **`0.338761`** | `0.613206` | `0.815259` | `0.034630` | `8940.4` | `0.242708` | `0.248958` | `0.599219` | `0.101042` | `0.632000` | `0.208640` |

Standard LoRA 从 Step-25 到 50 提高 `+0.024943`，paired 95% CI `[+0.008855,+0.042168]`，是三种方法中后半程提升最清楚的一组。Parse 增加 `+0.020971`，平均长度减少约 266 token。

### 8.2 I8 signal + random

| 方法 | Step | Macro Avg@k | Pass@k | Parse | Hit max | Mean len | AIME24 | AIME25 | AMC23 | HMMT | MATH500 | Minerva |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| I8：8 signal + 24 random，seed43 | 25 | `0.328308` | `0.591644` | `0.803118` | `0.031871` | `8838.9` | `0.255208` | `0.222917` | `0.576563` | `0.097917` | `0.632500` | `0.184743` |
| I8：8 signal + 24 random，seed43 | 50 | **`0.341696`** | `0.597693` | `0.836369` | `0.031733` | `8471.1` | `0.272917` | `0.188542` | `0.632031` | `0.093750` | `0.669000` | `0.193934` |

I8 后半程提高 `+0.013388`，但 paired 95% CI `[-0.001059,+0.027844]` 跨 0。Parse 增加 `+0.033251`，平均长度减少约 368 token；AIME25 和 HMMT 下降，而 AMC23、MATH500 上升。

### 8.3 I32 signal-only

| 方法 | Step | Macro Avg@k | Pass@k | Parse | Hit max | Mean len | AIME24 | AIME25 | AMC23 | HMMT | MATH500 | Minerva |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| I32：32 signal + 0 random，seed43 | 25 | **`0.335436`** | `0.606601` | `0.808637` | `0.035458` | `8943.0` | `0.296875` | `0.184375` | `0.600000` | `0.097917` | `0.634000` | `0.199449` |
| I32：32 signal + 0 random，seed43 | 50 | `0.333201` | `0.602923` | `0.824779` | `0.026904` | `8353.1` | `0.235417` | `0.198958` | `0.611719` | `0.102083` | `0.652500` | `0.198529` |

I32 从 Step-25 到 50 下降 `0.002235`。虽然 Parse 提高 `+0.016142`、平均长度减少约 590 token、hit-max 降低 `0.008554`，但 AIME24 下降 `0.061458`。这说明行为格式继续改善时，held-out Macro 仍可能停滞或回落。

相对 standard LoRA，I8 和 I32 的 Step-25 优势到 Step-50 分别收缩 `0.011556` 和 `0.027178`。当前 A6000 轨迹最直接支持的是“signal 初始化加快早期学习，普通 LoRA 后续追近”，而不是“signal 初始化提高 50-step 可达上限”。

## 9. Matched standard LoRA：本机 training seed42/43

这组只比较同一 standard-LoRA 方法的两个 training seed。两组使用完全相同的固定 LoRA 初始化，initial-state SHA-256 均为 `039db13cae337b17c4e233341afa4252fa46c645d16943493cf0b82e55f167ad`；training/data/dataloader/rollout seed 分别设为 42 和 43。

| 方法 | Step | Macro Avg@k | Pass@k | Parse | Hit max | Mean len | AIME24 | AIME25 | AMC23 | HMMT | MATH500 | Minerva |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Matched standard LoRA r32，seed42 | 50 | `0.330053` | `0.567031` | `0.801600` | `0.036010` | `8974.4` | `0.246875` | `0.197917` | `0.610156` | `0.092708` | `0.647000` | `0.185662` |
| Matched standard LoRA r32，seed43 | 50 | **`0.338761`** | `0.613206` | `0.815259` | `0.034630` | `8940.4` | `0.242708` | `0.248958` | `0.599219` | `0.101042` | `0.632000` | `0.208640` |

Seed43 的 Macro 高 `0.008708`，主要增量来自 AIME25 和 Minerva；AIME24、AMC23、MATH500 反而略低。两个 seed 只能显示当前两条 trajectory 的差异，不能估计完整 seed population。

Seed42 在 Step-42 计算期间因 worker 中断，从完整 Step-25 checkpoint 恢复并重算 Step 26--41。恢复包含 model、optimizer、extra state、RNG、scheduler 和 dataloader state，最终训练与评测均 verified；但该恢复边界必须随结果披露。Seed43 是 Step 0--50 无中断运行。

## 10. 与原 L40S Phase-1 的连续比较

这一节为对照 L40S 总结而保留，但不是纯方法消融。P2 basis、allocation、代码和评测 snapshot 均冻结；然而 I0 seed43 来自原 L40S，I8/I32 seed43 来自当前 A6000，因此方法与设备边界同时变化。

### 10.1 Seed43：L40S I0 与 A6000 后续结果

| 方法 | 设备 | Step | Macro Avg@k | Pass@k | Parse | Hit max | Mean len | AIME24 | AIME25 | AMC23 | HMMT | MATH500 | Minerva |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| I0：0 signal + 32 random，seed43 | L40S | 50 | `0.327747` | `0.591642` | `0.798565` | `0.037390` | `8951.1` | `0.269792` | `0.190625` | `0.578125` | `0.104167` | `0.628000` | `0.195772` |
| Matched standard LoRA r32，seed43 | A6000 | 50 | `0.338761` | `0.613206` | `0.815259` | `0.034630` | `8940.4` | `0.242708` | `0.248958` | `0.599219` | `0.101042` | `0.632000` | `0.208640` |
| I8：8 signal + 24 random，seed43 | A6000 | 50 | **`0.341696`** | `0.597693` | `0.836369` | `0.031733` | `8471.1` | `0.272917` | `0.188542` | `0.632031` | `0.093750` | `0.669000` | `0.193934` |
| I32：32 signal + 0 random，seed43 | A6000 | 50 | `0.333201` | `0.602923` | `0.824779` | `0.026904` | `8353.1` | `0.235417` | `0.198958` | `0.611719` | `0.102083` | `0.652500` | `0.198529` |

相对 L40S I0，I8/I32 的 Macro 分别为 `+0.013949/+0.005454`。I8-I0 的 paired 95% CI 为 `[+0.001000,+0.027188]`，但仅 3/6 benchmark 为正，没有满足预注册的至少 4/6 条件；I32-I0 的 CI `[-0.005884,+0.016747]` 跨 0。

冻结哈希降低了配置漂移风险，但不能把硬件、runtime 和方法效应完全分离。因此这张表适合描述项目连续结果，不适合声称 A6000 上 I8/I32 相对同机 I0 的严格增益。

### 10.2 Phase-1 signal 方法的 seed42/43

| 方法 | Training seed | 设备 | Step | Macro Avg@k | Pass@k | Parse | Hit max | Mean len | AIME24 | AIME25 | AMC23 | HMMT | MATH500 | Minerva |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| I8：8 signal + 24 random | 42 | L40S | 50 | `0.366161` | `0.605923` | `0.866584` | `0.032423` | `8639.7` | `0.278125` | `0.225000` | `0.699219` | `0.097917` | `0.694500` | `0.202206` |
| I8：8 signal + 24 random | 43 | A6000 | 50 | `0.341696` | `0.597693` | `0.836369` | `0.031733` | `8471.1` | `0.272917` | `0.188542` | `0.632031` | `0.093750` | `0.669000` | `0.193934` |
| I32：32 signal + 0 random | 42 | L40S | 50 | **`0.376479`** | `0.628601` | `0.872103` | `0.026214` | `8373.5` | `0.304167` | `0.235417` | `0.704688` | `0.105208` | `0.698000` | `0.211397` |
| I32：32 signal + 0 random | 43 | A6000 | 50 | `0.333201` | `0.602923` | `0.824779` | `0.026904` | `8353.1` | `0.235417` | `0.198958` | `0.611719` | `0.102083` | `0.652500` | `0.198529` |

I8 和 I32 从 seed42 到 43 的 Macro 分别变化 `-0.024465/-0.043278`，且 I8/I32 排序反转。由于 training seed 与机器迁移同时变化，最严格的表述是“signal-informed initialization 对 trajectory 条件敏感”；当前不能把全部差异归因于 seed，也不能把它解释成 A6000 性能退化。

## 11. Qwen3-4B：Base 与 standard LoRA

Qwen3-4B 是独立模型协议，不与 DeepSeek 1.5B 的绝对分数混排。Base 和 LoRA 使用相同六任务 snapshot 和生成设置；LoRA 训练使用 8 GPU、rank32、alpha64、dropout0、252 个 all-linear 模块、66,060,288 个可训练参数、training seed42 和 50 steps。

| 方法 | Step | Macro Avg@k | Pass@k | Parse | Hit max | Mean len | AIME24 | AIME25 | AMC23 | HMMT | MATH500 | Minerva |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B base，无 RL | 0 | `0.643853` | `0.773400` | `0.980546` | `0.025248` | `9481.3` | `0.690625` | `0.626042` | `0.943750` | `0.317708` | `0.819000` | `0.465993` |
| Qwen3-4B standard LoRA r32，seed42 | 50 | **`0.652255`** | `0.777676` | `0.981926` | `0.022075` | `9218.9` | `0.712500` | `0.656250` | `0.958594` | `0.307292` | `0.817500` | `0.461397` |

LoRA 相对 base 的 Macro delta 为 `+0.008402`，problem-paired bootstrap 95% CI `[+0.001514,+0.015556]`，`P(delta>0)=0.9920`。增益集中在 AIME24、AIME25 和 AMC23；HMMT、MATH500、Minerva 略降。Parse 提高 `0.001380`，平均长度减少约 262 token，hit-max 降低 `0.003173`。

该结果只说明当前机器上 Qwen3-4B 的 standard-LoRA 单 seed baseline 可运行且有小幅正向变化。当前没有 Qwen3-4B 的 P2 probe、I0 或 I8 完整评测，不能用该表支持 signal initialization 的跨模型泛化。

## 12. 未形成可比较 full-bench 的结果

- 原 L40S 上完成的 18 个结果仍由 [L40S 对照表](local_l40s_full_benchmark_comparison_2026_10_zh.md)单独维护；本文件只在第 10 节重复列出与迁移链直接相关的 I0/I8/I32。
- DeepSeek no-RL base `0.324348` 是复用并审计过的历史 records，不是当前 A6000 重新生成，因此没有加入后迁移主表。
- AdaLoRA step150 只有五个可用任务且 Minerva 不完整，已在第 1 节作为 D 级诊断单列，不能进入六任务排名。
- Stable-SNR v2 只有 probe；v3 只有 probe 和 50-step trainable-A pilot，没有 full-bench。Dominant uniform-r8 也只有训练日志，没有六任务结果。
- JustRL stable-rank oracle 只完成 rank map 和 dry-run，Ray 与正式训练没有启动。
- Qwen3-4B 的 4-GPU 和前两个 8-GPU 尝试在正式训练前或 vLLM wake-up 阶段终止，没有 checkpoint/full-bench，不是算法结果。
- Phase-0/0.5/0.6 只有 probe、capture 和 gate 诊断，没有训练后六任务评测。
- screen subset 不进入正式表；损坏或缺失记录只在能够明确修复时拼接，并必须像 RLPO-init 一样披露来源。

## 13. 跨实验主要判断

1. AdaLoRA 的唯一大评测来自已知有缺陷的早期接入，且任务不全；它只能说明 rank 分化发生过，不能判断标准 AdaLoRA 对 RL 是否有效。
2. RLPO-init 与 CHR 表明输入侧初始化 span 和静态低预算 rank 都能稳定训练。CHR 用约一半参数接近 r32，但缺少 matched fixed-r16，尚未证明异构分配本身带来优势。
3. Stable-SNR 说明逐窗口 RL 梯度难以稳定外推；Dominant-atoms 则说明极低 rank 有训练容量，但当前模块 allocator 的 held-out 排序失败。找 candidate span 和给模块分 rank 是两个问题。
4. GradTop 是最清楚的早期学习加速结果，但实际几乎仍是 r32。Macro 在 step100--150 达峰，后续主要改善 Parse、Pass、hit-max 与长度，训练 reward 不能替代 held-out accuracy。
5. Compact 支持在已有 `DeltaW` 上事后减少约 39% adapter 参数而只付出小幅点估计损失；chord 支持沿累计权重轨迹控制生成行为，但两者都不是训练前 adaptive-rank 已解决的证据。
6. 后迁移 seed43 实验同样显示 signal 初始化在 Step-25 的优势到 Step-50 收缩；standard LoRA 后 25 步追近，当前结论更接近“初始化改变学习速度”而非“提高最终上限”。
7. L40S seed42 到 A6000 seed43 的 signal 方法差异同时混入 seed 与设备边界，不能做单因素归因；Qwen3-4B 尚无 signal 初始化对照，跨模型有效性仍未闭环。

## 14. Artifact 路径

AdaLoRA 的不完整评测、复盘和实现问题记录：

```text
/home/wangls/Tina_orthres_run/outputs/full_bench_eval/
  adalora_r32_to_r8_step150_fullbench_32768/
runs/analysis/adalora_experiment_retrospective_20260914/
```

RLPO-init、CHR、Dominant 和 GradTop 系列的完整 summary：

```text
runs/outputs/full_bench_eval/
  rlpo_init_r32a64_b64m16n8_step270_fullbench_32768_v1/
  rlpo_init_r32a64_b64m16n8_step270_hmmt_feb_32768_fixed_v1/
  chr_rlpo_v0_d1act95_r8to28_mean16p78_step{200,270}_fullbench_32768_v1/
  dominant_atoms_v1_trainableA_pbudget8_step50_fullbench_32768_v1/
  gradtop_probe12_r8to32_mean31p63_step{50,100,150,200,250,270}_fullbench_32768_v1/
  gradtop_step100_act90_compact_r8to28_mean18p29_step150_fullbench_32768_v1/
  gradtop_step100_270_chord_lambda1p25_fullbench_32768_v1/
```

对应方法复盘、probe gate、子空间和反事实分析：

```text
runs/analysis/rlpo_oracle_gradient_probe_experiment_summary_20260921/
runs/analysis/rlpo_experiment_retrospective_20260914/
runs/analysis/stable_snr_v2_ckpt50_training_gate_20260918/
runs/analysis/stable_snr_v3_training_gate_20260918/
runs/analysis/dominant_atoms_v1_step50_fullbench_decision/
runs/analysis/gradtop_compact_step150_fullbench_comparison_20260918/
runs/analysis/gradtop_step100_270_chord_lambda1p25_fullbench_comparison_20260920/
```

DeepSeek 1.5B 当前机器的 summary、records 和 postverify 位于：

```text
runs/matched-standard-lora-r32-v2/outputs/full_bench_eval/
runs/matched-standard-lora-r32-v2/analysis/verification/
runs/phase1-signal-random-v1/outputs/full_bench_eval/
runs/phase1-signal-random-v1/analysis/verification/
runs/step25-trajectory-v1/analysis/verification/
```

Qwen3-4B 完整 summary 和 records 当前位于 NFS：

```text
/media/nfs_share/wangls/peft-for-rl-external-validity-v1/
  qwen3-4b-standard-lora-8gpu-safetensors-v3/outputs/full_bench_eval/
```

Qwen3-4B 的本地训练 manifest 和 paired comparison 镜像位于：

```text
runs/qwen3-4b-standard-lora-8gpu-safetensors-v3/analysis/
```
