# 本机 PEFT for RL 实验研究叙事

更新时间：2026-10-08（Asia/Shanghai）

本文不按目录或启动时间罗列实验，而是回答一条连续问题：在数学推理 GRPO 中，LoRA 的初始化子空间、有效 rank 和训练动态应如何设计，才能在有限参数和训练预算下更快学到可泛化更新。

实验主线为：

```text
固定 LoRA/OFT 摸底
  -> AdaLoRA：动态裁剪同时引入正交损失、调度和同步问题
  -> RLPO-init：先把持续正交约束缩减为一次性初始化
  -> D1/CHR：确认模块所需 rank 不同，但离线 oracle 不是可部署 allocator
  -> GradTop：用真实 RL 梯度寻找候选初始化 span
  -> Stable-SNR/Dominant：检验稳定方向和异构 rank map，发现 ordering/allocation 仍不可靠
  -> Full-gradient/covariance/token mask：拆分 candidate、打分与 token 信号
  -> Phase-0/1：冻结变量，比较 signal 与 random complement
  -> Step-25：确认主要收益是早期 trajectory bias，普通 LoRA 后期会追近
  -> Qwen3-4B：完成外部模型标准 LoRA 基线，尚未验证 signal init 可迁移
```

**机器分界。** 本文按研究问题而非启动日期组织，因此章节顺序不能直接当作机器时间线。阶段一至阶段六的主要结果来自迁移前的历史实验；当前 **4 x NVIDIA L40S** 机器上的新实验从 `SPAR v0` 开始：2026-09-20 完成环境与 1-step smoke，2026-09-21 完成 positive-probe 的 uniform-r8 / adaptive-eqr8 两组正式 50-step 训练。随后直到 Phase-1 seed42 全组和 seed43-I0，均在这台 L40S 主机上完成；从 seed43-I8 开始的补充确认、Step-25 和 Qwen3-4B 则迁移到后续机器，不能标成 L40S 结果。跨机器结果只在冻结代码、allocation、数据和评测 snapshot 后用于延续同一问题链。实际思路与设备分界为：

原 4 x L40S 上通过正式六任务完整评测的 18 个唯一结果，按可比逻辑拆分后的统一表格见：[原 4 x L40S 本机完整评测结果对照](local_l40s_full_benchmark_comparison_2026_10_zh.md)。

```text
SPAR：先用正确轨迹筛方向与分配 rank
  -> Full-gradient：改用真实完整 GRPO 梯度，并拆开 discovery/calibration/audit
  -> Windowed / r8 controls：检验稳定性、方向收益和极端低秩容量
  -> Token-mask covariance：比较均值、协方差与 token 选择
  -> Original random-B 复现：在 L40S 上重新建立历史 near-r32 参照
  -> Phase-0 + Phase-1 seed42 / seed43-I0 [原 4 x L40S]：冻结变量，比较 signal 与 random complement
  -> Phase-1 后续确认 + Step-25 [异地后续机器]：检查 training-seed 与收益时序
  -> Qwen3-4B [后续 8-GPU 机器]：检查跨模型边界
```

## 1. 证据口径

本机实验跨越多个代码版本、训练步数、dropout、benchmark snapshot 和训练 seed。为避免构造无统计意义的总排行榜，本文按以下等级使用结果。

| 等级 | 含义 | 用途 |
| --- | --- | --- |
| A | 冻结协议，records、contract、postverify 齐全 | 当前固定设置下的主要比较 |
| A-reference | 严格审计但不参与 paired training | 绝对位置参考 |
| B | 评测完整，但训练协议不完全匹配 | 历史量级和机制线索 |
| C | summary 完整，原始 records 或协议不足 | 描述性历史 |
| D | 已知实现缺陷、数据缺失或仅完成 screen | 工程诊断和失败记录 |

后文始终区分四个问题：

1. **Candidate span discovery**：是否找到包含有用更新方向的宽子空间。
2. **Atom ordering**：候选 span 内哪些方向应优先保留。
3. **Rank allocation**：固定总预算下，各模块应分配多少 rank。
4. **Training-time adaptation**：训练中是否需要裁剪、恢复或重分配。

GradTop 的主要正结果回答第 1 项；Stable-SNR、Dominant 和 SPAR 暴露第 2、3 项的问题；AdaLoRA 尝试第 4 项。不能用其中一项的正结果替代其余问题。

### 1.1 本文术语

对一个形状为 `[d_out, d_in]` 的线性层权重 `W`，LoRA 写成：

```text
W' = W + Delta W
Delta W = s B A
A: [r, d_in]
B: [d_out, r]
s = alpha / r
```

本文中的术语均基于这个表达式：

| 术语 | 本文含义 |
| --- | --- |
| Rank `r` | 单个线性模块允许的 LoRA 更新矩阵最大秩，不是整个网络只有 `r` 个方向 |
| Atom | A 的一行 `a_j` 及 B 的一列 `b_j` 组成的 rank-1 更新 `b_j a_j^T` |
| Candidate | probe 为一个模块生成、尚未决定全部保留的候选 A 方向；常见宽度为 32 |
| Span / 子空间 | 若干 A 行方向张成的输入侧子空间；“找到 span”不等于已经排好 atom 顺序 |
| Prefix / top-k | 按某个 score 排序后保留前 `k` 个 atom；它同时依赖 candidate 和 ordering |
| SVD | Singular Value Decomposition（奇异值分解）；右奇异向量对应线性层的输入侧方向 |
| Uniform-r8 | 所有模块都保留 8 个 atom |
| Adaptive-eqr8 | 各模块 rank 可以不同，但全模型 A/B 参数总量与 uniform-r8 相同；`eq` 指 equal parameter budget |
| Probe / 探测 | 正式训练前生成 rollout、反向计算梯度和统计子空间，但不执行 optimizer update |
| Observation | 一次 PPO mini-batch 完成全部 micro-batch backward 后，在 optimizer 边界捕获的一份模块梯度；不是一条 rollout |
| Window | 若干连续 observation 组成的统计单元；Stable-SNR/Dominant 先在窗口内汇总，再比较窗口间复现性 |
| Discovery | 用于生成 candidate 的 prompt；只能决定候选方向 |
| Calibration | 与 discovery 不相交，用于给候选 atom 打分、选择 rank 或超参数 |
| Audit | 再独立的一组 prompt，只用于报告选定方案的 held-out 质量，不参与选择 |
| Capture | 某子空间保留的目标能量比例；gradient capture 测梯度，DeltaW capture 测训练后权重更新，必须看清被测对象 |
| Gain-LCB | Gain Lower Confidence Bound；预测某方向在 held-out gradient 上的一阶改进，再减去统计标准误惩罚 |
| Reward AUC1:50 | 前 50 个 trainer steps 的训练 reward 算术平均，反映整段轨迹而不只看最后一点 |
| Macro Avg@k | 先在每道题的多个采样上求平均正确率，再对六个 benchmark 等权平均 |
| Pass@k | 一道题的 `k` 个采样中至少有一个答对的比例，反映采样覆盖度，不等同于平均正确率 |
| Parse / hit-max | 输出能否解析出答案 / 是否生成到长度上限；用于区分能力变化、格式变化和截断 |
| Effective rank | 根据奇异值分布得到的软谱宽度；与配置中的 nominal rank 不同 |
| Rank@95% energy | 从最大奇异值开始累计，达到总平方奇异值能量 95% 所需的最小分量数 |
| Paired CI | 在相同题目和 sample key 上比较两个方法，并按问题重采样得到的差值置信区间 |
| On-policy trajectory | 当前模型生成数据、再用这些数据更新自身所形成的训练路径；初始化会先改变采样，再反馈到后续梯度 |

`PEFT` 指 Parameter-Efficient Fine-Tuning。`GRPO` 指 Group Relative Policy Optimization：同一 prompt 的多条 response 在组内形成相对 advantage，再通过 PPO 风格的 clipped policy loss 更新模型。`OFT` 是 Orthogonal Finetuning；`AdaLoRA` 是训练中动态分配 rank 的 LoRA；`RLPO-init` 是本项目对“以基础权重右奇异子空间初始化 LoRA-A、B 置零”的称呼；`CHR-RLPO` 是使用静态异构 rank map 的 Calibrated Heterogeneous-Rank RLPO。BISO、BOET、GeoRA 和 Skew 是仓库中的其他结构化 adapter 方案，本文只把其历史分数用于量级参考，不从这些跨协议结果推断机制。

实现层术语中，`FSDP` 负责多卡参数/梯度分片，`vLLM` 负责 rollout 推理。二者必须使用同一份 adapter 权重和 scaling，否则采样 policy 与训练 policy 不一致。

## 2. 公共实验设置

### 2.1 1.5B 主训练协议

| 项目 | 设置 |
| --- | --- |
| Base model | DeepSeek-R1-Distill-Qwen-1.5B |
| 训练数据 | DAPO-Math-17k boxed |
| RL 算法 / reward | GRPO / boxed outcome accuracy |
| LoRA target | `all-linear`，196 个模块 |
| Global prompt batch | 64 |
| PPO mini-batch | 16 |
| 每 prompt rollout | 8 |
| PPO epoch | 1 |
| 最大 prompt / response | 1024 / 8192 |
| 学习率 / scheduler | `1e-6` / constant |
| Weight decay / warmup | 0 / 0 |
| KL reward / loss | 均关闭 |
| 常用 LoRA scaling | `alpha/r = 2` |
| 训练 shuffle / actor shuffle | True / False |

一个 trainer step 包含 64 个 prompt、每个 8 条 rollout，并通过 16-prompt mini-batch 产生 4 次 optimizer update。AdaLoRA 的调度必须按 optimizer update，而不是 trainer step 计数。

训练 loss 默认采用 `token-mean`：先把一个 PPO mini-batch 内所有 rollout 的有效 response-token loss 求和，再除以有效 token 总数。micro-batch 只用于显存切分，其 backward 梯度按比例累积，直到还原完整 mini-batch 梯度。因此长 response 通常贡献更多 token 项；这里不是“每条 rollout 等权算一个梯度分数”。

`boxed outcome accuracy` 只检查最终 `\boxed{...}` 答案是否正确，不给推理过程分。KL reward/loss 均关闭，表示训练目标没有额外加入“保持接近 base policy”的显式 KL 惩罚；因此初始化造成的早期采样差异更可能沿 on-policy trajectory 被后续训练放大。

所有 `grad_subspace` 正式训练都令 A/B 可训练。probe 只决定初始 A，B 初始为零，所以研究对象是初始化几何和早期路径依赖，不是永久冻结子空间。

### 2.2 正式六任务评测

| 项目 | 设置 |
| --- | --- |
| 任务 | AIME24、AIME25、AMC23、HMMT Feb、MATH500、Minerva |
| Problems / generations | 902 / 7248 |
| Temperature / top-p | 0.6 / 0.95 |
| Max new tokens | 32768 |
| Eval seed | 42 |
| 主指标 | 六任务等权 Macro Avg@k |

同时报告 Parse、Pass@k、hit-max 和平均长度。多轮实验表明，训练 reward、held-out accuracy、格式正确率和长度可能朝不同方向变化。

## 3. 阶段一：固定 PEFT 摸底

### 为什么先做

在讨论自适应 rank 前，先确认普通 LoRA、OFT 和已有结构化 adapter 在当前 GRPO 管线中能否学习，并确定随训练步数变化的大致量级。这一阶段用于建立背景，不作为后续 Phase-1 的严格基线。

### 重要结果

早期 stable LoRA 在 step 100/200/300 的 Macro 为 `0.331179/0.340859/0.346207`。另一条使用 dropout 0.05、训练更久且 records 完整的旧 LoRA 轨迹为 `0.389839/0.402729/0.409672`。

OFT 的 step 100/200/300/400/500/600 为：

```text
0.323712 / 0.337397 / 0.359758 / 0.378069 / 0.374849 / 0.366732
```

其他历史点包括 BISO step 200 `0.410365`、BOET step 200 `0.394733`、GeoRA 约 step 500 `0.385174`、Skew step 100 `0.386569`。

### 得到什么

- PEFT 在当前 RL 配置下可以稳定学习，性能不只由名义 rank 决定。
- 更长训练不保证单调提升，OFT 后期已经出现平台和回落。
- 这些 run 的 step、dropout、初始化和 snapshot 不统一，只能作为 B/C 级背景。

由此自然产生下一个问题：固定 rank 已经能学，动态 rank 能否更有效地分配预算？

## 4. 阶段二：AdaLoRA 为什么没有形成有效结果

### 为什么这样设置

AdaLoRA 是最直接的动态 rank 基线：为每个模块提供较宽的 SVD adapter，再按 importance 把全局预算从 `init_r=32` 压到 `target_r=8`。核心配置为 alpha 64、dropout 0.05、196 个模块、270 trainer steps，即 1080 次 optimizer update。

### 实现暴露的问题

1. 若把 270 trainer steps 当作 AdaLoRA total step，裁剪会快 4 倍。
2. verl 的 PPO loss 在模型外计算，PEFT 正交 loss 原本没有进入目标。
3. `orth_reg_weight=0.5` 时，加权正交项约 2.6，而 PG loss 常见绝对值仅 `0.01--0.15`，优化目标被正交性主导。
4. PEFT config 的继承字段 `r=8` 与真实宽度 `init_r=32` 不一致，若直接交给 vLLM，rollout scaling 会比 actor 大 4 倍。
5. `lora_E/ranknum` 不能直接同步给 vLLM，必须折入普通 LoRA 权重。
6. allocator 的 importance/uncertainty EMA、FSDP 全梯度和 checkpoint 状态都需显式处理。

### 结果与边界

唯一早期 step-150 merge 的五个有效任务 Macro 为 `0.3720`，但 HMMT 输入损坏、Minerva 少 36 条，训练 rollout 还经历过 scaling 错误，属于 D 级结果。有效 rank 均值 `9.878`，范围 `0--25`，3 个模块被裁空，只能用于诊断。

后续修复了调度、FSDP、vLLM 转换和正交 loss 接入，默认正交权重也降到 `1e-3`，但没有从 step 0 完成正式六任务闭环。

### 为什么转向 RLPO-init

AdaLoRA 同时混合了初始化、持续正交正则、动态裁剪和状态同步，既难归因，也容易让 SFT 正则压过 RL 信号。因此下一步先缩小问题：正交性是否只需用于初始化？

## 5. 阶段三：RLPO-init，只保留一次性正交初始化

### 设计动机与设置

RLPO-init 取消训练期正交 loss 和 allocator，只用 base weight 的主右奇异子空间初始化 LoRA-A：

```text
A0 = top-32 right singular vectors of W_base
B0 = 0
DeltaW0 = 0
```

rank 32、alpha 64，A/B 随后都训练。这样初始模型函数不变，也避免持续正交损失覆盖策略梯度。

### 结果

step 270 的修复后六任务 Macro 为 `0.39865`。旧 HMMT 数据曾损坏，因此该结果仍是历史描述，不是当前 A 级 matched run。

| 几何指标 | 结果 |
| --- | ---: |
| 最终 DeltaW 留在初始 top-32 right span 的能量 | `99.997%` |
| `B(A-A0)` 相对 `BA` | `0.709%` |
| DeltaW effective rank | `17.13` |
| rank@95% energy | `24.44` |
| 前 8 个 DeltaW 奇异分量能量 | `66.87%` |
| base top-4/8/16 prefix 对最终 DeltaW 的 capture | `13.53% / 26.03% / 50.83%` |

### 得到什么

完整初始化 span 几乎不旋转，功能更新主要由 B/left side 在该 span 内重组；最终更新也确实可压缩。但 base weight 的 singular-value prefix 不能正确排序 RL 更新原子。完整 top-32 span 有用，不代表 base top-8 就是最重要的 8 个方向。

因此下一步需要分别研究模块异构 rank 和更符合 RL 目标的方向排序。

## 6. 阶段四：D1 与 CHR，验证异构 rank 的存在

### 为什么这样设置

D1 要回答的是：训练完成后，不同模块实际需要多少个 `Delta W` 分量才能保持其在真实 hidden state 上的输出，而不是仅根据参数矩阵的奇异值判断 rank。

### D1 activation-weighted output energy 诊断流程

对每个模块，先把 RLPO step-270 adapter 的功能更新写成：

```text
Delta W = s B A = U diag(sigma) V^T
```

其中 `v_i` 是第 `i` 个输入侧右奇异方向。对进入该模块的 hidden state `h`，第 `i` 个分量造成的输出为 `sigma_i u_i(v_i^T h)`。由于 `u_i` 彼此正交，该分量在实际 activation 分布上的平均输出能量为：

```text
e_i = sigma_i^2 * E_h[(v_i^T h)^2]
```

具体过程如下：

1. **固定诊断对象。** 使用已经训练完成的 rank-32 RLPO adapter，不做 backward，也不更新参数。通过 A/B 的薄 QR 和一个 `32x32` core SVD 得到各模块的 `sigma` 与 `V`，无需物化完整 `Delta W`。
2. **选择固定 response。** 从正式六任务 records 中每题只取一条 response，共选 48 条、六个 benchmark 平衡；按确定性规则分成 24 条 calibration 和 24 条 validation。重放时只统计 response 区域，每条序列最长 4096 token，并均匀采样最多 256 个 token 位置。
3. **收集 activation 统计。** forward pre-hook 读取每个模块输入 `h`，立即投影为 `z=hV`，分别为 calibration/validation 累计小型 `32x32` 矩阵 `C=sum z^Tz`。hook 返回后不保留完整 hidden state；除每个模块的投影基外，新增 activation 统计只需 rank-space 的 `C`。
4. **计算分量能量。** 对每个 split 使用 `e_i=sigma_i^2 C_ii`。代码保存求和形式；若写成期望，只需统一除以采样 token 数，该因子在累计能量比例中抵消。普通 Frobenius 能量只看 `sigma_i^2`；D1 还乘上该方向在真实 response activation 中被使用的强度，所以同样大的奇异值可能得到不同的重要性。
5. **只在 calibration 选 rank。** 保持原始奇异值前缀顺序，取累计 `e_i` 达到 95% 的最小 `k`，再向上量化到 `8,12,16,20,24,28,32`。这一步为 196 个模块生成静态 rank map。
6. **在 validation 验证。** 不重新选 rank，直接测 calibration map 在 validation activation 上保留的能量，并与参数量最接近的 uniform-r16 比较。另行用 validation 独立计算一张 rank map，只用于报告两组模块 rank 排序的 Spearman；Spearman 越接近 1，说明所需 rank 的模块间排序对换一批 response 越稳定。

D1 还计算过把奇异分量按 `e_i` 重新排序的 component oracle，但它可以跳过原始前缀，只作为诊断上界。后续 CHR 使用的是可实现的 prefix rank map：它只迁移“模块 `m` 分配多少 rank”，从 base model 重新取 top-`r_m` 初始化方向，并没有把 step-270 `Delta W` 的奇异向量迁移回训练起点。

| 指标 | 结果 |
| --- | ---: |
| Mean rank / range | `16.78 / 8--28` |
| 相对 uniform-r32 参数/FLOP | `50.09%` |
| Calibration/validation rank Spearman | `0.9871` |
| D1 最差模块 retained energy | `94.62%` |
| Fixed-r16 最差模块 retained energy | `75.16%` |

这些数值说明 D1 map 的主要优势是保护 validation 上的低尾模块，而且模块 rank 排序在这次 split-half 中稳定；按模块能量加权的全局 retained energy 并未优于同参数量 fixed-r16。这里测量的也只是单个线性模块的 `Delta W h` 能量，不是端到端 logits、KL、reward 或 benchmark accuracy。

CHR-RLPO 使用这张静态 map 从 base 重训，各模块用 base top-`r_m` 方向初始化，并保持 `alpha_m/r_m=2`。

### 结果与限制

CHR 在 step 200/270 的 Macro 为 `0.380842/0.393234`。训练曾在 step 106 后从无 optimizer state 的 step-100 checkpoint 恢复，轨迹不严格连续。更重要的是，D1 使用训练完成后的 `Delta W` 反推各模块需要多少 rank，是 hindsight oracle，而不是训练开始前可用的预测器；同时缺少同初始化、同预算 fixed-r16 control。

D1 证明不同模块的可压缩程度不同，也说明约半数参数可能接近 r32 表现，但没有给出训练前可部署的 allocator。于是方向发现必须转向少量真实 RL rollout。

## 7. 阶段五：GradTop，用 RL 梯度寻找候选 span

### 为什么不继续使用 base top-r

RLPO 表明完整 top-32 span 稳定，但 base prefix ordering 与最终更新不匹配。GradTop 因此用小批 rollout 的 RL 梯度发现方向。对某个线性模块，记基础权重的策略梯度为 `G:[d_out,d_in]`，第一版 random-B probe 使用：

```text
A_probe = 0
B_probe = R,  R_ij ~ N(0, 1/r)
grad_A = s R^T G
S = grad_A^T / s = G^T R
```

`A_probe=0` 使 probe adapter 的 `Delta W=sBA` 始终为零，所以探测不会改变生成 policy；随机 `B_probe=R` 只充当读取完整梯度输入侧结构的投影矩阵。

### GradTop 的 rollout 与 observation 流程

1. 同一 prompt 的 8 条 rollout 先计算组相对 advantage，再进入 PPO clipped token loss。
2. `b16n8` probe 的一个 PPO mini-batch 含 16 个 prompt，即 128 条 rollout。所有有效 response token 按 `token-mean` 归约；多个 micro-batch 的 backward 累积到同一份 `A.grad`，多卡梯度再由 FSDP 汇总。
3. 全部 backward 完成、optimizer 边界到达后，collector 读取一次 `A.grad` 并构造 `S=G^T R`。这才算一个 observation；代码不保存 128 个逐 rollout GradTop score。
4. energy probe 随后刷新随机 `R`，对下一 observation 使用新的输出侧投影。probe 不执行参数更新，下一批 rollout 仍由同一个 policy 生成。
5. 对同一模块，不同 observation 的 `S_t` 沿列拼接并做容量为 64 的增量 SVD 压缩：

```text
S_pool = [S_1, S_2, ..., S_T]
S_pool S_pool^T = sum_t S_t S_t^T
```

因此需要区分两次聚合：同一 mini-batch 内，rollout 先通过 token-mean loss 形成一阶梯度，正负贡献可能抵消；不同 observation 之间累计的是 sketch 的未中心化二阶能量，方向相反不会像直接求和那样抵消。代码确实直接累加标量 `sum_t ||S_t||_F^2` 作为总能量分母，但不会计算 `S_1+...+S_T`。增量 SVD 会截断到容量 64，所以保留的是累计能量的主方向近似。

累计 12 个 observation 后，根据奇异值平方的累计能量选择 rank bin，把对应输入侧基写入正式 A；正式 B 清零，使正式训练 step 0 仍与 base policy 完全等价，随后 A/B 均继续训练。

### 重要结果

95% energy 规则几乎没有压缩：平均 rank `31.63`，196 个模块中 192 个为 r32。它发现的是宽 candidate span，不是有效 adaptive rank map。

| Step | Macro Avg@k |
| ---: | ---: |
| 50 | `0.391198` |
| 100 | `0.403326` |
| 150 | `0.404574` |
| 200 | `0.401638` |
| 250 | `0.399401` |
| 270 | `0.399411` |

GradTop 在 step 50 已达到较高水平，100--150 接近峰值，之后平台或轻微回落；训练 reward 后期仍会上升，但更多对应 parse、Pass@k 和长度变化。最终 DeltaW 几乎留在初始化 span 内，但按初始化 atom prefix 截断时顺序接近随机。

结论是：GradTop 找到了有用的宽 span，却没有解决 span 内 atom ordering，也没有给出可靠的模块 rank map。

## 8. 阶段六：Stable-SNR、Dominant 与压缩旁证

### Stable-SNR：稳定不等于有用

RL 梯度跨窗口重合低，Stable-SNR 尝试用多窗口、裁剪、均值/方差和独立 validation 选择稳定方向。

每个 observation 仍按上节生成一份 `S_t`。一个含 `W` 个 observation 的 window 先按 sketch Frobenius 范数的中位数阈值裁剪异常批次，再构造：

```text
H_w = sqrt(W) * mean_t(S_wt)                         # window signal
N_w = [S_w1-mean, ..., S_wW-mean] / sqrt(W-1)       # window 内 noise
```

对 `K` 个 discovery window 沿列拼接 `H=[H_1,...,H_K]`、`N=[N_1,...,N_K]`，寻找最大化下式的输入方向 `q`：

```text
J(q) = q^T H H^T q / q^T (N N^T + rho I) q
```

实现时先对 `H` 做薄 QR，只在至多 `K*probe_width` 维的 signal span 中求解广义特征问题 `A v=lambda B v`，再映射回原输入空间；`lambda` 越大，表示该方向的窗口均值能量相对窗口内波动越强。按 `lambda` 降序排列后再做 QR，得到行正交且保持有序前缀 span 的 LoRA-A candidate。calibration/留一窗口测各 rank 的 capture，并按单位新增参数的 capture 增益分配固定预算；validation window 只做独立报告。该方法排序的是“强且相对稳定”的方向，不是直接对窗口平均梯度排序。

- v2 使用 5 windows、15 observations，约束到 uniform-r16 预算；attention 全为 r32，MLP mean rank `11.05`，held-out capture `24.82%`，对最终 DeltaW retrospective capture 仅 `5.10%`。
- v3 使用 5 discovery、2 calibration、2 validation windows，mean rank 16；即使 rank32，validation capture 也只有 `21.59%`，选中 map 为 `11.75%`。

fixed-A 被 gate 拒绝，trainable-A 只做了 50-step pilot。低跨窗口 overlap 可能意味着多个功能近似的可学习子空间，而不必然代表初始化无用；把几何稳定性作为唯一效用代理过于保守。

### Dominant-atoms：低 rank 可学，但 allocator 不可靠

Dominant 不再拟合窗口内全部梯度能量：先对每个 discovery window 的 signal 做 SVD，只取 top-2 局部 atom；再用 atom 的窗口内能量占比、对各 observation 的中位 capture、谱间隙和跨窗口重现度加权。所有窗口的加权 atom 再做一次 SVD，得到按“局部强且反复出现”排序的 consensus basis；calibration capture 随后在等价 uniform-r8 的总参数预算下分配 rank。

该流程得到 mean rank `10.67`、范围 `4--16`。step-50 full-bench Macro 为 `0.373654`。

| Rank map | 最终 DeltaW capture |
| --- | ---: |
| Heterogeneous map | `47.67%` |
| Uniform-r8 | `54.49%` |
| Greedy oracle | `60.39%` |
| Family-shuffled map | `47.69%`，`p=0.605` |

异构 map 没有优于 uniform-r8，且与 family-shuffled 无差异。低秩 span 有学习价值，但“哪个模块应拿更多 rank”的排序未通过反事实检验。

### Compact 与 Chord 旁证

Compact 从 GradTop step 100 的实际 `Delta W` 出发，用 held-out response hidden state 测每个奇异分量对 `Delta W h` 的输出能量；每个模块选择达到 90% calibration 能量的最小奇异值前缀，并量化到允许的 rank bin。随后对 `BA` 做小型 QR+SVD，截取该 rank 后重新物化更窄的 A/B，再继续训练；它是依赖 step-100 权重和 activation 的事后压缩，不是训练前 allocator。

该过程得到 mean rank `18.29`，保留 r32 的 `60.77%` 参数；step-150 Macro 为 `0.397868`，未压缩 GradTop 为 `0.404574`，差 `-0.67 pp`，95% CI 跨 0。这支持“可压缩”，不支持“压缩提高准确率”。

Chord 先在功能权重空间构造 `DeltaW(lambda)=(1-lambda)DeltaW_100+lambda DeltaW_270`，再投影回同一正交 A anchor，只重建对应 B；它不是直接插值两组 LoRA A/B 因子。`lambda=1.25` 表示沿 step100→270 的功能更新方向再外推 25%，得到 Macro `0.403345`；格式提高约 `0.93 pp`、输出缩短约 433 token。它更像行为控制，不是准确率或方向预测证据。

## 9. 阶段七：Full-gradient、协方差与 token mask

### 为什么去掉 random-B

random-B 只观测 `G^T R`，输出侧随机压缩可能漏掉主能量。Full-gradient probe 直接从完整梯度构造 mean、covariance、hybrid 等候选，并划分 discovery/calibration/audit，避免同一数据既选方向又评价方向。这里的 mean 是跨 prompt 梯度均值；covariance 是跨 prompt 的输入侧梯度二阶统计；hybrid 是两类候选投影空间的组合。

这里的 `G` 是某个线性模块基础权重 `W` 的策略梯度，形状与 `W` 相同。我们需要的是它的输入侧方向，因为这些方向最终写入 LoRA-A。random-B probe 通过随机矩阵得到 `G^T R`，计算和存储便宜，但先在输出维做了随机压缩；full-gradient probe 则读取完整 `G`，之后只在数值分解阶段使用随机 Nyström sketch。Nyström 在这里是一种不显式存储巨大协方差矩阵、只近似求其主特征子空间的数值方法；它不再决定哪些输出侧信息能被观察到。

### 三种 probe 不要混为一谈

本机先后使用过三条探测路径，它们的训练初始化形式相似，但梯度来源不同。

| Probe | 梯度来源 | Candidate 如何产生 | 数据切分 |
| --- | --- | --- | --- |
| Random-B / GradTop | 真实 GRPO 梯度的随机输出侧 sketch | 累计 `G^T R` 后做 SVD | 早期 observation/window 设计 |
| **SPAR positive-probe（L40S 本机实验起点）** | 正确且可解析 response 的 teacher-forced CE 梯度；即给定完整目标 response，计算逐 token 交叉熵 | rank32 random-B sketch | 32 discovery + 32 calibration |
| Full-gradient RL probe | 完整 prompt-group GRPO 梯度 | mean SVD 或 covariance eigenspace | 初版 16/16/8；Phase-0 扩为 64/32/16 |

这里的“L40S 本机实验起点”按实际启动时间定义：先做 SPAR smoke，再完成 SPAR 正式训练；表中的 Random-B / GradTop 则是为解释方法谱系而保留在前的一条迁移前历史路径。SPAR 只保留 reward≥0.5、parse 成功且 prompt 唯一的 response；full-gradient probe 则使用同一 prompt 的全部 rollout 及其组相对 advantage。两者回答的问题不同：SPAR 更像“正确轨迹告诉我们什么”，full-gradient 更接近“模型在 step 0 实际会沿什么策略梯度更新”。

### Full-gradient probe 的完整流程

以下流程以一个模块 `m` 为例。正式 artifact 会对 196 个 all-linear 模块同时执行。

1. **冻结待探测 policy。** 使用 base policy 生成 rollout；probe 期间不执行 optimizer step，因此所有 prompt 都来自同一个 policy 起点。
2. **按 prompt 组成 GRPO group。** 每个 prompt 有 8 条 response。先计算完整组的相对 advantage，再以该组有效 token 总数为分母，逐 response backward 并累积成一个 token-mean 梯度。一个 prompt group，而不是单条 response 或多 prompt mini-batch，是一个统计样本。
3. **计算并裁剪完整模块梯度。** full-gradient 路径在 discovery 时不创建 LoRA，而是临时解冻目标基础权重。对每个有效 prompt，通过 FSDP 收集各模块完整梯度 `G_mp`；再按该 prompt 的全模型联合梯度范数得到一个公共裁剪系数 `c_p`，令 `Gclip_mp=c_p G_mp` 后立即转移到 CPU accumulator。所有模块共享 `c_p`，避免逐模块归一化破坏层间尺度。
4. **严格分割 prompt。** discovery 只生成方向；calibration 只打分和分配；audit 只做最终 held-out 报告。三个集合按 prompt 隔离，同一题的 rollout 不跨 split。
5. **从 discovery 生成 rank-32 candidate。** `mean` 直接逐元素累加裁剪后的 prompt-group 梯度，再计算 `M_m=(1/P)sum_p Gclip_mp`，对其做 SVD 并取前 32 个右奇异向量。`covariance` 累计的是二阶矩 `C_m=(1/P)sum_p Gclip_mp^T Gclip_mp`；centered 版本再减去 `M_m^T M_m`。为避免物化 `d_in x d_in` 矩阵，代码实际固定随机正交矩阵 `Omega`，只累计 `sum_p Gclip_mp^T(Gclip_mp Omega)`，再用 Nyström 求前 32 个特征向量。`hybrid` 最后组合 mean 与 covariance 两类投影空间；每个候选向量都是 A 的一行。
6. **在 calibration 上逐 atom 投影。** 对候选 `a_mj` 和 calibration prompt `n`，计算 `g_mj^(n)=G_mn a_mj^T`。它表示：如果 A 中保留该方向，第一次更新 B 时能看到多大的输出侧梯度，以及这个梯度在不同 prompt 上是否一致。
7. **计算 atom score。** 保存能量/稳定性分数 `F/S/R/P`，并计算有符号的一阶收益下界 Gain-LCB。此时 candidate 方向固定，只比较如何排序和保留。SPAR 与 full-gradient artifact 对字段名 `U` 的定义不同，见下文。
8. **生成 allocation。** uniform-r8 在每个模块内选 8 个 atom；adaptive-eqr8 允许模块 rank 在 2--32 之间变化，但总参数预算不超过 uniform-r8。
9. **物化正式 LoRA。** 把选中的方向写入 A，B 全零，因此 step 0 的 `Delta W=sBA=0`，模型输出与 base policy 相同。随后 A/B 都按普通 GRPO 训练，probe score 不再在线更新。
10. **只用 audit 验收。** 在完全未参与 candidate、排序和 allocation 的 prompt 上报告 capture、稳定性等指标。audit 不能用于回头改 `k`、mask 或 allocator。

因此 `mean` 会让跨 prompt 的相反梯度直接抵消，描述共同的一阶更新；uncentered covariance 中符号相反的梯度仍贡献相同的 `G^T G` 能量，描述反复出现的输入侧方向；centered covariance 则进一步去掉共同均值，强调 prompt 间变化。这里的“平均”始终以 discovery prompt group 为单位，不是对单条 rollout 等权平均。

这个流程把三个问题分开：discovery 决定“有哪些方向”，calibration 决定“保留哪些方向、各模块多少个”，audit 检查“选择是否能推广到未见 prompt”。如果三个 split 混用，capture 会高估，allocator 也会对小 probe 集合过拟合。

### `P*R` 到底是什么

`P*R` 在代码中叫 `stable_energy`，是 calibration 阶段的 atom utility，不是概率乘法，也不是 Phase-0 的方法编号 P0/P1/P2。对模块 `m` 的第 `j` 个候选 atom：

```text
g_mj^(n) = G_mn a_mj^T

F_mj = mean_n ||g_mj^(n)||_2^2
S_mj = ||mean_n g_mj^(n)||_2^2
R_mj = S_mj / (F_mj + eps)
P_mj = F_mj / (sum_k F_mk + eps)
U_stable,mj = P_mj * R_mj
```

各项含义如下：

- `F` 是该 atom 在 calibration prompt 上的平均投影梯度能量。值大说明这个方向“有信号”，但信号可能在不同 prompt 上互相抵消。
- `S` 先对投影梯度向量求均值再平方。只有不同 prompt 的 `g` 大致同向时，`S` 才会大。
- `R=S/F` 是稳定信号比例。若各 prompt 上方向一致，`R` 接近 1；若方向反复翻转、均值抵消，`R` 接近 0。
- `P` 是 `F` 在当前模块 32 个 candidate atom 内的相对能量占比。它把不同模块各自归一化，避免大矩阵或梯度尺度大的模块仅靠绝对值垄断排序。
- `U_stable=P*R` 同时要求“在本模块内能量重要”和“跨 prompt 方向稳定”。`P` 高但 `R` 低表示强而不稳定；`R` 高但 `P` 低表示稳定但太弱；只有两者都高才得到高稳定能量效用。

一个简化例子：某 atom 占模块投影能量的 20%，即 `P=0.20`，但跨 prompt 稳定比例只有 `R=0.10`，则 `P*R=0.02`；另一个 atom 只有 10% 能量，但 `R=0.80`，则 `P*R=0.08`，后者会优先保留。该规则有意牺牲一部分瞬时大梯度，偏向可重复方向。

需要特别注意 artifact 字段名：SPAR positive-probe 把 `P*R` 直接保存为 `U`；full-gradient artifact 则把 `U` 定义为 `max(Gain-LCB, 0)`，而 `stable_energy` allocator 从已保存的 `P` 和 `R` 现场重新计算 `P*R`。因此读取 safetensors 时不能跨两种 probe 把 `U` 当成同一个量。

uniform-r8 与 adaptive-eqr8 使用 `P*R` 的方式不同：

- **Uniform-r8**：每个模块独立按 `P*R` 取前 8 个 atom。
- **Adaptive-eqr8**：先保证每个模块至少 2 个 atom，再把所有剩余 `(module, atom)` 按 `(P*R)/c_m` 全局排序，其中 `c_m=d_in+d_out` 是增加一个 rank 的 A/B 参数成本，直到用完 uniform-r8 的总预算。

因此 adaptive 的平均 rank 可以大于 8：便宜的模块每增加一个 rank 所需参数更少，会获得更多 atom；昂贵模块相应减少。`eq-r8` 比较的是总参数量，不是模块平均 rank。

文中的 `calibration capture` 通常指选中 atom 的 `F` 之和占全部 candidate `F` 的比例：

```text
capture = sum_selected F / sum_all F
```

它回答“选中方向保留了多少投影梯度能量”，不回答“训练后 benchmark 一定提高多少”。本机多次观察到 capture 更高但 full-bench 没有同步提高，所以它只能作为 probe 诊断和 gate，不能替代正式训练评测。

Gain-LCB 是另一种排序思路。先用 discovery 平均梯度得到该 atom 的输出侧更新方向 `b_D,mj=M_D,m a_mj^T`；再对每个 calibration prompt 计算有符号对齐分数

```text
x_mj^(n) = <G_C,mn a_mj^T, b_D,mj>
gain_mj = mean_n x_mj^(n)
Gain-LCB_mj = gain_mj - z * sample_std(x_mj) / sqrt(N)
U_mj = max(Gain-LCB_mj, 0)
```

其中 `G_C,mn` 是模块 `m` 在第 `n` 个 calibration prompt 上的梯度，`N` 是 calibration prompt 数。`x>0` 表示 discovery 更新方向与该 calibration 梯度同向，按该方向做一步梯度下降时，一阶近似预测 calibration loss 降低；跨 prompt 波动越大，标准误惩罚越大。本机默认 `z=1`，即减去一个标准误，并非严格的 95% 置信下界。它只是一阶单步代理，不包含后续 Adam 状态、A 的更新或 on-policy 轨迹。`P*R` 只看 calibration 内的能量与稳定性，Gain-LCB 则检验 discovery 方向在 calibration 上是否保持有符号的一致性。比较 rank allocation 时，两组必须使用相同 score；否则结果同时改变了 atom ordering 和 rank map，无法归因。

### 为什么专门做 uniform-r8

最初的 random-B energy rule 目标是保留 95% probe energy，但结果几乎退化成高 rank：196 个模块中 192 个为 r32、3 个为 r8、1 个为 r28，平均 rank `31.6122`。它的训练结果即使较好，也无法区分两个解释：

1. probe 找到了更合适的初始化方向；
2. 方法只是给了几乎所有模块 r32 容量。

因此 r8 被用作有意设置的极端低秩控制，而不是预先假定的最优 rank。做法是复用同一个 rank-32 candidate artifact，只取每个模块排序最前的 8 个方向，物理创建 r8/alpha16 adapter，并保持 `alpha/r=2`。每模块的可达更新从最多 rank 32 压到 rank 8，A/B 参数从 r32 的约 36.93M 降到 `9,232,384`，约为四分之一。

选择 r8 有四个具体目的：

- **隔离方向与容量**：把既有 random-B candidate 强制截到 r8，观察大部分收益是否仍存在；正式评测再与同方法族的 near-r32 结果作近似容量对照。
- **检验 prefix ordering**：如果前 8 个方向仍有效，说明 probe 不只是找到宽 span，还提供了至少部分有用的前缀顺序。
- **建立等预算基线**：后续 adaptive-eqr8 必须严格使用 uniform-r8 的 A/B 参数预算，避免把“异构分配”与“更多参数”混在一起。
- **压力测试早期路径**：r8 是相对 r32 的 4 倍压缩。如果这种窄子空间仍能在 50 步内明显学习，就说明早期优化所需方向可能远少于 candidate width。

训练时曾把 vLLM 的全局容量设为 rank32/alpha64，以便同一运行栈兼容 adaptive map；真正的 `rank_pattern/alpha_pattern` 仍是每模块 r8/alpha16。因此 rollout 与 actor 的有效 scaling 都是 2，不能把全局容量配置误读为实际训练了 r32。

### r8 与 near-r32 的关键对照

最干净的历史链是 standard LoRA r8、random-B uniform-r8 和 original random-B near-r32。三者都使用 dropout 0.05，前两者 rank 完全相同；random-B r8 与 near-r32 又来自同类 probe 候选。

| 方法 | 实际 rank | Step | Macro Avg@k | 相对上一项 |
| --- | --- | ---: | ---: | ---: |
| Standard LoRA | uniform-r8 | 50 | `0.323111` | - |
| Random-B probe init | uniform-r8 | 50 | `0.358323` | `+0.035212` |
| Original random-B（L40S 复现） | mean `31.6122`，近似 r32 | 50 | `0.367304` | `+0.008981` |

表中的 `Original random-B` 是方法标签；`0.367304` 对应本机可审计的 `seed42_repro_v1` 复现，而不是旧机器上未带 `repro` 的同名目录。其配置如下：

- **Probe**：DeepSeek-R1-Distill-Qwen-1.5B 的 196 个 all-linear 模块；每个 observation 使用 16 个 prompt、每个 prompt 8 条 rollout，共 12 个 observation；random-B sketch width 8、capacity 64，以 95% 累积能量在 `{8,12,16,20,24,28,32}` 中选 rank，seed 42。最终 rank map 为 `3 x r8 + 1 x r28 + 192 x r32`，均值 `31.6122449`。
- **训练**：DAPO-Math-17k boxed 上做 GRPO；global prompt batch 64、mini-batch 16、每 prompt 8 rollouts，token-mean loss；A/B 均可训练，`alpha=2r`、dropout `0.05`、学习率 `1e-6`，计划训练 270 steps，此处报告 step 50。
- **评测**：step-50 checkpoint 在 AIME24、AIME25、AMC23、HMMT Feb、MATH500、Minerva 六任务上评测，共 902 题、7248 次生成；seed 42、temperature `0.6`、top-p `0.95`、max new tokens `32768`，得到六任务 Macro Avg@k `0.3673039216`。对应训练名为 `gradtop_probe12_r8to32_original_random_b_a2_b64m16n8_270_seed42_repro_v1`。

从 standard r8 到 near-r32 的总差为 `+0.044193`，其中 uniform-r8 probe init 已取得 `+0.035212`，约占这段点估计差距的 79.7%；把 rank 从 8 提高到近似 32，只再增加 `+0.008981`。逐任务上，random-B r8 相对 standard r8 在 6 个任务中的 5 个提高；near-r32 相对 random-B r8 也为 5 个提高，但 AIME25 反而下降。因此结果不是“所有任务都随 rank 单调变好”。

这三项共享主要训练口径，但 random-B r8 与正式 near-r32 复现不是位级相同的 candidate artifact，异步 rollout 也未做到逐请求配对。因此 `79.7%` 只描述点估计中收益的相对量级，属于 B 级机制证据，不是严格的 rank 因果分解。

full-gradient mean uniform-r8 在 step 50 得到 `0.358565`，与 random-B r8 的 `0.358323` 几乎相同。它使用 dropout 0，不能与前述链条作严格单变量比较，但提供了一个独立旁证：不同 probe 构造都能在仅 8 个方向下达到约 `0.358`，说明低秩结果不是 random-B 单一实现的偶然产物。

这组实验发现的不是“整个模型只有 8 维”，而是更窄且可辩护的现象：**对每个 LoRA 模块，少量经过 RL 梯度筛选的输入侧方向，已经足以承载大部分 50-step 初始化收益。** r8 是每个模块的局部 rank；全模型仍有 196 个 adapter，不能把它解释成全网络共享的 8 维子空间。

还需保留三个边界：

1. A/B 在正式训练中都可训练，r8 限制的是每模块更新矩阵的最大 rank，不是把 8 个初始方向永久冻结。
2. full-gradient r32 只有 step 150 的正式评测 `0.375749`，缺少同协议 step 50/100 结果，不能用它估计纯粹的 r8→r32 增益。
3. full-gradient mean r8 从 step 50 的 `0.358565` 降到 step 100 的 `0.340163`，同期训练 reward 上升。极低 rank 的证据更支持“方向正确时可以快速起步”，尚不证明长期上限与 r32 相同。

### 为什么随后比较 adaptive-eqr8

确认 r8 可以学习后，下一问题不是立即断言“r8 足够”，而是问：在完全相同的低参数预算下，把 rank 从不敏感模块转给高 utility 模块，是否优于所有模块统一 r8？

full-gradient `P*R` 对照中，两组读取同一个 candidate、使用同一个 atom score、相同 seed 和训练配置：

| 指标 | Uniform-r8 | Adaptive-eqr8 |
| --- | ---: | ---: |
| A/B 参数 | `9,232,384` | `9,231,360` |
| Rank min / mean / max | `8 / 8 / 8` | `2 / 14.2653 / 32` |
| Calibration capture | `0.843433` | `0.848773` |
| Audit capture | `0.838146` | `0.838414` |
| Reward AUC1:50 | `0.346875` | `0.357187` |

adaptive 的 reward AUC 高 `0.0103125`，但独立 audit capture 只高 `0.000268`，且异步 rollout 没有 request-level 固定随机性。另一条 SPAR positive-probe 等预算实验中，adaptive 的 mean rank 为 `13.7398`，但 AUC 只比 uniform 高 `0.000859`。两组结果都说明异构分配可以在相同参数预算下运行，却不足以证明 allocator 稳定优于 uniform-r8。

因此 r8 实验留下的核心发现分为两层：第一层较强，即有信息的低秩初始化方向能保留大部分早期收益；第二层仍未解决，即如何排序 atom、如何把有限 rank 正确分配到不同模块。

### Token-mask covariance

对 prompt-group 梯度 `G_p`，旧实验比较过两种二阶统计：

```text
mu = mean_p G_p
C_uncentered = mean_p (G_p^T G_p)
C_centered   = mean_p ((G_p-mu)^T (G_p-mu))
             = C_uncentered - mu^T mu
```

uncentered covariance 同时保留跨 prompt 的共同均值和变化能量；centered covariance 去掉共同均值，更强调 prompt 间变化方向。变化大可能代表任务多样性，也可能只是 rollout 噪声，因此 centered 并不天然更好。

token-mask 只在 probe 统计梯度时筛 token，不改变后续正式 GRPO loss。surprisal 定义为当前 policy 对实际采样 token 的负对数概率 `-log p(y_t)`；每条 response 保留 surprisal 最高的 50% token，同时至少保留 128 token 和末尾 128 token。末尾 token 用于保留答案格式区域，但这种规则也可能过度偏向异常高 surprisal token。candidate rank32、Nyström width40，adaptive 预算等价 uniform-r28，`r_min=16`、`r_max=32`。

| 方法 | Step | Macro |
| --- | ---: | ---: |
| Centered covariance adaptive eq-r28 | 100 | `0.382131` |
| Full-gradient mean uniform-r32 | 150 | `0.375749` |
| Centered covariance adaptive eq-r28 | 50 | `0.375451` |
| Uncentered covariance adaptive eq-r28 | 100 | `0.369779` |
| Random-B near-r32（L40S 复现） | 50 | `0.367304` |
| Full-gradient mean uniform-r8 | 50 | `0.358565` |
| Random-B uniform-r8 | 50 | `0.358323` |
| Standard LoRA r8 | 50 / 100 | `0.323111 / 0.315783` |

旧协议最稳妥的结论是：probe 初始化的 r8 明显高于同批 standard r8，说明方向重要；centered covariance 值得继续检查。但 centered/uncentered 同时改变过 mask scope 和 optimizer continuity，token mask、candidate 与 allocator 也同时变化，均不能作独立因果解释。

这些混杂促使后续不再继续堆方法，而是冻结变量做可审计实验。

## 10. 阶段八：Phase-0/1，冻结成可审计比较

**设备来源。** Phase-0、Phase-1 的四组 seed42 训练/评测，以及 seed43-I0，均在原 4 x L40S 主机 `app-ebbc492c8b20ba70` 上完成。之后项目做了异地 handoff：seed43-I8、seed43-I32、协议匹配的 standard LoRA r32 与后续补评测来自迁移后的机器，不属于原 L40S 实验。下表把这些结果放在一起，是因为它们复用冻结的 P2、allocation、训练协议和评测 snapshot，不表示运行硬件相同。

### Phase-0：先筛 candidate source

Phase-0 固定相同 64/32/16 个 discovery/calibration/audit prompts、每题 8 个 rollout、candidate rank32 和 sketch width40。

| 方法 | Estimator / mask | Calibration | Audit | Split cosine |
| --- | --- | ---: | ---: | ---: |
| P0 | centered / no mask | 0.571205 | 0.549364 | 0.717728 |
| P1 | centered / random mask | 0.570243 | 0.549561 | 0.707911 |
| P2 | centered / top-surprisal | **0.571859** | 0.549244 | **0.723547** |
| P3 | centered cross-fit / stable-band | 0.211234 | 0.207643 | 0.318512 |

这里的 P0/P1/P2/P3 只是方法编号，与 `P*R` 中的相对能量 `P` 无关。random mask 保留与 top-surprisal 相同数量的随机 token，用来判断收益来自“稀疏化本身”还是“选中了特定 token”。P3 的 stable-band selector 同时参考 advantage 和 token entropy，并排除极端 surprisal 尾部；cross-fit 则先用完整 8-response group 固定 advantage，再把 rollout 拆成半组，只保留跨半组可重复的协方差信号。

四组的 token selector 只作用于 discovery；calibration 和 audit 都用完整 response-token 梯度。这样 held-out capture 测的是“筛选后得到的子空间能否解释未筛选梯度”，而不是在同一个 mask 下自我评价。

P0/P1/P2 的 centered 与同数据 uncentered audit 差异都小于 `4.4e-5`；P2 提高 discovery capture，却未提高 audit。P3 的 audit 和稳定性明显下降，预注册判定 no-go。

Phase-0.5 再把 2/4/8/16 个 P3 正谱方向注入 P2。随着 P3 方向增加，calibration、audit 和 split overlap 全部单调下降，所有 hybrid 均未通过 gate。Phase-0.6 replay 又因 basis/atom-score invariants 大面积失败被标记为 invalid，相关 C candidates 禁止进入训练。

最终 Phase-1 只使用冻结并审计过的 P2，而不是从大量候选中事后挑最高分。

### Phase-1 设置

四组均为 uniform-r32、alpha64、dropout0、50 个 trainer steps（每步 4 次 optimizer update）、196 个模块；B 为零初始化，A/B 均训练。唯一变量是初始 A 中 P2 signal 与随机正交补空间的组成。

| 方法 | Signal | Random complement | 角色 |
| --- | ---: | ---: | --- |
| I0 | 0 | 32 | Matched random-orthogonal baseline |
| I8 | 8 | 24 | Mixed candidate |
| I16 | 16 | 16 | Mixed negative screen |
| I32 | 32 | 0 | Signal-only endpoint |

这不是 adaptive rank 实验，而是隔离“RL 梯度 signal 是否适合构造初始化 span”。四组 rank 和参数量完全相同。

构造时，每个模块先固定 P2 的有序正交 signal basis。`Ik` 原样保留前 `k` 行，再从同一个确定性高斯随机矩阵中逐行取向量，投影掉已选 signal 和已有随机方向并归一化，补足到 32 行；I0/I8/I16/I32 因而分别含 0/8/16/32 个 signal 方向。四组使用相同 rank、scaling 和零初始化 B，差别只在 A 的 signal 比例，而不是模型初始输出或参数量。

### 50-step 主结果

结果来源为：I0 的 seed42/43 和 I8/I16/I32 的 seed42 属于原 L40S；I8/I32 的 seed43 以及 matched standard LoRA r32 两个 seed 属于异地后续机器。因此 seed43 的 `I8/I32 - I0`、以及 signal 方法对 matched standard LoRA 的比较同时跨机器；冻结哈希降低了配置漂移风险，但不能把硬件和运行时差异完全从方法差异中剥离。

| 方法 | Seed 42 | Seed 43 | 两-seed mean |
| --- | ---: | ---: | ---: |
| Base，无 RL | `0.324348` | n/a | n/a |
| Matched standard LoRA r32 | `0.330053` | `0.338761` | `0.334407` |
| I0 | `0.330162` | `0.327747` | `0.328955` |
| I8 | `0.366161` | `0.341696` | `0.353928` |
| I16 | `0.313498` | 未运行 | n/a |
| I32 | `0.376479` | `0.333201` | `0.354840` |

| 比较 | Seed 42 | Seed 43 | 结论 |
| --- | ---: | ---: | --- |
| I8 - I0 | `+0.035999` | `+0.013949` | 两 seed 为正，但 seed43 仅 3/6 tasks 为正 |
| I32 - I0 | `+0.046317` | `+0.005454` | Seed43 paired CI 跨 0 |
| I8 - standard LoRA | `+0.036108` | `+0.002934` | Held-out seed 优势很小 |
| I32 - standard LoRA | `+0.046426` | `-0.005560` | 跨 seed 反转 |
| I8 - I32 | `-0.010318` | `+0.008494` | Mixed 与 signal-only 排序反转 |

seed43 的 I8-I0 paired delta 为 `+0.013949`，95% CI `[+0.001000,+0.027188]`；I32-I0 为 `+0.005454`，CI `[-0.005884,+0.016747]`。I8 宏平均为正，但只有 3/6 benchmark 为正，未满足预注册的至少 4/6 gate。

I0 在原 L40S 上从 seed42 到 43 只变化 `-0.002416`；I8/I32 跨到后续机器后分别下降 `-0.024465/-0.043278`。代码、allocation 和 subspace hash 一致，使“signal init 放大 training-seed 交互”仍是合理解释，但由于方法与机器边界同时变化，现有结果不能严格排除设备或运行时迁移的影响。

matched standard LoRA seed42 曾在 step41 后从完整 step25 checkpoint 恢复并重算 26--41；最终结果 verified，但引用时必须披露该恢复边界。

## 11. Step-25：收益发生在什么时候

**设备来源：异地后续机器，非原 4 x L40S。** 为区分早期加速和最终性能，在 seed43 的相同 snapshot 上评测 standard LoRA、I8、I32 的 step25；三组均有 7,248 条 records 并通过 postverify。

| 方法 | Step 25 | Step 50 | 25→50 | Parse 25→50 |
| --- | ---: | ---: | ---: | ---: |
| Standard LoRA | `0.313818` | `0.338761` | `+0.024943` | `.7943→.8153` |
| I8 | `0.328308` | `0.341696` | `+0.013388` | `.8031→.8364` |
| I32 | `0.335436` | `0.333201` | `-0.002235` | `.8086→.8248` |

相对 standard LoRA，I8/I32 在 step25 分别领先 `+0.014490/+0.021618`，到 step50 只剩 `+0.002934/-0.005560`。对应 difference-in-differences 为 `-0.011556/-0.027178`。

因此最有根据的机制描述是：signal 初始化提供有利的早期 trajectory bias，但没有证据表明它提高 50-step 可达上限。I32 早期最激进，也最容易在后续或不同 seed 中反转；I8 的随机补空间可能起缓冲作用，但两个 seed 不足以证明其稳定优于 I32。

## 12. 外部模型检查：Qwen3-4B

**设备来源：后续 8-GPU 机器，非原 4 x L40S。** 此前的 4-GPU 与早期 8-GPU 尝试均未形成算法结果；下表的 base 与正式 standard-LoRA step50 来自后续 8-GPU 环境。

### 为什么先跑 standard LoRA

1.5B 上 signal init 有明显 seed interaction，所以不能把 I8 直接搬到第二个模型后只看单点。先完成 Qwen3-4B base 与同协议 standard LoRA，确认训练栈、资源需求和可学习量级，再决定是否为 252 个模块生成新的 P2 artifact。

Qwen3-4B standard LoRA 使用 8 GPU、rank32、alpha64、dropout0、252 个 all-linear 模块、66,060,288 个可训练参数和 50 个 trainer steps。4-GPU 与早期 8-GPU 尝试因资源/加载路径失败，不属于算法结果；最终 8-GPU run 完成，耗时约 26 小时 15 分。

| 模型 | Macro Avg@k | Parse | 平均长度 |
| --- | ---: | ---: | ---: |
| Qwen3-4B base | `0.643853` | `0.980546` | 9481 |
| Standard LoRA step50 | `0.652255` | `0.981926` | 9219 |

paired delta 为 `+0.008402`，95% CI `[+0.001514,+0.015556]`，`P(delta>0)=0.9920`。逐任务并不统一：AIME24/AIME25/AMC23 为正，HMMT/MATH500/Minerva 为负。

该结果只证明 Qwen3-4B standard-LoRA 基线可运行且单 seed 小幅改善；目前尚无 Qwen3-4B 的 P2、I0 或 I8 正式结果，不能声称 signal init 跨模型迁移。

## 13. 串联全部实验后的认识

### 13.1 初始化 span 是当前最可靠的正结果

RLPO 和 gradient probe 都显示，LoRA-A 初始化 span 在训练中高度稳定，主要功能更新发生在 B/left side。GradTop、旧 r8 probe 和 Phase-1 step25 共同说明，较好的初始化方向能加快早期学习。

但 span 不唯一。Stable-SNR 的低 overlap 与 GradTop 的强训练效果可以同时成立：不同小批梯度可能给出功能近似但几何不同的子空间。几何稳定性不能直接等同于训练效用。

### 13.2 可压缩不等于 allocator 已解决

RLPO 最终 DeltaW effective rank 约 17，前 8 个奇异分量占 66.87%；Compact 也能用约 60.77% 参数保持接近表现。这证明最终更新存在谱集中。

但 base prefix、GradTop atom prefix、Stable-SNR map 和 Dominant map 都没有稳定给出正确 ordering/allocation。当前缺口不是“是否存在低秩解”，而是训练前如何判断哪些原子和模块值得预算。

### 13.3 动态裁剪不是第一优先级

AdaLoRA 把 importance、正交损失、裁剪调度和 rollout 同步绑定在一起，难以归因，也容易让正则主导 RL。现有证据更支持先做静态、可审计的 probe、ordering 和 allocation，再考虑低频重分配。

### 13.4 训练 reward 不能替代 held-out full-bench

GradTop 后期 reward 上升而准确率平台；centered/uncentered 的 reward 难区分但 full-bench 不同；I8/I16 训练指标接近而评测相差明显。方法选择至少要联合看 held-out Macro、逐任务方向、parse、length、hit-max 和 paired interval。

### 13.5 Signal init 与 on-policy trajectory 强交互

Phase-1 中 I0 跨 seed 接近，I8/I32 更敏感；step25 又显示 signal 方法早期领先但被普通 LoRA 追近。signal init 不是把模型放到恒定更优的静态点，而是改变早期采样分布，再由 on-policy 数据反馈放大或消解差异。

## 14. 当前结论边界

### 可以声称

- RL-gradient-informed 初始化能改变并经常加快早期学习。
- RLPO/GradTop 的初始化 A span 基本稳定，B/left-side 重组承担主要功能更新。
- 最终 DeltaW 有明显谱集中，静态压缩具有可行性。
- 不同模块的可压缩程度不同，但现有训练前 allocator 尚未证明优于同预算 uniform rank。
- signal init 的收益对 training seed、训练阶段和 benchmark 敏感。

### 不能声称

- AdaLoRA 在当前 RL 设置下已有有效正式分数。
- I8 或 I32 稳定优于 matched standard LoRA。
- random complement 明确优于 signal-only，或反过来。
- token mask、centered covariance 或 adaptive rank 的独立贡献已经隔离。
- Qwen3-4B 已验证 signal init 的跨模型泛化。
- 更高训练 reward 意味着更高六任务准确率。

## 15. 下一步

下一轮不应同时修改 probe、atom score、rank map 和训练动态。最小顺序是：

1. 固定已审计的 full-gradient/P2 candidate 框架，保留 discovery/calibration/audit 三分割。
2. 在 uniform rank 下单独研究 atom ordering，比较 mean、covariance、dominant 和 signal/random prefix。
3. ordering 通过后，再以相同 atom score 比较 uniform-r8/r16 与等参数 heterogeneous map。
4. 静态 map 有优势后，才做低频、带 held-out gate 的 reallocation，而不是每步裁剪。
5. Qwen3-4B 必须重新生成 252 模块的 probe，并在模型内做 I0/I8 配对；不能复用 1.5B directions。

当前最稳的研究主线不是“已经解决 adaptive rank”，而是：**RL 梯度可以构造有效的初始化 span 并带来早期加速，但 candidate discovery、atom ordering 和 rank allocation 是三个不同问题；on-policy trajectory 又会放大初始化与 seed 的交互。**

## 16. 复现入口

- 原 4 x L40S 完整评测对照：[local_l40s_full_benchmark_comparison_2026_10_zh.md](local_l40s_full_benchmark_comparison_2026_10_zh.md)
- Full-gradient probe：[full_gradient_rl_probe_v1_zh.md](full_gradient_rl_probe_v1_zh.md)
- SPAR 静态分配：[spar_lora_v0_static_zh.md](spar_lora_v0_static_zh.md)
- Windowed consensus：[windowed_adam_consensus_probe_zh.md](windowed_adam_consensus_probe_zh.md)
- RLPO-init：[verl_rlpo_init_zh.md](verl_rlpo_init_zh.md)
- Stable-SNR：[verl_rl_gradient_subspace_stable_snr_v2_zh.md](verl_rl_gradient_subspace_stable_snr_v2_zh.md)
- Dominant-atoms：[verl_rl_dominant_gradient_atom_probe_zh.md](verl_rl_dominant_gradient_atom_probe_zh.md)
- AdaLoRA 实现说明：[verl_adalora_code_walkthrough_zh.md](verl_adalora_code_walkthrough_zh.md)
- seed-43 I8 迁移说明：[phase1_seed43_i8_migration_zh.md](phase1_seed43_i8_migration_zh.md)

主要路径：

```text
Repository:       /root/peft-for-rl
Runtime root:     /data/peft-for-rl-runtime
Training data:    /data/peft-for-rl-runtime/datasets/verl_dapo/data/dapo-math-17k-boxed.parquet
Evaluation root:  /data/peft-for-rl-runtime/outputs/full_bench_eval
```
