# 本机 PEFT for RL 实验研究叙事

更新时间：2026-10-08 UTC

本文不是按文件夹或启动时间罗列实验，而是回答一条连续的研究问题：在数学推理 GRPO 中，LoRA 的初始化子空间、有效秩和训练动态应当怎样设计，才能在较少可训练参数下保持甚至加快学习。

最简主线如下：

```text
固定 LoRA/OFT 摸底
  -> 直接迁移 AdaLoRA：工程复杂，而且 SFT 正交正则压过 RL 信号
  -> RLPO-init：把持续正交约束改成一次性正交初始化
  -> D1/CHR：验证模块所需秩不同，但离线 oracle 不等于可部署 allocator
  -> GradTop：不用 base principal 顺序，改用 RL 梯度发现初始化 span
  -> Stable-SNR/Dominant：稳定梯度与低秩 atom 有用，但 rank allocator 不可靠
  -> Full-gradient/token-mask：去掉 random-B 投影瓶颈，拆分均值、协方差和 token 信号
  -> Phase-0/1：冻结 signal/random 组成，做可审计的同协议比较
  -> Step-25 轨迹：确认主要收益是早期 trajectory bias，普通 LoRA 会追近
  -> Qwen3-4B：先完成外部模型 standard-LoRA 基线，尚未证明 signal init 可迁移
```

## 1. 如何阅读这些结果

本机实验跨越多个代码版本、设备、训练步数、dropout、benchmark snapshot 和训练 seed。本文使用以下证据等级，避免构造一个没有统计意义的总排行榜。

| 等级 | 含义 | 可以支持的结论 |
| --- | --- | --- |
| A | 冻结 Phase-1 协议，records、contract 和 postverify 齐全 | 当前固定模型、任务和训练预算下的主要比较 |
| A-reference | 严格审计但不参与 paired training 的基准 | 绝对位置参考 |
| B | records 和评测口径可复用，但训练协议不完全匹配 | 历史量级和机制线索 |
| C | summary 完整，但协议或原始 records 不足 | 描述性历史 |
| D | 已知实现缺陷、任务缺失、损坏或只完成 screen | 工程诊断和失败记录 |

任何跨等级的数字都只能帮助形成假设，不能直接证明方法优劣。尤其需要区分四个问题：

1. **Candidate span discovery**：是否找到了一个包含有用更新方向的候选子空间。
2. **Atom ordering/scoring**：候选 span 内哪些方向应优先保留。
3. **Rank allocation**：总预算固定时，各模块分别应拿多少 rank。
4. **Training-time adaptation**：训练过程中是否需要裁剪、恢复或重新分配 rank。

GradTop 的成功主要回答第 1 项；Dominant/SPAR 的负结果主要暴露第 2、3 项；AdaLoRA 尝试第 4 项，但当前没有形成有效的方法结果。不能用其中一项的正结果替代另外三项。

## 2. 公共训练与评测口径

### 2.1 1.5B 主实验协议

大多数 1.5B 数学 RL 实验使用以下设置。历史 run 若偏离，会在相应章节注明。

| 项目 | 设置 |
| --- | --- |
| Base model | DeepSeek-R1-Distill-Qwen-1.5B |
| 训练数据 | DAPO-Math-17k boxed |
| RL 算法 | GRPO |
| LoRA target | `all-linear`，共 196 个模块 |
| Global prompt batch | 64 |
| PPO mini-batch | 16 |
| 每 prompt rollout | 8 |
| PPO epoch | 1 |
| 最大 prompt/response | 1024 / 8192 |
| 学习率 | `1e-6` |
| weight decay / warmup | 0 / 0 |
| scheduler | constant |
| KL reward / KL loss | 均关闭 |
| 常用 scaling | `alpha/r = 2` |
| 训练 shuffle / actor shuffle | True / False |

这里的一个 trainer step 包含 64 个 prompt、每个 8 条 rollout，并通过 16-prompt mini-batch 产生 4 次 optimizer update。这个差异对 AdaLoRA 的剪枝时间轴尤其关键。

### 2.2 正式六任务评测

| 项目 | 设置 |
| --- | --- |
| 任务 | AIME24、AIME25、AMC23、HMMT Feb、MATH500、Minerva |
| Problems / generations | 902 / 7248 |
| temperature / top-p | 0.6 / 0.95 |
| max new tokens | 32768 |
| eval seed | 42 |
| 主指标 | 六任务等权 Macro Avg@k |

同时必须看 Parse rate、Pass@k、hit-max 和平均输出长度。多轮实验已经证明：on-policy reward、held-out accuracy、格式正确率和长度可以朝不同方向变化。

## 3. 阶段一：先确认固定 PEFT 的基本量级

### 为什么先做

在讨论自适应 rank 前，需要确认普通 LoRA、OFT 和已有结构化 adapter 在当前 GRPO 管线中能否学习，以及分数随步数大致怎样变化。这一阶段的主要价值是建立量级，不是给后来的 Phase-1 提供严格 baseline。

### 重要设置与结果

最早 stable LoRA 文档轨迹为 step100/200/300 的 `0.331179/0.340859/0.346207`。另一条有完整 records、但使用 dropout 0.05 和更长训练预算的旧 LoRA 轨迹，在 step100/200/300 达到 `0.389839/0.402729/0.409672`。

OFT 的 step100/200/300/400/500/600 为：

```text
0.323712 / 0.337397 / 0.359758 / 0.378069 / 0.374849 / 0.366732
```

其他旧方法代表点包括 BISO step200 `0.410365`、BOET step200 `0.394733`、GeoRA 约 step500 `0.385174`、Skew step100 `0.386569`。

### 得到什么

- PEFT 在此 RL 配置下可以稳定学习，且性能并不只由名义 rank 决定。
- 更长训练不保证单调提升，OFT 和后来的 GradTop 都出现平台或回落。
- 这些 run 的 step、dropout、初始化和 snapshot 不统一，只能作为 B/C-grade 历史背景，不能与 50-step Phase-1 直接排名。

这一步引出的问题是：如果固定 rank 已经能学，动态 rank 是否能更有效地分配预算？

## 4. 阶段二：直接迁移 AdaLoRA，为什么没有形成有效结果

### 为什么这样设置

AdaLoRA 是最直接的动态 rank 基线：先为每个模块提供较宽的 SVD 式 adapter，再依据 importance score 在训练中把全局预算从 `init_r=32` 逐渐压到 `target_r=8`。它可以检验 SFT 中成熟的动态裁剪机制能否直接迁移到 GRPO。

核心配置为：

| 项目 | 设置 |
| --- | --- |
| 实现 | 官方 PEFT AdaLoRA 接入 verl |
| 初始 / 目标 rank | 32 -> 8 |
| alpha / dropout | 64 / 0.05 |
| target | all-linear，196 modules |
| trainer steps | 270 |
| optimizer updates | 1080，即每 trainer step 4 次 |
| 正确调度轴 | `total_step/tinit/tfinal/deltaT` 全部按 optimizer update |

### 实现中发现的关键问题

1. **调度时间错位**：若把 270 trainer steps 直接作为 AdaLoRA total step，裁剪会快 4 倍。allocator 必须在每次成功 `optimizer.step()` 后调用。
2. **PPO loss 在模型外计算**：PEFT 官方正交 loss 只有模型收到 labels 并返回 loss 才会进入目标。verl 原路径只取 logits，因此最初的正交 loss 实际没有优化。
3. **正交项尺度压倒 PG**：旧 fixed run 使用 `orth_reg_weight=0.5`。raw orth loss 约 `5.2`，加权后约 `2.6`，而常见 PG loss 绝对值仅 `0.01--0.15`。这会让优化主要服务于 A/B 正交，而不是 reward。
4. **actor 与 rollout 缩放不一致**：PEFT config 的继承字段 `r=8`，真实 A/B 宽度由 `init_r=32` 决定。若原 config 直接交给 vLLM，rollout 使用 `alpha/8`，actor 初始使用 `alpha/32`，adapter 被放大 4 倍，污染采样、old log-prob 和 PPO ratio。
5. **额外权重不被 vLLM 接受**：`lora_E/ranknum` 不是普通 LoRA 权重，需要在同步边界折入 A，并过滤额外 key。
6. **FSDP 与 checkpoint**：allocator 必须看到全参数和全梯度；importance/uncertainty EMA 也必须显式保存，否则中途 resume 会改变裁剪轨迹。
7. **不能用任意 step 的 `E != 0` 统计预算**：mask 后 Adam 可能让 E 再次非零，必须分别记录 scheduled rank、mask 时 rank 和当前 nonzero rank。

### 唯一评测及其边界

早期 official step150 merge 的五个有效任务 Macro 为 `0.3720`，但它是 D-grade：HMMT 输入损坏，Minerva 少 36 条，而且训练 rollout 经历过上述缩放问题。有效秩均值 `9.878`，范围 `0--25`，3 个模块被裁空。它只能用于定位异常，不能作为 AdaLoRA 方法分数。

后续 official-fixed/isolated 版本修复了调度、FSDP、vLLM 转换和正交 loss 接入，但仍使用过大的 `orth_reg_weight=0.5`，且没有从 step0 完成正式六任务 run。当前代码默认已降到 `1e-3`，但实验闭环仍缺失。

### 得到什么，以及为何转向 RLPO-init

AdaLoRA 暴露了两个根本问题：SFT 的持续正交正则在 RL 中可能比策略梯度大一个数量级；动态 allocator 又同时引入调度、状态同步和 rollout 等价性风险。因此下一步不再同时解决“初始化、正交、动态裁剪”三个问题，而先问更小的问题：正交性是否只需要放在初始化时？

## 5. 阶段三：RLPO-init，把正交约束改成一次性初始化

### 设计动机

RLPO-init 取消训练中的正交 loss 和 allocator，只使用 base weight 的主右奇异子空间初始化 LoRA-A：

```text
A0 = top-32 right singular vectors of W_base
B0 = 0
DeltaW0 = B0 A0 = 0
```

A、B 随后都可训练。这样既避免初始输出扰动，也避免持续正交损失压过 RL 信号。rank32、alpha64，scaling 保持 2。

### 结果

step270 的修复后六任务 Macro 为 `0.39865`。原评测的 HMMT 数据损坏，旧值 `0.378684` 无效；`0.39865` 来自其余任务加 HMMT-only 修复，因此仍属于历史描述，不是当前 A-grade matched run。

几何诊断比单一分数更重要：

| 指标 | 结果 |
| --- | ---: |
| 最终 DeltaW 留在初始 top-32 right span 的能量 | `99.997%` |
| `B(A-A0)` 相对 `BA` 的比例 | `0.709%` |
| DeltaW effective rank | `17.13` |
| rank@95% energy | `24.44` |
| 前 8 个 DeltaW 奇异分量能量 | `66.87%` |
| base top-4/8/16 prefix 对最终 DeltaW 的 capture | `13.53% / 26.03% / 50.83%` |

注意这里的 `DeltaW` 是训练产生的 `BA - B0A0`；由于 `B0=0`，它数值上等于 `BA`，但不能把 A 本身当成权重更新。

### 核心结论

- 完整的初始化 A span 几乎不旋转，训练主要通过 B/left side 在这个 span 内重组。
- 最终 DeltaW 的谱确实聚集，说明可压缩。
- 但 base weight 的 singular-value prefix 不能正确排序 RL 更新原子。完整 top-32 span 有用，不代表 base top-8 就是最终最重要的 8 个方向。

这一步把后续问题明确成：既然最终 rank 有冗余，如何为不同模块选择 rank，并找到真正有用的方向顺序？

## 6. 阶段四：D1 与 CHR-RLPO，先用离线 oracle 验证异构 rank

### 为什么这样设置

不同层、attention/MLP 家族和模块尺寸不同，统一 rank 很可能浪费预算。D1 不直接训练 allocator，而是先对已训练 RLPO 模型做 activation-weighted output energy 诊断：每个模块选择能够保留 95% 输出能量的最小 rank，并离散到 `8,12,16,20,24,28,32`。

探测使用 48 条平衡序列，24 条 calibration、24 条 validation。结果为：

| 指标 | 结果 |
| --- | ---: |
| mean rank | `16.78` |
| rank range | `8--28` |
| 相对 uniform-r32 参数/FLOP | `50.09%` |
| calibration/validation rank Spearman | `0.9871` |
| D1 最差模块 retained energy | `94.62%` |
| fixed-r16 最差模块 retained energy | `75.16%` |

D1 的优势主要是保护最差模块，不是提高全局平均 retained energy。

CHR-RLPO 随后使用这张静态 map，从 base 重新训练；每个模块使用 base top-`r_m` right singular vectors，`B0=0`，并固定 `alpha_m/r_m=2`。

### 结果与限制

| Step | Macro Avg@k |
| ---: | ---: |
| 200 | `0.380842` |
| 270 | `0.393234` |

训练在 step106 后因 vLLM wake-up OOM，从 step100 且没有 optimizer state 的 checkpoint 恢复。因此轨迹不严格连续。更关键的是，D1 使用了训练后模型的信息，是 oracle/hindsight rank 诊断，不是训练前可部署的 rank 选择器；同时缺少同初始化、同预算 fixed-r16 control。

### 得到什么

D1 支持“不同模块的可压缩程度不同”和“约半数参数仍能接近 r32”的假设，但不证明异构 map 优于统一低 rank。后续必须让 rank 依据训练前或少量 rollout 信号产生。

## 7. 阶段五：GradTop，从真实 RL 梯度寻找候选 span

### 为什么不继续使用 base top-r

RLPO 的完整 top-32 span 稳定，但 prefix ordering 与最终 DeltaW 不匹配。于是改用小批 rollout 的 RL 梯度探测更新方向，而不是按 base singular value 排序。

第一版 random-B/GradTop 使用：

```text
A_probe = 0
B_probe = R,  R_ij ~ N(0, 1/r)
grad_A = R^T G
S = grad_A^T = G^T R
```

累计 12 个 observation 的输入侧 sketch，SVD 后将右子空间写入正式 LoRA-A，正式 B 清零，A/B 均继续训练。这样在不显式保存每个全量梯度矩阵的情况下得到候选方向。

### 重要现象

95% energy 规则几乎没有压 rank：平均 rank `31.63`，196 个模块中 192 个为 r32。这说明宽 candidate span 容易找到，但不能据此声称实现了 adaptive rank。

其正式训练轨迹是：

| Step | Macro Avg@k |
| ---: | ---: |
| 50 | `0.391198` |
| 100 | `0.403326` |
| 150 | `0.404574` |
| 200 | `0.401638` |
| 250 | `0.399401` |
| 270 | `0.399411` |

这是本机最清楚的“初始化改变学习速度”证据之一：step50 已达到较高水平，step100--150 接近峰值，之后准确率平台化或轻微回落。与此同时训练 reward 仍可上涨，主要伴随 parse 改善、Pass@k 变化和输出缩短，而非 held-out accuracy 持续上升。

子空间分析显示 DeltaW 几乎完全留在 probe 初始化 span 内，但按初始化 atom prefix 截断时顺序接近随机。也就是说，GradTop 找到了有用的宽 span，却没有解决 span 内 atom ordering，更没有得到可靠的模块 rank map。

## 8. 阶段六：Stable-SNR 与 Dominant-atoms，为什么“更稳定”仍不够

### Stable-SNR

RL rollout 梯度高方差、跨 step 方向近似正交，因此 Stable-SNR 尝试用多窗口、裁剪、均值/方差和独立 validation 来选择稳定方向。

v2 使用 5 windows、15 observations，并约束到 uniform-r16 等预算。结果出现 attention/MLP 预算塌缩：attention 全为 r32，MLP mean rank 仅 `11.05`；held-out capture `24.82%`，对最终 DeltaW 的 retrospective capture 仅 `5.10%`，因此没有启动正式训练。

v3 使用 5 discovery、2 calibration、2 validation windows，mean rank16、范围 8--32。即使 rank32，validation capture 也只有 `21.59%`；选中 map capture `11.75%`。fixed-A 方案因此被 gate 拒绝，trainable-A 只做了 50-step pilot，没有 full-bench。

这里的负结果说明：低跨窗口重合不一定意味着初始化无用。它也可能意味着有多个近似等价的可学习子空间，或者小批 RL 梯度只覆盖局部 trajectory。用“跨窗口稳定”作为唯一效用代理会过度保守。

### Dominant-atoms

随后把目标从重建完整高方差梯度，改成保留每个窗口最强的少数局部方向：每窗口 top-2 atoms，总预算等价 uniform-r8，得到 mean rank `10.67`、范围 `4--16`。

step50 训练 reward 为 `0.35949`，full-bench Macro 为 `0.373654`。低 rank 确实能训练，但 allocator 的反事实检验失败：

| Map | 最终 DeltaW capture |
| --- | ---: |
| heterogeneous map | `47.67%` |
| uniform-r8 | `54.49%` |
| greedy oracle | `60.39%` |
| family-shuffled map | `47.69%`，`p=0.605` |

rank map 与旧 rank@90 的 Spearman 为 `-0.156`。这说明被选中的低秩 span 有学习价值，但“哪个模块应获得更多 rank”的排序没有通过检验。

### Compact 与 Chord 两个旁证

GradTop step100 后用 activation-rank 压缩到 mean rank `18.29`，只保留 r32 的 `60.77%` 参数。step150 Macro 为 `0.397868`，未压缩 GradTop 为 `0.404574`，差 `-0.67 pp`，95% CI 跨 0。它支持可压缩，不支持压缩提高准确率。

对 step100 到 step270 的 DeltaW 做 chord 外推，`lambda=1.25` 得到 Macro `0.403345`。准确率改进不确定，但 format 提高约 `0.93 pp`、输出缩短约 433 token。线性外推更适合作为行为控制，而不是梯度方向预测或准确率改进证据。

## 9. 阶段七：Full-gradient、协方差与 token mask

### 为什么去掉 random-B

random-B 实际观测的是 `G^T R`，输出侧被随机矩阵压缩，可能漏掉全梯度的主能量。Full-gradient probe 直接在 FSDP full-gradient 上构造候选，并把 discovery、calibration、audit 分开，减少同一数据既选方向又评价方向的偏差。

主要候选包括：

- `mean`：对分组 GRPO 梯度做稳健累计后取 top right singular vectors。
- `covariance`：用输入侧 sketch 估计跨 prompt 梯度协方差，寻找反复出现的方向。
- `hybrid`：组合 mean 与 covariance 的投影空间。
- `Gain-LCB`：用有符号的预期改进下界排序 atom。
- `stable_energy P*R`：结合模块内相对能量 P 和跨 prompt 稳定比例 R。

同一个正式 artifact 使用 16 discovery、16 calibration、8 audit valid prompts，共覆盖 80 prompts/640 rollouts。A 作为训练初始化，默认不是永久冻结；静态的是 probe artifact 和 rank map。

### 同分数、同参数预算的 rank 对照

为避免把 atom score 与 rank map 混为一谈，uniform-r8 与 adaptive-eq-r8 都使用 `P*R` 选择 atom，并匹配约 9.23M 参数：

| 指标 | uniform-r8 | adaptive eq-r8 |
| --- | ---: | ---: |
| reward@50 | `0.390625` | `0.406250` |
| reward AUC1:50 | `0.346875` | `0.357187` |
| 参数量 | `9,232,384` | `9,231,360` |
| mean rank | `8` | `14.2653` |
| calibration capture | `0.843433` | `0.848773` |
| audit capture | `0.838146` | `0.838414` |

Adaptive 的 reward AUC 高 `0.0103125`，但 audit capture 几乎完全相同；异步 rollout 也没有逐请求固定。这只能作为 allocator 筛选信号，不能证明 adaptive rank 显著优于 uniform rank。

### Token-mask covariance

考虑到 RL 更新可能主要来自少数高 surprisal 或关键答案 token，token-mask 方案保留每条 response 的 top 50% surprisal token，同时至少保留 128 token 和末尾 128 token。候选 rank32、Nyström sketch width40，adaptive 预算等价 uniform-r28，`r_min=16`、`r_max=32`。

9 月旧协议中的主要 full-bench 结果为：

| 方法 | Step | Macro | Parse | 平均长度 |
| --- | ---: | ---: | ---: | ---: |
| token-mask centered covariance adaptive eq-r28 | 100 | `0.382131` | `0.895557` | `7471.6` |
| full-gradient mean/Gain-LCB uniform-r32 | 150 | `0.375749` | `0.887417` | `6609.4` |
| centered covariance adaptive eq-r28 | 50 | `0.375451` | `0.870999` | `8295.7` |
| uncentered covariance adaptive eq-r28 | 100 | `0.369779` | `0.877759` | `7382.6` |
| random-B near-r32 | 50 | `0.367304` | n/a | n/a |
| full-gradient mean uniform-r8 | 50 | `0.358565` | `0.846164` | `8655.5` |
| random-B uniform-r8 | 50 | `0.358323` | n/a | n/a |
| SPAR adaptive eq-r8 | 50 | `0.348220` | n/a | n/a |
| windowed consensus uniform-r8 | 50 | `0.348194` | `0.845199` | `8598.9` |
| standard LoRA r8 | 50 / 100 | `0.323111 / 0.315783` | n/a | n/a |

最稳妥的解释是：在这一批旧协议中，探测初始化的 r8 明显高于同批 standard LoRA r8，说明方向选择很重要；centered covariance 也值得继续。但 centered/uncentered 同时改变过 mask scope 和 optimizer continuity，不能把差值单独归因于中心化。token mask、candidate 和 allocator 同时变化时，同样不能声称 token mask 的独立贡献。

## 10. 阶段八：Phase-0 到 Phase-1，把方法冻结成可审计实验

### Phase-0/0.5/0.6 的作用

前述实验最大的问题不是没有正信号，而是变量太多。Phase-0 因此先筛 candidate source：P3 未同时改善 held-out capture 与稳定性，判定 no-go；P2 成为唯一 validated source。Phase-0.5 的 P2+P3 hybrid H2/H4/H8/H16 均未通过 calibration gate。Phase-0.6 crossfit replay 的 basis/atom-score invariants 大面积失败，诊断为 `crossfit_estimator_failure`，相关 C candidates 禁止进入训练。

这些失败记录很重要：最终 Phase-1 只使用被冻结并审计过的 P2，而不是从大量候选中事后挑最高分。

### Phase-1 的设计

四组均使用 uniform-r32、alpha64、dropout0、50 optimizer steps，196 个模块和完全一致的训练/评测协议。每个 LoRA-A 都是 32 维正交 basis，B 为零初始化，A/B 均训练；唯一变量是 P2 signal directions 与随机正交补空间的组成：

| 方法 | Signal | Random complement | 角色 |
| --- | ---: | ---: | --- |
| I0 | 0 | 32 | matched random-orthogonal baseline |
| I8 | 8 | 24 | mixed candidate，seed42 screen 后选中 |
| I16 | 16 | 16 | mixed negative screen |
| I32 | 32 | 0 | signal-only endpoint，不参与原 mixed selection |

这样设置不是 adaptive rank 实验，而是先隔离“梯度 signal 是否适合构造初始化 span”。所有方法 rank 和参数量完全相同。

### 50-step 主结果

| 方法 | Seed42 | Seed43 | 两-seed mean |
| --- | ---: | ---: | ---: |
| Base, no RL | `0.324348` | n/a | n/a |
| Matched standard LoRA r32 | `0.330053` | `0.338761` | `0.334407` |
| I0 | `0.330162` | `0.327747` | `0.328955` |
| I8 | `0.366161` | `0.341696` | `0.353928` |
| I16 | `0.313498` | 未运行 | n/a |
| I32 | `0.376479` | `0.333201` | `0.354840` |

关键比较如下：

| 比较 | Seed42 | Seed43 | 判断 |
| --- | ---: | ---: | --- |
| I8 - I0 | `+0.035999` | `+0.013949` | 两 seed 为正，但 seed43 仅 3/6 tasks 为正 |
| I32 - I0 | `+0.046317` | `+0.005454` | seed43 paired CI 跨 0 |
| I8 - standard LoRA | `+0.036108` | `+0.002934` | held-out seed 优势很小 |
| I32 - standard LoRA | `+0.046426` | `-0.005560` | 跨 seed 发生反转 |
| I8 - I32 | `-0.010318` | `+0.008494` | signal-only 与 mixed 排序反转 |

Seed43 的 paired bootstrap 中，I8-I0 为 `+0.013949`，95% CI `[0.001000,0.027188]`；I32-I0 为 `+0.005454`，CI `[-0.005884,0.016747]`。I8 虽然宏平均为正，但只有 3/6 benchmark delta 为正，没有满足预注册的至少 4/6 gate。

I0 从 seed42 到 43 仅变化 `-0.002416`，I8 和 I32 分别下降 `-0.024465/-0.043278`。代码、allocation 和 subspace hash 已核对一致，因此更符合“signal-informed initialization 改变 on-policy trajectory，并放大 training-seed 交互”，而不是统一设备偏移。

Matched standard LoRA seed42 在 step41 后中断，从完整 step25 checkpoint 恢复并重算 26--41；model、optimizer、extra state、RNG、scheduler 和 dataloader state 都恢复。其最终结果 verified，但引用时必须披露这一恢复边界。

## 11. Step-25 轨迹：收益发生在什么时候

为区分早期加速和最终性能，使用 seed43 已有 checkpoint，在完全相同的六任务 snapshot 上评测 standard LoRA、I8、I32 的 step25。三组均为 7248 records，且通过 postverify。

| 方法 | Step25 | Step50 | 25 -> 50 | Parse 25 -> 50 | Length 25 -> 50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard LoRA | `0.313818` | `0.338761` | `+0.024943` | `.7943 -> .8153` | `9207 -> 8940` |
| I8 | `0.328308` | `0.341696` | `+0.013388` | `.8031 -> .8364` | `8839 -> 8471` |
| I32 | `0.335436` | `0.333201` | `-0.002235` | `.8086 -> .8248` | `8943 -> 8353` |

相对 standard LoRA：

| 方法 | Step25 delta | Step50 delta | Difference-in-differences |
| --- | ---: | ---: | ---: |
| I8 | `+0.014490` | `+0.002934` | `-0.011556` |
| I32 | `+0.021618` | `-0.005560` | `-0.027178` |

I8-standard 在 step25 的 paired 95% CI 为 `[-0.000852,+0.031424]`，step50 为 `[-0.010343,+0.016290]`；都不能称为显著最终优势。I32 在 step25 最领先，随后略回落；普通 LoRA 后 25 步继续稳定提升并追近。

因此当前最有根据的机制描述是：signal 初始化给模型一个有利的早期 trajectory bias，并更快改善某些任务和生成行为，但没有证据说明它提高了 50-step 可达上限。全 signal 的 I32 早期最激进，也最容易在后续或不同 seed 中反转；I8 的 random complement 可能提供缓冲，但现有两个 seed 仍不足以证明它稳定优于 I32。

## 12. 外部模型检查：Qwen3-4B 到了哪一步

### 为什么先跑 standard LoRA

在 1.5B 上 signal 初始化存在明显 seed interaction，因此不能直接把 I8 搬到第二个模型后只看一个结果。先在 Qwen3-4B 上完成 no-RL base 与同协议 standard LoRA，确认训练栈、资源和可学习量级，再决定是否为 252 个模块重新生成 P2 signal artifact。

### 配置与工程过程

Qwen3-4B standard LoRA 使用 8 GPU、rank32、alpha64、dropout0、252 个 all-linear 模块、66,060,288 个可训练参数，50 steps，其他 GRPO 设置与主协议一致。标准 PEFT 初始化为 A Kaiming uniform、B zero。

4-GPU 首次尝试在 step0 被 SIGTERM，随后 8-GPU 两个版本在 vLLM wake-up 阶段触发主机内存上限，Ray 记录约 498--502/503.7 GB 使用。最终使用冻结的 safetensors 模型资产、8 GPU 和 NFS checkpoint 路径完成 50 步，耗时约 26 小时 15 分。前述失败是资源/加载路径诊断，不是算法结果。

### 结果

| 模型 | Macro Avg@k | Macro Pass@k | Parse | Hit-max | 平均长度 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B base | `0.643853` | `0.773400` | `0.980546` | `0.025248` | `9481` |
| standard LoRA step50 | `0.652255` | `0.777676` | `0.981926` | `0.022075` | `9219` |

paired Macro delta 为 `+0.008402`，problem-bootstrap 95% CI `[+0.001514,+0.015556]`，`P(delta>0)=0.9920`。逐任务并不统一：AIME24/AIME25/AMC23 为正，HMMT/MATH500/Minerva 为负。

这只证明 Qwen3-4B 的 standard-LoRA baseline 在当前单 seed 下可运行且有小幅正向变化。它不证明 I8 跨模型迁移，也不提供 multi-seed robustness。当前本机尚没有 Qwen3-4B 的新 P2 probe、I0 或 I8 正式结果。

## 13. 把全部实验串起来后的核心认识

### 13.1 初始化 span 是当前最可靠的正结果

RLPO 和多轮 gradient-probe 都显示，LoRA-A 初始化 span 在训练中高度稳定，主要学习发生在 B/left side 的重组。GradTop、旧 r8 probe 和 Phase-1 step25 又共同说明，好的初始化方向能明显加快早期学习。因此“先用少量 RL rollout 发现候选 span”值得保留。

但 span 并不唯一。Stable-SNR 的低跨窗口 overlap 与 GradTop 的强训练效果可以同时成立：不同小批梯度可能给出不同但功能近似的子空间。不能把几何稳定性直接等同于训练效用。

### 13.2 谱可压缩，不等于 allocator 已经解决

RLPO 最终 DeltaW effective rank 约 17，前 8 个奇异分量占 66.87%；Compact 也能用约 60.77% 参数保持接近表现。这些都是“最终更新可压缩”的证据。

但 base singular-value prefix、GradTop atom prefix、Stable-SNR map 和 Dominant heterogeneous map 都没有稳定给出正确 ordering/allocation。当前最缺的不是“是否存在低秩解”，而是训练前如何判断哪些原子和模块值得预算。

### 13.3 动态裁剪不是当前第一优先级

AdaLoRA 将 importance、正交损失、裁剪调度和 rollout 同步绑在一起，既难归因，也容易在 RL 中让正则项主导。现有结果更支持先做静态、可审计的 probe + allocation，再考虑低频重分配；没有可靠 atom score 时，加入 regrowth 只会扩大噪声和同步开销。

### 13.4 训练 reward 不能替代 held-out full-bench

GradTop 后期 reward 上升而准确率平台，centered/uncentered 的训练 reward 难以区分但 full-bench 不同，Dominant 的训练 reward 也不能验证 allocator。后续选择方法至少要联合看：held-out Macro、逐任务方向、parse、length、hit-max 和 paired interval。

### 13.5 初始化收益与 on-policy trajectory 强交互

Phase-1 中 I0 跨 seed 接近，而 I8/I32 对 seed 更敏感；Step-25 又显示 signal 方法早期领先但被普通 LoRA 追近。这说明 signal init 并非简单地把模型放到一个恒定更优的静态点，而是在改变早期采样分布，随后通过 on-policy 数据反馈放大或消解差异。

## 14. 当前可以和不可以声称什么

### 可以声称

- 在本机旧 r8 协议和冻结 Phase-1 中，RL-gradient-informed 初始化能改变并经常加快早期学习。
- RLPO/GradTop 的初始化 A span 在训练中基本稳定，B/left-side 重组承担主要功能更新。
- 最终 DeltaW 存在明显谱能量聚集，静态压缩具有可行性。
- 不同模块的可压缩程度确实不同，但现有训练前 allocator 尚未证明优于同预算 uniform rank。
- signal 初始化的收益对 training seed、训练阶段和 benchmark 敏感。

### 不能声称

- AdaLoRA 在当前 RL 设置下已有一个有效正式分数。
- I8 或 I32 稳定优于 matched standard LoRA。
- random complement 明确优于 signal-only，或反过来。
- token mask、centered covariance 或 adaptive rank 的独立因果贡献已经被隔离。
- Qwen3-4B 已验证 signal initialization 的跨模型泛化。
- 训练 reward 更高就意味着六任务准确率更高。

## 15. 由现有证据自然导出的下一步

下一轮不应再同时修改 probe、atom score、rank map 和训练动态。最小而有信息量的顺序是：

1. **先固定候选 span**：保留已经审计过的 full-gradient/P2 框架，使用 discovery、calibration、audit 三分割，不复用 audit 选超参。
2. **单独研究 atom ordering**：在 uniform rank 下比较 mean、centered covariance、局部 dominant 和 signal/random mixture，先问 prefix 是否优于随机 prefix。
3. **排序通过后再研究 rank allocation**：用完全相同的 atom score 比较 uniform-r8/r16 与等参数 heterogeneous map；不能再让 score 和 rank map一起变化。
4. **静态 map 有优势后再做训练时 adaptation**：优先低频、带 held-out gate 的 reallocation，而不是每步 importance 裁剪；显式计入 full-gradient、SVD、FSDP gather 和 vLLM sync 开销。
5. **Qwen3-4B 需要 fresh probe**：252 个模块不能复用 1.5B 的 P2 directions。先完成 Qwen-specific probe 和 I0/I8 配对，结果只在 Qwen3-4B 内比较。

从论文叙事看，目前最稳的主线不是“已经解决 adaptive rank”，而是：RL 梯度信号可以构造有效的初始化 span并产生早期加速，但 candidate discovery、atom ordering 和 rank allocation 是三个不同问题；on-policy trajectory 又会放大初始化与 seed 的交互。这个结论既解释了 GradTop/RLPO 的成功，也解释了 AdaLoRA、Stable-SNR 和 Dominant allocator 的失败。

## 16. 主要证据入口

- 方法与结果登记：[method_and_results_registry_zh.md](method_and_results_registry_zh.md)
- 时间线：[experiment_timeline_202607_202610_zh.md](experiment_timeline_202607_202610_zh.md)
- 9 月梯度子空间实验：[local_gradient_subspace_experiments_2026_09_zh.md](../local_gradient_subspace_experiments_2026_09_zh.md)
- Full-gradient probe：[full_gradient_rl_probe_v1_zh.md](../full_gradient_rl_probe_v1_zh.md)
- RLPO/Oracle/Gradient 总结：[report.md](../../runs/analysis/rlpo_oracle_gradient_probe_experiment_summary_20260921/report.md)
- AdaLoRA 复盘：[report.md](../../runs/analysis/adalora_experiment_retrospective_20260914/report.md)
- Phase-1 方法矩阵：[method_and_results_matrix_20261005.md](../../runs/phase1-signal-random-v1/analysis/method_and_results_matrix_20261005.md)
- Step-25 postverify：[verification](../../runs/step25-trajectory-v1/analysis/verification)
- Qwen3-4B paired comparison：[comparison report](../../runs/qwen3-4b-standard-lora-8gpu-safetensors-v3/analysis/comparisons/qwen3_4b_standard_lora_r32a64_b64m16n8_8gpu_safetensors_step50_seed42_v3_fullbench_32768_seed42_vs_qwen3_4b_base_fullbench_32768_seed42_v1.md)
