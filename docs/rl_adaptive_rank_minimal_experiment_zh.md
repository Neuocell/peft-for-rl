# 面向 RL 的自适应 Rank LoRA：诊断优先的最小实验设计

## 1. 当前结论

第一轮不直接实现完整的动态 rank 算法。先回答三个更基础的问题：

1. RL 更新是否真的表现出稳定的模块间 rank 差异？
2. 某个 batch 上选出的 rank，能否预测另一个 batch 上的函数扰动？
3. rank 信号是否稳定到足以执行硬裁剪，而不是追逐 policy-gradient 噪声？

只有三个问题都得到肯定答案，才值得加入动态 mask、KL 保护和 regrowth。

## 2. 已有离线证据

### 2.1 GeoRA 连续轨迹

GeoRA rank-16 step20-160 的完整诊断表明：

- raw `B_t A_t` 的 step20/160 cosine 为 0.999995，原始初始化子空间几乎不动；
- 真正训练差分 `B_t A_t - B_20 A_20` 持续增长；
- 相邻 20-step move 的 cosine 只有 0.049-0.108；
- 上一个 move 的 top-8 双侧子空间只能覆盖下一个 move 的约 1.0%-1.8%；
- left/right 稳定性高度不对称，MLP expanded dimension 一侧旋转最快。

因此不能从 raw GeoRA atom 的幅值或稳定性判断真实训练重要性。

### 2.2 普通 LoRA 和 RLPO 端点对照

| 轨迹 | 差分区间 | 差分 effective rank | rank@95 | rank@99 | top-8 energy | raw left/right overlap@16 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| GeoRA r16 | 20->40 | 23.21 | 24.92 | 30.46 | 51.13% | 1.0000 / 1.0000 |
| PC LoRA r32 | 50->100 | 20.70 | 25.99 | 30.60 | 57.20% | 0.6709 / 0.9263 |
| RLPO-init r32 | 250->270 | 21.31 | 25.72 | 30.26 | 57.41% | 0.9364 / 0.9815 |

结论：两个 checkpoint 的差分接近 rank 32 是普遍现象，不能直接作为“应该保留
30 个 rank”的证据。差分混合了多个 batch、多个 optimizer update 和子空间旋转。
GeoRA 的特殊问题是 raw basis 被非零初始化大项锁住，而不是差分 rank 高本身。

### 2.3 已训练 RLPO 的绝对谱

RLPO rank-32 step270 的 raw `Delta W = BA`：

- mean effective rank：17.13；
- mean rank@95：24.44；
- mean rank@99：29.89；
- top-8 平均保留 66.87% energy；
- 各模块 mean rank@95 只在 23.39-25.79 之间；
- early/middle/late mean rank@95 为 23.09/25.63/24.64。

纯 Frobenius 谱只显示温和的异构性，而且倾向保留很高的 rank。仅凭它做阈值裁剪，
大概率只能把平均 rank 从 32 降到约 24-26，不能支撑复杂动态算法。

### 2.4 RLPO 初始化子空间到最终更新的偏移

用 base weight 的 top-32 右奇异子空间作为 RLPO step-0 anchor，对 checkpoint-250/270
的全部 196 个模块重新计算 principal overlap。更早的 step50-200 目录只有 `data.pt`，
完整 adapter 已被 checkpoint retention 删除，因此不能恢复早期轨迹。

step-270 的结果呈现出两个不同层次：

- 最终 `A` span 在初始 top-32 span 内的能量加权 overlap 为 99.9974%；
- 最终 `DeltaW` 右子空间 overlap 为 99.9975%，其 99.9970% Frobenius energy 仍在
  初始 top-32 span 内；
- `||AA^T-I||_F/sqrt(r)` 均值仅 0.0011，A 仍接近行正交；
- 按 `B_tA_t = B_tA_0 + B_t(A_t-A_0)` 直接分解，A 漂移项的全局 Frobenius norm
  仅为总更新的 0.709%，且 `B_tA_t` 与 `B_tA_0` 的 cosine 为 0.9999748；
- 但最终 top-4/8/16 与 base top-4/8/16 的 overlap 只有
  15.56%/25.14%/46.90%，平均 principal angle 为 70.21/63.60/47.43 度；
- base top-4/8/16 实际只能捕获最终 `DeltaW` 总能量的
  13.53%/26.03%/50.83%。

因此不能简单说“RLPO 主子空间没有变化”。准确结论是：rank-32 的大 span 几乎被
固定住，但 RL 通过 B 在该 span 内重新混合出与 base singular-value 排序明显不同的
主方向。对较小 rank 直接取 base top-r prefix 会预先排除 rank32-r 个后来被使用的
方向；这使 batch-informed 子空间探测值得做，但单个高方差 RL batch 是否足够稳定仍需
held-out mini-batch 诊断，不能由 hindsight step-270 结果直接推出。

完整结果位于 `runs/analysis/rlpo_init_r32_step250_270_vs_init_subspace/`。

## 3. 先做的诊断实验

### D1：Activation-weighted 截断诊断

对象使用已经训练好的 RLPO-init rank-32 checkpoint-270，不重新训练。

从一批 on-policy 或现有评测 response 中取固定 token 样本。对模块 `m` 的输入
activation `x_m`，比较普通奇异值能量和实际前向输出误差：

```text
R_m(k) = DeltaW_m - TruncSVD_k(DeltaW_m)

fro_tail_m(k) = ||R_m(k)||_F^2 / ||DeltaW_m||_F^2

act_tail_m(k) = E[||R_m(k) x_m||_2^2]
                / E[||DeltaW_m x_m||_2^2]
```

实现不保存大 activation。对 `DeltaW = U Sigma V^T`，hook 中只累计
`V^T x` 的 `r x r` covariance，因此每个模块的统计状态只有 `O(r^2)`。

比较以下三个 rank map：

- `r_fro(m)`：保留 95% Frobenius energy；
- `r_act(m)`：保留 95% activation-weighted output energy；
- 固定 rank 对照：取与 `r_act` 相同平均 rank。

需要输出：

- rank 的均值、标准差、分位数和模块/层分布；
- 两个独立 calibration half 得到的 rank map 一致性；
- `r_fro` 与 `r_act` 的差异；
- 每种 map 的参数量和理论 LoRA FLOPs。

这一步回答：任务访问到的 activation 子空间是否让不同模块呈现出比普通奇异值
更明显、更稳定的有效 rank 差异。

#### D1 实测结果（2026-09-12）

D1 已在 RLPO-init rank-32 checkpoint-270 上完成。输入是该 checkpoint 已有的
full-bench response，六个数学 benchmark 均衡抽取 48 条序列，固定分成 24 条
calibration 和 24 条 validation；每条 response 均匀采样 256 个 token，因此两半
各有 6144 个 token。统计覆盖全部 196 个 LoRA 模块，目标能量为 95%，rank bin
为 `{8, 12, 16, 20, 24, 28, 32}`。

activation-prefix map 的结果：

- mean/median/std rank：16.78/16/6.43，范围 8-28；
- 相对 rank-32 的参数量与理论 LoRA FLOPs：50.09%；
- calibration/validation rank MAE：0.29，exact match 92.86%，within-4 100%；
- rank 的 split-half Spearman：0.9871；
- validation 全局加权能量保留：98.1637%；
- validation 模块能量保留的 mean/min/p10：96.2922%/94.6174%/95.1164%。

相同参数量下，固定 rank-16 的全局加权能量保留更高，为 98.6774%，但模块级
mean/min/p10 只有 94.0860%/75.1605%/87.1520%。因此 activation map 没有在
全局能量指标上胜过 fixed-rank，却显著减少了少数模块被严重欠分配的问题。

模块差异也很明确：`q_proj` 和 `gate_proj` 的 mean rank 分别是 13.14 和 13.29，
`v_proj` 是 23.57；early/middle/late 分别是 15.83/19.09/15.07。普通 Frobenius
prefix 的 mean rank 则为 25.80，需要 rank-32 的 81.01% 参数量，说明仅看权重谱
过于保守。按 activation energy 重排分量的 oracle mean rank 为 16.51、参数量
48.85%，但它可能保留非前缀奇异分量，目前只作为诊断上界。

D1 的结论是：模块局部的 activation-rank 信号稳定且确实异构，但“保护最差模块”
是否能转化为更小的策略扰动，不能由 adapter-output energy 直接推出。下一步应先做
D2 的 token KL/logprob 对照；在 D2 前不据此启动裁剪训练。

完整产物在
`runs/analysis/d1_rlpo_r32_step270_activation_rank/`，可复现入口是
`scripts/local/run_d1_activation_weighted_rank.sh`。

### D2：Held-out policy-distortion 验证

D1 的前半数据只用于选择 rank，后半数据用于验证，避免同一批数据自证。

把候选 rank map 临时折叠到 adapter，计算：

```text
token_kl = KL(pi_full-r32 || pi_truncated)
logprob_mae
top1_token_flip_ratio
PPO surrogate loss change
```

不需要为每个模块分别运行模型。每种完整 rank map 只做一次 teacher-forced
forward。第一轮只比较三种 map：`activation threshold`、`Frobenius threshold`、
`same-average fixed rank`。

判定依据不使用拍脑袋的绝对 KL 常数，而是与正常训练中一个 optimizer/global step
产生的 policy KL 比较：

```text
truncation_KL <= normal_update_KL
```

如果 activation map 在相同平均 rank 下不能稳定优于 fixed-rank，它就不值得进入
训练实现。

### D3：一个 GRPO batch 内的梯度稳定性

仅在 D1/D2 有正向结果后做。使用一个正常的 64-prompt、8-rollout GRPO batch，
分别记录 4 个 mini-batch 的 module-local projected gradient energy。

第一步不做完整 dense weight-gradient SVD，也不修改训练参数。只利用已有 LoRA
factor gradient 和小矩阵统计，检查：

- 同一个 global batch 内，4 个 mini-batch 的 rank 排序是否一致；
- activation rank 与 gradient-weighted rank 是否大体一致；
- 单 batch rank map 在另一半 trajectory 上是否仍能预测 loss/KL 扰动。

如果 4 个 mini-batch 之间的 rank map 都剧烈变化，就说明瞬时 importance、Fisher
或 gradient threshold 都不适合直接硬裁剪，后续必须使用跨 global-step EMA。

### D4：30-step observer-only 轨迹

只有 D3 的信号可用时，才运行一次 30-step RLPO rank-32 probe。它只记录，不裁剪：

```text
step 0 / 5 / 10 / 20 / 30:
  activation-weighted recommended rank per module
  rank-map change
  spectrum
  left/right projected novelty
  policy KL and reward
```

这个 probe 是最后一个诊断，不加入 gate、正则或 regrowth。它回答 rank map 的时间
尺度到底是一个 mini-batch、一个 global step，还是十几个 global step。

## 4. 第一版训练方法：一次校准的异构 Rank RLPO

第一版不是完整动态算法，而是验证核心假设的最小版本：

```text
Calibrated Heterogeneous-Rank RLPO (CHR-RLPO v0)
```

流程：

1. 从 base model 启动一个短的 rank-32 RLPO probe，运行 20 steps；
2. 使用 D1 验证过的 activation-weighted threshold，为每个模块独立生成 rank map；
3. rank 量化到 `{8, 12, 16, 20, 24, 28, 32}`，只设 `r_min=8`、`r_max=32`；
4. 丢弃 probe 权重，从同一个 base model 重新开始正式训练；
5. 每个模块用自己的静态 rank，并采用 RLPO 正交 A 初始化、B=0；
6. 正式训练期间不再改变 rank。

这里短 probe 只提供 rank map，不提供最终 adapter 权重或初始化子空间。这样第一版只
检验“RL 数据能否决定模块 rank”，不会把数据驱动子空间初始化混进同一个实验。

### 4.1 模块独立阈值

每个模块自己选择能使 activation-weighted tail energy 不超过 5% 的最小 rank：

```text
r_m = min { k : act_tail_m(k) <= 0.05 }
```

不设置全局 rank budget，不跨层排序，也不计算 AdaLoRA importance score。最终报告
实际平均 rank、总参数量和各模块 rank 分布。

### 4.2 固定缩放

不同 rank 模块必须保持相同 adapter scaling，避免把 rank 和更新尺度混为一谈。
基线 `r=32, alpha=64` 的 scaling 为 2，因此应使用：

```text
alpha_m = 2 * r_m
scaling_m = alpha_m / r_m = 2
```

不能对所有模块固定 `alpha=64`，否则 rank 8 模块的 scaling 会从 2 变成 8。

### 4.3 第一版明确不加入的内容

- 不加可学习 gate；
- 不加 L1、nuclear norm 或正交 loss；
- 不做周期性 SVD refactor；
- 不做训练时硬裁剪；
- 不做 KL reward/loss；
- 不做 regrowth；
- 不做全局 rank budget；
- 不同时引入 gradient-SVD 初始化子空间。

第一版只有一个变量：`per-module calibrated rank map`。如果它不能优于相同参数量的
固定-rank 对照，就应该停止这条分支，而不是继续增加动态机制补救。

## 5. 最小对照与验收

第一轮只需要两个训练对象：

| 实验 | 初始化 | rank | 作用 |
| --- | --- | --- | --- |
| Fixed-rank control | RLPO | 选择总参数量最接近的单一 rank | 控制参数/FLOPs 预算 |
| CHR-RLPO v0 | RLPO | calibration 得到的 `r_m` | 检验异构分配 |

已有 rank-32 RLPO 结果作为上界参考，不重跑。暂不做 rank 8/16/24/32 网格，也不做
多个 seed。

固定对照不能直接使用算术平均 rank，因为不同模块的 `d_in+d_out` 不同。应选择：

```text
r_fixed = argmin_r |sum_m r * (d_in_m + d_out_m)
                       - sum_m r_m * (d_in_m + d_out_m)|
```

同时报告两者实际参数量与 LoRA FLOPs 差异；无法完全相等时取最近的单一 rank，而不
为了精确配平再增加第二套复杂配置。

继续开发的最低条件：

1. calibration 两半得到的 module rank map 有明显一致性；
2. activation map 在同平均 rank 下的 held-out KL/loss 扰动小于 fixed-rank；
3. CHR-RLPO v0 的训练 reward、entropy 和长度没有异常；
4. 至少在一次正式评测中优于同参数量 fixed-rank control。

## 6. KL 与 Regrowth 的后续位置

KL 在后续版本中首先应作为 rank 操作的安全检查，而不是额外训练 loss：

```text
candidate rank change
  -> cached validation batch 上计算 policy KL
  -> KL 超过近期正常 update KL：拒绝裁剪或回退 rank
  -> KL 可接受：提交 rank change
```

这样每次 rank 变更只增加候选模型的一次 teacher-forced forward。若每 20 个 global
step 才检查一次，摊销开销约为一次 forward / 20 steps，而不是每步增加 reference
policy forward。

Regrowth 只在证明 rank map 会发生有意义的慢变化后加入。候选条件应是连续多个窗口
都出现的 projected residual，而不是一次梯度尖峰：

```text
EMA residual > high_threshold for P checks -> grow
EMA residual < low_threshold  for Q checks -> prune
```

`high_threshold > low_threshold` 构成 hysteresis，同时设置最短 rank-change interval。
实现上保留 `r_max=32` 的 dormant slots，只有确实需要 regrowth 时才重新初始化 B=0
和新的正交 A 方向。该部分需要 mask、optimizer-state 和 vLLM 同步支持，因此明确不
进入第一版。

## 7. 开销边界

| 组件 | 预期开销 | 第一版是否启用 |
| --- | --- | --- |
| compact `r x r` SVD | 很小，196 个 32x32 SVD | 仅离线校准 |
| activation covariance hook | 每模块 O(r^2) 状态，前向增加低秩投影 | 仅 D1/probe |
| 每种 rank map teacher-forced KL | 一次额外 forward | 仅 D2 |
| gradient-side hook | 一个正常 backward 上的小矩阵统计 | 仅 D3 |
| 动态 mask/gate | 每个训练 forward 和同步路径都有额外逻辑 | 否 |
| reference-policy KL | 额外模型或额外 reference forward | 否 |
| regrowth SVD/optimizer surgery | 周期性同步和状态迁移 | 否 |

诊断 hook 的目标是额外显存低于 1 GB、observer step 墙钟开销低于 5%。超过 10% 时，
即使信号有效，也需要先改成 token sampling 或更小的 covariance sketch，再考虑在线化。

## 8. 决策树

```text
D1 是否产生稳定异构 rank map？
  否 -> 停止 adaptive-rank；保留固定 rank RLPO
  是 -> D2

D2 是否优于同平均 rank 的固定-rank截断？
  否 -> rank proxy 无效，停止
  是 -> D3

D3 单 batch 梯度信号是否稳定？
  否 -> 不做瞬时动态 rank；先做 CHR-RLPO v0
  是 -> D4 observer-only trajectory

CHR-RLPO v0 是否优于同参数量 fixed-rank control？
  否 -> 停止增加复杂度
  是 -> 才进入带低频 KL guard 和 hysteresis regrowth 的 v1
```
