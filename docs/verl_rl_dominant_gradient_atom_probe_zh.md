# RL Dominant Gradient Atom Probe

## 1. 动机

此前 stable-SNR 探针优化的是独立窗口完整梯度 sketch 的 Frobenius 能量覆盖率。这个目标对
RL 过强：不同 prompt、rollout、token 和 advantage 产生的大量正交尾部方向会降低 held-out
capture，但这些方向未必决定策略收益。旧 rank-32 梯度初始化即使跨 step 梯度不稳定，仍然
显著加快早期训练，说明“初始化子空间解释全部梯度”不是必要条件。

下一版把目标改为：找到每个模块中少量、局部占优、跨窗口反复出现的梯度原子。探针只负责
提供有用起点和低秩容量分配，不要求冻结后的 A 能重构未来全部梯度。

## 2. 局部 dominant atoms

仍使用零函数随机投影：

```text
A_probe = 0
B_probe ~ N(0, 1 / p)
Y_w,t = G_w,t^T B_probe,w
```

窗口内固定 `B_probe,w`，对两个或三个 batch 做 robust mean，得到 `Y_w`。每个模块只对
`d_in x p` 的薄矩阵做 SVD：

```text
Y_w = U_w Sigma_w V_w^T
```

不再保留全部 p 个方向。默认只取前两个，并记录：

```text
dominance_i = sigma_i^2 / sum_j sigma_j^2
gap_i       = sigma_i / max(sigma_{i+1}, eps)
agreement_i = median_t capture(u_i, Y_w,t)
```

原子权重为：

```text
local_weight_wi = dominance_i * agreement_i * clip(gap_i, 1, gap_max)
```

`agreement_i` 防止单个异常 batch 的大奇异值进入候选集。原子只在方向意义上使用，因此符号
不影响结果。

实现中还显式计算跨窗口支持度：

```text
support_wi = mean_{v != w} max_j (u_wi^T u_vj)^2
weight_wi  = local_weight_wi * max(support_wi, 0.05)^2
```

平方支持度使一个窗口内幅值很大、但在其他窗口不复现的方向受到强降权。`0.05` 下限只保留
少量探索权重，避免所有局部新方向被数值上完全删除。

## 3. 跨窗口投影共识

对每个模块累计低秩投影矩阵：

```text
C = sum_wi weight_wi * u_wi u_wi^T
```

实际实现不构造 `d_in x d_in` 矩阵，而是把加权原子拼成薄矩阵
`Z=[sqrt(weight_wi) u_wi]`，再对 Z 做小型 SVD。`C` 的特征值不表示梯度总能量，而表示
一个方向被多个窗口重复支持的程度。

rank 的依据改为 recurrence spectrum：

```text
smallest r such that sum_{i<=r} lambda_i / sum_i lambda_i >= tau_consensus
```

第一版候选 rank bins 使用 `{4, 8, 12, 16}`，目标等价 rank 8 或 12，不再默认从 8 一直
放到 32。若有效原子不足，剩余维度从 pooled dominant span 补齐；不使用坐标轴式的任意
正交 completion。

rank 分配使用独立 calibration window 的 dominant capture，并按每增加一个 rank 所需的
`d_in + d_out` 参数成本进行全局贪心分配。每个模块独立竞争预算，不启用按模块类型强制均分。
validation window 完全不参与 basis、排序或 rank 选择，只用于最终诊断。

## 4. 校准指标

校准和验证不再用完整 sketch 能量，而用 held-out dominant energy：

```text
dominant_capture_w(r) =
    sum_{i<=k_local} sigma_i^2 ||A_r u_wi||^2 /
    sum_{i<=k_local} sigma_i^2
```

同时保留旧 full-sketch capture 作为噪声/漂移诊断，但它不再作为训练启动门。需要报告：

- dominant rank-1、rank-2 capture；
- recurrence eigenvalue 与 eigengap；
- 原子跨窗口支持次数；
- calibration/validation dominant capture；
- full-sketch capture（仅诊断）；
- 每种模块类型和层段的 rank 分布。

## 5. 行为信号与 token mask

第一版先不直接修改训练 loss。探针窗口额外记录 reward variance、advantage RMS、parse/format
比例，并对 advantage 几乎为零的窗口降权。第二版再加入 token mask：

- constructive mask：高 `|advantage|` 且高 entropy/surprisal 的 response token；
- noise mask：低 `|advantage|` 或在不同 rollout 间高度冲突的 token；
- 初始化子空间从 constructive sketch 构造；
- noise sketch 只用于方向降权或软正交惩罚，不直接扩充 rank。

这比要求所有 token 梯度跨 batch 对齐更符合 GRPO：同一 response 的标量 advantage 会传播到
全部 token，但真正改变策略行为的通常只是少数决策 token。

## 6. 探测开销

建议第一版：

```text
probe width          = 8
window size          = 2
discovery windows    = 4
calibration windows  = 1
validation windows   = 1
total actor batches  = 12
local atoms/window   = 2
max exported rank    = 16
```

相对刚完成的 27-batch stable-SNR v3，rollout 开销下降约 56%。每个窗口只增加一次
`d_in x 8` 薄 SVD，CPU 开销可忽略。

最重要的是落盘原始小对象：

```text
window_sketches.safetensors
window_atoms.safetensors
window_metrics.json
```

这样 `tau_consensus`、local top-k、rank bins、全局预算和模块类型预算可以离线重算，不再为
每个设置重新进行 rollout。

## 7. 50 步训练判据

短程训练每 10 步保存。不能只看单步 reward，使用 5-step rolling mean，并与旧 rank-32
gradtop 轨迹对齐：

- reward/accuracy 上升速度；
- parse success、boxed rate；
- mean/P90 response length 和 hit-max ratio；
- entropy、grad norm、PG loss、clip fraction；
- `span(A_t, A_0)` 和 `||B(A_t-A_0)|| / ||BA_t||`；
- DeltaW 谱的 top-1/top-2 能量和有效秩。

解释规则：

| 训练结果 | A 旋转 | 结论 |
| --- | --- | --- |
| 快 | 小 | 少量初始化原子已足够，适合继续降 rank 或尝试 fixed A |
| 快 | 大 | rank 容量有效，但初始方向需要在线修正 |
| 慢 | 大 | dominant atom 选择仍不准确 |
| 慢 | 小 | rank/优化容量不足，或 B 在固定尺度下难以起步 |

只有短程行为结果和 A 旋转共同决定是否继续到 100/270 步。完整梯度 held-out capture 不再单独
否决训练。

## 8. 已实现入口

探测方法已作为 `gradient_probe_method=dominant_atoms` 接入现有 verl actor optimizer 边界。
默认第一版配置为 4 个 discovery、1 个 calibration、1 个 validation window，每个 window
包含 2 个 actor batch，总计 12 步：

```bash
bash scripts/local/start_gradient_probe_dominant_atoms_4gpu.sh
```

artifact 生成后，等效 rank-8、A/B 均可训练的 50-step 实验入口为：

```bash
bash scripts/local/start_dominant_atoms_trainable_a_4gpu.sh
```

训练入口要求探测目录中同时存在 `rank_map.json` 和 `subspaces.safetensors`，缺失时会在启动
模型前退出。默认保持 `alpha/r=2`、global batch 64、mini-batch 16 和每 prompt 8 rollouts。
