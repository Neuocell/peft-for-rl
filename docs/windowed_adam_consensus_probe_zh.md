# Windowed Adam Consensus Probe 设计

## 目标

该实验只检验“初始化子空间是否具有 RL 泛化性”，第一轮固定使用 uniform-r8，
不同时引入自适应秩。训练设置保持历史强基线的 global batch 64、mini batch 16、
每个 prompt 8 条 rollout、seed 42。

## 梯度估计

仍为每个 prompt 生成 8 条 rollout，并用完整组计算 GRPO advantage。Backward 时按
稳定 hash 均匀选择一条 response。若完整 prompt-group loss 为：

```text
L_p = -sum_i,t A_pit log pi_pit / D_p
```

则单 response 估计为：

```text
Lhat_p = -n * sum_t A_pi*t log pi_pi*t / D_p
```

其对 response 采样的条件期望等于 `L_p`。16 个 prompt 的估计再取平均形成一个
时间窗口梯度。因此实际 loss 缩放为 `n / 16`。不能把 rollout.n 改成 1，因为
当前 GRPO 的组内 advantage 会退化。

第一版不做启发式 token 丢弃。后续若加入 token 筛选，必须使用 Bernoulli inclusion
及 `1/q_t` 修正，并用未筛 token 的 held-out full gradient 审计。

## 候选方向

Discovery 使用 8 个窗口。对经过窗口级范数裁剪的 full-weight 梯度维护 CPU 虚拟
Adam 一、二阶矩，并做 bias correction。每个模块、每个窗口分别从 raw momentum
和 Adam update 中提取 4 个局部右奇异 atom。

局部 atom 的 recurrence 定义为它与其他非相邻窗口局部子空间的最大 squared cosine
的均值。按窗口内奇异值相对能量乘 recurrence 加权局部 projector，再取 SVD，导出：

- `raw_momentum`
- `adam_update`
- `consensus_hybrid`

每套候选均有 32 个正交 atom。这里的虚拟 full-weight Adam 不是 LoRA factor Adam
的严格等价物，因此三套候选都保留，不能只根据 Adam 空间自证。

## Cross-fit 评价

随后 2 个 calibration 窗口和 2 个 audit 窗口不再修改候选方向，并统一使用原始
full policy gradient 评分。对每个窗口计算 atom 投影能量以及模块内相对能量，保存
均值、标准误和 LCB。allocation utility 为：

```text
U_j = recurrence_j * max(P_mean_j - z * P_se_j, 0)
```

audit 不参与 atom 选择，仅用于比较三套候选的未来 capture 和 calibration-audit gap。
artifact 同时保存 `F/S/R/P/P_se/P_lcb/recurrence/U`，可在 CPU 独立校验和重分配。

## 第一轮实验

Probe 完成后分别生成三套 uniform-r8 allocation，比较 calibration/audit capture。
选择 audit capture 较高且 gap 较小的候选；没有明显赢家时使用
`consensus_hybrid`。训练时 A 从所选候选初始化并继续训练，B 为 0，所有模块 rank=8、
alpha=16，故有效 scaling 恒为 2。训练 50 步，在 step 25 和 50 保存 model、optimizer
和 extra state。
