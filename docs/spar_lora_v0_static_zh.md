# SPAR-LoRA v0-static 设计与实验说明

## 1. 目标和边界

SPAR-LoRA（Stable Probe-and-Allocate LoRA）v0-static 在训练开始前完成一次正样本探测和静态 atom 分配。对线性模块 `m`：

```text
Delta W_m = sum_j z_mj B_m[:, j] A_m[j, :]
```

`r_max=32` 只是离线候选 atom 数。`z_mj` 是离线确定的 0/1 选择；进入正式训练前，A/B 已按被选 atom 物理压缩，不保留连续 gate 或元素级 mask。训练期间 A、B 都可训练，B 从零开始，因此 step 0 的 `Delta W` 严格为零。所有模块都要求 `alpha/r=2`，与 rank32/alpha64 基线相同。

v0 明确不包含 AdaLoRA importance、正交 loss、固定 A、训练期 pruning/regrowth、错误 rollout、负 advantage token 或 destructive-space 约束。

## 2. 数据和正样本筛选

probe 使用 base policy 的现有 vLLM rollout 和 boxed reward。一个 rollout 只有同时满足以下条件才可进入 probe：

1. boxed reward 不小于 0.5；
2. `parse_success=True`；
3. 对应 prompt 尚未被选过。

每个 prompt 最多保留一个正 rollout。前 32 个唯一 prompt 用于 discovery，后 32 个用于 calibration，因此两个集合不重叠。正样本不足时继续采样；artifact 记录总 prompt 数、总 rollout 数、正 rollout 数、正样本率及所选 response 的长度分布。

对每个被选 response 单独执行 teacher-forced CE backward。loss 是 response mask 上的 token mean，而不是 token sum，避免长 response 仅因 token 更多获得更大权重。该路径不执行 optimizer step。

## 3. Discovery：候选子空间

probe 阶段使用 rank32 LoRA，初始化为 `A=0` 和确定性随机 `B`。因为 B 非零，`grad_A` 可作为全参数梯度右子空间的随机 sketch；因为 A 为零，初始 adapter 输出仍严格为零。对模块 `m`：

```text
grad_A_m^T / scaling = G_m^T B_m
```

每个样本 backward 后，只把该低秩 sketch 搬到 CPU，不长期保存完整参数梯度。所有模块 sketch 的联合 L2 范数作为样本范数；以历史样本范数的滚动中位数乘 `2.5` 为阈值，对整个样本使用同一缩放系数。这样保留模块间的相对比例，同时限制单个爆炸样本的影响。

低内存增量压缩复用 `GradientSubspaceAccumulator`。32 个 discovery 样本完成后，每个 all-linear 模块得到 32 行正交候选 A，并记录对应奇异值。随后把候选 A 安装到模型并将 B 清零，进入只评分、不再修改候选 A 的 calibration 阶段。

## 4. Calibration：held-out atom 评分

对 calibration 样本 `n` 和候选 atom `a_mj`：

```text
g_mj^(n) = G_mn a_mj^T
F_mj = mean_n ||g_mj^(n)||_2^2
S_mj = ||mean_n g_mj^(n)||_2^2
R_mj = S_mj / (F_mj + eps)
P_mj = F_mj / (sum_k F_mk + eps)
U_mj = P_mj R_mj
```

安装候选 A 且 B=0 后，`grad_B/scaling` 正好是 `G A^T`，因此无需构造完整 `G`。F 表示总梯度能量，S 表示跨样本一致信号，R 是稳定信号比例，P 是模块内相对能量，U 是 v0 allocator 使用的效用。F/S/R/P/U 全部保存，rank map 只是该原始评分 artifact 的一个可重算派生物。

## 5. Artifact 格式

probe 目录包含：

```text
probe_summary.json
candidates.safetensors
atom_scores.safetensors
```

`candidates.safetensors` 以规范化模块名为 key，tensor shape 为 `[32, d_in]`。`atom_scores.safetensors` 对每个模块保存 `F`、`S`、`R`、`P`、`U` 和 `singular_values`。`probe_summary.json` 保存模型模块 shape、正交误差、样本计数、prompt split、长度分布、CE loss、样本范数裁剪信息和文件路径。

`validate_spar_artifact()` 完全在 CPU 上加载 artifact，并检查：

- discovery/calibration prompt 不相交；
- 每个 A shape 正确且行近似正交；
- 所有 score shape 正确且无 NaN/Inf。

## 6. 静态 allocation

两个配置必须由同一个 probe artifact 一次生成。

### uniform-r8

每个模块选择 U 最大的 8 个 atom。模块 rank 恒为 8，alpha 恒为 16，scaling 恒为 2。

### adaptive-eqr8

令一个模块 atom 的训练参数成本为 `c_m=d_in+d_out`。预算为 uniform-r8 的精确 A+B 参数量：

```text
budget = sum_m 8 c_m
```

先为每个模块选择 U 最大的 `r_min=2` 个 atom，再将其余 `(module, atom)` 按 `U_mj/c_m` 全局降序扫描；只有加入完整 atom 后仍不超过预算时才分配。每个模块 rank 范围是 `[2, 32]`。每模块 alpha 设置为 `2*rank`，同时训练 launcher 使用全局 rank32/alpha64 作为 PEFT/vLLM 容量上限，保证 actor 和 rollout 都应用 scaling 2。

allocation 目录包含：

```text
rank_map.json
subspaces.safetensors
allocation_summary.json
```

summary 记录实际训练参数量、预算余量、rank mean/min/max、层段和 module family 平均 rank、全局 calibration energy/U capture，以及逐模块 rank、atom indices、shape 和参数成本。

## 7. verl/FSDP/vLLM 接线

`peft_type=spar_probe` 只负责采样和生成 artifact。trainer 在 boxed reward 后筛选样本；每个选中样本复制到所有 actor data-parallel rank。FSDP actor 在每个 rank 上用相同样本同步 backward，并通过 `summon_full_params(with_grads=True)` 读取完整 LoRA factor 梯度。calibration 完成后只由 global rank 0 原子写入共享 artifact，其余 rank 在 distributed barrier 等待文件发布完成。

正式训练复用 `peft_type=grad_subspace`：PEFT 根据 `rank_pattern` 创建不同物理 shape 的 A/B，加载选中的候选 A，并将 B 清零。A 不冻结。checkpoint 和 rollout 同步仍走已有 PEFT 标准 LoRA state dict，只包含 adapter A/B 和标准配置；probe 的候选、score 和 JSON 元数据不进入 rollout adapter。

## 8. 第一轮实验协议

共同设置来自 dominant-atoms uniform-r8 launcher，方法实验只改变 rank map 和实验名：

```text
data seed = 42
PPO dataloader seed = 42
CUDA_VISIBLE_DEVICES = 0,1,2,3
total steps = 50
save steps = 25, 50
```

运行顺序固定为 uniform-r8、adaptive-eqr8。训练日志至少保留 reward mean、boxed accuracy、response length、entropy、KL、policy loss、grad norm、clip fraction、step time、峰值显存、active parameter count 和静态结构指标。比较 reward AUC 1:20 与 1:50，并报告 step 20、step 50，而不是只看最终单点。

## 9. 验收和停止条件

训练前验证两组 step 0 输出与 base 在浮点容差内一致、每模块 scaling 为 2、adaptive 参数量不超过 uniform、A 行正交且 B 为零。训练链路验证标准 LoRA 权重同步、checkpoint 保存/恢复和 merge。

若 adaptive 发生任一情况，50 步后停止，不扩展到 270 步：

- reward AUC 1:50 比 uniform 低超过 0.02；
- rank 几乎全部卡在 r_min 或 r_max；
- 参数预算不一致；
- scaling 随模块 rank 改变；
- probe calibration capture 不高于 uniform；
- 出现 NaN、entropy 爆炸、rollout 同步异常或明显 step-time 异常。

## 10. seed42 第一轮结果

4 卡 L40S 环境中，stable LoRA 1-step smoke、SPAR uniform/adaptive 1-step smoke 和两组正式
50-step 训练均正常完成。probe 共检查 112 个 prompt、896 条 rollout，得到 315 条正 rollout；
最终选取 32 条 discovery 和 32 条 calibration，两个 prompt 集合无交集。196 个模块候选 A 的
最大行正交误差为 `1.97e-5`。

```text
                              uniform-r8       adaptive-eqr8
trainable parameters          9,232,384        9,230,848
rank min / mean / max         8 / 8 / 8        2 / 13.7398 / 32
calibration energy capture    0.860903          0.880820
U score capture               0.795626          0.865641
reward@20                     0.367188          0.382812
reward@50                     0.392578          0.402344
reward AUC1:20                0.356836          0.359180
reward AUC1:50                0.373477          0.374336
mean step time (s)            524.904           534.263
```

VERL 日志中的 `perf/max_memory_allocated_gb` 是 4 个 data-parallel actor rank 的聚合值；
汇总脚本保留 aggregate 原值，并在对照表中报告除以 world size 后的单 rank 峰值。两组分别为
13.1523 GiB 和 13.1526 GiB。

adaptive 的 AUC1:50 比 uniform 高 `0.000859`，同时少 1,536 个训练参数；rank 未集中卡在
`r_min/r_max`，capture 更高，scaling 始终为 2，训练无 NaN、entropy 或同步异常，因此本轮
未触发停止条件。step25/50 checkpoint 均保存成功；step50 adapter 只包含标准 LoRA A/B，
可以 safe merge，也已验证从 step50 恢复后不执行额外 update。
