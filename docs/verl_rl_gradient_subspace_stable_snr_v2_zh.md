# RL 梯度子空间探测 v3：独立验证、分组预算与固定 A 训练

## 目标

v1 使用每个 batch 独立随机的输出投影，并累计无符号 sketch 能量。它能够找到一个宽的候选
输入子空间，但不能可靠地区分跨 batch 一致的学习方向和 RL 高方差噪声，最终有 192/196 个
模块选择 rank 32。

v2 只保留跨 batch 符号一致的信号，并把方向选择和全局 rank 预算分开。但第一次实测出现
两个问题：同一批窗口同时影响方向和 rank 选择，且按绝对 capture/参数量做全局分配使所有
attention 投影取 rank 32、MLP 平均只有 rank 11.05。v3 将方向发现、rank 校准和最终验证
完全拆开，并在七种模块类型内部独立维持等价 rank-16 预算。

## 探针参数化

探针仍使用函数保持初始化：

```text
A_probe = 0
B_probe ~ N(0, 1 / probe_width)
DeltaW_probe = 0
grad_A / scaling = G^T B_probe
```

探针 optimizer 不更新参数。v3 使用 9 个窗口，每个窗口 3 个 actor batch；窗口内保持同一个
`B_probe`，窗口结束后再刷新。默认 `probe_width=8`，共 27 个观测。窗口严格分为：

- 5 个 discovery window：只构造候选 A 子空间；
- 2 个 calibration window：只选择每个模块的 rank；
- 2 个 validation window：只报告最终质量，不参与方向或 rank 选择。

5 个 discovery window 允许 leave-one-window-out 时仍保留 `4 * 8 = 32` 个候选方向。

每个窗口内先按 sketch Frobenius norm 做 robust clipping，然后构造：

```text
F_signal,w = sqrt(3) * mean_t(Y_w,t)
F_noise,w  = concat_t(Y_w,t - mean_t(Y_w,t)) / sqrt(2)
```

窗口之间再做一次 signal-norm clipping。最终只在 `F_signal` 的薄 QR span 中求解：

```text
S v = lambda (N + ridge I) v
S = F_signal F_signal^T
N = F_noise F_noise^T
```

得到的方向按广义特征值排序，再做保持 prefix span 的 QR 正交化，因此导出的每个 A 都满足
`A A^T ~= I`。

噪声协方差 ridge ratio 从 `0.05` 提高到 `1.0`，避免广义特征分解把“绝对信号极小、但估计
噪声更小”的方向错误排在前面。

## 独立 rank 分配与验证

discovery 窗口内部的 leave-one-out 只测方向稳定性。最终 A 仅由全部 discovery window 构造；
两个 calibration window 测量各 rank prefix 的 capture，中位数用于分配 rank。两个 validation
window 在选择结束后才测量 `validation_capture_selected`、`validation_capture_at_max_rank` 和
`validation_retention_of_max`，因此这些数值是严格样本外指标。

rank bins 为 `{8,12,16,20,24,28,32}`。所有模块先取 rank 8，再按下式逐级增加：

```text
marginal utility = normalized calibration capture gain / added LoRA parameters
normalized capture = capture(rank) / capture(rank=32)
```

预算不再允许不同模块类型互相借用。`q/k/v/o/gate/up/down` 各自在 28 层内部使用等价
uniform rank 16 的参数预算，因而仍可跨层自适应，但 attention 不会再挤占 MLP 容量。七组
预算相加仍精确等于全模型 uniform rank 16。这里的 16 是 `equivalent_uniform_rank`，不一定
等于异构分配后 rank 的算术平均。产物同时记录：

- 实际 `rank_mean/min/max`；
- `equivalent_uniform_rank` 与预算利用率；
- leave-one-window-out rank MAE；
- discovery LOO capture、calibration capture、独立 validation capture；
- fold overlap、signal/noise energy；
- batch/window clipping 次数和 SNR 特征值。

## 固定 A / B-only 正式训练

正式训练从 base model 重新开始：

```text
A = v2 导出的正交输入基底，并永久冻结
B = 0，只优化 B
DeltaW = (alpha / rank) * B A
alpha / rank = 2
```

固定 A 不妨碍 B 在该 span 内重组，只禁止输入/right subspace 旋转。启动时严格验证只有 B
可训练；optimizer 只接收 `requires_grad=True` 参数。adapter checkpoint 仍保存 A 和 B，恢复
训练时必须再次传入 `lora_freeze_a=true`。

## 启动入口

v3 探针：

```bash
bash scripts/local/start_gradient_probe_stable_snr_4gpu.sh
```

默认产物：

```text
runs/analysis/rl_gradient_probe_stable_snr_w5x3_c2_v2_pbudget16_v3/
  rank_map.json
  subspaces.safetensors
  summary.json
```

固定 A 正式训练：

```bash
bash scripts/local/start_stable_snr_fixed_a_4gpu.sh
```

正式训练脚本会在启动前检查两个探针文件，并保持原版 GRPO 的 batch、rollout、长度、LR、
seed、dropout 和 270 步配置不变。
