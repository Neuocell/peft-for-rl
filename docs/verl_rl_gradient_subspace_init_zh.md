# 基于 RL 累积梯度的 LoRA 子空间初始化

## 1. 第一版实验目标

这一版只回答一个问题：在不先训练一份完整 LoRA、也不依赖 base weight 主奇异空间的情况下，能否用少量真实 GRPO batch 找到稳定的 policy-gradient 子空间，并据此同时决定每个模块的 rank 和初始化方向。

实现分为两个完全隔离的阶段：

1. `grad_probe`：base policy 上运行 6 到 12 个小 batch，只采集梯度子空间，不更新模型；
2. `grad_subspace`：重新从 base model 启动正式 270 步训练，加载探针导出的异构 rank 和正交 `A`，令 `B=0`。

第一版不加入正交 loss、在线裁剪、KL 触发或 regrowth。这样结果可以直接回答“RL 梯度子空间初始化是否有效”，不会混入多套机制。

## 2. 为什么不用 dense base-weight gradient

对某个线性层记 policy loss 对原权重的梯度为

```text
G_t = d L_t / d W,  G_t in R^(d_out x d_in)
```

直接保存所有 `G_t` 等价于为 1.5B base model 保留 dense gradient，不符合 PEFT 的显存目标。激活协方差虽然便宜，但它只描述输入分布，不包含 advantage、reward、token mask 和 policy loss 的方向信息。

本实现给每个目标线性层临时加入宽度为 `w` 的普通 LoRA：

```text
A_probe = 0                       shape = [w, d_in]
B_probe ~ N(0, 1 / w)             shape = [d_out, w]
DeltaW_probe = B_probe @ A_probe = 0
```

由于 `A_probe=0`，adapter 输出严格为零，探针 policy 与 base policy 完全相同。反向传播给出：

```text
grad_A = scaling * B_probe.T @ G_t
Y_t = grad_A.T / scaling = G_t.T @ B_probe
```

因此只读取 LoRA `A.grad`，就得到 `G_t` 的输入侧随机子空间 sketch，无需构造 `G_t`。

当 `B_probe` 的元素方差为 `1/w` 时：

```text
E[B_probe @ B_probe.T] = I
E[Y_t @ Y_t.T] = G_t.T @ G_t
```

跨 batch 累积的是：

```text
C = sum_t Y_t @ Y_t.T  ~=  sum_t G_t.T @ G_t
```

这里刻意不累加带符号的 `sum_t G_t`。RL 小 batch 的 advantage 和采样会造成方向翻转，直接求均值可能把持续出现、但符号变化的高能方向抵消；二阶能量累积保留这些方向。

## 3. 探针阶段的数据与优化边界

默认配置：

| 项目 | 值 |
| --- | ---: |
| prompt batch | 16 |
| rollout / prompt | 8 |
| PPO mini-batch | 16 |
| PPO epoch | 1 |
| probe width | 8 |
| sketch capacity | 64 |
| 最少 trainer steps | 6 |
| 最多 trainer steps | 12 |
| seed | 42 |

`prompt batch == PPO mini-batch` 且 `PPO epoch=1`，所以每个 trainer step 只有一次 optimizer boundary。当前 policy 与 rollout policy 相同，GRPO loss 在 ratio 为 1 的位置提供真正的 on-policy policy gradient。

每次 optimizer boundary 的顺序固定为：

```text
all micro-batch backward
    -> FSDP 完成梯度归并
    -> 逐个 leaf FSDP layer 召回完整 LoRA A/B 和 A.grad
    -> 保存 Y_t = A.grad.T / scaling
    -> A 清零，B 换成下一组确定性随机投影
    -> 只计算/裁剪 grad norm
    -> 跳过 optimizer.step
    -> zero_grad
```

所以探针阶段没有任何参数更新。`B` 的随机 seed 只由模块名、probe seed 和观测序号决定，所有 data-parallel rank 使用相同投影。实现会检查 196 个 `all-linear` 模块是否每次恰好捕获一次；漏层或重复都会立即报错。

探针分支强制使用 FSDP1 `use_orig_params=True`，因为 PyTorch 只在该模式支持 `summon_full_params(with_grads=True)`。它关闭普通 LoRA 使用的 A/B 叶子 auto-wrap，改为按 transformer layer 包裹和召回。这个设置只用于短探针；正式 `grad_subspace` 训练仍沿用现有普通 LoRA FSDP 路径。

## 4. 增量子空间分解

每个模块在 CPU 保存一个最多 `capacity=64` 列的压缩因子 `F`，使：

```text
F @ F.T ~= sum_t Y_t @ Y_t.T
```

新 sketch 到来后，拼接 `[F, Y_t]`，先做 thin QR，再只对小矩阵 `R` 做 SVD：

```text
[F, Y_t] = Q R
R = U S V.T
F_new = Q U[:, :capacity] S[:capacity]
```

这避免对 `d_in x d_in` 协方差做特征分解，也不需要长期保存所有历史 sketch。

## 5. 每个模块独立选择 rank

对模块 `m` 的累计谱记为 `s_(m,1) >= s_(m,2) ...`。先找覆盖目标能量的原始 rank：

```text
r_raw,m = min r such that
sum_(i<=r) s_(m,i)^2 / total_sketch_energy_m >= 0.95
```

然后向上量化到：

```text
[8, 12, 16, 20, 24, 28, 32]
```

每个模块独立决定 rank，不做跨层 importance 排序，也没有全局 budget 强制重分配。这符合不同层、不同投影模块对 RL 更新需求不同的假设。日志会给出：

```text
gradient_probe/rank_mean
gradient_probe/rank_min
gradient_probe/rank_max
gradient_probe/retained_energy_mean
gradient_probe/retained_energy_min
gradient_probe/module_coverage
```

如果某个模块在 rank 32 下仍达不到 95%，它会保留 rank 32，并通过 `retained_energy` 明确暴露，而不是伪装成已经满足目标。

## 6. 稳定判定和自动停止

每个 trainer step 都比较当前累计子空间与上一步累计子空间。模块重合度为：

```text
overlap_m = ||V_prev.T @ V_now||_F^2 / min(r_prev, r_now)
```

同时计算逐模块 rank 的平均绝对变化。满足以下条件才记一个稳定窗口：

```text
trainer_step >= 6
module coverage == 100%
overlap 的模块 p10 >= 0.98
rank MAE <= 1.0
```

连续 3 个窗口满足即导出；否则最多运行到第 12 步。连续累计子空间的重合度是工程停止条件，不应被解释为统计显著性证明。若第一轮结果有收益，下一步应补独立 batch split 或 held-out gradient sketch 做 D2 验证。

## 7. 导出物和正式初始化

探针目录生成：

```text
rank_map.json
subspaces.safetensors
summary.json
```

`rank_map.json` 含每个 PEFT target 的 `rank_pattern`、`alpha_pattern` 和固定 `alpha_m/r_m=2`。`subspaces.safetensors` 中每个 tensor 的形状是 `[r_m, d_in]`。

正式训练重新加载 base model，构造异构普通 LoRA，并设置：

```text
A_m = V_m.T
B_m = 0
alpha_m = 2 * r_m
dropout = 0.05
```

`A_m A_m.T = I`，初始 `DeltaW=0`。训练后 `A/B` 都是普通可学习参数，不加正交约束，也不在线裁剪。它与 vLLM 的接口仍然是普通 A/B LoRA。

## 8. 计算和内存开销

探针 backward 相当于 rank-8 LoRA，而不是 dense base-model finetuning。对当前 Qwen 1.5B、28 层、每层 7 个 target 的近似开销：

- GPU：普通 rank-8 LoRA forward/backward；adapter 输出虽然为零，低秩算子仍会执行；
- FSDP：optimizer boundary 时逐个 transformer layer 召回 LoRA 参数和梯度，不召回整模型；
- CPU 长期 sketch：`sum(d_in) * capacity * fp32`，约 124 MiB；
- CPU 临时 sketch：probe width 8 时约 15.5 MiB/step；
- 分解：每个模块一次 `d_in x <=72` 的 QR 和小矩阵 SVD；
- rollout：16 prompts x 8 responses，最多 12 步，是主要墙钟开销；
- 正式 270 步训练：参数量和算力由导出的平均 rank 决定，不再有 QR/SVD 开销。

探针不迁移 optimizer state，因为正式训练从 base policy 和全新 optimizer 开始。

## 9. 启动方式

当前 0-3 卡上的 CHR 训练结束后，启动探针：

```bash
cd /home/wangls/peft-for-rl
tmux new-session -d -s rl_grad_probe \
  'bash scripts/local/start_gradient_probe_4gpu.sh'
tail -f runs/logs/verl/rl_gradient_probe_b16n8_w8_cap64_e95_s6to12_v1.log
```

检查 `summary.json` 的覆盖率、平均 rank、最低能量保留率与停止原因。确认 artifact 后启动正式训练：

```bash
tmux new-session -d -s rl_grad_subspace \
  'bash scripts/local/start_gradient_subspace_init_4gpu.sh'
tail -f runs/logs/verl/gradient_subspace_init_r8to32_mean_tbd_a2_b64m16n8_270_v1.log
```

两个启动脚本默认都绑定当前仓库、`peft-for-rl` conda 环境和 `runs/` 目录。正式训练脚本在 artifact 缺失时会在启动前失败，不会退化成随机 LoRA。

## 10. 第一轮判定标准

先不要求超过所有已有方法。第一轮只检查：

1. 探针每一步模块覆盖率为 100%，无 non-finite sketch；
2. 子空间是否在 12 步内达到稳定条件；
3. rank 分布是否非退化，且 rank 32 模块的 `retained_energy` 是否仍明显低于 0.95；
4. 正式训练初始 actor/rollout logits 一致，reward、length、entropy 和 grad norm 无异常；
5. 与固定 rank-32 RLPO 和 CHR 在相同步数下比较训练 reward 及后续 6-task 评测。

若没有可见收益，优先诊断 gradient sketch 的跨 batch 一致性和 probe batch 代表性，而不是立刻叠加在线裁剪、KL controller 或 regrowth。
