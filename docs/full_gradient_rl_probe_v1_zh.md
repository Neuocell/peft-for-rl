# Full-Gradient RL Probe v1

## 目标

本实现移除旧梯度探针的输出侧随机投影。旧路径通过 `A=0, B=random` 观测
`G^T B`；v1 直接读取 all-linear base weight 的完整梯度 `G`，再构造静态 LoRA
右子空间。探针不执行 optimizer step，rollout policy 始终等于 base policy。

目标模块限定为 `q/k/v/o/gate/up/down`。probe 模式不创建 LoRA adapter，仅临时解冻这些
base weight；FSDP 使用 `use_orig_params=True`，每次 capture 后逐 leaf summon 完整梯度并立即
转移到 CPU accumulator。

## RL 探测信号

一个 prompt 及其全部 rollout 是一个统计样本。对同一 prompt 的 GRPO advantage 使用：

```text
L_p = -sum_i,t A_pi,t log pi(y_pi,t | x_p, y_pi,<t) / sum_i,t mask_pi,t
```

在 base policy 上 PPO ratio 为 1，clip 未激活，因此这是实际 step-0 policy-gradient 方向。
一个 group 内的 response 逐条 forward/backward，但共享完整 group 的 token-mean 分母；这样既
保持原始目标，又避免 8 条长 response 同时进入显存。

全模型梯度范数为每个 prompt 产生一个 clip scalar。所有模块使用同一个 scalar，避免逐模块
归一化破坏跨模块尺度。advantage RMS 近零的 prompt 不进入任何 split。

## Discovery

Discovery 同时计算两类统计：

```text
M_m = mean_p G_mp
C_m = mean_p G_mp^T G_mp
```

`mean` 候选是 `M_m` 的 top-r 右奇异向量。`covariance` 不显式构造 `C_m`，而是用固定输入侧
随机矩阵 `Omega_m` 流式累计：

```text
Y_m = sum_p G_mp^T (G_mp Omega_m) = C_m Omega_m
```

完成后用单遍 Nyström 分解恢复 top-r eigenspace。这里每个 `G` 的全部输出维都参与 `G^T G`，
随机矩阵只用于数值低秩分解，不是旧方法的 `G^T B_random` 输出观测瓶颈。

`hybrid` 对归一化后的 mean/covariance 投影矩阵求共同 top-r 子空间。一次 probe 同时导出三套
候选，后续比较不需要重新 rollout。

## Calibration 与 Audit

候选 `a_mj` 固定后，保存 discovery 与 held-out 的投影：

```text
b_D,mj  = M_D,m a_mj^T
gain_mj = mean_p <G_C,mp a_mj^T, b_D,mj>
LCB_mj  = gain_mj - z * standard_error_mj
U_mj    = max(LCB_mj, 0)
```

`gain` 是 `B=0` 时第一次 B 更新在 calibration loss 上的一阶预测改进。artifact 同时保存
`F/S/R/P`、gain、standard error、LCB、Adam 首步近似和 audit 对应指标。Audit prompt 不参与
候选或 atom 选择。

## 运行

```bash
bash scripts/local/start_full_gradient_rl_probe_4gpu.sh

python scripts/analysis/build_full_gradient_uniform_allocation.py \
  --artifact-dir runs/analysis/full_gradient_signed_grpo_probe_v1_seed42 \
  --output-dir runs/analysis/full_gradient_signed_grpo_probe_v1_seed42/mean_uniform_r8 \
  --candidate-method mean \
  --uniform-rank 8

python scripts/analysis/build_full_gradient_uniform_allocation.py \
  --artifact-dir runs/analysis/full_gradient_signed_grpo_probe_v1_seed42 \
  --output-dir runs/analysis/full_gradient_signed_grpo_probe_v1_seed42/mean_adaptive_eqr8 \
  --candidate-method mean \
  --allocation-mode adaptive \
  --adaptive-utility stable_energy \
  --uniform-rank 8 \
  --r-min 2

CANDIDATE_METHOD=mean \
ALLOCATION_DIR=runs/analysis/full_gradient_signed_grpo_probe_v1_seed42/mean_uniform_r8 \
bash scripts/local/start_full_gradient_uniform_r8_4gpu.sh
```

`candidate-method` 可取 `mean`、`covariance` 或 `hybrid`。三个 uniform-r8 实验必须读取同一个
probe artifact，并保持 `B=0`、A 可训练、`alpha/r=2`。

adaptive allocator 先给每个模块分配 `r_min` 个完整 atom，再按 utility/
`(d_in+d_out)` 在全模型范围分配剩余 atom。`gain_lcb` 使用 discovery 更新方向与 calibration
梯度的一致改进下界；`stable_energy` 使用模块内相对能量与跨 prompt 稳定比例 `P*R`，避免
原始层间梯度尺度直接支配全局分配。预算严格取同候选方法 uniform-r8 的 A/B 可训练参数量；
rank map 中每个模块仍写入 `alpha=2*rank`，因此有效缩放恒为 2。Audit score 只用于分配后的
泛化诊断，不参与排序。

旧的 random-B energy probe 需要在相同容量下比较。新导出的 `summary.json` 会保存每个模块
每个 rank 的 A/B 参数成本，可将其主方向统一截断为 rank 8：

```bash
python scripts/analysis/build_gradient_probe_uniform_allocation.py \
  --artifact-dir runs/analysis/rl_gradient_probe_b16n8_w8_cap64_e95_s6to12_v1 \
  --output-dir runs/analysis/rl_gradient_probe_b16n8_w8_cap64_e95_s6to12_v1/uniform_r8 \
  --uniform-rank 8
```

该转换只截断已有候选方向，不重新 probe；输出仍为标准 `rank_map.json` 和
`subspaces.safetensors`，并固定 `alpha/r=2`。
