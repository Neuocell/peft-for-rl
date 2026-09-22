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

比较 `stable_energy` adaptive 的**全局 rank 分配本身**时，uniform 也必须用相同的
`P*R` 分数选择每个模块的 top-8 atom。默认 uniform 是 `gain_lcb`，用于检验有符号
step-0 改进分数；二者不能当作只改变 rank map 的实验。匹配分数的对照可另行导出：

```bash
python scripts/analysis/build_full_gradient_uniform_allocation.py \
  --artifact-dir runs/analysis/full_gradient_signed_grpo_probe_v1_seed42 \
  --output-dir runs/analysis/full_gradient_signed_grpo_probe_v1_seed42/mean_uniform_r8_stable_energy \
  --candidate-method mean \
  --uniform-rank 8 \
  --uniform-utility stable_energy
```

训练这组匹配对照时设置 `LORA_RANK=32 LORA_ALPHA=64`，使 vLLM 的
`max_lora_rank` 与 adaptive 一致；`rank_pattern`/`alpha_pattern` 仍物理实现
uniform-r8，每个模块的有效缩放仍为 2。先前的 gain-LCB uniform 保留原来的
全局容量 8，属于独立筛选实验，不能仅改 rank map 后直接当成严格的 rank 对照。

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

## 2026-09-22 第一轮筛选

正式 full-gradient artifact 使用相同的 16 discovery、16 calibration、8 audit prompt，
共采样 80 prompt / 640 rollout，其中 209 条 boxed reward 为正。三个 uniform-r8 初始化
均使用这一个 artifact，各有 196 个模块、9,232,384 个可训练参数、`alpha/r=2`。

| 候选 | calibration F capture | reward AUC1:10 | reward@10 | 平均 step(s) |
|---|---:|---:|---:|---:|
| mean | 0.7574 | 0.38828 | 0.42188 | 197.49 |
| covariance | 0.7878 | 0.37734 | 0.39844 | 195.04 |
| hybrid | 0.7757 | 0.36719 | 0.42188 | 195.23 |

这些训练使用相同 data/PPO seed，但 vLLM 的异步采样没有 request-level 固定种子，
step-1 reward 分别为 0.53906、0.43750、0.43750，不能把单次 AUC 差异全归因于子空间。
排除 step 1 后，mean 与 covariance 的 step 2-10 平均 reward 分别为 0.37153 与 0.37066，
因此目前仅选 mean 进入 50-step 检查，不宣称已显著优于 covariance。

`gain_lcb` 全局 adaptive 虽用满 rank-8 参数预算，calibration F capture 只有 0.4011，
55/196 个模块停在 rank 2，25/196 个达到 rank 32，按预设条件暂停该配置。改用
calibration `P*R` 的 `stable_energy` 分配后，参数为 9,231,360，calibration F capture
为 0.8488，独立 audit F capture 为 0.8384；uniform 对应的 audit F capture 为 0.7490。
这里的 uniform 使用 `gain_lcb` 选择 atom，不能用它隔离 rank 分配的效果。

为隔离 rank 与 atom 选择的影响，离线计算相同 `P*R` top-8 uniform 的 calibration F
capture 为 0.84343、audit F capture 为 0.83815；adaptive 对应为 0.84877 和 0.83841。
原 `gain_lcb` uniform 的 audit F capture 只有 0.74903，因此与 adaptive 之间观察到的
大部分 audit capture 差异来自 atom 排序分数的改变，不能归因于全局 rank 分配。
匹配 `P*R` 的两组现已完成 50-step 训练，对照结果见下文。

### Gain-LCB uniform 50-step 探索基线

`full_gradient_mean_uniform_r8_50_seed42` 已完成 50 步：reward@20 为
`0.21875`，reward@50 为 `0.41406`，AUC1:20 为 `0.34570`，AUC1:50 为
`0.34750`，step 2-50 均值为 `0.34550`；平均 response 长度约 6,480、
entropy `0.91244`、step time `196.46s`，可训练参数为 9,232,384。
历史 random-B 日志前 50 步均值约 `0.35539`，但平均 rank 约 31.63，
与这里的 rank 8 不是等预算对照。不能从这个探索基线推断全局 rank 分配的收益。

### 同分数 uniform/adaptive 50-step 对照

两组读取同一个 full-gradient probe artifact，均按 calibration `P*R` 选择 atom，
vLLM 的全局 LoRA 容量均为 rank32/alpha64；实际每模块 `alpha/r=2`。
seed、GRPO/reward、batch、优化器和 rollout 配置一致。下表的 AUC 是逐步 reward
的算术平均，KL 是日志中的 PPO KL，并非相对 base policy 的 reference KL。

| 指标 | uniform-r8 | adaptive-eqr8 |
|---|---:|---:|
| reward@20 | 0.257812 | 0.273438 |
| reward@50 / boxed accuracy@50 | 0.390625 | 0.406250 |
| reward AUC1:20 | 0.341406 | 0.357422 |
| reward AUC1:50 | 0.346875 | 0.357187 |
| reward mean 2:50 | 0.345823 | 0.356505 |
| response length mean 1:50 | 6465.17 | 6453.71 |
| entropy mean 1:50 | 0.903619 | 0.900485 |
| PPO KL mean 1:50 | 0 | 0 |
| A/B 可训练参数 | 9,232,384 | 9,231,360 |
| 模块平均 rank | 8 | 14.2653 |
| 平均 step time (s) | 204.925 | 202.932 |
| actor 分配显存峰值/卡 (GiB) | 14.5304 | 14.6305 |
| 观测整卡显存峰值 (GiB) | 32.3447 | 32.2275 |

adaptive 的 AUC1:50 高 `0.0103125`，且参数少 1,024 个；其 196 个模块中
9 个 rank=2、13 个 rank=32、174 个位于内部。calibration F capture 为
`0.848773`，略高于同分数 uniform 的 `0.843433`。独立 audit F capture 分别为
`0.838414` 和 `0.838146`，仅差 `0.000268`，不能将训练差异归因于显著提升
的 held-out 梯度覆盖率。vLLM 异步 rollout 未固定 request-level 随机种子，
因此这是一轮筛选结果，不是可归因的显著性证据。

两组都正常完成 50 步，未见非有限指标、entropy 爆炸、同步故障或步时异常。
adaptive 未触发预设的 AUC 落后超过 0.02、预算/scaling 不一致、rank 集中在边界、
calibration capture 不高于 uniform 等停止条件；仍不直接扩展到 270 步，
下一轮需恢复历史训练的 global batch，检查小 rank 是否接近历史大 rank 基线。

adaptive 的 step-25/50 checkpoint 均包含四份 FSDP model shard、extra state 和
标准 PEFT adapter；step-50 adapter 的 392 个权重键仅含 LoRA A/B，逐模块
`alpha/r` 均为 2。真实 1.5B base model 的 FP32 短输入 merge 前后 logits
最大绝对误差为 `1.01e-4`。另用四卡 `resume_mode=auto` 只加载 step-50，
四个 rank 成功读取 model、RNG 和调度器，trainer 判定已达到总步数并退出，
没有额外更新。恢复检查日志为
`runs/full-gradient-v1/logs/verl/full_gradient_mean_adaptive_stable_energy_resume_check_seed42.log`。
和 uniform 一样，此配置只保存 `model + extra`，**不保存 optimizer state**；
上述检查证明模型状态可恢复，但不能据此声称中途恢复后优化器轨迹完全一致。

原始训练日志位于 `runs/full-gradient-v1/logs/verl/`，文件名分别为
`full_gradient_mean_uniform_r8_stable_energy_50_seed42.log` 和
`full_gradient_mean_adaptive_stable_energy_eqr8_50_seed42.log`。包含逐步指标的
完整对照为 `runs/full-gradient-v1/analysis/mean_stable_energy_uniform_vs_adaptive_50_seed42.json`。

### batch64 小 rank 与历史大 rank

上面的 full-gradient 50-step 筛选使用 global batch16，而历史 random-B energy
强基线 `/root/gradtop_probe12_r8to32_mean31p63_a2_b64m16n8_270_v1.log`
使用 batch64、mini-batch16、n8、actor/inference token budget 12288、
vLLM `max_num_seqs=256`。历史基线前 50 步 reward AUC 为 `0.355391`，
reward@50 为 `0.378906`，平均 rank 约 31.63；原 batch16 结果不能用于判断
rank8 在该训练配置下是否达到大 rank 水平。

下一组直接复用同一个 full-gradient probe artifact 的 mean 候选、模块内
gain-LCB top-8 atom；其 10-step 筛选在 mean/covariance/hybrid 中 AUC 最高。
`scripts/local/start_full_gradient_mean_gain_lcb_b64_50_4gpu.sh` 固定以上历史
训练参数、seed42 和 50 步，设置 vLLM 容量 rank32/alpha64，但实际每个模块的
rank map 均为 rank8/alpha16，A/B 参数量为 9,232,384，缩放恒为 2。
这组的主要验收目标是小 rank 的 reward AUC1:50 能否接近历史大 rank，
不要求先完成等预算 random-B 对照。历史实验与当前机器的异步 rollout
无法做到逐请求配对，单 seed 结果仍需谨慎解读。

## Checkpoint 与显存口径

gain-LCB uniform 的 step-25 checkpoint 有完整的 4-rank FSDP model shard 和 extra-state，
另导出标准 PEFT adapter（196 组 A/B，共 392 个权重键，不含 probe 元数据）。
在 CPU 上从实际 1.5B base model 加载该 adapter 后，FP32 短输入的 merge 前后 logits
最大绝对误差为 `2.29e-5`，均值为 `3.33e-6`。BF16 merge 的绝对误差更大，
因此不能拿 BF16 的逐 logit 相等作为验收标准。

stable launcher 的 checkpoint `save_contents`/`load_contents` 均为 `model + extra`：
模型、调度器和 RNG 可恢复，但未保存优化器状态，恢复训练时优化器会重新建立。
另以独立日志 `runs/full-gradient-v1/logs/verl/full_gradient_mean_uniform_r8_resume_check_seed42.log`
执行 4 卡 `resume_mode=auto`、目标 50 步的只加载检查：四个 rank 成功加载
`global_step_50` 的 model、RNG 和调度器，trainer 明确判定已达到 50 步并退出，
没有再执行 optimizer update 或改写原训练日志。
verl 日志的 `perf/max_memory_allocated_gb` 是 actor PyTorch 分配量；另外的
`record_gpu_memory.py` CSV 记录逐卡整卡占用，包含 vLLM。当前 gain-LCB uniform 的
整卡采样从训练中途开始，不能视为覆盖完整 50 步的峰值。
