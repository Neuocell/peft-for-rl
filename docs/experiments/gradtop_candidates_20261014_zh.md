# GradTop 备选实验：方案审查与实现

日期：2026-10-14（Asia/Shanghai）。状态：实现与 CPU 验证；尚无本轮正式训练结果。

2026-10-15执行优先级已调整：先跑 [原聚合GradTop-r32干净对照](gradtop_r32_clean_baseline_20261015_zh.md)，Prompt-sign和Masked只保留为已实现备选。本文件记录原候选实现，不自动触发两条候选训练。

本轮选择 **Prompt-sign GradTop** 和 **Masked-GradTop**，各自作为独立方法，不叠加。
GradTop-Core 保留为下一轮候选。本文件给出选择理由、修正后的数学对象、冻结配置和运行入口。
训练与评测沿用仓库的 [RLVR_PROTOCOL.md](../../RLVR_PROTOCOL.md)，不以旧实验的绝对分数代替 matched 对照。

## 1. 历史依据及其边界

- 旧 GradTop 的 probe 保留 GRPO advantage，读取 optimizer 边界处的累计梯度；跨 observation 累计 sketch 二阶能量，而不是将全部梯度求均值后做一次 SVD。正式训练 A/B 均可训练。旧 step50/100/150 Macro 为 39.12%/40.33%/40.46%，mean rank 约31.63。依据：[本地实验记录第七阶段之前的 GradTop 记录](../local_gradient_subspace_experiments_2026_09_zh.md)、[历史方法审计](../../audits/ppt_method_trace_2026_10_09.md)。审计仍指出原 probe tensor 身份未确认；旧主轨迹不是本轮严格 matched 基线。
- Top-surprisal + uncentered/centered covariance 的 step50 Macro 约35.08%/37.55%，确有正式训练记录。旧配置同时改变 covariance、rank 分配、mask scope 和部分训练条件，不能单独归因于 mask。后续 cross-fit/stable-band 被诊断拒绝也不能否认先前 mask 正式短跑。
- 当前冻结训练为64 prompts/step、8 responses/prompt、16 prompts/PPO minibatch、1 epoch。**正式 step50 是200次逻辑 optimizer update**，不是50次。当前 LR 档为 `1e-6/5e-6/1e-5`，alpha64、dropout0、uniform rank32 为首轮配置。

## 2. 为什么这轮先做两条

| 方案 | 判断 | 本轮处理 |
| --- | --- | --- |
| Prompt-sign | 批量 backward 可估计逐题二阶矩；最有机会检验有限 probe 预算是否够用，但需要处理完整分组和归一化 | 实现，8 observations |
| Masked-GradTop | 能复用既有 selector 与 GradTop 收集/初始化，历史正式结果提供继续尝试的依据 | 实现，16 observations |
| GradTop-Core | 条件输出基的能量目标成立；工程上需第二遍相同 rollout、Core 参数化、FSDP/vLLM/snapshot/resume 全链路 | 暂缓实现，下一轮优先候选 |
| GRPO-LoRA-GA | 初始切空间论证成立，但非零 A/B 的 base 补偿需与 rollout 和独立 adapter 导出严格对齐 | 暂缓 |
| GRPO-SB-Zero | 需要 dense 梯度、模拟 Adam 更新和 Core 接入；首步 sign 型更新还抹去梯度幅值 | 暂缓 |

选择前两条的理由是能通过现有普通 LoRA 初始化、训练与推理导出链路形成完整实验，并在首轮隔离各自机制。
这不是对 Core 容量的负面实验结论。

## 3. Prompt-sign 的数学对象和必要修正

记一个完整 prompt 组的损失为

\[
L_p=\frac1n\sum_{i=1}^{n}\frac1{T_{pi}}\sum_t m_{pit}\ell^{\rm GRPO}_{pit},\quad G_p=\nabla_W L_p.
\]

`ell` 保留原 GRPO advantage 和 vanilla dual-clipped PPO 规则，包括负 advantage 的 `clip_ratio_c=10`。
仅在 probe 使用等 prompt、等 response、response 内长度归一化；全对/全错的零 advantage 组保留在分母中。

\[
L_{\rm sign}=P^{-1/2}\sum_p\xi_pL_p,\qquad
\mathbb E_\xi[Z^TZ]=P^{-1}\sum_pG_p^TG_p.
\]

符号乘在**完成 clipping 后的 token loss**上，不能先翻转 advantage，否则可能改变 PPO clipping 分支。
每题的8条回答使用同一个符号；符号由 seed、trainer step、minibatch index 和稳定的 prompt 位置确定，不依赖随机 UUID、DP rank 或 microbatch 排列。

现有 probe 的 `A=0, B=random` 给出 `grad_A.T/scaling=G.T@B`，其中 Gaussian B 方差为 `1/width`，满足 `E[BB.T]=I`。
因此还对输出投影取期望后，sketch 的二阶矩对应上述逐题目标。**无偏的是该随机二阶矩，有限样本 SVD 选出的子空间并不无偏；8次观察的效果必须实测。**

必须与附件说明区分三点：

1. 当前默认 response 长度均衡会把一题的回答拆到不同 observation，导致统计量变成 partial-prompt 的平方和。本轮全部新 probe 禁用这一步，保持连续完整题组；正式训练继续原来的长度均衡。
2. Prompt 等权和 response 长度归一化本身也是变化，所以增加无符号 `gradtop_prompt_mean` 对照。仅与 token-mean 对照比较不能把收益全归于 sign。
3. 这仍使用仓库原 GRPO advantage。REAL 另有 reward contrast、population std、至少两正两负筛选和不同 sketch；本轮不是 REAL 复现，也不宣称估计“无采样噪声的真实梯度”。

新 probe 在打包 microbatch 之前统计全局 token 数和完整 prompt 数，补偿 FSDP 对梯度取平均的 world-size 因子，避免目标随 microbatch 长度拆分而变化。
`gradtop`/`gradtop_short` 采用真正的全局 token mean；旧 actor 的 microbatch-token-mean 再按样本数加权可能与之不同。
因此这两个控制应称为**当前协议下重新定义并固定的 GradTop 控制**，不能声称逐位复现旧 GradTop。

## 4. Masked-GradTop 的边界

复用历史 `build_probe_token_mask(mode="top_surprisal")`：每条 response 取约50%的有效 token，至少128个，并预留末尾128个。
它是在固定保留数量内预留末尾 token 再补高 surprisal token，**不是对严格 top50% 再额外并集末尾128个**；短回答可能全部保留。
selector 使用冻结策略的 detached log probabilities；只将未选 token 的 loss 项置零，分母仍为原始完整有效 token 数。

随后执行同一 GradTop sketch/SVD。正式训练使用原完整 token-mean GRPO，不携带 probe mask。
该设计检验的是 token 选择对初始化的作用；surprisal 不是权重梯度范数，mask50%也不意味着 backward 成本减半。
若未来需要声称“高 surprisal 优于任意同密度筛选”，仍应补 random mask 对照，本轮只比较有无该完整 selector。

## 5. 首轮冻结实验矩阵

候选只有两条；其余三行是必要的机制/预算控制。默认全部 `rank32, alpha64, LR1e-5, seed42, step50`，A/B 均训练、B 从零开始。
如选择另一档协议允许的 LR，应在跑结果前整体统一，不能仅为某方法结果差而事后加扫。

| METHOD | 角色 | Probe reduction | observations | rollout steps | prompts / responses |
| --- | --- | --- | ---: | ---: | ---: |
| `gradtop` | Masked 的同预算控制；长预算控制 | 全局 token mean | 16 | 4 | 256 / 2048 |
| `gradtop_short` | 缩短 probe 预算的控制 | 全局 token mean | 8 | 2 | 128 / 1024 |
| `gradtop_prompt_mean` | Sign 的同归一化、同预算控制 | prompt/response mean，无符号 | 8 | 2 | 128 / 1024 |
| `gradtop_prompt_sign` | 候选1 | prompt/response mean，Rademacher sign | 8 | 2 | 128 / 1024 |
| `gradtop_masked` | 候选2 | 全局原 token 分母，surprisal selector | 16 | 4 | 256 / 2048 |

全部 probe：width8、capacity64、uniform rank32、输入基正交、无参数更新、无梯度裁剪、无稳定性提前终止。
若 sketch 的非零支持不足32维，沿用既有确定性坐标 QR 补齐，不设置新的谱覆盖验收 gate；artifact 逐模块记录 `sketch_supported_rank`、`basis_completion_rank` 和补齐方法，contract 记录补齐总数。补齐方向不能被描述为由梯度信号发现的方向；正式结果需同时披露该数量。
新 artifact 明确区分 `observations` 和 `optimizer_updates=0`，保存实际 runtime contract 和每次完整题组/token 数。
旧 collector 中名为 `optimizer_updates` 的计数实际是 observations；新方法的公开日志和导出不再混淆二者，旧 artifact 字段保留兼容。
每次正式训练从原 seed42 sampler 起点重新开始，probe 数据消费不推进正式训练的 dataloader。

主要比较：

- `prompt_sign` vs `prompt_mean`：相同逐题目标权重下的随机符号作用。
- `prompt_mean` vs `short`：归一化作用；`short` vs `gradtop`：原预算是否冗余。
- `masked` vs `gradtop`：同预算 token selector 作用。
- `prompt_sign` vs `gradtop`：半 probe 预算的整体方案能否接近长预算控制；该比较本身不能独立归因于 sign。

正式训练每组 step50：3200 prompts、25600 responses、200次 optimizer update。
按冻结网格 step10/50 和同一六榜7248 records评测，报告 Macro、逐题配对统计、parse、hit-max、长度，以及 probe 墙钟时间、采样数和 backward 次数。
首轮不预设子空间跨批次稳定，不以 coverage/calibration gate 代替正式短跑效果。

## 6. 运行与导出

入口：[run_gradtop_candidates_4gpu.sh](../../scripts/local/run_gradtop_candidates_4gpu.sh)，默认 dry run。

```bash
cd /root/peft-for-rl
export PYTHON_BIN=/home/node/anaconda3/envs/peft-for-rl/bin/python

# 预览 probe → 初始化验证 → 正式训练配置
METHOD=gradtop_prompt_sign PHASE=all LORA_RANK=32 LR=1e-5 TARGET_STEP=50 \
  bash scripts/local/run_gradtop_candidates_4gpu.sh

# GPU 空闲后执行同一流程；每种 METHOD 单独运行
DRY_RUN=0 METHOD=gradtop_prompt_sign PHASE=all LORA_RANK=32 LR=1e-5 TARGET_STEP=50 \
  bash scripts/local/run_gradtop_candidates_4gpu.sh

DRY_RUN=0 METHOD=gradtop_masked PHASE=all LORA_RANK=32 LR=1e-5 TARGET_STEP=50 \
  bash scripts/local/run_gradtop_candidates_4gpu.sh
```

`PHASE=probe` 只构造 artifact；`PHASE=train` 复用对应的已完成 artifact。
`GRADTOP_ARTIFACT_DIR` 可指定独立产物目录。正式训练仍由唯一冻结启动器 `run_rlvr_protocol_1p5b_4gpu.sh` 执行，验证全部模块、形状、正交性、rank/alpha、实际 probe 设置及预算，再将三个初始化文件的 SHA256 写入 contract。
缺失 artifact 的 `PHASE=all` dry run 只预览下一步，不能视为已经通过 artifact 验证。
默认禁止覆盖已有 probe 产物，避免把旧方向误标为新方法。

训练参数化为现有 `grad_subspace`，因此复用现有标准 LoRA 的 rollout 同步、每10步 snapshot 和完整恢复点；并未引入新的推理 wrapper。

## 7. Core 的下一轮具体要求

固定输入基 Q 后，第二遍应读取 `grad_B/scaling=GQ`，并从累计 `(GQ)(GQ)^T` 选择 U；不能遗漏 alpha/r 的缩放。
第一遍的压缩 sketch 不足以重建 `GQ`，必须保存并重放完整 rollout、old logp、advantage 和 mask，才能声称不增加 rollout。

将训练参数化设为 `delta_W=(alpha/r) U R Q.T, R0=0` 后，需实现：固定 U/Q、仅 R 训练的检查；FSDP 包装与恢复；逐层 LR 参数组；vLLM 同步时 `A=Q.T, B=UR` 的物化；推理 snapshot round-trip；完整 checkpoint 保存 U/Q/R 和 optimizer state。
最小对照是同 Q 的 fixed-A、B-only GradTop；否则输出压缩和冻结 A 的变化混在一起。

当前196模块的 AB-r32 为36,929,536个训练参数，fixed-A B-only 为20,643,840，Core32×32为200,704。
Core 相对 AB 减少184倍，相对 B-only 减少约102.86倍；固定基仍存36,929,536个数值，标准 A/B 导出也不按184倍缩小。
`sqrt(d_out/32)` 在当前模块分别约为2.828、6.928、16.733，统一10倍只是近似，不能称为逐层公式实现，也不能保证等 policy KL。
应事前注册实际倍率或另立协议，不静默突破当前三档学习率规则。

## 8. 验证范围

CPU 验证覆盖：符号与逐题 dense 梯度等价、符号枚举的二阶矩恒等式、clipping 后乘符号、mask 保持分母、动态拆分/reorder不改变目标、多 rank 梯度平均、零函数 probe、固定预算导出、重新初始化保持基座函数，以及无效 artifact/协议参数拒绝。
旧 GradTop/REAL 的相关回归检查通过；合并执行结果为 **155 passed**（另有1条现有 Ray API deprecation warning）。
四个涉及的 shell 启动器通过 `bash -n`，修改的 Python 模块通过 `py_compile`，`git diff --check` 通过。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/node/anaconda3/envs/peft-for-rl/bin/python -m pytest -q \
  tests/test_gradtop_objectives.py tests/test_gradtop_collector.py \
  tests/test_gradtop_protocol.py tests/test_gradient_subspace_init.py \
  tests/test_gradient_probe_uniform_allocation.py tests/test_real_probe_integration.py \
  tests/test_real_protocol.py tests/test_full_gradient_rl_probe.py
```

本机默认冻结协议预检存在两项**既有资产不一致**，尚不能通过正式启动前检查：

1. 缺少协议指定的 `runs/datasets/verl_dapo/data/dapo_math_boxed_heldout637_seed42_v1/train_indices.csv`。
   本地另有 `rlvr_protocol_seed42_v1/sampler_prefix_indices.csv`，其顺序可核对，但文件哈希不同，不能假装它就是原冻结资产。
2. 当前 `dapo-math-17k-boxed.parquet` 的哈希前缀为 `8e3c9314`，协议要求 `dc653393`。
   文件名相同不代表训练数据字节一致，未据此改写原协议中的路径/哈希。

默认入口会在 Ray/GPU 启动前拒绝这两种不一致。需要恢复原冻结资产，或另行明确登记当前资产的协议；
仅修改临时协议做命令拼装检查不能作为正式预检成功的证据。
实现完成时本机4张 L40S 正在执行已有任务，未占用 GPU 启动新的 probe/50-step 训练；因此本轮没有奖励提升结论，也未完成真实4卡吞吐/显存验证。
