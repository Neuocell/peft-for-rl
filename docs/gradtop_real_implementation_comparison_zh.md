**GradTop 与 REAL：实现细节、统计目标和工程差异**

2026-10-18状态入口：[近期实验与精确版本总表](experiments/real_gradtop_summary_20261018_zh.md)、[已发现问题及修复状态](../audits/real_gradtop_implementation_audit_20261018_zh.md)。下文保留10月16日的方法对照；新增REAL-P256为相同REAL定义、4步/256题/87有效组，已训练至210，当前GradTop已完成270。

整理日期：2026-10-16，Asia/Shanghai。依据当前工作树源码、已登记配置、实际初始化产物，以及用户提供的 [REAL 论文《Reward-Gradient Aligned Low-Rank Adaptation for Post-Training》](../../1683_Reward_Gradient_Aligned_L.pdf) §3.2–3.4、Algorithm 1。下文 GradTop 指 2026-10-15 的当前协议 uniform-r32 版本；REAL 指本仓库的二元正确性实现。源码和产物相互核对，历史版本的区别在文末单列。

核心区别是：**GradTop 先把多个 prompt 的梯度聚合成 minibatch 梯度，再累计其输入方向能量；REAL 先对每个有效 prompt group 的梯度分别计算二阶能量，再在题组之间平均。** 当前实现还同时改变了 response 长度权重、投影宽度、子空间分解器和正式训练中 A 是否可训练。

**1. 当前配置对照**

| 维度 | 当前 GradTop-r32 | 本地 REAL-r32 |
| --- | --- | --- |
| METHOD / 观察阶段 peft_type | `gradtop` / `grad_probe` | `real` / `real_probe` |
| 正式训练 peft_type | `grad_subspace` | `grad_subspace` |
| 共同模型与覆盖范围 | DeepSeek-R1-Distill-Qwen-1.5B，196 个 attention/MLP linear 模块 | 相同 |
| 观察时 policy | base policy，零函数 LoRA，不更新参数 | 相同 |
| 每题回答数 | K=8 | K=8 |
| 一次统计 observation | 全局 16 个完整 prompt group，共 128 responses | 一个完整、通过筛选的 prompt group，共 8 responses |
| 组筛选 | 不按正确/错误数量筛题；零 advantage 组仍计入 token 分母 | 至少 2 正确、2 错误；所有 response 非空；已接收题组去重 |
| advantage | 沿用训练器的 GRPO advantage | 用原始二元 reward 重新计算 population-std advantage |
| loss 长度归一化 | 整个逻辑 minibatch 的有效 token 总数作分母 | 每条 response 先除自身长度，再对 K 条平均 |
| 输出随机投影宽度 | 8 | 128，即 4r |
| 输入随机投影 | 无；保留 sketch 的完整输入坐标 | 固定高斯 Ω，宽度 128，即 4r |
| 输出投影刷新 | 每个 minibatch observation 后 | 每个完整 prompt group 后 |
| CPU 子空间估计 | 拼接原幅值 sketch，薄 QR + 小矩阵增量 SVD，capacity=64 | 流式累计 Y，PSD Nyström，最后对恢复因子做 SVD |
| 观察预算 | 4 rollout steps、16 observations、256 prompts | REAL64：64 rollout steps、4096 attempted prompts；REAL32：前 32 steps 统计前缀 |
| 最终 rank / alpha / scaling | uniform32 / 64 / 2 | 相同 |
| 正式初始化 | A=Qᵀ，B=0 | A=Pᵀ，B=0，论文 ν=1 |
| 正式可训练参数 | A+B，共 36,929,536 | B-only，共 20,643,840 |
| 信号维数不足时 | 确定性坐标 QR 补齐并记录 | 不补齐；数值秩不足 r 时直接报错 |
| 当前额外 prompt sign / token mask | 无 / 无 | 无 / 无 |

GradTop 的 probe rank8 是临时随机投影宽度，正式 adapter rank 为 32。REAL32、REAL64 的后缀表示观察 rollout steps，两者正式 rank 都为 32。

**2. 两者共用的零函数梯度探针**

对一个线性层，令 W∈R^(d_out×d_in)，LoRA 增量为 ΔW=sBA，其中 s=alpha/r。观察阶段设置：

```text
A_probe = 0
B_probe = Ψ，Ψ_ij ~ N(0, 1 / sketch_width)
ΔW_probe = 0
```

因此 forward 与 base model 相同。对观察 loss 的权重梯度 G=∇W L，有

```text
grad(A_probe) / s = ΨᵀG
S = grad(A_probe)ᵀ / s = GᵀΨ
E[ΨΨᵀ] = I
```

读取 A 的梯度即可得到完整权重梯度的随机投影，无需显式构造 dense G。两者观察阶段实际都令临时 alpha=sketch_width，因此 s=1；正式训练才使用 rank32/alpha64/scaling2。

REAL 明确冻结 base 和 B，只令 A 可导。GradTop 使用普通 PEFT LoRA 参数配置，在 A=0 时 B 的梯度理论上为零；观察路径不会调用 Adam 更新。两者都在收集后清梯度、刷新 Ψ，并保持 A=0。这里的“冻结 A”是正式 REAL 的行为，观察 REAL 恰好需要 A 可导。

来源：[随机投影和零初始化](../verl/utils/peft_gradient_subspace.py#L70)、[两类临时 adapter 构造](../verl/workers/fsdp_workers.py#L623)、[REAL 参数冻结](../verl/utils/peft_real.py#L93)。

**3. GradTop 的完整执行过程**

每个 rollout step 从固定初始 policy 采 64 题，每题 8 条回答。Trainer 给题组分配 ID，关闭 probe 的 trainer 长度 balance，保证完整组归属于同一个 DP rank 和同一个逻辑 observation。一个 step 形成 4 个全局 minibatch，每个 minibatch 16 题。

GradTop 保留 vanilla dual-clipped PPO 分支，当前 clip low=0.20、high=0.28、C=10，无 KL、entropy 或其他梯度正则。令 ℓ_it 为实际 clipped policy loss，m_it 为 response mask，其显式目标为：

```text
L_t = Σ_(i,token) m_it ℓ_it / T_t
T_t = 该全局逻辑 minibatch 的有效 response token 总数
G_t = ∇W L_t
```

分母在动态 microbatch 拆分前跨 DP 求和并固定；每个 DP rank 的局部 loss 乘以 dp_size，抵消 FSDP 的梯度平均。显式 probe 将后续 microbatch loss_scale_factor 设为 1，避免再次按样本数缩放。因此改变 microbatch 拆分不改变这个数学目标。

在初始同策略且 importance ratio=1 时，上式梯度退化为按全局 token 总数归一化的 GRPO score 梯度。advantage 使用项目 GRPO 路径：`torch.std` 的样本标准差，再加 `epsilon=1e-6`，与 REAL 的 population std 定义不同。全对/全错组的 advantage 为 0，但 response tokens 仍留在分母中。

每个 observation 完成所有 microbatch 的 backward 和 FSDP 归约后，收集 S_t=G_tᵀΨ_t；此时不做 gradient clipping，也不调用 optimizer step。随后刷新输出投影，继续下一个 observation。

CPU 保存未经单位范数归一化的 sketch。每个 rollout step 末将本 step 的 4 份 sketch 与已有因子拼接：

```text
F_k = truncate_64([F_(k-1), S_(4k-3), S_(4k-2), S_(4k-1), S_(4k)])

truncate(X):
    Q, R = thin_QR(X)
    U_small, σ, Vᵀ = SVD(R)
    return (Q @ U_small[:, :64]) * σ[:64]
```

中间因子保存 UΣ，因而保留累计能量的幅值；最终 A 只取 top32 单位输入方向的转置，不携带奇异值。忽略截断误差时，F Fᵀ 对应 Σ_t S_t S_tᵀ，输出投影期望对应 Σ_t G_tᵀG_t。

观察固定在第 4 个 rollout step 结束，共 16 observations，不走稳定性提前停止。最终 rank 由 `rank_bins=[32]` 固定。`target_energy=0.95` 仍用于诊断，但不能保证每层捕获 95% 能量。实际产物的模块平均 retained sketch energy 为 **71.4392%**，最低为 **44.2307%**；它们都是本次训练内 sketch 的诊断，不能与 REAL 的 Nyström energy fraction 直接横比。

16×8 份累计列方向足以生成 rank32；单次 width8 不会把最终 rank 限制为 8。capacity64 则限制了增量压缩保留的方向数。当前算法没有显式遗忘或时间衰减，policy 在整个 probe 中保持不变，增量 SVD 的截断顺序依赖不能被解释为在线跟踪策略漂移。

来源：[完整组与实际归一化契约](../verl/trainer/ppo/ray_trainer.py#L1433)、[显式 loss](../verl/utils/gradtop_objectives.py#L120)、[GRPO advantage](../verl/trainer/ppo/core_algos.py#L265)、[原幅值增量累计](../verl/utils/peft_gradient_subspace.py#L192)、[收集与刷新](../verl/utils/peft_gradient_subspace.py#L1394)、[绕过 clipping/Adam](../verl/workers/actor/dp_actor.py#L1756)。

**4. REAL 的完整执行过程**

REAL 从初始 policy 采样并判分，观察只使用原始 correctness reward R_i∈{0,1}。K=8 时只接收 2≤n_correct≤6 的题组，并要求每条 response 有有效 token。已接收题组按稳定 prompt ID 跨 batch 去重。

对于一个题组 p，重新定义 advantage 和目标：

```text
â_i = (R_i - mean(R)) / std_population(R)
L_i = Σ_token m_it
J_p = (1/K) Σ_i â_i [Σ_token m_it logπ_it / L_i]
G_p = ∇W J_p
```

实现最小化 -J_p，因此读到的 sketch 为 -Ψ_pᵀG_p；这个整体负号不影响二阶矩。REAL 不直接复用训练器 advantage，也不使用 GradTop 的全局 token 分母。前者按每条 response 的平均 token score 赋权，后者让较长 response 累计更多 token 项。

同一个 prompt 的 8 条 response 逐条 forward/backward，先累计成完整 H_p，再做一次二次统计：

```text
H_p = Ψ_pᵀG_p                        # [s_out, d_in]
Ω ~ N(0,1)，每模块固定              # [d_in, s_in]
Y_sum += H_pᵀ(H_p Ω)                 # [d_in, s_in]
Y = Y_sum / accepted_prompt_groups
```

输出 Ψ_p 每个题组刷新，输入 Ω 在所有题组之间共享。组内不能在每条 response 后刷新 Ψ，也不能分别对 response sketch 平方再相加；否则会丢掉正确/错误 response 之间的交叉项。

它估计的目标是未中心化题组二阶矩：

```text
C_prompt = mean_p G_pᵀG_p
max trace(Pᵀ C_prompt P), subject to PᵀP = I
```

它既不对 prompt 之间的梯度去均值，也不要求梯度在不同题组方向一致。输出随机压缩给出 C_prompt 的无偏二阶矩估计，但有限宽度 Nyström 最终得到的子空间不因此具有无偏保证。

导出时计算 Nyström 恢复：

```text
M = sym(ΩᵀY)
F = Y M^(†/2)
P = top-r left singular vectors of F
A = Pᵀ
```

代码在 CPU float64 上缩放后分解，对 M 做对称化，按相对谱阈值 1e-6 求伪逆；显著负特征值、零能量、core 或恢复因子的数值秩不足 r 均报错。最终导出的 basis 为 float32，不加入未观察到的补齐方向。

默认 s_out=s_in=4r；r32 时都是 128。观察在指定 step 数或有效题组上限到达时结束。`REAL_PROBE_PROMPTS=0` 表示无有效组上限，按 step 结束；给定上限也不保证在结束前一定采够。实际 REAL64 的 4096 attempted prompts 中接收 1499 组；REAL32 来自 REAL64 第 32 step 的精确累计统计缓存，接收 735 组。

来源：[reward 系数](../verl/utils/peft_real.py#L49)、[题组筛选](../verl/trainer/ppo/ray_trainer.py#L1652)、[逐 response 累计完整组梯度](../verl/workers/actor/dp_actor.py#L517)、[流式 Y](../verl/utils/peft_real.py#L199)、[Nyström 数值处理](../verl/utils/peft_real.py#L140)、[统计前缀导出](../scripts/local/real_probe_cache.py)。

**5. “先聚合再平方”究竟改变了什么**

为单独说明聚合的作用，暂时假设两者采用相同的每题 loss，并令 G_t=Σ_p w_pG_p。则：

```text
G_tᵀG_t = Σ_p w_p² G_pᵀG_p + Σ_(p≠q) w_p w_q G_pᵀG_q
```

GradTop 保留同一 minibatch 内的题间交叉项；REAL 的逐题二阶矩不包含这些跨题项。若同一 minibatch 内有 G_1=G、G_2=-G 且权重相等，聚合梯度消失，GradTop 在这个 observation 中看不到相应能量；REAL 则保留两个题组的 GᵀG。

但若 G 和 -G 分别出现在两个不同 GradTop observations，因子拼接会累计两者各自的二阶能量，二者不会因一次总平均而抵消。因此当前 GradTop 也属于二阶能量方法，准确比较是 **minibatch 级二阶矩与 prompt 级二阶矩**。

这使 GradTop 更直接反映所选分组和权重下的聚合梯度，REAL 更直接反映各题组独立的输入投影能量。二者都不能只凭初始化能量保证未来的 Adam 更新收益；实际实现的 loss、采样筛选和训练参数化又不同，最终准确率差异不能只归因于交叉项。

**6. 正式训练和工程路径的差异**

两者观察结束都重新创建正式 adapter/optimizer/dataloader，不把 probe 消耗的位置当成正式训练进度。训练从 seed42 数据顺序起点开始，加载正交 A 并令 B=0，rank32/alpha64/scaling2。正式训练共享本地 DAPO-style GRPO 设置：64 prompts/step、K8、PPO minibatch16、1 epoch、LR1e-5、dropout0。

GradTop 的 A、B 均可训练，观察得到的输入子空间只是起点，之后可以旋转。REAL 冻结 A，仅训练 B，权重更新始终形如 sBPᵀ，输入行空间保持在选定 P 内。rank 相同不代表可训练参数相同：

```text
GradTop: Σ_modules r * (d_in + d_out) = 36,929,536
REAL:    Σ_modules r * d_out          = 20,643,840
```

冻结 A 仍需保存和加载 A、B 两个因子。推理与 rollout 同步必须包含完整 adapter，不能只传 B。两者使用同一套标准 adapter 保存/加载路径，这不是一个只属于 REAL 的额外推理模块。

共同正式训练里的 `token-mean` 实现是“每个 microbatch 内 token mean，再按 response 样本比例累计并跨 DP 平均”。一般不等于 GradTop 显式 probe 的全局 token mean，也不等于 REAL 的逐 response 长度归一化。当前 probe 梯度不能直接称作正式训练 Adam 实际接收的梯度。

| 工程维度 | GradTop | REAL |
| --- | --- | --- |
| DP 观察分工 | 各 rank 处理不同完整题组，归约得到同一个 minibatch 梯度后再平方 | 各 rank 复制同一个完整题组，避免不同题的梯度被 FSDP 平均后再平方 |
| backward 调度 | 动态 microbatch 累计，一个逻辑 observation 可含多次物理 backward | 当前每条 response 一次 backward，8 次累计为一组，再收集一次 |
| CPU 持久统计 | capacity64 的输入侧因子，另有本 step 待压缩 sketch | 每模块 Ω 和 Y 两个 d_in×128 矩阵；该次产物记录合计 521,142,272 bytes |
| 正式 FSDP 配置 | 当前协议 `use_orig_params=False` | 当前协议 `use_orig_params=True` |
| REAL 观察支持范围 | — | FSDP1、sequence parallel size1；Qwen2 确定性观察保留已适配的 activation checkpointing |

REAL 当前的组复制和逐 response 调度会增加本地吞吐成本，这属于现有执行方式。论文描述一个组一次逻辑 backward；本地实现分 8 次 backward 得到同一组累计梯度，不应把本地串行开销当作数学目标的必然代价。

REAL 不计算或收集 dense base dW，但冻结参数与可训练参数共处同一 FSDP flat leaf 时，FSDP 仍可能为冻结 base 提供全零梯度存储。上表 CPU sketch 大小或论文线性 buffer 公式均不能替代整卡峰值显存测量。

来源：[正式初始化](../verl/utils/peft_gradient_subspace.py#L1710)、[B-only 验证](../verl/utils/peft_gradient_subspace.py#L1667)、[统一训练入口](../scripts/local/run_rlvr_protocol_1p5b_4gpu.sh#L155)、[REAL 观察模式](../verl/utils/real_probe_mode.py#L12)、[FSDP 限制](../verl/workers/fsdp_workers.py#L896)。

**7. 实际预算、产物与论文版本边界**

| 当前已完成产物 | rollout steps | observations / accepted groups | attempted prompts | sampled responses | probe wall time |
| --- | ---: | --- | ---: | ---: | --- |
| GradTop-r32 | 4 | 16 minibatches，覆盖 256 组 | 256 | 2048 | 2250.33 s，约 37.51 分钟 |
| REAL32-r32 | 32 | 735 accepted groups | 2048 | 16384 | 无独立采集计时；复用 REAL64 的统计前缀 |
| REAL64-r32 | 64 | 1499 accepted groups | 4096 | 32768 | 40085.41 s，约 11.135 小时 |

这两个实测完整 probe 的墙钟比约 17.81，但 REAL64 的 attempted prompts 是 GradTop 的 16 倍，还存在投影宽度和执行调度差异。该数字描述当前选定预算下的成本，不能用作同样本、同目标、同估计质量的算法加速倍数。

两者复用三个初始化文件：`rank_map.json`、`subspaces.safetensors`、`summary.json`。前者提供各模块 rank/alpha，中间文件保存 [rank,d_in] 的 A，summary 记录观察定义与诊断。REAL metadata 另要求 `freeze_a=true`；正式 launcher 也须实际设置 `lora_freeze_a=True`。

当前 GradTop 代码允许信号维数不足时用坐标 QR 补齐，实际此次产物的 **completed_basis_directions=0**，196 个模块都有 sketch 支持的 rank32。REAL 当前产物均通过严格数值秩检查。

直接证据：

- [GradTop 产物 summary](../runs/analysis/gradtop_r32_o16_w8_c64_seed42_20261015/summary.json)。
- [REAL32 产物 summary](../runs/analysis/real_probe_r32_s32_seed42_replayed/summary.json)。
- [REAL64 产物 summary](../runs/analysis/real_probe_r32_s64_p0_seed42_20261013/summary.json)。
- [GradTop 当前冻结协议](../configs/experiments/gradtop_r32_clean_20261015.json)。
- [成本记录与训练结果交接](experiments/gradtop_real_handoff_20261016_zh.md)。

本地 REAL 对齐论文的 Eq.6–9、二正二负筛选、双侧随机压缩、ν=1 和 fixed-A，但正式训练采用本地 GRPO/rank32/alpha64/LR1e-5/无KL 配置。论文 Qwen3 主实验采用 rank16、GSPO、LR1e-6；distilled 分支采用 rank64、GSPO-token、KL0.01；观察计入其最初 64 个 training steps。本地是观察和正式更新分别计数，因此这里是本地协议下的方法实现与比较，未构成完整论文实验复现。

**8. 版本与启动入口中容易混淆的地方**

当前 GradTop 采用 16 observations、uniform32、显式全局 token mean、完整题组和固定预算。历史 GradTop 常见 12 observations、192 题、按能量分档的 heterogeneous rank（均值约31.63）、每 observation 压缩及稳定性提前停止；历史 loss 归约也受 microbatch 划分影响。二者共享 random-B sketch 和增量能量 SVD 的核心计算路径，但不是同一条冻结配置。

仓库还注册了 `gradtop_short`、`gradtop_prompt_mean`、`gradtop_prompt_sign`、`gradtop_masked`。它们是独立变体，当前本文 GradTop 没有额外 prompt sign 或 token mask。`run_gradtop_candidates_4gpu.sh` 的默认 METHOD 是 `gradtop_prompt_sign`，复查当前版本必须显式指定 `METHOD=gradtop`。

以下为已有入口的预览用法，`DRY_RUN=1` 不启动训练：

```bash
# 当前 GradTop 配置：观察与正式训练预览
METHOD=gradtop LORA_RANK=32 LR=1e-5 TARGET_STEP=50 DRY_RUN=1 \
  PROTOCOL_FILE=configs/experiments/gradtop_r32_clean_20261015.json \
  bash scripts/local/run_gradtop_candidates_4gpu.sh

# REAL 观察配置预览
REAL_PROBE_RANK=32 REAL_PROBE_STEPS=64 DRY_RUN=1 \
  bash scripts/local/start_real_gradient_probe_4gpu.sh

# 以已有 REAL 产物预览统一协议训练
METHOD=real LORA_RANK=32 LR=1e-5 TARGET_STEP=50 DRY_RUN=1 \
  REAL_ARTIFACT_DIR=runs/analysis/real_probe_r32_s64_p0_seed42_20261013 \
  bash scripts/local/run_rlvr_protocol_1p5b_4gpu.sh
```

GradTop 的 PHASE=train 消费已有产物时须提供 `GRADTOP_ARTIFACT_DIR`；重新观察须选择新的 run/artifact 目录。REAL 的低层 `start_real_fixed_a_4gpu.sh` 用于工程调试，正式方法对照使用统一协议入口。

本次公式与执行语义的现有 CPU 检查为 **59 passed**，覆盖显式归一化、零函数 sketch、增量产物、REAL 组梯度、二阶矩和 Nyström。使用以下四个测试文件：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=. python -m pytest -q \
  tests/test_gradtop_objectives.py tests/test_gradtop_collector.py \
  tests/test_real_probe.py tests/test_real_probe_integration.py
```

另逐项核对三份实际 summary 的 196 模块覆盖、rank32、GradTop 无补齐、REAL 数值秩和可训练参数量；文档 UTF-8、中文、链接与代码行号检查通过。
