# GradTop 吸收 REAL 的下一轮实验设计

日期：2026-10-14（Asia/Shanghai）。范围：研究方案、原始证据核对、CPU 数学检查；本文没有启动新的 GPU probe、训练或评测。

**2026-10-15 优先级修订：** 当前主线改为 [原batch聚合的GradTop-r32干净对照](gradtop_r32_clean_baseline_20261015_zh.md)，研究问题是 [逐题能量与有效共享更新方向的区别](gradtop_update_geometry_research_20261015_zh.md)。不预设REAL目标更优。下文保留10月14日的候选设计与推导作为历史备选，Prompt-sign降为机制对照；fixed-A、reward-scatter与mask均不进入这次基线。下文原有“优先”“本轮”措辞应按这一修订理解，不再代表当前执行顺序。

本文补充 [已有候选实现说明](gradtop_candidates_20261014_zh.md)，以 [RLVR_PROTOCOL.md](../../RLVR_PROTOCOL.md) 为训练和评测标准。新增方法名称只是设计名称，未自动注册为可运行的 METHOD。

## 1. 当前证据：应当超过哪条基线

以下都是当前本地资产绑定下的 r32、alpha64、LR=1e-5、seed42；step50 是 200 次逻辑 optimizer update。主指标为六任务等权 Macro Avg@k。

| 方法 | 正式训练 step | Macro Avg@k | 相对同 step LoRA | 证据边界 |
| --- | ---: | ---: | --- | --- |
| LoRA | 50 | 38.7224% | — | 当前对照 |
| LoRA | 100 / 150 | 38.3390% / 38.8141% | — | 已有完整轨迹点 |
| REAL，64-step probe | 50 | 40.0388% | +1.3164 pp；题目 bootstrap 95% CI [0.1646, 2.4712] pp | 单训练 seed |
| REAL，32-step probe | 50 | 40.3361% | +1.6137 pp；95% CI [0.2854, 2.9320] pp | 单 seed、独立登记的 probe-budget 消融 |
| 旧 GradTop | 50 / 100 / 150 | 39.1198% / 40.3326% / 40.4574% | 不作当前 matched 差值 | 旧协议、近 r32，原 artifact 身份仍有审计限制 |

来源：[LoRA 轨迹](../../runs/ops/rlvr_protocol/rlvr_v1_lora_r32_a64_lr1em5_b64m16n8_seed42.step50_100_150_comparison.md)、[REAL64 配对报告](../../runs/ops/real_r32_lr1e5_seed42_after_eval/paired_comparison.md)、[REAL32 汇总](../../runs/ops/real_probe32_r32_lr1e5_seed42_after_real64_eval/summary_comparison.json)、[REAL32 配对报告](../../runs/ops/real_probe32_r32_lr1e5_seed42_after_real64_eval/paired_comparison.md)、[历史登记](method_and_results_registry_zh.md)。REAL32 相对 REAL64 的点估计差约 0.2973 pp；现有两个 CI 都以 LoRA 为对照，不能据此声称 REAL32 显著更好。

REAL32 正在从 step50 续到150；本次读取的状态文件于北京时间 23:08 更新到 step79，100/150 尚无评测结果。状态是时点记录，后续以 [运行状态](../../runs/ops/real_probe32_continue_step150/status.json) 为准。

另有值得研究的行为分离：REAL64/32 的 Macro Pass@k 相对 LoRA 分别下降约 4.953/2.315 pp；HMMT Pass@32 从 50.0% 变为 26.67%/30.0%；平均输出长度从 7749.7 变为 5088.3/5338.1。它提示应同时检查平均正确率、解题覆盖和长度，但单 seed 结果不足以认定“探索坍塌”或解释因果。来源：[REAL64 行为指标](../../runs/ops/real_r32_lr1e5_seed42_after_eval/summary_comparison.json)、[REAL32 行为指标](../../runs/ops/real_probe32_r32_lr1e5_seed42_after_real64_eval/summary_comparison.json)。

`probe_budget_comparison.md` 中 44.5/46.2/46.4 是按 response 数加权的 Avg，不能混入上述 Macro 表。

## 2. 统一协议和实际资产

首轮主配置固定：DeepSeek-R1-Distill-Qwen-1.5B、4 GPU、bf16、all-linear、uniform rank32、alpha64、dropout0、LR1e-5、64 prompts/step、每题8条回答、PPO minibatch16、1 epoch、1024/8192 token、token-mean GRPO、原 clipping 和无 KL/entropy/额外长度奖励。所有 probe 都保持初始函数不变、无参数更新，正式训练从原 seed42 sampler 起点开始。

新候选使用预先声明的 step10/50 六榜评测，step50 为主比较点；如果利用 step10 选 checkpoint，则必须为所有主表对照补同样的网格。后续主方法及对照共同续到100/150，保存完整轨迹；每10步快照、每50步和计划终点保存完整恢复点，按协议精确续训。

当前 canonical JSON 的旧 parquet 字节哈希和索引路径与本机不一致，但已有 LoRA/REAL 正式 run **已登记 local_asset_binding**：本地 parquet 为 `8e3c9314...`，sampler-prefix 为 `9c926a98...`，并记录官方来源、转换程序和语义行哈希。新实验应从这份共同资产绑定与当前方法定义生成新的不可变 run contract，保留父协议哈希及差异；不能仅使用旧 local JSON 而漏掉新 GradTop 定义，也不能静默改 canonical 的资产哈希。

资产依据：[LoRA local protocol](../../runs/ops/rlvr_protocol/rlvr_v1_lora_r32_a64_lr1em5_b64m16n8_seed42.local_protocol.json)、[REAL local protocol](../../runs/ops/real_r32_lr1e5_seed42_after_eval/protocol.json)。已有登记提供了继续实验的明确路径，候选文档中的默认预检失败不代表必须重建整个数据集。

rank16/64 只作为主线成立后的扩展，alpha 仍为64，因此有效 scaling 会变化，不能同时声称跨 rank scaling 恒定。当前主协议固定 seed42；跨 training seed 确认需另行预声明协议扩展，不能静默改 seed。第一轮不为每个候选搜索全部 LR；确认阶段给入选方法和强对照相同的三档 LR 预算。

## 3. 可以从 REAL 吸收什么

| 内容 | 本轮取舍 | 原因 |
| --- | --- | --- |
| 完整 prompt 内的 reward contrast | 明确保留 | 同题正确/错误回答之间的关系是 RL 信号，不能先拆散再平方 |
| 逐题未中心化二阶矩 | 主线 | 相反方向的题目梯度可共享有用输入子空间，均值抵消不代表无用 |
| 每 response 长度归一化 | 单独对照 | 它改变长答案权重，不能把收益都归因于 sign |
| fixed-A、B0 | 用同一 artifact 做消融 | 可减少训练参数，但论文和本地历史都不能保证冻结一定更优 |
| 逐题串行 backward、宽度4r | 保留作 REAL baseline | 不必成为 GradTop 的强制实现方式；需要同目标桥接实验 |
| Nyström | 作为估计器控制 | 现有 GradTop 增量 SVD 可复用；换估计器不等于换统计目标 |
| 正确性之外的长度 utility | 不作为新颖主贡献 | REAL Appendix E 已包含 probe-only 长度偏好分支 |

REAL 本地定义及论文来源见 [REAL 实现说明](../real_reward_gradient_probe_zh.md)，论文为 `/root/1683_Reward_Gradient_Aligned_L.pdf`，ICLR 2027 匿名投稿版本。REAL 没有收集 dense base gradient；GradTop 的贡献不能描述成“首次避免 full-gradient”。

## 4. 方案一：Prompt-sign GradTop，优先跑已有实现

令完整题组的 probe loss 为

\[
L_p=\frac1K\sum_{i=1}^K\frac1{T_{pi}}\sum_t m_{pit}\ell^{\mathrm{GRPO}}_{pit},\quad G_p=\nabla_W L_p.
\]

在包含 P 题的逻辑 observation 中，对每题采一个独立 Rademacher 符号：

\[
Z=P^{-1/2}\sum_p\xi_pG_p,\qquad
\mathbb E_\xi[Z^TZ]=P^{-1}\sum_pG_p^TG_p.
\]

使用零函数 probe `A=0, B=Ψ`，得到 `grad_A/scaling=Ψ.T @ Z`；Gaussian Ψ 的元素方差为 `1/width`，所以对输出投影再取期望也保持上述二阶矩。这样可以在一次逻辑批量梯度累计中近似逐题平方和，避免跨题确定性的相互抵消。

这里保留 GRPO advantage，当前实现不等于 REAL 的 population-std contrast + 至少两正两负筛选。符号必须乘在 clipping 后的 loss 上；一题8条回答同号；完整组跨 microbatch 累计，分母事先固定；FSDP 平均要补偿；不同 observation 刷新输出投影。数学无偏性属于未压缩二阶矩，不属于有限样本选出的 SVD basis。

固定K=8且忽略epsilon时，sample/population std只差一个全局常数，本身不改变理想子空间；更实质的差别是response/token归一化和题组筛选。

已实现并已做 CPU 验证的五条方法如下；本轮尚无其正式训练结果。

| METHOD | 统计定义 | Observations / rollout steps | Attempted prompts / responses |
| --- | --- | ---: | ---: |
| `gradtop` | 全局 token mean | 16 / 4 | 256 / 2048 |
| `gradtop_short` | 同上 | 8 / 2 | 128 / 1024 |
| `gradtop_prompt_mean` | prompt/response mean，无符号 | 8 / 2 | 128 / 1024 |
| `gradtop_prompt_sign` | 同归一化，加 prompt sign | 8 / 2 | 128 / 1024 |
| `gradtop_masked` | 原 token 分母，surprisal selector | 16 / 4 | 256 / 2048 |

共有 width8、capacity64、输出 rank32、A/B 均训练、B0。8 observations 是2次采样 step，物理 backward 次数还取决于 microbatch，不能写成“仅8次反传”。新 GradTop 控制修正了旧 microbatch reduction，因此不能称为旧实验的逐位复现。

最低机制对照为 `sign vs prompt_mean`、`prompt_mean vs short`、`short vs gradtop`。第一项回答随机符号是否有用，第二项回答归一化是否有用，第三项回答预算能否缩短。`sign vs gradtop` 可以回答整体效率，但不能单独解释机制。

**可证伪点：** 8次随机混合可能方差过大，width8 也可能丢方向；若 sign 不能优于同预算 prompt_mean，就不能宣称符号带来了收益。先报告这一结果，再在预声明的预算消融中增加 observations；不要在结果后任意换符号 seed。

## 5. 方案二：同 basis 的 fixed-A 与严格 REAL-sign 桥接

### 5.1 fixed-A：最小新增训练改动

对同一个 GradTop/Psign artifact，分别训练 A+B 和 B-only：`A=Q.T, B0=0, scaling=64/32=2`。两条保持 LR、数据和 steps 一致。增加随机正交 A 的 B-only 对照，避免把“正交缩放/冻结”本身当 reward basis 的收益。

196个模块的 r32 参数数为：A+B 36,929,536，B-only 20,643,840，少约44.10%。固定 A 仍需保存和参与前向，激活、模型权重、rollout cache 仍存在；这不是总显存减少44.10%的保证。若要声称 REAL 的优势主要来自 basis 而非冻结，还应对同一个 REAL artifact 加 A+B 对照。

这项消融可利用现有 fixed-A 和标准 adapter 导出链路，工程成本低；目前公开 GradTop METHOD 固定 `freeze_a=false`，实际执行前需另注册方法定义及对应验证，不可直接修改原 artifact metadata 来绕过协议。

### 5.2 REAL-sign：把“同目标、更便宜”讲清楚

新增设计 `real_sign` 固定 REAL Eq.6 loss、population std、2≤正确条数≤6、输入/输出 sketch128、同一 Nyström、同 rank32/fixed-A，仅替换逐组收集方式。首先在新建的同一份完整 rollout cache 上做小规模估计器比较，再比较训练效果。

每批 b 个有效题组令 `Z_b=Σ ξ_p G_p / sqrt(b)`、`H_b=Ψ_b.T @ Z_b`。累计 `Σ_b b H_b.T(H_b Ω) / M`，其中 M 是有效题组总数。b 不等时必须保留该权重，否则不同有效组密度会改变目标。

一次标准 LoRA-A backward 需要同批共用 Ψ；标准 REAL 则每题独立 Ψ。因此两者具有相同期望目标，却不是相同随机估计量。小规模桥接增加“共用 Ψ、仍逐题平方”的控制，分别隔离 projection 共享与 sign 混合。

对任意 v，忽略输出压缩时：

\[
\operatorname{Var}_\xi(\|Z_bv\|^2)
=\frac4{b^2}\sum_{p<q}\langle G_pv,G_qv\rangle^2.
\]

因此相似梯度较多时，一次大 batch 的 sign 不会自动得到低相对方差。观察次数、输出宽度和完整题组分块都是需要记录的统计预算。

桥接后再分别比较128→8输出宽度、Nyström→capacity64 SVD、32/64→2/4采样steps；不要一次全部改变后归因于 batched estimator。完整大矩阵不必都进入训练，先用同 cache 的 held-out projection energy、主角度和实际耗时筛查估计器精度；这些诊断不代替最终短跑。

现有 REAL cache 保存的是累计 Y/Omega，不包含完整 rollout、逐组 G 或 H；它只能导出保存过的前缀，不能重算 sign、reward、mask 或任意子集。需保存新的 token IDs、attention/response mask、raw correctness、稳定 prompt ID，以及使用 PPO probe 时的 old logp/advantage和生成配置。已有32-step artifact可复用做训练对照，不能冒充已存在的重放轨迹。

## 6. 方案三：按 reward 条件化的梯度可靠性排序

这是优先建议新增的 RL-specific 候选，暂称 `reward_scatter`，尚未实现。

### 6.1 研究问题和统计定义

同一道题的多条 rollout 提供了正确/错误分组。REAL/Prompt-sign 的高二阶能量既可能来自稳定的正确—错误梯度差异，也可能来自同一正确性类别内部的回答波动。后者是估计 contrast 时的不确定性来源，但也可能包含有价值的多样解法，不能无条件称为有害噪声。

首版与 REAL 使用同一筛选：K=8，至少两正两负；与同样筛选、同归一化的 signal-only 方法比较，避免筛选差异混入排序收益。定义每条回答的 score 矩阵：

\[
S_{pi}=\nabla_W\left[\frac1{T_{pi}}\sum_t m_{pit}\log\pi(y_{pit}\mid q_p,y_{pi,<t})\right].
\]

令 n₊/n₋ 为正确/错误计数，`a_p=sqrt(n₊n₋)/K`，`Sbar_pc` 为同题同类别均值，则 REAL contrast 为

\[
G_p=a_p(\bar S_{p+}-\bar S_{p-}),\quad C=\mathbb E_p[G_p^TG_p].
\]

构造 contrast 的类内方差估计：

\[
N_p=a_p^2\sum_{c\in\{+,-\}}
\frac{\sum_{i\in c}(S_{pi}-\bar S_{pc})^T(S_{pi}-\bar S_{pc})}
{n_c(n_c-1)},\quad N=\mathbb E_p[N_p].
\]

条件于题目、正确性计数，若同类别 rollout 可视为独立同分布，该 N 估计 contrast 均值的矩阵方差，且

\[
\mathbb E[G_p^TG_p-N_p\mid n_+,n_-]
=a_p^2(\mu_{p+}-\mu_{p-})^T(\mu_{p+}-\mu_{p-}).
\]

该等式针对上述长度归一化 score contrast；不能把它升级成无偏的总体 expected-reward 梯度或实际 Adam 更新收益保证。经验采样相关、reward/parser错误和有限样本都会削弱这一解释。

### 6.2 保持低成本：class-centered 随机符号

每个 prompt、每条回答采独立 Rademacher ζ，按同题同类别中心化：`zeta_tilde_pi=zeta_pi-mean_class(zeta)`。令

\[
V=P^{-1/2}\sum_p\sum_c
\frac{a_p}{\sqrt{n_c(n_c-1)}}\sum_{i\in c}\tilde\zeta_{pi}S_{pi}.
\]

对 ζ 有 `E[V.T V]=mean_p N_p`；再用独立输出投影 Ψ 的零函数 probe，仍可用批量 backward 估计 N，不需要逐回答导出 dense 梯度。类内中心化在数学上消掉类别共同方向，随机性消掉跨类别、跨题交叉项。

先用 signal pass 建 Q64，再将同一缓存重放一次得到 N 的 sketch，并投影到 Q64 中累计64×64统计。**新增一次反传遍历，不新增 rollout；全部额外 backward tokens、收集和分解时间计入成本。** 需要缓存或保留第一遍 signal sketches，保证 C_Q 与 signal-only 控制来自同一统计量。若低成本收集截断了谱，必须把这种近似两边保持一致。

首版具体保存16份原始signal H及各槽位b，Q仍按capacity64生成；Q确定后统一重算`C_Q=sum_b b*(H_b Q).T@(H_b Q)/M`。增量SVD的截断因子不能恢复这个精确的投影统计。当前模型全部196模块、16×width8、FP32原始H上限约248.5 MiB，计入CPU存储；不保存dense梯度。noise只需投影后的小矩阵，额外signal控制也直接在固定Q中累计。

### 6.3 首版固定参数和导出

建议首版固定 attempted256 prompts、K8、16 signal observations、width8、Q最多64维、uniform输出r32；noise pass同一prompt顺序、每组一份独立class-centered sign，独立输出投影，固定seed42并用不同随机流标签。无 mask、不叠加随机补空间比例搜索。

这里16指预定的完整题组minibatch槽位；若某槽位没有符合筛选的题，则记为零有效observation，不额外延长采样。signal/noise均按实际accepted数加权，使累计目标对应有效题组等权；记录支持维数d和补齐方向数，不能把坐标补齐叫reward信号。C_Q/N_Q的小矩阵累计和广义分解使用CPU FP64。

具体累计规则为：每槽位若使用`1/sqrt(b)`系数，则贡献`b*H.T@H`，最后除总accepted数M；第二遍signal合并第一遍后除2M。若M=0则本次probe失败。若模块有效支持d<32，三种方法对该模块统一使用相同的signal基加确定性QR补齐，不做广义排序；单独报告这些模块，避免将没有可选择余量的退化情形解释为可靠性排序的收益。

noise系数偶然为零的response首版也不跳过forward/backward，以保持三组之间预定的反传token口径；实际物理次数和耗时仍分别测量。小矩阵先对称化，明显负谱视为实现/数值异常，不能事后任意截谱修复。

在 Q 的 d≤64 维空间内解

\[
C_Qv=\lambda(N_Q+\rho I)v,\quad C_Q=Q^TCQ,\quad N_Q=Q^TNQ.
\]

预先固定 `rho=max(trace(N_Q)/d, 1e-6*trace(C_Q)/d)`，不在首轮扫 ridge。若 C 无有效信号，按预声明的失败/补齐规则处理并披露；N=0 时该式退化为 signal 谱排序。取前32个广义特征向量后对 `Q V32` 做 Euclidean QR，导出正交 A，B0、alpha64，首版 B-only。QR 保留所选 span，使缩放与对照一致；不能继续声称导出的向量逐列满足原 N-metric 归一化。

首版不直接做 C−N 后截负谱。C、N 分别 PSD，强 ridge 限制小方差方向被异常放大。在总体近似 C=signal+N 下，该排序偏向 signal 相对噪声更强的方向；有限样本和子空间截断仍可能失效。

同一个 Q64 内的核心比较和扩展控制：

| 对照 | 用途 | 是否首批训练 |
| --- | --- | --- |
| C_Q 普通前32谱方向、fixed-A | 隔离候选span、筛选和冻结 | 必须 |
| reward-scatter 广义排序、fixed-A | 检验条件化可靠性 | 必须 |
| 相同cache额外16次独立signal观测，在原Q内精化C_Q，再普通排序 | 与16 signal+16 noise匹配backward预算，检验额外计算是否值得用于N | 必须 |
| 不按正确性分组的总回答scatter，或同计数随机类别scatter | 判断收益是否特有于reward条件化 | 机制诊断；主候选有效后补训练 |

三种方法的Q均只由第一遍16次signal构成。额外signal控制固定原Q，不重新扩展candidate span；新旧signal统计按同权平均。三种方法的unique prompts、缓存与rank一致，分别报告一遍和两遍backward的成本。若reward-scatter只胜过一遍signal，却不胜过两遍signal，则不能声称其成本效益来自条件方差信息。

可与 `real_sign` 共享真实 correctness、筛选和缓存，但“width8+SVD”版本应有独立方法定义。不要把它与现成的未筛选 Psign 直接比较后声称纯排序贡献。

### 6.4 与历史失败的关系及否证标准

历史 Stable-SNR 用跨 batch 均值/残差，把 prompt 异质性也纳入噪声；这里的 N 来自**同题、同 reward 类别内部**，并按 contrast 均值方差缩放。它是新的估计对象，不是把原失败 artifact 换名。数学结构接近正则化判别分析/广义特征排序，论文新颖性须落在 RL contrast、便宜估计和有效证据上，不能宣称首创 SNR/LDA。

优先检查：同缓存枚举/小矩阵数值正确性、PSD、重放一致性、排序是否被极小 N 支配、独立题组的 contrast capture 与长度/正确性分层表现。最终需要同预算训练 Avg@k 改善，且 Pass@k/难题覆盖没有系统性损失，才支持“去掉了无益 rollout 波动”的解释。

这里的独立audit指与basis构造不重叠的probe题组，不是把协议剩余637行改成held-out训练集；另行采样/反传的诊断成本也需记录。若用audit挑超参，它便成为calibration，必须另留最终audit。

若只提高 discovery capture，独立 audit/训练不提高，则判为拟合 probe 噪声；若 Pass@k下降更严重，则保留“抑制了有用解法多样性”的可能性，不包装成全面 RL 能力提升。不能用 benchmark reward 反过来挑 ridge、mask 或方向。

## 7. 其他候选的优先级

**Masked-GradTop：保留，优先级低于 sign 和 fixed-A。** 当前已实现 top-surprisal50%、min128/final128，分母仍用原完整 token 数；和16-observation GradTop比较。它检验过程 token 选择与初始化效果。surprisal不是因果推理重要性，也不是梯度范数；将loss置零不会让整条序列反传成本减半。若有效，补同密度random mask后才声称选token规则本身有效。暂不与sign或reward-scatter叠加。

**同题独立 rollout 交叉矩：后续研究备选。** 两组各K8、各自计算advantage，才有清楚的独立性；同组先算8条advantage再拆4+4会引入共同随机归一化。交叉矩有限样本可能不PSD，而且固定总rollout预算会减少unique prompts，需要auto-moment对照。旧 P3/Phase0.6 有明确 no-go/invalid；本方案不复用其 C artifact。优先选上面的PSD类内scatter，避免第一轮同时引入更多采样和负谱处理。

**在线换基/动态 rank/Core：暂后置。** On-policy分布漂移是合理动机，但换基时还要保持当前ΔW、迁移Adam状态、同步vLLM和完成snapshot/resume。Core需要第二遍GQ、固定输出基及新参数化；都不适合用来回答第一轮的probe效率。先从已有50/100/150轨迹测固定子空间的投影覆盖是否下降，再决定是否值得做在线适应。

## 8. 如何证明低成本 probe 是优势

| Probe | Rollout steps | Attempted prompts | Responses | 有效题组 | Width |
| --- | ---: | ---: | ---: | ---: | --- |
| REAL64 | 64 | 4096 | 32768 | 1499，已测 | 输入128/输出128 |
| REAL32前缀 | 32 | 2048 | 16384 | 735，已测 | 输入128/输出128 |
| GradTop16 | 4 | 256 | 2048 | 需报告informative组数 | 输出8，capacity64 |
| Psign8 | 2 | 128 | 1024 | 需报告informative组数 | 输出8，capacity64 |
| reward-scatter提案 | 4 | 256 | 2048 | 2≤n+≤6，待测 | signal/noise输出8，Q64 |

来源为两个 REAL [完整summary](../../runs/analysis/real_probe_r32_s64_p0_seed42_20261013/summary.json) / [32步summary](../../runs/analysis/real_probe_r32_s32_seed42_replayed/summary.json) 和已有 GradTop 配方。不同筛选下有效组数含义不同，不能只比accepted数量。

REAL64 已实测 wall=40085.41秒≈11.13小时，4卡预留约44.54 GPU-hours；[执行记录](../../runs/ops/real_r32_lr1e5_seed42_after_eval/probe_execution.json)。64条日志计时合计约11.09小时，其中 generation4.51小时、old-logp0.82小时、actor observation5.76小时。因此在同一批采样下，单独加速反传不能消掉生成成本。REAL32源自64步运行保存的统计前缀，32步实际采集成本应从原日志前缀计算；离线导出时间不是完整probe时间。

Psign8相对REAL32/64的**采样条数**为1/16、1/32；GradTop16为1/8、1/16。宽度8相对128也小16倍。这些是配置比例，不是已测墙钟加速或显存比例。当前 REAL 逐response反传、各DP rank复制同一组，存在工程开销；正式效率结论应区分小采样预算、批量统计和REAL实现并行效率，至少补同缓存小预算REAL控制，并披露或优化其DP复制。

必须记录四套成本：

1. 采样：attempted/accepted/unique prompts、responses、有效和生成token数。
2. 反传：backward tokens、实际microbatch backward数、逻辑observation/梯度收集次数。
3. 系统：采样、logp、actor、收集、CPU分解/导出各自wall time；总wall/GPU-hours；每卡峰allocated/reserved与进程显存。
4. 完整流程：probe+正式训练到固定step的总成本，以及共同checkpoint网格下达到预声明质量门槛的成本。

主要图应为 Macro Avg@k 对 probe GPU-hours、Macro 对 probe+training GPU-hours、Avg@k与Pass@k/长度的联合变化。不能把oracle/full-gradient审计或额外重放从研究成本中隐藏；一次性诊断与部署所需probe可以分别列出。

效率成立有两种结果：同训练预算下精度更高且probe更省；或者精度在事前确定的非劣界内且总成本显著下降。建议新候选使用0.5 pp作为初步非劣界，以相对REAL32的题目配对CI下界>−0.5 pp为单seed筛查标准，同时披露CI宽度；这不等于跨training-seed非劣证明。没有显著差异也不等于等效。

## 9. 分阶段执行单

| 阶段 | 工作 | 主要问题 | 规模 |
| --- | --- | --- | --- |
| A：现成主线 | gradtop、short、prompt_mean、prompt_sign | 预算、归一化、sign贡献 | 4条×step50；已有LoRA/REAL复用符合口径的结果 |
| A2：并行备选 | gradtop_masked | token selector是否值得继续 | 1条×step50 |
| B：冻结方式 | 同Psign artifact fixed-A、同GradTop artifact fixed-A、random-orthogonal fixed-A | 廉价probe能否同时减少训练参数 | 3条×step50；已有AB各作对照 |
| B2：REAL归因 | 同REAL32 artifact A+B | REAL basis和冻结的作用 | 1条×step50，按论文claim需要补 |
| C：严格桥接 | REAL loss同cache direct/shared-Ψ/sign；宽度和预算分开消融 | 统计目标一致下的成本/误差 | 先小probe诊断，再选预声明方案训练 |
| D：新RL-specific | 同Q64 signal-only、reward-scatter、重复signal的等反传预算控制 | 同reward类内波动是否应被抑制，是否值得额外反传 | 3条×step50；都按相同筛选/fixed-A |
| E：轨迹确认 | 入选GradTop、LoRA、REAL32共同100/150网格 | 持续收益还是早期加速 | 精确续训；复用已有对照轨迹 |

A阶段如预算只能容纳4条，先做四个sign链条控制；A2后置。各方法的预定50-step都完整完成，不能仅因step10的训练reward低就截停形成不公平筛选。不得直接将新方法提升到270再与50-step对照比较。

新机制先预声明固定参数和主要比较，再看结果。若A阶段Psign无效，仍可报告“归一化有效/缩短预算足够”等实际结论；不要强行把所有收益汇到sign。D阶段只有在估计器数值、完整缓存重放和协议验证完成后才启动；这里的要求是保证测到所声称的方法，不以主观几何gate代替训练效果。

论文确认阶段需要共同LR预算，及另行登记的跨trajectory或第二模型验证。当前的单seed题目bootstrap只描述评测题目不确定性；如果资源暂不支持更多训练seed，清楚限定为当前配置的screen，不能把它写成robust superiority。

## 10. 代码落点与验证清单

| 工作 | 主要文件 | 完成条件 |
| --- | --- | --- |
| 现成Psign/Mask运行 | `scripts/local/run_gradtop_candidates_4gpu.sh`、`verl/utils/gradtop_objectives.py` | 绑定同local assets，四卡probe/训练/snapshot实际通过 |
| fixed-A新方法登记 | protocol JSON、`scripts/analysis/verify_rlvr_protocol.py`、统一launcher | 同basis哈希，A无梯度/不更新，B0，完整导出和恢复 |
| REAL-sign与cache | `verl/workers/actor/dp_actor.py`、`verl/trainer/ppo/ray_trainer.py`、`verl/utils/peft_real.py` | 有效组计数加权、共享Ψ粒度明确、按相同rollout比较 |
| reward-scatter | 新的objective/accumulator模块，接入现有collector | class-centered系数、全局分母、C/N PSD、ridge/QR、同Q对照 |
| 成本与指标 | run contract、probe summary、逐样本eval records | 成本分项、Macro口径、problem-cluster paired bootstrap、行为指标 |

必要验证应直接对应数学风险：极小矩阵dense梯度与sketch一致；sign枚举消掉跨题项；类内centered符号估计N；完整组跨microbatch/DP不改变目标；无优化器更新/初始函数不变；同缓存重放结果一致；FSDP收集、导出/加载、精确恢复和vLLM输出一致。冻结方式还应验证optimizer只更新预期参数，不能仅看requires_grad日志。

本次用CPU float64随机矩阵做了完整符号枚举：4题的prompt-sign二阶矩恒等式最大误差约8.9e-16；8条回答按2正确/6错误划分的类内方差估计，枚举256种符号后最大误差约6.7e-16，估计矩阵为PSD。这只确认代数恒等式，不证明GPU执行正确、有限width估计可靠或下游准确率提升。

当前已有候选文档记录155项CPU测试通过；本文未重新运行这些已有实现测试，也未生成新的模型性能结果。对新增方法的GPU验证及正式训练仍属于后续实施工作。

## 11. 论文论点的边界

可以检验的主命题是：**在分组RL中，用批量随机投影保留逐题reward contrast的二阶结构，能否以更低采样/反传成本找到足够好的rank32适配空间；再利用同题同reward回答的波动，是否能改善有限rollout下的方向选择。**

其中，Prompt-sign提供便宜的统计估计；fixed-A提供与REAL对齐的训练参数化；reward-scatter提供额外的RL统计结构。三者分别需要机制对照，不能仅靠一个组合方法的最高分证明。若新增可靠性排序不奏效，保留低成本初始化的独立结论；若只能匹配准确率却显著省时间，也可以是有价值的效率结果。

不应预先声称：已超过REAL、32倍墙钟加速、彻底去除RL噪声、首次使用reward-gradient sketch、fixed-A普遍更好、mask恢复了真实token credit、或方法跨seed稳定。
