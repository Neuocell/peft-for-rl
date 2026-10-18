# GradTop 研究路线：共享策略更新、逐题能量与 RL 中的子空间漂移

日期：2026-10-15（Asia/Shanghai）。状态：研究问题与可执行诊断设计；本轮只写文档，不实现新收集器，不运行 GPU probe、训练或评测。

本文把**原始 GradTop 的 batch 聚合更新**作为研究起点。核心问题是：在有限 rank 的 RL 适配中，应该保留每道题各自较大的梯度方向，还是保留经过 rollout 和题目平均后、实际推动共享策略改善的更新方向？两者何时一致、何时分离，分离来自采样波动、题间异质性，还是训练中的策略变化？

这是待验证的研究问题，不是已经成立的新方法。Prompt-sign 只作机制对照，不作为默认改进；fixed-A、mask、归一化开关及更换矩阵分解器，也不单独承担论文贡献。本文的优先级取代 [10 月 14 日下一轮设计](gradtop_real_next_round_design_20261014_zh.md) 中“优先推进 sign”的建议；原文件保留为历史讨论。

## 1. 先固定对原方法的理解

原始 GradTop 先在一个逻辑 optimizer minibatch 内聚合梯度，再跨 observation 累积 sketch 二阶能量。它不是把全部数据梯度平均成一张矩阵后做一次 SVD。

REAL 则先形成每道题完整回答组的梯度，再累积其未中心化二阶矩。其 loss、回答长度归一化、有效组筛选、sketch 宽度和正式训练的冻结方式还与当前 GradTop 不同，因此现有两条端到端结果不能直接隔离“先聚合还是先平方”。

| 维度 | 原 GradTop 主线 | 当前 REAL 实现 | 本轮如何控制 |
| --- | --- | --- | --- |
| 平方前的统计单位 | 一个 batch 的聚合梯度 | 一个 prompt 的完整回答组梯度 | 同 loss、同 cache、同题组，仅改变平方前聚合 |
| 输出侧压缩 | 随机投影梯度 | 随机投影梯度 | 使用相同投影与宽度，另留独立评估投影 |
| 累积方式 | 流式低秩因子、增量截断 SVD | 流式 Nyström 统计 | 主诊断先排除分解器差异；之后才比较工程估计器 |
| 时间遗忘 | 原 energy 累积没有显式时间权重 | 默认累计统计没有显式时间权重 | 不把流式处理当作时间适应 |
| 正式训练 | 当前注册 GradTop 为 A/B 均训练 | 当前 REAL 固定 A、只训练 B | 几何诊断不混入冻结方式的收益 |

**增量 SVD 本身不会自动识别“新方向”，也不自动遗忘过时策略。** 它压缩截至当前的累计信息；有限 capacity 的反复截断可能带来顺序依赖和近似误差，这些不等于有目的的时间遗忘。REAL 也已经是流式算法，不能把“流式、免存 dense gradient、使用随机投影”写成区别于 REAL 的首次贡献。代码依据见 [GradTop 收集与累积](../../verl/utils/peft_gradient_subspace.py)、[REAL 实现说明](../real_reward_gradient_probe_zh.md)。

## 2. 三个可证伪假设

**H1：逐题能量中有相当一部分来自有限 rollout 的组级估计波动；batch 聚合能抑制这部分能量，并更好地保留独立样本上的共享更新。** 如果逐题能量方向在独立重复上同样可复现，且同预算下保留更多未来更新，H1 不成立。不能把所有题间差异都叫噪声。

**H2：部分被 batch 平均削弱的方向代表真实的题间/题型差异，其价值随训练阶段变化。** 如果这些方向仅有当前样本能量，没有独立重复、题型一致性或后续更新证据，就不能据此主张应当保留它们。反之，如果它们稳定承载困难题或未来阶段的更新，单纯强化共享均值也可能有害。

**H3：固定初始化子空间的主要限制可能是 on-policy 更新几何随策略变化，而非初始估计精度。** 只有跨 checkpoint 的变化超过同一 checkpoint 内的重复采样误差，并且旧子空间在新策略下损失可重复的更新覆盖，才进入在线适应研究。仅观察主角度改变不够。

这三个假设允许不同结果：原 GradTop 可能已经选对统计目标；REAL 的逐题目标可能更合理；两者可能各适合不同阶段；也可能所有几何指标都不能预测最终 RL 效果。最后一种结果同样是停止堆叠局部修改的依据。

## 3. 理论参照：为何原聚合有直接依据

先考虑等权、独立同分布的完整 prompt-group 梯度 \(G_p\)。令

\[
\mu=\mathbb E[G_p],\qquad
\Sigma=\mathbb E[(G_p-\mu)^\top(G_p-\mu)],\qquad
\bar G_m=\frac1m\sum_{p=1}^mG_p.
\]

则

\[
C_{\rm batch}(m)=\mathbb E[\bar G_m^\top\bar G_m]
=\mu^\top\mu+\frac1m\Sigma,
\qquad
C_{\rm prompt}=\mathbb E[G_p^\top G_p]
=\mu^\top\mu+\Sigma.
\]

给定正交输入子空间 \(Q\)，\(C_{\rm batch}\) 的 top-r eigenspace 最小化

\[
\mathbb E\|\bar G_m-\bar G_mQQ^\top\|_F^2.
\]

因此原聚合有“保留实际 batch 梯度更新”的直接动机。它既保留共同方向，也保留按 batch 大小缩小的变化方向。交叉项在期望中包含 \(\mu^\top\mu\)，不能一概当作应该删除的污染。

随机符号 \(Z=m^{-1/2}\sum_p\xi_pG_p\) 满足

\[
\mathbb E_\xi[Z^\top Z\mid G_{1:m}]
=\frac1m\sum_pG_p^\top G_p.
\]

它把目标换为逐题二阶矩，使题间方差相对共同均值的权重增大。这个恒等式不保证新目标更有用，也不保证有限 sign 次数或有限 sketch 得到的子空间更好。

反例：\(\ell_\xi(w)=\tfrac12(w_1-1)^2+\tfrac12(w_2-2\xi)^2\)，\(\xi=\pm1\)。在 \(w=0\) 时，逐样本二阶矩为 \(\mathrm{diag}(1,4)\)，rank1 选择只有采样波动的第二轴；\(m=16\) 的 batch 二阶矩为 \(\mathrm{diag}(1,1/4)\)，选择通向总体最优点 \((1,0)\) 的第一轴。它说明“减少抵消”不是独立的正确性依据。

上述 \(G_p\) 指现有 loss 与采样机制实际产生的梯度。GRPO 组内 advantage 标准化、clipping 等使其未必是原始期望回报梯度的无偏估计，不能自动将 \(\mu\) 等同于 \(\nabla J\)。SGD 投影解释也不能直接推广为 Adam 最优性。

## 4. 最小诊断样本与层选择

第一阶段只回答 base policy 下的 H1/H2；通过后再用已有严格协议 LoRA step50 测 H3。step100 为事前保留的扩展，不因为某个 checkpoint 更好看而替换时间点。

| 阶段 | 策略状态 | 固定题目数 | 每题独立回答组 | 每组 K | 新生成 responses |
| --- | --- | ---: | ---: | ---: | ---: |
| D0，必需 | base policy，记为 t=0 | 128 | 2 | 8 | 2048 |
| D1，D0 通过后 | 现有严格 LoRA rank32、LR1e-5、seed42 的 step50 | 同一128题 | 2 | 8 | 2048 |
| D2，条件扩展 | 同一 LoRA 轨迹的 step100 | 同一128题 | 2 | 8 | 2048 |

D0+D1 共4096条回答；完整 D0+D1+D2 共6144条。它们是研究诊断预算，不是宣称新方法实际部署时需要的 probe 预算。每组仍是 K=8；不能把两组并为 K=16 后重算 advantage，否则研究对象发生变化。

128题从正式训练源数据中用预先固定的稳定 ID/hash 顺序选取，64题 discovery、64题 audit，先保存题目 manifest 再生成回答。audit 只表示不参与子空间拟合；它不是额外 benchmark，也不是训练 held-out 集。不得把协议中自然剩余的637条改称验证集。按训练 sampler 记录这些题在各 checkpoint 前是否已被消费，避免把训练暴露差异误写成纯策略变化。

同一题的两组使用不同且可追踪的诊断 RNG 子流：由 base seed42、prompt ID、checkpoint、repeat ID 派生。不能两次重置相同生成 seed 而得到相同样本；多卡复制同一组也不算独立重复。诊断子流不改变正式训练 seed42 或正式 sampler/RNG 状态。

代表模块预先固定为28层模型中的第0、13、27层，各取 `self_attn.q_proj` 与 `mlp.down_proj`，共6个模块，覆盖早/中/晚层及 attention/MLP。诊断结果只支持这些模块，不能直接外推到全部196个目标模块。

对每个模块采用零函数探针，冻结策略参数，读取 \(H=\Psi^\top G\)，不形成 dense base gradient。拟合投影 \(\Psi_A\) 和评估投影 \(\Psi_B\) 各宽32，互相独立；可以拼为总宽64的一次收集，分块保存。两组回答、全部题目和各 checkpoint 在同一模块内共用相应的 \(\Psi_A/\Psi_B\)。探针在安装前后必须保持 logits 一致，不更新 optimizer。

FP32 保存两组投影下的完整输入侧 H，单 checkpoint 的张量预算约为

\[
128\times2\times(32+32)\times
3(1536+8960)\times4\ \text{bytes}\approx1.92\ \text{GiB}.
\]

这是完整 prompt-group H 的 CPU/磁盘统计量预算，不含模型、激活、rollout cache、临时分解内存，以及第5节正式执行权重分支所需的额外统计量，也不是 GPU 显存估计。仅凭这些 H 和每组 token 数，不能重建当前正式 actor 的全部梯度。若需更小的首轮筛查，可另用固定、独立于数据的128维输入投影画低维统计图；该图只诊断随机坐标，不替代完整输入侧的 rank32 选基和覆盖评估。

## 5. 同 cache、同 loss：分开处理机制与真实训练权重

每个策略状态只生成一次上述双重复 cache。保存 token IDs、attention/response mask、原始正确性、稳定题目 ID、repeat ID、真实 token 数、生成设置、生成策略身份，以及该组的 old logp 和原 GRPO advantage。以后比较聚合方式均重放同一 cache；改变随机符号不重新生成回答。

主诊断不使用 REAL 的“至少两正两负”筛选。全对/全错造成的零 advantage 组也必须保留，否则会改变要分解的采样分布。

为避免 token 权重与聚合方式混淆，保留三个明确分支：

1. **等题机制分支。** 令 \(L_p\) 为完整 K8 组内部的 token-mean GRPO loss，\(G_p=\nabla L_p\)。保持相同 clipping、advantage 和回答 token，比较等题 \(C_{\rm prompt}\) 与 \(C_{\rm batch}(m)\)，\(m\in\{1,4,16,64\}\)。这是用于解释恒等式的离线诊断，不宣称其等题权重等于正式 actor 的执行权重。
2. **理想全局 token-mean／当前显式 GradTop probe 分支。** 对固定16题 batch，令 \(T_p\) 为该组有效 token 数，\(T_B=\sum_pT_p\)，全局 token-mean 梯度为 \(G_B=\sum_p(T_p/T_B)G_p\)。定义同 batch 内贡献 \(F_{p|B}=16(T_p/T_B)G_p\)，比较 \(G_B^\top G_B\) 与 \(16^{-1}\sum_pF_{p|B}^\top F_{p|B}\)。两者使用同一组贡献和权重，只改变聚合/平方顺序。这对应当前显式 GradTop probe 修正后的全局分母，不等于当前正式训练 actor 的 reduction。
3. **正式执行权重重放分支。** 当前正式 actor 仍在每个 microbatch 内做 token mean，随后乘该 microbatch 的 response 样本数占 local minibatch 的比例，最后由 FSDP/DP 平均。需要复现实际长度均衡、DP分配、逻辑 update 成员和动态 microbatch 打包，逐 token 重建其真实系数；不能用第2分支的 \(T_p/T_B\) 代替。

具体地，动态 batch 路径下，令 d 为 DP rank，j 为该 rank 的 microbatch，\(n_{dj}\) 为其 response 数，\(N_d\) 为实际配置的 local minibatch response 数，\(T_{dj}\) 为有效 token 数。在 D 个 rank 平均的前提下，属于该 microbatch 的有效 token 的 loss 系数为

\[
c_{dj}=\frac1D\frac{n_{dj}}{N_d}\frac1{T_{dj}},\qquad
G_{\rm formal,raw}
=\sum_{d,j}c_{dj}\sum_{i\in\mathcal M_{dj},t}
m_{it}\nabla\ell_{it}.
\]

这是梯度裁剪和 Adam 处理之前的累计梯度公式；所有计数、分母、world size 及归约约定都从实际运行读取，不由“16 prompts”推断。非动态路径另按实际 `1/gradient_accumulation` 系数登记。代码依据为 [正式 actor 与显式 GradTop 分支](../../verl/workers/actor/dp_actor.py)。不同 microbatch 的平均回答长度不同时，\(c_{dj}\) 通常不同于统一的全局 token 分母。

完整组 \(H_p=\Psi^\top\nabla L_p\) 已把组内回答按统一分母混合。如果同一题的回答进入不同 microbatch，甚至不同逻辑 update，仅存 \(H_p,T_p\) 会丢失重加权所需的信息。正式执行权重分支必须选择并登记以下实际收集方式：按真实 \(c_{dj}\) 分别累计每个逻辑 update 内的 prompt/partial-prompt weighted sketches，保留足够的成员与分母元数据；或者在同 cache 上额外执行完整正式打包路径的直接 batch backward。后者足以得到 batch 累计梯度，但若要再比较“逐题先平方”的同权重对照，仍需额外保留该 update 内各题/partial-prompt 的贡献，不能从 batch 总和逆推出它们。

这些额外 weighted sketches、直接 backward、CPU/磁盘和 backward tokens 单独计入诊断成本，不包含在第4节1.92 GiB估算里。重放固定 checkpoint、无 optimizer step 可以验证 reduction 和几何；它不会自动复现正式训练在连续更新后产生的真实梯度轨迹。需要时用第10节的训练 instrumentation 补充后者。

第2/3分支权重依赖 rollout 长度及打包，也可能与梯度相关，不直接套用 iid 等权公式。对确定、独立于梯度的权重才有 \(C=\mu^\top\mu+(\sum_pa_p^2)\Sigma\)。第6节的双重复分解首先适用于第1分支定义的完整组估计器；迁移到正式执行权重时须重新声明条件分布与部分组贡献，不能直接沿用结论。

REAL 原 loss 仅作为后续敏感性分支：在预先固定的32题上重算该 loss，并对两种聚合使用完全相同的定义和 cache。若沿用 REAL 筛选，则报告两次重复都通过的题数及该条件选择的总体；不能将结果外推到128题全体。该分支额外 backward 成本单独记账，不能和主 GRPO 分支混合归因。

## 6. 用两组独立 K8 分离组内波动与题间差异

对固定策略、固定题 p，写作

\[
G_{pa}=\mu_p+\epsilon_{pa},\qquad
G_{pb}=\mu_p+\epsilon_{pb},\qquad
\mathbb E[\epsilon_{pc}\mid p]=0.
\]

两组在给定题和策略下独立，\(\mu_p\) 是“该算法 K8 组级梯度估计器”的条件均值。不要把8条回答内部的梯度当作8个独立 GRPO 梯度，因为 advantage 共享组统计。

用 \(\operatorname{sym}(X)=(X+X^\top)/2\)，在 P 题上定义

\[
\widehat U=\frac1{2P}\sum_p(G_{pa}^\top G_{pa}+G_{pb}^\top G_{pb}),
\]

\[
\widehat N=\frac1{2P}\sum_p(G_{pa}-G_{pb})^\top(G_{pa}-G_{pb}),
\qquad
\widehat S=\frac1P\sum_p\operatorname{sym}(G_{pa}^\top G_{pb}).
\]

有限样本中已经有精确代数关系 \(\widehat U=\widehat S+\widehat N\)。其期望分别对应逐题总能量、K8 rollout 估计波动、可跨独立回答组复现的逐题能量。

再令 \(\bar G_a=P^{-1}\sum_pG_{pa}\)、\(\bar G_b=P^{-1}\sum_pG_{pb}\)，定义

\[
\widehat M=\operatorname{sym}(\bar G_a^\top\bar G_b),
\qquad
\widehat H=\widehat S-\widehat M.
\]

条件于固定题目库，其期望为共享均值能量 \(M=\bar\mu^\top\bar\mu\) 和题间异质性 \(H=P^{-1}\sum_p(\mu_p-\bar\mu)^\top(\mu_p-\bar\mu)\)。由该有限题目库无放回抽取 m 题时，理想 batch 二阶矩为

\[
M+\frac{P-m}{m(P-1)}H+\frac1mN.
\]

这一区分很关键：当 m=P，题目抽样变化消失，但每题只采一组 K8 带来的 rollout 波动仍存在。

\(\widehat S,\widehat M,\widehat H\) 在有限样本下可能不半正定。保留并报告负特征值、交叉能量与不确定性，不能先裁成 PSD 再声称完成无偏分解。可在预先指定的方向、固定候选空间内报告 trace/交叉能量；接近零的分母不计算不稳定比例。

**同一投影是交叉矩成立的前提。** 对 \(H_{pa}=\Psi^\top G_{pa}\)、\(H_{pb}=\Psi^\top G_{pb}\)，共享且独立于 rollout 的 \(\Psi\) 满足 \(\mathbb E[\Psi\Psi^\top]=I\)，于是对投影取期望，交叉矩对应 \(G_{pa}^\top G_{pb}\)。若两组各用独立、零均值的 \(\Psi_a,\Psi_b\)，直接计算 \(H_{pa}^\top H_{pb}\) 会在投影期望中趋于零，无法估计共享信号。

同一投影也用于跨题、跨时间内积。输出投影的随机性与 rollout 随机性分别登记；同 cache 换投影可以检验 sketch 稳定性，不能充当新的 rollout 重复。

## 7. 从题间异质性进一步检查题型差异

只有题间方差不等于发现题型机制。先检查源数据是否有可信类别；若没有，可在看 rollout/reward 前，按问题文本固定标注代数、几何、数论、组合/其他，允许“无法判定”，不根据结果重新分组。小于8题的类别不单独作稳定结论。

令 \(\bar G_{ca},\bar G_{cb}\) 为类别 c 两次重复的均值，\(w_c=P_c/P\)，则

\[
\widehat B=\sum_cw_c\operatorname{sym}(\bar G_{ca}^\top\bar G_{cb})-\widehat M,
\]

\[
\widehat W=\widehat S-
\sum_cw_c\operatorname{sym}(\bar G_{ca}^\top\bar G_{cb}),
\qquad \widehat H=\widehat B+\widehat W.
\]

B/W 对应类别间与类别内部的条件均值差异。若大部分 H 在 W 中，不能用粗题型解释；如果按同一次 rollout 的成功率分难度，会引入结果选择，应只作描述性图，不代替这个预先分组的分解。

## 8. 子空间拟合、独立评估与 sign 的位置

只在64道 discovery 题、投影 A 上拟合 rank32 的 \(Q_{\rm batch}\) 与 \(Q_{\rm prompt}\)。主要 batch 大小是16；m=1/4/64仅作缩放规律检查。m=64 的独立 batch 数很少，须报告样本/谱秩限制，不能把其训练内高覆盖当作可靠结果。

两种目标共用相同的数值求解精度和输出 rank。离线可使用保存的 H 以矩阵乘法算子求 top32，避免显式构造 \(d_{in}\times d_{in}\) 矩阵；检查特征残差。如果沿用截断 SVD，必须同 capacity、同顺序，另在小规模上量化截断误差。不要在这一步同时用“GradTop width8/capacity64”和“REAL width128/Nyström”，否则统计目标与近似质量再次混在一起。

在64道 audit 题、独立投影 B 上评估：

- 总梯度能量覆盖，以及等题、全局 token-mean／显式 GradTop probe 两种 batch 梯度覆盖；完成第5节额外收集后，才另报正式执行权重下的 batch 梯度覆盖。
- 双重复交叉能量覆盖，即 \(\operatorname{tr}(Q^\top\widehat S Q)\) 与 \(\operatorname{tr}(Q^\top\widehat M Q)\)，同时列绝对值、分母和区间。
- 被选择方向上的 \(\widehat N,\widehat H\) 能量及题型分布。
- rank32 空间的主角度、有效谱秩、缺少信号支持的维数。

再交换 A/B 的角色作预先声明的敏感性检查；不挑选更好看的投影。按 prompt 成对 bootstrap，两组回答和不同时间点一起重采样，输出2000次重采样的区间；这只是单训练 seed 的诊断不确定性，不是训练稳定性证据。

sign 对照在已经保存的 H 上离线计算 \(m^{-1/2}\sum_p\xi_pH_p\)，同一题完整 K8 共用符号，先固定8套符号，另用256套只验证趋近逐题目标。它不增加本轮 rollout，也不需要为每套符号重跑 backward。这里可精确检查无偏恒等式和有限符号方差；不能把离线已有逐题 H 的成本说成 sign 部署时的批量反传成本。

如果以后验证 sign 的工程省时，需要单独实现并测量真实批量 backward，同时补偿 microbatch/DP 分母，并与同 loss、同 cache 的逐题路径比较。这项工程对照不改变本研究的统计假设。

## 9. 策略漂移：跨时间变化必须超过重复噪声

D1 在同一128题上，用 step50 策略独立生成两组 K8，重复 D0 全部统计；投影和题目 split 固定。主要参考轨迹先选已有 LoRA，避免只在由某候选方法塑造的策略上证明该方法自己的目标正确。需要时再把同样审计用于 REAL/GradTop 轨迹。

比较 t=0/50 的 \(M,S,N,H\)、rank32 空间，以及 \(Q_0\) 在 t=50 独立 audit 更新上的覆盖。每个 t 内用 repeat a/b 各自拟合得到重复采样的基线差异；跨时间主角度和覆盖下降须与该基线一起报告。如果跨时间变化没有超过同时间重复差异，只能判为证据不足。

on-policy 比较中，模型参数和采样回答分布会共同变化。若需区分两者，在预先固定的16道 discovery 题上补 \(\theta\in\{\theta_0,\theta_{50}\}\) 与 cache 来源 \(D\in\{D_0,D_{50}\}\) 的2×2重放：对角为各自 on-policy，非对角重用 cache 所属策略的 old logp、advantage 和 token。非对角是 off-policy surrogate 梯度，必须记录 importance ratio、clipping 比例和非有限值，不能称为对应策略的真实 on-policy 更新。

这两个非对角单元增加 \(16\times2\times8\times2=512\) 条 response 的反向重放，不新增生成；计入诊断 backward tokens。若 clipping 使这些单元几乎全失活，就不能靠这个重放解释参数与分布的独立因果作用。

## 10. 实际 Adam 更新：只能用真实状态回答

原梯度几何与实际 Adam 更新几何分开报告。A/B 因子上的 Adam，不等于对 dense W 的梯度做一个假想 Adam 更新；也不等于 \(GQQ^\top\) 加同一个逐坐标预条件器。

若以后在严格协议训练中增加只读 instrumentation，对一次真实 optimizer update 保存对应模块更新前后的因子。有效权重增量为

\[
\Delta W=s[(B+\Delta B)(A+\Delta A)-BA]
=s(B\Delta A+\Delta B A+\Delta B\Delta A).
\]

可直接用低秩因子计算其右子空间覆盖，不必存 dense \(\Delta W\)。同时登记真实 Adam 的 first/second moments、step、epsilon、betas、weight decay、梯度裁剪及跳过更新。仅有每10个 trainer steps 的 adapter snapshot，只能计算区间累计权重变化，不能恢复中间每次 Adam 更新；完整 optimizer state 的存在也不能恢复未记录的历史梯度。

本轮不重跑训练来补这项数据。若目前只有 snapshot，就把它标成“累计有效权重位移”，并保留它可能已经被原 LoRA 参数化限制的边界。后续方法成立的证据应包括真实更新覆盖与最终 reward/benchmark；不能仅靠虚拟 dense Adam 或当前 surrogate loss 下的一阶内积。

## 11. 结果如何决定是否值得发展成方法

| 诊断结果 | 允许的解释 | 后续研究动作 |
| --- | --- | --- |
| batch 空间在独立 audit 上保留更多共享/未来更新；prompt 空间主要增加 N | 逐题能量混入与共同更新无关的采样波动 | 研究如何以低成本估计、保留可预测策略改善的更新空间，原 GradTop 作强基线 |
| prompt 空间增加稳定 H/B，并覆盖后续阶段的真实更新 | 平均会削弱某些真实任务方向 | 研究按未来更新需求保留多任务能力；不能仅根据初始共享均值剪除这些方向 |
| 初始空间已足够，但跨时间覆盖显著退化，且超过重复噪声 | 非平稳性可能是主要限制 | 才进入在线跟踪/更换子空间，明确触发准则、预算、保持当前函数及迁移 optimizer 状态 |
| 聚合差异小于投影/重复采样误差 | 当前样本不足以支持机制 | 预先允许一次128→256题扩展；保留原 split 成员并分别追加，不据此挑新超参 |
| 几何改善不预测实际更新或正式 RL 效果 | 代理目标可能选错 | 否定该故事，不把更多 mask/sign/冻结开关堆成方法 |

几何筛查以预先指定的 m=16、rank32 为主要比较，逐模块报告区间和效应量。只有方向在两次投影角色交换下稳定、在多数代表模块和独立 audit 上可复现，才推进正式方法设计；这些是探索性门槛，不当作统计显著性或最终论文结论。

真正的新方法必须回答“为何该空间能更好地支持 RL 策略改善”，并给出一个围绕该目标的完整估计与适应机制、理论边界、足够强的反例和对照。换一个累计公式、把 SVD 改成 Nyström、加一次冻结或缩短 probe，都不足以自动回答这个问题。若最终结果仅是同一 REAL 目标的更便宜估计，应如实定位为效率研究并引用 REAL，不包装成不相干的新目标。

## 12. 正式实验协议、成本与执行交付

后续正式训练以 [RLVR_PROTOCOL.md](../../RLVR_PROTOCOL.md) 和 [机器协议](../../configs/experiments/rlvr_dapo_math_1p5b_protocol_v1.json) 为准：DeepSeek-R1-Distill-Qwen-1.5B、4 GPU、bf16、all-linear、uniform rank32、alpha64、dropout0、LR1e-5、全链路 seed42、64 prompts/trainer step、K8、PPO minibatch16、1 epoch、1024/8192 token、原 token-mean DAPO-style GRPO配置、原 clipping、无额外 KL/entropy/长度奖励。这里的 token-mean 配置保留当前正式 actor 的 microbatch token mean、response 样本比例加权和 DP 平均执行方式，不能宣称已经等于统一的全局 token mean。本轮诊断不借机修改正式 reduction；若以后修改，须作为新的共享协议变化独立登记，不能只改某个候选方法后仍称严格 matched。

一个 trainer step 对应4次逻辑 optimizer update；step50为200次。正式训练从原 sampler 起点开始，诊断不推进训练 RNG/sampler。先固定 step10/50 的共同评测网格；晋级方法和强对照共同续到100/150。rank16/64及跨训练 seed 只在主线成立后另行预声明，不能混入当前 seed42 主对照。

新增方法执行前登记方法定义、内部预算、artifact 哈希和不可变 run contract，并沿用已审计的共同 `local_asset_binding`；不静默修改 canonical 资产哈希，不使用旧 local JSON 漏掉新的方法定义。

成本分成三张账：

- **一次性研究诊断：** 本文双重复生成、逐题反传、两组投影、正式执行权重所需的额外 weighted sketches/直接 batch backward、敏感性 loss 和跨时间重放；明确 GPU-hours、rollout/backward tokens、CPU/磁盘/峰值显存。
- **方法实际 probe：** 新用户执行该方法必需的采样、梯度收集、分解、同步和导出，不能借用诊断 oracle 却不计成本。
- **正式训练与评测：** 固定 optimizer 更新数、总 wall/GPU-hours、六榜 Macro Avg@k、Pass@k、parse/hit-max、输出长度；同时报告 probe+training 总成本。

执行时的最小交付顺序如下；相关新收集/分析入口目前尚未实现，不给出可直接运行的虚构命令。

1. 固化 `diagnostic_manifest.json`：协议/资产/策略身份、128题 ID、64/64 split、6个模块、loss、两个 K8 子流、投影种子及64总宽、m网格、r32、时间点和成本口径。
2. 保存两组 cache，确认题目、组、old logp、advantage、token mask 和生成身份可重放；小样本验证探针零函数、无 optimizer 更新、微批次分母和完整组不被误拆。
3. 输出逐题 `H_A/H_B`、必要元数据和成本记录。先验证双重复/投影身份，再做交叉矩；检查 \(\widehat U=\widehat S+\widehat N\) 数值残差。全局 token-mean 重组对照显式 GradTop backward；正式执行权重另收 weighted sketches 或额外直接 backward，并按真实打包验证，不能仅由完整组 H 推断其正确性。
4. 只在 discovery 上拟合，冻结 Q 后输出 audit 的 `geometry_metrics.json`、题目成对 bootstrap、模块表和图。报告全部预声明统计，不只展示最有利方向。
5. D0 结果支持机制后，执行 D1；根据第11节选择继续研究、扩样或否定。形成可审查的具体方法定义后，再实现正式 probe 和训练链路。

本文件没有把上述诊断写成已完成实验，也没有预设 GradTop 必须优于 REAL。首要产出是明确哪一类梯度方向值得有限 rank 长期保留，以及现有聚合与逐题能量各自会在哪些 RL 条件下失效。
