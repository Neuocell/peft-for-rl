# 当前 GradTop-r32 / REAL 方法审计

日期：2026-10-16（Asia/Shanghai）。范围仅为当前实验 `rlvr_v1_gradtop_r32_a64_lr1em5_b64m16n8_seed42_20261015` 的算法、协议、probe与运行代码，以及当前REAL的数学定义和原始PDF依据。本次只读核验，唯一写入文件为本报告，没有启动probe、训练、评测或GPU测试。

## 1. 当前实验身份与运行代码

当前正式方法为 `METHOD=gradtop`、`gradient_probe_variant=token_mean`：**batch聚合、无prompt sign、无token mask、正式训练A/B均可训练**。名称中的r32指正式adapter rank；probe输出sketch宽度为8。

当前experiment的step10/50训练与评测已经完成，正在从完整step50恢复到100，先评测100，再从完整100恢复到150并评测。审计读取的[运行状态](../gradtop_r32_continue_step150_20261015/status.json)时间为 `2026-10-16T00:46:18.854090+08:00`，状态 `train_to100`、日志步数86、已完成评测仍为10/50。该状态为读取时点，后续以status为准，不能把计划100/150写成已完成。

当前运行[plan](../gradtop_r32_continue_step150_20261015/plan.json)的32个 `code_sha256` 已逐个重新读取计算，**全部一致，零漂移**。plan SHA256为 `a98a83cb423040b55122855c5622491b437bab6820e6b03d688d94db65f3bdb6`。controller运行期间不要编辑被锁定的源码；新会话所需说明应写入独立文件。

其中每条rollout的确定性采样种子实现在 [agent_loop.py:477](/root/peft-for-rl/verl/experimental/agent_loop/agent_loop.py:477)，函数 `trajectory_sampling_seed` 位于第869行，键为base seed、global step、sample index、rollout序号及validate标识。该文件已纳入当前32个源码锁定，SHA256为 `cf9fa73556b2c5d7143330ceef880ddcec24cb3ad4900e2b70c1c116376072dc`。

主要执行依据：[机器协议](../../../configs/experiments/gradtop_r32_clean_20261015.json)、[当前GradTop实验说明](../../../docs/experiments/gradtop_r32_clean_baseline_20261015_zh.md)、[续训登记](../../../docs/experiments/gradtop_r32_continue_step150_20261015_zh.md)。

## 2. 当前 GradTop probe 的数学对象

设模块权重 `W ∈ R^(d_out×d_in)`，一个observation的聚合loss梯度为 `G_t=∂L_t/∂W`。probe临时LoRA的 `A_probe=0`，`B_t∈R^(d_out×8)` 为每个observation独立刷新的Gaussian投影，每元素方差 `1/8`。probe保持初始模型函数不变，收集

```text
S_t = grad(A_probe).T / scaling = G_t.T @ B_t     # [d_in, 8]
F_k = rank64_QR_SVD([F_(k-1), S_(4k-3), ..., S_(4k)])
```

先对拼接因子作薄QR，再对小R作SVD，累计因子保留 `U*singular_values`。最终取前32个单位输入方向，导出 `A=Q.T`、`B=0`；A不携带奇异值幅度，正式训练A/B都更新。

对独立输出投影取期望，未截断时累计的是observation聚合梯度的 `sum_t G_t.T G_t`。同一observation内不同prompt之间的交叉项仍在；跨observation保存能量，没有先把全部G求成一个总均值。该对象与REAL的逐题 `sum_p G_p.T G_p` 不同。

| 当前probe字段 | 固定定义 |
| --- | --- |
| observation | 16个完整prompt × 每题8 responses，全局128 responses |
| 总预算 | 16 observations =4个64-prompt rollout step =256 prompts、2048 responses |
| sketch / capacity | 输出宽度8 / 累计capacity64 |
| 压缩时点 | observations 4、8、12、16之后 |
| loss | GRPO advantage与PPO分支；显式全局有效response-token分母；含DP平均补偿 |
| 分组 | probe禁用trainer级length balance，完整K=8题组先分配到同一observation；内部microbatch可拆分 |
| 筛选 | 不采用REAL至少两正两负筛选；零advantage题组仍保留在分母 |
| sign / mask | 无 / 无 |
| 更新 / clipping | 没有Adam更新，没有梯度裁剪 |
| rank | 196模块统一32；`rank_bins=[32]` |
| energy threshold | `.95`只作诊断，不控制本轮rank，不能保证每层捕获95%能量 |
| 不足秩处理 | 确定性坐标QR补齐；本轮artifact记录补齐方向总数0 |
| 停止条件 | 固定4个rollout step，不使用稳定性提前停止 |

当前累计器没有指数遗忘、时间衰减或单位范数重权。有限capacity的重复截断可能产生顺序依赖，但没有实现主动适应policy漂移的机制。

精确代码入口：

- [peft_gradient_subspace.py:70](/root/peft-for-rl/verl/utils/peft_gradient_subspace.py:70)：随机B与零A；第136行薄QR/SVD；第192行累计器；第228行接收sketch；第238行压缩；第1394行capture/refresh；第1494行step末压缩；第1525行导出；第1710行加载A并将B置零。
- [gradtop_objectives.py:62](/root/peft-for-rl/verl/utils/gradtop_objectives.py:62)：完整题组、全局分母、DP补偿；第120行显式probe loss；第158行无sign的token_mean分支。
- [dp_actor.py:1566](/root/peft-for-rl/verl/workers/actor/dp_actor.py:1566)：显式probe令额外loss_scale_factor=1；第1758行只capture、不调用optimizer/clipping。
- [ray_trainer.py:1433](/root/peft-for-rl/verl/trainer/ppo/ray_trainer.py:1433)：probe完整题组前置检查。
- [run_gradtop_candidates_4gpu.sh:20](/root/peft-for-rl/scripts/local/run_gradtop_candidates_4gpu.sh:20)：`METHOD=gradtop`映射token_mean/16 observations。该脚本的默认METHOD并非gradtop，执行当前方法必须显式指定 `METHOD=gradtop`。

实际probe依据：[artifact summary](../../analysis/gradtop_r32_o16_w8_c64_seed42_20261015/summary.json)、[probe验收](../gradtop_r32_clean_20261015/probe_artifact_verification.json)、[执行记录](../gradtop_r32_clean_20261015/probe_execution.json)。执行记录的完整probe wall time为2250.3308秒，即约37.51分钟；16 observations不表示仅16次实际backward。

## 3. 当前正式训练与评测协议

| 项目 | 当前固定值 |
| --- | --- |
| 模型 / precision | DeepSeek-R1-Distill-Qwen-1.5B / bf16 |
| GPU | 4×L40S |
| adapter | 196个all-linear模块，rank32 / alpha64 / scaling2 / dropout0 |
| 初始化 / 可训练状态 | `A=Q.T, B=0`；A/B均训练，36,929,536可训练参数 |
| 优化 | LR1e-5，constant scheduler，warmup0，weight decay0 |
| rollout / PPO | 64 prompts/step，每题8 responses，PPO minibatch16，1 epoch |
| token上限 | prompt1024 / completion8192 |
| 正式loss | token-mean DAPO-style GRPO，clip low.20 / high.28 / C10 |
| 其他训练项 | 无KL reward、actor KL、entropy bonus、overlong shaping |
| seeds | data、rollout、FSDP、trainer、初始化均42 |
| step口径 | 每trainer step包含4次逻辑optimizer update；step50/100/150分别200/400/600次 |
| 保存 | 每10step adapter snapshot；每50step及计划终点完整恢复点；完整恢复点保留最近2份 |
| 评测 | 六榜7248 records，902题，seed42、temperature.6、top_p.95、max_new_tokens32768、max_model_len34816 |
| 采样次数 | AIME24/AIME25/AMC23/HMMT为32次；MATH500/Minerva为4次 |
| 报告 | 逐任务Avg@k/Pass@k、六任务等权Macro、parse/hit-max/平均长度；配对problem-cluster bootstrap |

**probe梯度与正式actor训练梯度须区分。** 当前probe采用global token mean；正式actor每个microbatch内部取token mean，再按microbatch样本占比累积，最后跨DP平均，见 [dp_actor.py:1566](/root/peft-for-rl/verl/workers/actor/dp_actor.py:1566)。二者一般不等，不能直接把probe的G称为真实Adam输入梯度。后续机制诊断必须控制归约、length balance、microbatch系数，以及Adam预条件与LoRA参数化的影响。

续训只延长同experiment预算，不重新probe、不更换artifact。完整恢复包含model、optimizer、RNG、scheduler及dataloader；独立adapter snapshot不作为完整恢复点。正式训练的constant scheduler加warmup0使延长目标不改变LR曲线。

## 4. 当前 REAL 的原始论文身份

原始PDF：[1683_Reward_Gradient_Aligned_L.pdf](/root/1683_Reward_Gradient_Aligned_L.pdf)，SHA256 `dd89c53528b0d4a657b5009fb3e1df9d34e4eb28f651112df725531004afbd7f`。标题为 *Reward-Gradient Aligned Low-Rank Adaptation for Post-Training*。

PDF首页直接标注 **“Under review as a conference paper at ICLR 2027”**、**“Anonymous authors”**、**“Paper under double-blind review”**。据此可称“所提供的ICLR2027匿名在审版本”。不能按当前日历年份猜成ICLR2026，也不能称已录用或已发表。

第5–6页§3.2–3.4给出统计目标、两侧压缩与pipeline；第16页Appendix B Algorithm1明确 `A←νP.T, B←0; freeze A`，随后只更新B。第10页Reproducibility Statement（印刷行499–500）称 **“Upon acceptance, we will release our code, training configurations, and evaluation scripts”**。PDF没有提供可核验的作者源码链接；本次在仓库文档和本机已知位置未找到可归属REAL作者的独立checkout。当前repo remote是 `Neuocell/peft-for-rl`，其中 `verl/utils/peft_real.py` 应称本地paper-based实现，不能称作者官方代码。

“未找到作者repo”是本次本地证据边界，不能证明作者当前仍未公开代码。OpenReview论坛ID、首发日期、正式投稿日期、录用状态均未核实；文件名1683不能代替官方投稿身份链接。

第17–18页Appendix E已有仅观察阶段使用、正确性条件化的相对长度奖励，随后训练只用正确性reward。因此“观察时加入length-aware reward”本身不能声称为相对REAL的新颖点。当前本地REAL实现是二元正确性probe，未把该长度奖励分支混入对照。

## 5. 当前 REAL 的数学定义和代码

对完整prompt组p的K条responses，使用原始正确性 `R_i∈{0,1}`，至少两正两负，每条response至少一个有效token。使用population标准差：

```text
a_i = (R_i - mean(R)) / std_population(R)
J_p = (1/K) sum_i a_i [sum_t mask_it logpi_it / L_i]
G_p = grad_W J_p
C = mean_p G_p.T G_p                  # 未中心化逐题二阶矩
```

它按response长度归一化再平均，不是global token mean；目标不是中心化协方差，也不是 `mean(G).T mean(G)`。本地代码最小化 `-J_p`，符号不影响二阶矩。不能直接复用正式GRPO预计算的advantage作为REAL Eq.6，因为std的correction/epsilon及reward处理可能不同。

输出投影每题刷新 `Psi_p∈R^(d_out×s_out)`，元素方差 `1/s_out`；输入投影 `Omega∈R^(d_in×s_in)` 为标准Gaussian、跨题固定：

```text
H_p = Psi_p.T @ G_p
Y = mean_p H_p.T @ (H_p @ Omega)
M = sym(Omega.T @ Y)
F = Y @ M^(dagger/2)
P = top-r left singular vectors of F
A = P.T, B = 0; freeze A, train B
```

代码入口：

- [peft_real.py:49](/root/peft-for-rl/verl/utils/peft_real.py:49)：reward系数；第93行零输出random-B probe；第140行Nyström；第199行流式累加器；第263行逐题collector。
- [dp_actor.py:517](/root/peft-for-rl/verl/workers/actor/dp_actor.py:517)：每题逐response backward，先完整累计H再平方，最后刷新Psi；不能逐response平方，否则丢失正确/错误回答之间的交叉项。
- [ray_trainer.py:1591](/root/peft-for-rl/verl/trainer/ppo/ray_trainer.py:1591)：完整组筛选和去重。
- [run_rlvr_protocol_1p5b_4gpu.sh:152](/root/peft-for-rl/scripts/local/run_rlvr_protocol_1p5b_4gpu.sh:152)：REAL固定A。

REAL也不构造或收集dense base gradient。本地用 `A_probe=0, B_probe=Psi_p` 读取 `grad_A/scaling=-H_p`，冻结base与B，只对A求导；所有DP rank处理相同完整prompt组，避免FSDP先平均不同题再平方。本地逐题逐response执行与DP复制带来的吞吐代价，未经作者代码核实，不能归为作者实现必然具有的开销。FSDP冻结参数仍可能分配零gradient buffer，因此估计器持久缓冲小不等于整卡峰值同比下降。

当前本地REAL为r32、`s_out=s_in=128`、alpha64、正式训练LR1e-5/GRPO，B-only20,643,840参数。当前GradTop为A+B共36,929,536参数；同rank比较不等于等可训练参数比较。论文Qwen3实验用r16、distilled用r64，并含GSPO设置；本地对齐共享训练协议，不是论文原实验完整复现。论文观察阶段的总预算表述与本地独立probe后重新启动正式训练的计数方式也须分开说明。

## 6. 当前方法表述的边界

1. 当前GradTop保留无sign的batch聚合；不能称已执行prompt-sign或REAL逐题二阶矩。
2. 增量QR/SVD只压缩累计sketch能量，没有主动识别新方向或遗忘策略的机制。
3. REAL已有免dense gradient、流式随机投影和Nyström，不能把这些性质单独写成当前GradTop首次提出的贡献。
4. probe的global token mean不等于正式actor的microbatch加权归约，更不等于Adam预条件后的有效更新。
5. rank32对rank32不等于等参数、等probe预算或等总计算；效率主张应同时报告实际wall/GPU-hours、rollout/backward tokens、有效组、峰值显存与效果。
6. `target_energy=.95`不保证本轮uniform-r32每层捕获95%能量。
7. 运行计划、代码存在和预检通过不能替代完整artifact、checkpoint、records与verification所提供的执行证据。
