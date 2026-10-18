# 当前 GradTop 结束于270步：REAL 等 prompt 预算长跑与下降原因诊断

登记：2026-10-17，Asia/Shanghai；执行状态更新于12:43。当前 GradTop 已完成270步训练及六榜评测：250段于04:09完成，270段于07:09完成，控制器和后继队列均为complete。251首轮实际四GPU adapter验收通过，actor/engine/ready/request各4份回执齐全。用户随后要求先做诊断，诊断于11:06启动、12:42全部完成，结果见第7节；REAL新长跑仍未启动。本文只使用当前 GradTop `rlvr_v1_gradtop_r32_a64_lr1em5_b64m16n8_seed42_20261015`；不混入旧 GradTop 版本。

最新结果见[270终点及完整轨迹表](../../runs/ops/gradtop_r32_resume270_syncverified_20261017/results.md)。270的Macro Avg@k为41.6845%，相对250为+0.0811pp（题目配对95%区间[-1.0473,+1.2553]pp），相对150为−0.7125pp（[-1.7373,+0.2848]pp），均跨零；后段没有明确性能收益。评测解析率从150的99.03%降至250的96.96%、270的95.50%，需纳入后续行为诊断，不能只概括为始终稳定。检查时四GPU均空闲，磁盘剩余约14GiB。

## 1. 已登记执行：250评测后续到270并结束

保留同一probe、rank32/alpha64、A/B均训练、LR1e-5 constant、64 prompts×8 responses、minibatch16及seed42。从完整250恢复model、Adam、scheduler、四rank RNG和dataloader。270累计1,080次逻辑Adam更新、17,280次训练prompt访问、138,240条训练回答。真实训练数据17,917行，batch64、drop_last产生279批/epoch，因此270仍在第一epoch，无需改epoch逻辑。

执行顺序固定为：250训练完成 → 250六榜评测及报告验收 → 原控制器退出 → 绑定实际250源文件SHA → 270控制器预检 → 验证恢复后首轮实际GPU adapter → 训练到270 → 六榜评测及结果汇总 → 停止。

排队时不预填尚未生成的250源哈希。已冻结50个代码/协议文件、前序控制器PID及启动标识；只有前序状态、完整源checkpoint、评测报告和同步证据全部通过后，才生成最终续训plan及新的同步token。本次已经按此流程完成源绑定与执行。任何失败停止后继启动。依赖逻辑8个分支检查已通过，包括前序仍运行、退出竞态、错误退出、身份不符、终点不完整及缺失同步证据。

251首轮重新检查源checkpoint→actor全部392个A/B张量→vLLM注册adapter→CUDA活动槽位→首次生成请求，未复用201的验收回执。201及251的实际四GPU验收均已通过，251源adapter SHA为`c2006b6d9b01c6c438dbe69e9a10c73fbf00c44de2246ce668aae64fd8817947`。270后的报告已比较270−250、270−200及270−150，使用题目配对bootstrap 10,000次、seed42。

每10步保留adapter快照；270完整保存后保留完整250/270恢复点。50/100/150/200/250已评测adapter另行保留，原始评测records和审计材料保留。磁盘预检不提前计入尚未发生的checkpoint回收，并为checkpoint与快照各留10%增长裕量及2GiB余量。

执行入口：

- [270队列状态](../../runs/ops/gradtop_r32_resume270_syncverified_20261017/queue_status.json)、[队列登记](../../runs/ops/gradtop_r32_resume270_syncverified_20261017/queue_plan.json)、[依赖检查](../../runs/ops/gradtop_r32_resume270_syncverified_20261017/queue_checks.json)
- [当前250状态](../../runs/ops/gradtop_r32_resume250_syncverified_20261016/status.json)、[当前训练日志](../../runs/ops/gradtop_r32_resume250_syncverified_20261016/train_to250.log)
- [270队列程序](../../runs/ops/gradtop_r32_resume270_syncverified_20261017/enqueue.py)、[续训控制器](../../runs/ops/gradtop_r32_resume270_syncverified_20261017/controller.py)

执行文件已全部生成于同一270目录：`plan.json`在源250完成验收后生成；训练日志为`train_to270.log`，首轮证据为`first_rollout_verification.json`，终点表为`results.md`。`report_verification.json`确认270完整checkpoint、四rank optimizer/scheduler/RNG、数据游标270批/17,280题及全部评测验收通过。

## 2. 预算按实际尝试的prompt数对齐

| 设置 | Probe rollout steps | 尝试prompts | 生成responses | 纳入题组 | 分解及冻结方式 | 实测probe耗时 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 当前GradTop | 4 | 256 | 2,048 | 256 | width8/capacity64增量SVD；正式A+B训练 | 2,250.33秒，37.51分钟 |
| 旧REAL32 | 32 | 2,048 | 16,384 | 735 | 输入/输出sketch宽128；Nyström；固定A | 从64步累计缓存重建，无独立32步耗时 |
| 旧REAL64 | 64 | 4,096 | 32,768 | 1,499 | 同上 | 40,085.41秒，11.135小时 |
| 拟议REAL-P256 | 4 | 256 | 2,048 | 实跑后记录 | 保留原REAL定义与宽度128、固定A | 待测 |

REAL32/64命名中的32/64指probe rollout steps，正式rank均为32。GradTop的16 observations是16个minibatch梯度观察，每个rollout step含4次观察，故总计4步，不能误设为REAL 16步。REAL保留K8下至少2正2负筛选；“同prompt数”按筛选前尝试数计算，不为凑256个有效题组额外采样。

旧REAL32/64分别用了当前GradTop的8倍/16倍prompt预算。REAL64与GradTop约17.81倍的实测端到端probe耗时比包含预算差异，不能当作同预算速度优势。新实验另记生成tokens、有效题组、梯度计算次数、峰值显存、probe墙钟/GPU小时，以及正式训练和评测成本。相同prompt数不等于相同计算量；REAL与GradTop正式可训练参数也分别为20,643,840和36,929,536。

现有REAL缓存仅有step032/048/064的累计量及omega，没有step004缓存或逐题梯度。不能通过截取累计矩阵列、除以8等操作得到真实前4步子空间；P256必须重新probe。

依据：[当前GradTop probe](../../runs/analysis/gradtop_r32_o16_w8_c64_seed42_20261015/summary.json)、[REAL32重建probe](../../runs/analysis/real_probe_r32_s32_seed42_replayed/summary.json)。当前GradTop详细定义仍以[当前协议对照](gradtop_r32_clean_baseline_20261015_zh.md)为准。

## 3. 旧REAL下降：已核实的现象

对象为`rlvr_v1_real_probe32_r32_a64_lr1em5_b64m16n8_seed42`，不是REAL64。重新核对50/100/150六榜records，均为902题、7,248回答。

| REAL checkpoint | Avg.（按回答加权） | Macro Avg@k | Macro Pass@k | Parse | Hit-max | 平均输出tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 47.8339% | 42.0021% | 64.0748% | 96.3162% | 1.6694% | 5,952.61 |
| 150 | 45.5022% | 39.4078% | 61.7490% | 91.9840% | 3.0767% | 5,399.94 |
| 150−100 | −2.3317 pp | −2.5943 pp | −2.3258 pp | −4.3322 pp | +1.4073 pp | −552.67 |

Macro下降的题目配对95% bootstrap区间为[-3.7915, -1.4357]个百分点。各任务Avg@k差异依次为AIME24 −6.7708、AIME25 −0.4167、AMC23 −0.3906、HMMT +0.6250、MATH500 −0.8000、Minerva −7.8125个百分点。下降主要集中在AIME24和Minerva，并非六榜统一崩溃。该区间只反映固定训练seed下题目抽样不确定性。

训练1–150步完整连续；所检查的指标均有限，未发现NaN、OOM、Traceback或持续梯度爆炸。LR始终1e-5，PPO clipfrac约4–5×10^-5。下面是训练日志窗口均值，不能与上面的32k评测行为指标混用：

| 方法与窗口 | Reward | Parse | 平均长度 | Hit-max | Entropy | Grad norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| REAL 81–100 | 0.51289 | 89.814% | 5,236.77 | 14.482% | 0.86292 | 0.07275 |
| REAL 111–120 | 0.51758 | 96.230% | 4,762.39 | 4.590% | 1.06866 | 0.08178 |
| REAL 131–150 | 0.47520 | 95.439% | 4,608.83 | 4.756% | 1.15506 | 0.08276 |
| 当前GradTop 81–100 | 0.50557 | 89.150% | 5,449.65 | 16.367% | 0.85052 | 0.06717 |
| 当前GradTop 131–150 | 0.50010 | 92.051% | 5,482.69 | 13.184% | 0.78601 | 0.06897 |
| 当前GradTop 181–200 | 0.49951 | 95.771% | 5,190.05 | 7.881% | 0.75812 | 0.07249 |

REAL后段更准确的描述是熵上升、输出缩短、reward回落及验证性能下降，而非熵坍缩或数值训练崩溃。不同step的prompt组成变化，窗口相关性本身不能建立因果关系。当前GradTop也不是持续涨分：150→200的Macro为42.3971%→42.0933%，差值−0.3038个百分点，配对区间[-1.3862,+0.7508]跨零；目前支持后段训练相对平稳、评测进入平台。

### 3.1 恢复实现是必须排除的混杂因素

| 轨迹与step | Parse | Grad norm | Hit-max |
| --- | ---: | ---: | ---: |
| REAL 50 | 89.453% | 0.06935 | 12.109% |
| REAL 51，恢复首轮 | 59.961% | 0.14759 | 42.383% |
| REAL 52 | 93.555% | 0.07536 | 11.523% |
| REAL 100 | 88.281% | 0.07055 | 17.773% |
| REAL 101，恢复首轮 | 55.664% | 0.21885 | 48.438% |
| REAL 102 | 91.992% | 0.07667 | 12.891% |
| 当前GradTop 51，旧恢复实现 | 58.398% | 0.18011 | 42.969% |
| 当前GradTop 101，旧恢复实现 | 55.273% | 0.26432 | 50.977% |

REAL和当前GradTop的旧计划绑定了相同`fsdp_workers.py`哈希`109c3297fd6ffa5ffbfda5f5d7a22a7012af83a0447cee5861096a146fdf3298`。REAL日志确认`grad_subspace`、`freeze_a=True`、`lora_adapter_path=None`、dummy加载，走相关native LoRA恢复路径。结合GradTop已发现的恢复首轮base-only问题，REAL的相似异常高度可疑。

REAL51/101没有历史实际GPU张量回执，因此不能声称已运行时证实REAL的每次错误，也不能把100→150整个下降归因于一个恢复批次。现在重跑REAL的首要作用是排除这项实现混杂。当前GradTop151/201已按实际源权重验证，251将再次验证；这无法抹除当前轨迹51/101的历史异常。修复后新REAL与这条GradTop长跑只能先作探索对照，论文主结论还需要从头跑当前GradTop配置的干净轨迹。

### 3.2 “A可训练所以持续旋转”目前缺乏支持

对实际保存的196个模块adapter作CPU检查：REAL的A在50/100/150完全相同；GradTop的A确实变化。聚合有效权重增量范数（scaling=2）如下：

| 方法 | step50的‖ΔW‖F | step100 | step150 |
| --- | ---: | ---: | ---: |
| REAL32 | 1.34234 | 1.83885 | 2.30654 |
| 当前GradTop | 1.30823 | 1.80758 | 2.20703 |

使用小Gram精确计算`‖2BA‖F²=4·trace((BᵀB)(AAᵀ))`并跨模块求和，未见明显权重范数爆炸。对GradTop保存的Aᵀ做QR，50→150子空间重叠`‖Q50ᵀQ150‖F²/32`平均为0.9999412，最小0.9997123；A相对Frobenius变化平均0.7755%、最大1.7692%。因此不能把优势预先解释成明显子空间旋转。固定A与A+B仍是一个可检验因素，小幅变化也可能有影响，但需要独立消融和灵敏度证据。

审计文件：[训练审计及源SHA](../../runs/ops/real_gradtop_budget_audit_20261017/training_audit.json)、[REAL逐步CSV](../../runs/ops/real_gradtop_budget_audit_20261017/real32_training_steps.csv)、[当前GradTop逐步CSV](../../runs/ops/real_gradtop_budget_audit_20261017/gradtop_training_steps.csv)、[REAL评测汇总](../../runs/ops/real_gradtop_budget_audit_20261017/real_evaluation_summary.json)、[150−100配对统计](../../runs/ops/real_gradtop_budget_audit_20261017/real_step150_vs100.json)、[adapter范数](../../runs/ops/real_gradtop_budget_audit_20261017/adapter_geometry_audit.json)、[GradTop A漂移](../../runs/ops/real_gradtop_budget_audit_20261017/gradtop_A_drift.json)。

## 4. 可执行的下一轮REAL设计

### 4.1 优先做的两个探索实验

| 实验 | 初始化来源 | 正式训练 | 首要回答 | 状态 |
| --- | --- | --- | --- | --- |
| REAL-P256 | 新probe，4 rollout steps；256 prompts×8 | 修复后的实现，用户于10月18日改为0→210 | 同GradTop prompt预算下REAL的性能及后段轨迹 | 210已评测；15:48按用户要求中断200、取消150，GPU已释放，见第10节 |
| REAL-P2048-repaired | 校验后复用原REAL32 artifact | 新建B=0、optimizer及data游标，0→270 | 同一REAL初始化在修复实现后是否仍下降 | 计划 |
| 当前GradTop-P256 | 原16 observations artifact | 当前轨迹结束于270 | 已有探索参照，保留恢复历史说明 | 已完成训练及评测 |

第二项从头正式训练，不能直接从旧REAL150继续，因为那样不能排除早期错误影响。复用原大预算artifact节省probe费用，同时尽量固定旧REAL的初始子空间。两个REAL新跑之间仍有probe采样实现/随机实现差异，不是严格单变量预算消融。

若探索支持继续研究预算，下一阶段采集一次新的32步REAL probe，事先保存step4和step32的累计统计、投影矩阵及采样标识，导出共享嵌套前缀的P256/P2048。同seed、同随机投影、同前256题及回答，只改变累计样本数。不要先导出4步后重启一个“同seed”的32步probe就假定随机实现逐位相同。必要时增加同配置GradTop干净重跑和额外训练seed；未补齐前只报告单seed探索结果。

### 4.2 共享正式协议及REAL定义

模型DeepSeek-R1-Distill-Qwen-1.5B、4×L40S、bf16、rank32/alpha64/scaling2、196个all-linear模块、dropout0；训练数据及benchmark资产哈希沿用当前协议。LR1e-5 constant、warmup0、weight decay0，64 prompts/step、K8、minibatch16、PPO epochs1，prompt上限1024、response上限8192。沿用当前token-mean DAPO-style GRPO和clip low0.20/high0.28/C10，不加KL reward、actor KL、entropy bonus或overlong shaping。data/rollout/FSDP/trainer及初始化seed均42。正式训练270步，probe不推进正式训练data游标、不携带optimizer状态。

REAL保留逐题reward contrast、response长度归一化、未中心化二阶矩、输入/输出sketch128、Nyström、至少2正2负及固定A/B可训练。低预算主对照不同时缩sketch宽度、不取消筛选、不解冻A。若4步不足以产生所需有效秩，逐模块披露信号秩和原实现的补齐/拒绝行为；不得静默增加prompt预算或换算法后仍标为同一个REAL-P256。

正式新跑优先连续0→270保存快照，训练结束后离线评测50/100/150/200/250/270，减少人为恢复点。主终点预先定270，同时报告100→150及150→270变化，不能事后只挑最好checkpoint。每10步adapter供110–160局部复核；完整恢复点按已有磁盘保留策略管理。

六榜仍902题、7,248回答，eval seed42、temperature0.6、top-p0.95、max_new_tokens32768；逐任务Avg@k/Pass@k、加权Avg.、等权Macro及parse/hit-max/length全部报告。AIME24/25、AMC23、HMMT每题32回答，MATH500及Minerva每题4回答。方法差异与时间差异均按题配对bootstrap 10,000次；保留完整records，不把题目区间当作跨训练seed证据。

启动前将这些值转为独立冻结协议，并验证REAL与当前同步机制兼容。新建B=0的初始策略不能直接套用“恢复源B必须非零”的检查：初始化检查A正确、B为零及与base一致，首次更新后检查非零adapter同步；任何非零checkpoint恢复使用源→actor→vLLM实际GPU→请求ID的完整验收。这里是待完成的REAL启动集成要求，不表示新REAL已获运行时验证。

结果表每行必须带方法、probe尝试/有效prompt数、probe tokens/GPU小时、A冻结方式、step、训练seed、同步验收及恢复历史，再给上述分数。预算相同、参数相同、计算量相同三个口径分别表述。

## 5. 从观察走向机制：保留逐题能量，还是保留可靠的共享更新？

**2026-10-17执行更新：用户已明确先做诊断，下述原建议顺序不再作为执行前置条件。** 已启动现有checkpoint的固定策略诊断，具体配置、状态入口及缺失点见第6节。REAL修复后重跑仍是后续用于归因的对照，当前诊断不等待该训练完成。

研究起点沿用[共享更新方向研究路线](gradtop_update_geometry_research_20261015_zh.md)：REAL估计逐题reward-gradient二阶能量；当前GradTop先在16-prompt minibatch内聚合，再通过随机sketch与增量SVD累计。这两种统计量对跨题抵消的处理不同。需要判断差别来自采样噪声、题型异质性还是策略阶段漂移，以及哪些保留方向能预测后续真实收益，而非先假定REAL目标有问题。

以下按依赖顺序推进，避免一次增加许多训练消融：

1. **先隔离恢复问题。** 比较相同旧REAL初始化在修复实现下的100→150行为及评测变化。如果原下降消失，应优先归因于实现/运行路径差异；如仍下降，才加大对目标和子空间机制的投入。单次复跑仍不能排除训练随机性。
2. **再检验probe样本量。** 对嵌套P256/P2048记录有效题组数、难度/题型分布、谱与有效秩、初始子空间重叠及独立prompt上的梯度覆盖。有效题组变多但下游270无改善，才能讨论边际预算收益有限；一次“不显著”不能证明等效。若作非劣结论，预先采用Macro 1个百分点的探索容忍界，要求P256−P2048的配对区间下界高于−1pp，并另外披露训练seed不足。
3. **测量共享更新的可靠性。** 预先从不参与probe的训练源划出64个固定诊断prompts，在step100/150/270分别按各自策略采两组独立K8回答，共3,072条诊断回答/方法；这部分预算另列，不进入训练。保留题型、难度分层和两组采样标识，固定公共随机sketch，不按结果换题。
4. **将指标变为可反驳的预测。** 若REAL逐题能量保留率高，但共享梯度保留率或独立rollout间一致性在150前显著下降，且与后续固定诊断reward及验证退化同向，才支持“高能量方向未必是有效共享更新”的解释。若两项均稳定，转查reward/解析行为、题型迁移及优化参数化，不强行套用该机制。

诊断必须在同一policy、同一rollout cache及同一loss归约下比较候选子空间；不能直接拿两个方法不同策略产生的梯度作因果差异。对每个受检policy，既投影到REAL的初始A子空间，也投影到GradTop的初始A子空间，另记录各自当前A。主诊断使用公共正式训练目标；原REAL contrast及GradTop probe全局token分母作为分开的目标对照。正式实现的microbatch-token-mean加权与probe全局token-mean并不等价，逐题分解时须保留实际microbatch和minibatch的权重。

可在不存完整G的情况下使用同一个固定输出sketch `H_p=ΨᵀG_p`。对输入正交基Q，同时估计：

- 逐题能量保留：`E_ind(Q)=Σp‖H_pQ‖F² / Σp‖H_p‖F²`。
- 共享更新保留：`E_shared(Q)=Σb‖(Σp∈b w_p H_p)Q‖F² / Σb‖Σp∈b w_p H_p‖F²`；16-prompt分组和w取公共训练归约规则，先聚合再平方。
- 两组独立rollout对相同prompt/minibatch的sketch内积、余弦及投影后重复性，用来区分一致信号与采样噪声；同时报告分母和零梯度比例，避免低能量组造成虚高比率。

固定64题诊断用于发现机制，不用于宣称泛化；若上述信号成立，再扩大prompt集和seed，并用匹配的短程分支训练测试“保留可靠共享方向”是否实际改善后续reward。冻结A对照只在需要判别参数化时添加，明确命名REAL-P256-A+B，不将其混入原REAL主基线。由于现有GradTop A子空间几乎不动，初始子空间质量与目标聚合方式目前比“大幅动态旋转”更值得优先探测。

增量SVD继续作为独立估计器差异保留：同一份sketch流分别离线做一次性SVD、当前capacity64分块增量SVD及顺序置换，测谱误差、子空间覆盖和CPU/内存成本；只有固定统计目标和sketch后，才能把差异归给分解器。增量SVD本身不会自动提供RL特有创新，论文方法需要由上述可靠性/策略漂移证据推出。

## 6. 本轮诊断执行配置与进度入口

**12:42:33完成：五个策略全部完成采样、反传和汇总，三组跨阶段比较已生成。** 共5120条回答、20份GPU adapter加载检查、20份HF加载/零函数probe检查；所有反传NLL有限，原始回答、统计输入、保留sketch及Gram哈希复核通过。见[最终验收](../../runs/ops/gradtop_real_gradient_diagnostics_20261017/completion_verification.json)。以下保留两次工程恢复记录，未改变诊断配置或重采样。

**12:38恢复记录：前4个策略已完成；GradTop270的1024条回答已完成并开始反传。** 12:32读取最后一批JSONL时，回答字符串内合法的U+2028被`str.splitlines()`误当作记录边界，导致解析报错；文件本身未损坏。改为逐物理行读取后，全部5120条记录完整且唯一，20个原始回答文件SHA均与采样结束记录一致。两处读取程序一并修复，原计划/代码/错误状态保存在`recovery_jsonl_unicode/`，新计划登记变更；不重新采样或更改科学配置。恢复时按真实数组尺寸核算剩余阶段磁盘峰值，保留2GiB余量，避免用首轮固定10GiB门槛阻挡可安全完成的末段。此前4个策略的所有GPU/HF验收、反传输入哈希和保留sketch/Gram已复核。

**11:30进度：GradTop100已完成，累计1/5个策略状态；队列已恢复并进入REAL100。** 11:24首个策略在CPU汇总时遇到PEFT adapter键名前缀差异，原始1024条回答和四份反传数据均已完成。修复仅将当前A的读取改为仓库已有canonical命名，采样、梯度和统计定义不变；五个策略的6个A键名及形状均已核查，GradTop100原始文件/归约输入哈希和保留sketch/Gram均已复核。旧计划、代码和错误状态保留于诊断目录`recovery_adapter_key/`，新计划记录前后哈希与恢复原因。11:29首点汇总成功，之后复用已完成材料继续队列。

逐点结果见[诊断结果表](../../runs/ops/gradtop_real_gradient_diagnostics_20261017/results.md)。11:30时只有GradTop100；现已全部补齐，跨阶段结果及解释见第7节。

2026-10-17 11:06正式启动，初始阶段为GradTop100的四GPU独立rollout。每个策略依次进行采样、正式归约权重重建、四GPU反传、CPU统计汇总；五个策略完成后自动做跨时间配对比较并结束。不进行optimizer更新，不改变训练checkpoint。

| 项目 | 本轮实际登记 |
| --- | --- |
| 策略顺序 | GradTop100 → REAL100 → GradTop150 → REAL150 → GradTop270 |
| REAL270 | 旧REAL只有150以内权重，270缺失；不新训练、不补造该点 |
| 固定题目 | 从原训练源按稳定ID/hash选64题，排除正式seed42 sampler前4096行，覆盖原两方法probe范围；同时记录各checkpoint前的训练暴露位置 |
| 标签 | 保留源ability=MATH；文本关键词标签预先保存，仅作启发式描述，不能称为人工验证题型；难度在采样前没有可靠标签 |
| Rollout | 每题2组独立K8；每策略1024条，5策略共5120条；temperature1、top-p1、top-k−1、response8192、prompt1024左截断 |
| RNG | seed由42、policy、稳定prompt ID及repeat派生；两组不合并成K16 |
| 代表模块 | 原研究方案已选定的第0/13/27层q_proj和down_proj，共6个；结论限定这些模块 |
| 共同输出投影 | 每模块固定Gaussian投影，两独立半区A/B各宽32；两次rollout及各策略共用相同投影 |
| 梯度采集 | 冻结所有原模型/adapter参数，额外安装A=0的零函数probe；一次response-mean NLL反传经标量权重重建各分支，无dense base梯度和optimizer更新 |
| 主目标 | 同一冻结policy下on-policy PPO在ratio=1处的梯度；保留正式GRPO样本标准差归一化（含epsilon1e-6） |
| 归约分支 | 等题组内token mean；固定16题全局token mean；正式4DP长度均衡、动态microbatch和4个逻辑update的partial-prompt加权重放 |
| REAL敏感性 | 预先选定前32题，在同cache上用原population-std、逐response长度归一化、至少2正2负contrast；统计两次都通过的有效题数 |
| 候选子空间 | 原GradTop初始化A、原REAL初始化A、受检策略当前A；同一policy/cache内共同评价 |
| 增量SVD | 同一梯度流，比较一次性SVD、capacity64增量SVD及逆序；拟合repeat0/投影A，评价repeat1/投影B。此估计器诊断width32，不作为正式width8 probe速度证据 |
| 统计 | 逐题能量、共享更新保留率、双重复交叉能量/余弦、噪声与异质性分解；题目配对bootstrap2000次；跨100→150及GradTop150→270变化 |

正式归约分支先对每次64题×8回答复现生产长度均衡和4DP分配，再逐逻辑update计算动态microbatch数及每条回答的系数`c=n_micro/(4×32×T_micro)`。同一题可分到不同update，分别保存partial-prompt贡献，避免将它错误当成完整16题组。所有update都冻结在同一checkpoint，不复现历史连续Adam更新；两组rollout长度不同也会改变正式打包，因此正式分支的重复性包含该变化。等题分支用于更直接的噪声分解。

启动检查已通过：零函数probe与显式dense-gradient投影恒等式；仓库PPO在ratio=1的导数；动态microbatch加权导数；Gram能量与直接矩阵计算、双重复分解恒等式。短样本GPU开发检查通过392个HF adapter源张量比对、源→vLLM注册及实际CUDA槽位比对、零probe前后logits逐项相同及反传梯度有限。开发检查不进入研究结果。正式每个policy/每个GPU还会重新进行adapter加载验收。

存储仅保留当前任务必要材料：原回答/token/logprobs/reward、正式打包系数、等题FP32完整输入侧sketch、各分支完整成对Gram及结果。正式partial-prompt和REAL敏感性临时sketch在完整统计保存后删除；其原token与系数仍可重新反传。无需生成合并模型权重。队列持有现有GPU资源锁，代码、权重、子空间和题目均绑定哈希；失败写error并停止后继阶段。

实际入口：

- [诊断状态](../../runs/ops/gradtop_real_gradient_diagnostics_20261017/status.json)、[控制器日志](../../runs/ops/gradtop_real_gradient_diagnostics_20261017/controller.log)、[冻结计划](../../runs/ops/gradtop_real_gradient_diagnostics_20261017/plan.json)
- [固定题目与训练暴露信息](../../runs/ops/gradtop_real_gradient_diagnostics_20261017/prompts.json)
- [采样/反传程序](../../scripts/analysis/run_gradtop_real_diagnostics.py)、[统计汇总程序](../../scripts/analysis/summarize_gradtop_real_diagnostics.py)
- 各policy子目录的`generate.0.log`至`generate.3.log`记录采样，`backward_status.0.json`至`backward_status.3.json`记录逐response反传进度；`analysis.json`在该policy全部完成后生成。
- 统一`runs/ops/gradtop_real_gradient_diagnostics_20261017/results.md`在第一个policy完成后生成并逐点更新；`trajectory.json`在五个policy全部完成后生成。尚未生成的结果不填数值。

## 7. 完成结果与当前能支持的解释

以下是正式动态microbatch归约下的共享更新能量保留率，按6个模块和两个独立投影等权平均。每一行的两个候选空间都在**同一policy、同一回答cache、同一loss**上评价；数值不是准确率。

| 受检policy | GradTop初始化子空间 | REAL初始化子空间 |
| --- | ---: | ---: |
| GradTop100 | 41.6873% | 47.2466% |
| REAL100 | 43.0857% | 49.5672% |
| GradTop150 | 43.4537% | 48.8018% |
| REAL150 | 39.2990% | 45.1432% |
| GradTop270 | 45.1482% | 51.6421% |

本轮结果不支持“REAL只保留单题能量、忽略共享更新，而GradTop在共享覆盖上更优”的简单解释。在这些模块和策略上，REAL初始化子空间对逐题能量和共享更新的保留率都更高。两种初始化来自不同probe预算（REAL2048、GradTop256 prompts），因此也不能据此断言REAL的统计目标本身更优或等预算下仍然占优。

REAL100→150存在可测的覆盖下降。等题分支中，REAL初始化空间的逐题能量保留率从48.752%降至44.952%，差值−3.8001pp，题目配对95%区间[-5.4688,−1.3823]pp；同一REAL策略上的GradTop初始化空间也下降3.2317pp，区间[-4.4676,−1.4601]pp。正式归约的共享覆盖点估计则分别下降4.4240pp和3.7868pp。两个空间同时下降，支持这一策略轨迹的梯度分布发生变化，尚不能将其归因于REAL特有的聚合缺陷、固定A或旧恢复异常中的某一项。

GradTop100→150的等题覆盖变化区间跨零；150→270两种初始化空间的等题覆盖反而上升：GradTop空间+3.7626pp，[+2.1623,+5.2730]pp；REAL空间+4.1512pp，[+2.6222,+5.5770]pp。与此同时，正式六榜Macro由42.3971%变为41.6845%。梯度能量覆盖上升并未对应六榜性能上升，当前覆盖指标不能单独充当未来reward改善的代理。

两组K8的可重复交叉能量整体接近零，多数逐模块区间跨零；部分负点估计和负区间照实保留。64题、两组回答和有限输出投影下，对稳定梯度信号的估计仍弱，不能把有限样本负值解释成“真实信号为负”，也不能据此直接宣布共享信号不存在。

固定同一统计目标与sketch流后，capacity64增量SVD和逆序增量SVD接近一次性SVD。在5个策略×6个模块的对照中，独立回答/投影上的共享能量保留率差值绝对值最大约0.0444个百分点。它支持这份诊断流上的分解近似误差较小；此处width32，不能代替正式width8 probe成本测试，也没有显示增量SVD本身带来额外性能收益。

诊断仅覆盖6个预选模块、64个训练源题目和单训练轨迹。旧REAL缺失270点；诊断完成时尚未补做REAL等预算probe或修复实现后的从头长跑，后续启动记录见第8节。证据详见[完整表](../../runs/ops/gradtop_real_gradient_diagnostics_20261017/results.md)、[跨阶段配对统计](../../runs/ops/gradtop_real_gradient_diagnostics_20261017/trajectory.json)及各policy的`analysis.json`。

## 8. REAL-P256长跑启动记录（2026-10-17）

北京时间13:01:13开始执行新probe。实验名为`rlvr_v1_real_p256_r32_a64_lr1em5_b64m16n8_seed42_20261017`。本次启动REAL-P256；REAL-P2048-repaired仍是待执行的独立方案。

冻结配置为[real_p256_r32_longrun270_20261017.json](../../configs/experiments/real_p256_r32_longrun270_20261017.json)，由最新GradTop实际使用的`gradtop_r32_clean_20261015.json`派生，`assets`、`training`、`evaluation`三部分逐字段相同。只把REAL observation设为4步，并登记50/100/150/200/250/270的正式评测网格。不要用当前通用协议替换这个文件：通用协议中的数据及sampler指纹与本轮冻结资产不同。

| 项目 | 已登记设置 |
| --- | --- |
| Probe预算 | 新采样4×64=256个筛选前prompts，K8，共2048条responses；不复用旧缓存 |
| REAL统计与分解 | 原逐题reward contrast、population std、response长度归一化、未中心化二阶矩；输入/输出sketch均128；Nyström；至少2正2负 |
| 有效组数 | probe完成后读取`summary.json`；不把筛选后的有效组数当作采样预算 |
| 初始化与训练 | rank32、alpha64，196模块，A固定，B归零；新optimizer及正式data游标；从0连续训练到270 |
| 正式预算 | 270 rollout steps、1080 Adam更新、17280 prompt visits、138240 responses |
| 保存 | 每10步adapter snapshot，每50步完整checkpoint及终点270；完整actor保留最新2份 |
| 评测 | 训练结束后依次评测50/100/150/200/250/270；原六榜、902题、每点7248条responses、32768生成上限 |
| 失败处理 | probe秩不足、资产/代码指纹不匹配或同步校验失败则停止；不自动扩充probe、不自动续训 |

本次为从零开始，原有“非零B恢复检查”不能直接用于初始化。新增检查按两个阶段进行：首次正式rollout前，将actor全部392个A/B张量与独立从probe构造的零B adapter比较；第一步训练完成后，将四个actor rank的导出与rank0实际更新后的adapter比较，确认A仍等于probe、B已经非零。两个阶段都复用实际vLLM注册adapter和CUDA packed buffer逐张量回读，并核对四个replica的生成请求确实带着对应adapter ID。阶段各留独立回执。这里的第二份source是实时actor导出，不是独立保存的训练checkpoint，证据中明确标记来源。正式训练尚未开始时，GPU回执状态为待执行，不能提前写为通过。

同步与恢复相关CPU测试57项通过（`tests/test_lora_resume_verification.py`、`tests/test_rollout_adapter_resume.py`），包含零初始化、首次更新、A错误变化、零B假更新、旧设备权重及过期回执拒绝。启动前资产核验、probe dry run、GPU空闲检查和26 GiB峰值空间预算检查通过。完整checkpoint的保留逻辑在写入下一个之前删除超出上限的旧actor，因此预算按同时保存2份计算。

磁盘清理释放约20.28 GiB，启动前可用约28.06 GiB。只删除已评测REAL旧50/150及当前GradTop250的完整模型分片；这些点的adapter、optimizer、RNG/data及评测证据继续保留，GradTop270完整checkpoint保留。删除清单和实际释放空间见[cleanup_result.json](../../runs/ops/real_p256_r32_longrun270_20261017/cleanup_result.json)。被清理的旧点不再具备完整续训状态，目录内已留说明。

运行入口及后续核查：

- [计划、代码SHA与资源预算](../../runs/ops/real_p256_r32_longrun270_20261017/plan.json)；同目录`source_snapshot/`保存本次代码副本。
- [实时状态](../../runs/ops/real_p256_r32_longrun270_20261017/status.json)和[控制器日志](../../runs/ops/real_p256_r32_longrun270_20261017/controller.log)。
- [Probe日志](../../runs/ops/real_p256_r32_longrun270_20261017/probe.log)；正式训练启动后生成同目录`train.log`和`train_metrics.log`。
- Probe通过后生成`artifact_binding.json`和独立`initial_adapter/`；正式训练前两轮验证结果保存在`sync_receipts/initial/`、`sync_receipts/after_step1/`。
- 评测后逐点更新`results.md`与`results.json`，按既定Avg.及六榜Avg@k / Pass@k格式输出，并补充Macro。

本实验先检验小预算REAL能否构造有效子空间、训练后段是否仍出现下降，以及实测probe成本。相同prompt数不代表相同反向计算成本；需分别记录probe与正式训练墙时。与已有GradTop轨迹比较仍保留第3.1节的历史恢复混杂说明，不能仅凭一次新REAL长跑将差异归因于统计目标或固定A。

## 9. 用户调整：210步停止，按210→200→150评测（2026-10-18）

用户明确要求当前REAL训练在210步结束，评测顺序为210、200、150。本项覆盖第8节原定270终点及六点评测计划；第8节及原冻结配置保留为启动历史，不改写正在运行的进程配置。

11:22:05新控制器完成接管，当时已完成209步。原控制器已退出，训练worker继续执行当前210步。原训练器没有动态终点接口，新控制器每0.1秒检查210步日志与原子落盘的adapter snapshot；二者齐备后，只冻结并终止携带本次训练stage token的进程，然后释放GPU、启动顺序评测。训练期间不更换adapter、不恢复checkpoint、不调整优化器或数据游标。终点对应840次逻辑Adam更新。

210步保存完整的196模块/392张量adapter及manifest，可用于原六榜评测。210不是原定完整checkpoint保存点，因此此操作不声称取得210步optimizer/RNG/data续训状态；最近完整可续训点为200。150和200的snapshot已核对存在且metadata匹配。旧实验目录名称中`longrun270`仅用于保持文件引用，实际终点以新的用户覆盖计划为准。

- [覆盖计划](../../runs/ops/real_p256_stop210_eval_20261018/plan.json)：目标210、顺序`[210, 200, 150]`，包含原controller身份、原计划SHA及训练stage token。继承的原`formal_training.steps=270`字段仅表示启动时预算；本次实际目标由`target_step=210`覆盖。
- [当前状态](../../runs/ops/real_p256_stop210_eval_20261018/status.json)、[当前控制日志](../../runs/ops/real_p256_stop210_eval_20261018/controller.log)。原目录`status.json`同步指向新控制目录。
- 原计划、接管时状态及控制器源文件已备份为新目录下`original_*`。原训练日志保持在`real_p256_r32_longrun270_20261017/train.log`。
- 210完成后写入`terminal_snapshot_verification.json`，记录step、840更新和adapter SHA；随后按210/200/150逐点验收7248条回答并更新新目录`results.md`及`results.json`。

## 10. 停止后续评测及早期checkpoint检查（2026-10-18 15:48）

按用户要求，当前step200评测已停止，排队的150评测取消，控制器停止，四张GPU已释放。210完整六榜结果保留：Avg.41.1010%、Macro35.8401%、解析率82.0778%、32768触顶率10.7340%、平均输出10656.2 tokens。200的部分生成没有形成完整有效评测，不能填入结果表。本轨迹尚无早期完整评测，不能把210较差直接表述为已经证明“从早期到后期下降”。

CPU检查覆盖step0以及每10步至210的全部22个snapshot，392张量完整且有限、A与probe一致，权重增量范数平滑增长；probe通过87组、196模块数值秩均足以导出rank32。前50步REAL训练正确率与最新GradTop接近，解析率略高，未发现一开始就明显失常的训练迹象。这些检查不等同于早期泛化表现正常。

建议优先完整评测50；若50已弱，再查10定位最早变化；若50正常，再查100定位中期转折。候选均已保存并核验。当前没有启动或排队新的GPU评测。

详细窗口对照、初始化检查、各snapshot SHA及条件式评测方案见[早期checkpoint检查报告](../../runs/ops/real_p256_early_checkpoint_audit_20261018/audit.md)和[原始审计数据](../../runs/ops/real_p256_early_checkpoint_audit_20261018/audit.json)。停止进程身份及取消记录位于`runs/ops/real_p256_stop210_eval_20261018/user_stop_processes.json`与`status.json`。
