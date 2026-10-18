**当前 GradTop、REAL 与 LoRA：实现审查、空间测量及研究决策**

日期：2026-10-16，Asia/Shanghai。唯一 GradTop 对象为 `rlvr_v1_gradtop_r32_a64_lr1em5_b64m16n8_seed42_20261015`；对照仅采用交接文档登记的同协议 LoRA、REAL32、REAL64。本文没有使用其他 GradTop 版本的实验结果。所有新增测量均为 CPU 上的源码核对、既有 records 重聚合、保存权重几何分析或明确标识的事后格式敏感性检查；没有启动 GPU probe、训练、生成评测，也没有改原实验源码、协议、checkpoint 或主结果。

**结论：保留当前 GradTop，优先研究“有限 rollout 下的可复现更新是否比一次采样的梯度能量更能预测真实策略改善”。当前结果支持继续研究，不足以直接建立一个区别于 REAL 的完整方法贡献。** 最需要先修复的是恢复后的首轮采样策略错误。最需要先澄清的是格式收益、梯度统计目标与实际 LoRA-Adam 更新之间的关系。

本轮已拉取并使用 [CCFA-Skills](https://github.com/mikubaka88/CCFA-Skills) 的 Humanization、Common、Idea Reviewer、Idea Optimizer、Experiment Designer、Literature Searcher；安装版本与文件校验见 [skill_provenance.json](../../runs/ops/gradtop_research_review_20261016/skill_provenance.json)。采用其“问题—机制—可证伪证据”要求；以下提出的目标与实验属于研究方案，未冒充执行结果或顶会接收保证。

**0. 实现和协议审查：算法主路径基本符合登记，整条实验轨迹尚未闭合。**

| 事项 | 本轮判断与证据 | 对结论的影响 |
| --- | --- | --- |
| 恢复首轮 rollout | GradTop step51/101、REAL32 step51/101 的配置与冻结源码均进入首次只同步 base、未注册 adapter 的路径。LoRA 单进程连续运行1–150，无这两个恢复边界 | 这是行为改变的软错误；恢复 actor/optimizer/RNG/data 成功，不能保证生成策略连续。GT 与 REAL 都有错误也不意味着偏差抵消 |
| 当前源码身份 | GradTop 首轮29项、续训32项冻结文件 SHA 全部与当前文件一致 | 核心实现身份可核验；清单外的历史 vLLM 下游状态仍不等于直接测得 engine 权重 |
| GradTop 零函数探针 | `A=0, B=Ψ`，读 `grad(A).T/scaling`；B 梯度为零，不做 optimizer 更新，不裁剪，无额外随机 sign | 符合 `S=GᵀΨ`；每个完整 observation 捕获后才刷新 Ψ，没有把不同投影坐标误相加 |
| 能量累计 | 16个 observation，每个16题×K8；每4个 observation 做一次 QR/SVD，保留 `UΣ`，capacity64，导出rank32 | batch 内可以抵消；不同 observation 因子拼接后的相反梯度不会直接抵消。符合登记的中等 batch 二阶矩 |
| REAL | 完整K8题组先反传累积，再平方；2–6个正确才接收；response 长度归一化，population std；Nyström；A固定、B训练 | 本地实现与所声明的 REAL probe 逻辑相符；它估计的是筛选后的逐题目标，不能写成全训练分布的无条件二阶矩 |
| 正式 actor 的 token mean | 每个 microbatch 取 token mean，再按 response 数累积并跨DP平均 | 与 GradTop probe 的全局 token mean、REAL 每response均值均不完全相同，是机制归因的实质混杂 |
| 初始化尺度和参数化 | 普通 LoRA 的 Kaiming A 每行期望平方范数1/3；GT/REAL 的 A 为单位正交；GT/LoRA训练A+B，REAL只训练B | 同rank、alpha、LR不等于同有效更新尺度/同可训练参数；更大的参数梯度范数不能直接解释为更强RL信号 |
| 评测记录 | 11组共79,728条记录重新聚合，与原summary一致；重放原解析器也一致 | 当前端点分数是真实评测结果，未发现表格抄算错误；但指标定义中混入格式是否被接受的问题 |

恢复问题的链条是：`load_format=dummy → base_sync_done=False → collect_lora_params(False)` 排除 `lora_` 参数 → rollout 第一次只加载 base → request 没有 adapter ID → 随后才置 `base_sync_done=True`。old log-prob 却由恢复后的 actor 重算。因此 PPO 的 old policy 与实际行为策略不一致，且不会靠正常执行或 checkpoint 验收自动报错。新训第一步 B=0，与 base 函数等价；恢复时 B 已非零，问题才暴露。

本轮进一步闭合了 REAL32 的历史证据：把当前 `fsdp_workers.py` 中后来新增的六行 GradTop guard 仅在内存移除，所得 SHA 与 REAL32 续训冻结 SHA 完全相同，说明相关同步路径并非用当前源码无依据地猜测历史。证据仍止于配置、冻结源码与日志，尚无当时 engine tensor/hash 的直接测量。详细链路与行号见 [implementation_audit.md](../../runs/ops/gradtop_research_review_20261016/implementation_audit.md)。

下一轮修复的验收对象应是**恢复后的第一次请求**：确认 adapter tensor 与checkpoint相符、已注册、请求携带对应ID，并在固定输入上与独立加载该adapter的引擎比对输出分布。还需在一个不中断/中断恢复的短对照上检查首个 rollout 的行为一致性。只检查 `base_sync_done=True` 或 actor 参数不够。论文主轨迹应在修复后重跑；从150再继续无法消除已进入优化器的51/101错误采样。

正式训练的动态 microbatch 路径可写为

\[
G_{\rm actor}=\sum_{d,j}\frac1D\frac{n_{dj}}{N_d}\frac1{T_{dj}}
\sum_{i\in\mathcal M_{dj},t}m_{it}\nabla\ell_{it}.
\]

这里 `n` 是microbatch response数，`N` 是local minibatch response数，`T` 是microbatch有效response token数。只有各microbatch的平均长度相同等特殊条件下才与全局token mean一致；每个microbatch只有一条response时，它就是response长度归一化。当前长回答/动态打包使这一区别值得实测，但现有日志不足以恢复每个token的真实权重。不能把完整题组梯度与总token数当作足以还原正式 actor 的数据。

对当前路径的 CPU 代数核对确认了零函数sketch恒等式、DP缩放及microbatch无关性、跨observation不抵消；另在实际实验Python/torch2.8.0环境运行20项必要CPU测试，20 passed。它们没有替代真实FSDP/vLLM恢复验收。源文件与数值记录见 [cpu_audit_checks.json](../../runs/ops/gradtop_research_review_20261016/cpu_audit_checks.json)，已有测试输出见 [cpu_existing_tests.log](../../runs/ops/gradtop_research_review_20261016/cpu_existing_tests.log)。

一个较小的新增数值边界是，冻结probe的4步日志中 `pg_clipfrac≈2.30e−5–3.90e−5`，并非逐token严格ratio=1。BF16、packing和不同重算路径是待核原因；没有证据据此认定policy漂移或实质损害。保留当前PPO分支的准确描述即可，不能把它写成运行时精确无clipping的reward梯度。

**1. Batch 和逐题能量的差别，首先是保留什么，而非怎样求 SVD。**

对线性权重 `W∈R^(d_out×d_in)`，令同一loss下的完整题组梯度为 `G_p`。在等题权重、独立抽题的理想参照中，写成

\[
G_{p,a}=\mu_p+\varepsilon_{p,a},\quad
\mu=\mathbb E_p\mu_p,\quad
M=\mu^\top\mu,
\]
\[
H=\mathbb E_p[(\mu_p-\mu)^\top(\mu_p-\mu)],\quad
N=\mathbb E_{p,a}[\varepsilon_{p,a}^\top\varepsilon_{p,a}].
\]

`H` 是真实题间异质性，`N` 是给定题、给定策略、K8组梯度估计器的rollout波动；这里的条件均值不保证等于原始期望reward的无偏梯度。于是

\[
C_{\rm prompt}=M+H+N,\qquad
C_{\rm batch}(m)=M+\frac{H+N}{m}.
\]

逐题能量将“各题分别很大”的方向保留；batch能量将“这些题合在一次共享更新中有多大”的方向保留。batch平均同时压低噪声与真实异质性，无法自行分辨二者。REAL的逐题损失、筛题和长度权重与当前GT不同，因此上式是受控比较的参照，不是把两份现有artifact直接套入的精确等式。

一个判别性的例子是 `G_p=(1, ±a)` 且 `1<a²<16`：逐题二阶矩优先第二轴，m16的batch二阶矩优先第一轴。第二轴既可能是真实题型冲突，也可能只是同题重新采样就变化的噪声。两种世界能产生相同的一次采样统计；需要同题独立重复才能区分。不能把“有抵消”直接解释为信息损失，也不能把“被平均掉”直接解释为去噪。

当前GT实际用每observation独立宽8输出投影：`S_b=G_bᵀΨ_b`，条件期望 `EΨ[S_b S_bᵀ]=G_bᵀG_b`。共16个sketch，在累计128列的过程中压缩到64，最终取32。REAL每题用宽128输出投影，另有固定宽128输入投影Ω，累计 `Y≈CΩ`，再由Nyström导出空间。二者都不必形成dense dW，也都是流式估计；这些不是GT独有的贡献。

输出投影刷新位置非常关键：当前GT先对完整batch反传，再刷新Ψ，保持了题间交叉项。如果未来诊断先逐题取 `H_p=Ψ_pᵀG_p`，却让每题Ψ独立，再离线把H相加，交叉项在投影期望中会消失，所谓batch实验实际退化为逐题能量实验。同题重复交叉矩也必须使用同一个Ψ；两套独立投影只用于fit/audit隔离，各套内部保持共同坐标。

**直接查看当前保存的探测空间，得到以下新测量。** 所有196模块用double-precision QR把保存A的行空间正交化，以 `overlap=||QᵀP||²_F/32` 比较；这是一组主角度余弦平方的均值，不是“百分之多少梯度相同”。

| 保存空间比较 | 平均 overlap | 含义 |
| --- | ---: | --- |
| GT init vs REAL32 init | 0.40733 | 存在明显空间差异，同时有共同成分 |
| GT init vs REAL64 init | 0.40784 | REAL增大probe前缀后，与GT仍相差很大 |
| REAL32 init vs REAL64 init | 0.93742 | 两份同目标、嵌套缓存空间接近；不是独立采样稳定性检验 |
| GT init vs GT step50 A | 0.9999847 | A虽可训练，当前短轨迹行空间几乎没动 |
| GT init vs GT step100 A | 0.9999581 | 同上 |
| GT init vs GT step150 A | 0.9999245 | 同上；不能据此说A被冻结 |

GT/REAL32的模块均值在down_proj为0.2310，在v_proj为0.5647。独立Haar rank32空间的随机参照 `32/d_in` 为约0.00357或0.02083，现有重合远高于该参照，但它不是显著性检验。目标、采样、loss、宽度和数值算法同时不同，当前低重合不能单独归因于batch聚合。

进一步计算 `||BA(I−Q0Q0ᵀ)||²_F/||BA||²_F`，按模块有效权重能量汇总，GT在50/100/150分别只有0.00217%/0.00574%/0.01030%的适配权重能量落在初始空间外。step150 A相对Frobenius位移的模块均值0.8825%。这为“当前训练仍强烈受初始空间影响”提供实测支持；它既不是未来梯度覆盖，也不保证小权重位移对logits影响小。

GT artifact的最终rank32保留原累计sketch能量：模块均值71.44%、最小44.23%。175/196模块的现有流式因子即使计入保留的64方向也未达到记录总sketch能量的95%；`raw_energy_rank=65`在这里是未达阈值的哨兵值，不能当作未压缩真实谱的95%秩。部分能量可能在前次截断中已经丢失。配置中的0.95并不表示实际每层保留95%梯度能量。

所有测量、定义、输入SHA及复现代码见 [saved_space_geometry.json](../../runs/ops/gradtop_research_review_20261016/saved_space_geometry.json) 与 [inspect_saved_spaces.py](../../runs/ops/gradtop_research_review_20261016/inspect_saved_spaces.py)。现有GT产物只保存最终A与summary，没有原始16份S，不能用它们逆推出噪声/异质性分解，或无损重做一次性SVD。

**2. 与 REAL 的差异和可主张的贡献。**

REAL原文《Reward-Gradient Aligned Low-Rank Adaptation for Post-Training》为本机ICLR 2027匿名在审PDF。其§3.2、Eq.8已经讨论batch聚合的抵消，Appendix G已经比较“先聚合梯度后求子空间”的REAL Gradient，并设置A+B训练的对照。因此“考虑reward梯度”“发现聚合抵消”“不存dense梯度”“流式probe”“训练A+B”都不能单独作为区别于REAL的新主张。

本地REAL的核心估计器吻合论文公式(6)–(9)和Algorithm 1，但应准确命名为本地共同协议下的correctness-only REAL。论文distilled模型分支在probe额外采用λ=0.5的长度效用，本地当前对照没有此项；作者的rank、训练目标及probe预算计入方式也不同。因此现有结论是本地同协议方法比较，尚不是对作者distilled完整分支或原论文成绩的复现。

真正可区分的是：REAL的逐题二阶矩、跨全部probe聚合后的一张平均梯度矩阵、当前GT的中等batch二阶矩，是三个不同对象。GT在跨batch累计时保留二阶能量，并没有退化为REAL Gradient的全probe一次平均。REAL对每题自身局部SGD增益的几何解释，不直接推出共享batch更新或LoRA-Adam的最优性；这提供了值得研究的目标选择问题，不代表REAL推导本身错误。

| 可保留的差异 | 当前证据 | 顶会层面需要补足的内容 |
| --- | --- | --- |
| 中等batch二阶矩对应共享更新的统计对象 | 代码和数学定义明确；当前空间与REAL不同 | 同cache、同loss、同求解精度比较，解释噪声和异质性各自的作用，并预测真实策略效用 |
| 很小probe预算下仍有有效端点 | GT2048 responses、37.51分钟；REAL64为32768 responses、11.135小时 | 同总rollout/backward预算、优化过的REAL实现、相近估计质量的成本曲线 |
| 窄输出sketch加增量SVD | 实现确实不同；原幅值保留，没有在线遗忘 | 排除预算与目标差异后量化误差/资源收益，若仅替换分解器则贡献偏工程 |
| RL有限组采样导致能量与可重复更新分离 | 数学上可定义，当前数据没有同题重复证据 | 揭示适用条件、失败条件与因果效用，形成一个统计目标而非多个局部开关 |

现有17.81倍probe墙钟比包含16倍尝试题数差距、本地REAL逐题逐response反传/DP复制开销，不能写成同样本算法加速17.81倍。REAL32来自64-step缓存的32-step精确前缀，没有独立采集计时。最终方法表应同时报告rollout数、forward/backward tokens、逻辑Adam更新、probe+训练总成本和全卡物理显存；现有日志的allocator峰值字段不宜直接当物理峰显存。

公开邻近工作也限制了新颖性：LoRA-GA已有梯度对齐初始化，GoRA已有梯度驱动rank/初始化，GaLore和Flora已有低秩/随机投影梯度处理。检索到的LoRA-GA² v1涉及多步probe，但其后续版本已标withdrawn，只能作为公开想法重叠线索，不能当可靠实验结论或正式接收论文。具体来源、原文段落与覆盖边界由 [paper_geometry.md](../../runs/ops/gradtop_research_review_20261016/paper_geometry.md) 保存；本轮检索不能支持任何“首次”或独立优先权主张。

按CCFA的概念评审口径：问题重要性和可解释性较强；当前新颖性主要是一个待验证的RL统计目标问题；random-B、QR/SVD与初始化组合本身的新颖性有限。建议为 **accept-to-develop / 继续发展该问题**，而非直接以当前实现包装投稿。最值得保留的是低预算batch探测；最薄弱的推理链是“几何覆盖更好，所以真正推理能力更强”。

**3. 现有结果揭示的问题，以及一个优先研究问题。**

当前主结果仍应按原协议报告：

| Step | LoRA Macro Avg@k | REAL32 Macro Avg@k | GradTop Macro Avg@k | GT−REAL32，题目配对95%CI，pp |
| --- | ---: | ---: | ---: | --- |
| 50 | 38.7224% | 40.3361% | 41.3282% | +0.9920 [−0.1483,+2.1325] |
| 100 | 38.3390% | 42.0021% | 41.3425% | −0.6596 [−2.1011,+0.8246] |
| 150 | 38.8141% | 39.4078% | 42.3971% | +2.9892 [+1.6527,+4.4124] |

GT150相对LoRA150为+3.5830pp，CI[+2.2411,+4.9986]；GT150相对自身100为+1.0545pp，CI[−0.2529,+2.3554]。这支持当前单seed端点差异，不说明跨训练seed稳定优胜，也不说明100→150稳定提升。51/101恢复偏差须与主表一并披露。

**新增结果一：格式遵循解释了相当一部分Minerva优势。** 原评测的 `no_answer_rate` 实际等于没有被当前parser解析的比例；不能等同于没有写答案。LoRA150的364条Minerva未解析回答中363条包含`Answer:`，只有1条触顶32768；REAL32-150的339条中332条含`Answer:`，8条触顶。训练parser与评测parser另有一个A–D字母答案的微小差异；全79,728条仅10条受此差异影响，且stored_correct均为False，无法解释主要差距。

本轮先固定一个额外规则，再对11组全部Minerva统一执行：只处理原未解析回答，仅接受末非空行的单一`Answer:`，限制长度、排除显式多答案/叙述歧义，复用原 `math_verify` 规范化与strict验证。2815个候选全部完成；原records与原分数未修改。这是事后敏感性分析，不是替换主指标，也不是推理过程正确性的认证。

| 方法/step | 原Minerva Avg@4 | 额外保守抽取后 | 新识别正确回答 |
| --- | ---: | ---: | ---: |
| LoRA150 | 22.70% | 28.40% | 62 |
| REAL32-150 | 23.07% | 30.06% | 76 |
| GT50 | 32.54% | 34.01% | 16 |
| GT100 | 32.81% | 34.10% | 14 |
| GT150 | 33.64% | 33.64% | 0 |

GT150相对REAL150的Minerva优势从10.57pp降至3.58pp，相对LoRA150从10.94pp降至5.24pp；仍有剩余优势。GT自身在这项扩展指标上的50/100/150为34.01/34.10/33.64%，原来的小幅增长不再出现。它直接说明一些“错误”是正确答案不符合抽取格式；它不能因果地证明其余差距由什么机制产生。规则、逐条结果和来源见 [minerva_answer_sensitivity.json](../../runs/ops/gradtop_research_review_20261016/minerva_answer_sensitivity.json)。

只替换这一个任务的抽取结果、其他五任务维持原评分后，GT150−REAL150的六任务Macro差由+2.989pp变为+1.825pp，探索性题目配对95%CI[+0.523,+3.200]；GT150−LoRA150由+3.583pp变为+2.633pp，CI[+1.318,+4.016]；GT50−REAL50则由+0.992pp变为+0.195pp，CI[−0.940,+1.329]。这些10,000次bootstrap区间仍不包含训练seed不确定性，也没有消除事后选择诊断的限制；保留原主结果，不将其改写为新的预注册主指标。复现见 [paired_sensitivity.json](../../runs/ops/gradtop_research_review_20261016/paired_sensitivity.json)。

**新增结果二：RL训练的长度、reward与探索变化，不能由一个entropy故事概括。** 末段131–150的训练均值为：

| 方法 | reward | parse | hit8192 | entropy |
| --- | ---: | ---: | ---: | ---: |
| LoRA | 46.56% | 78.72% | 26.49% | 0.7323 |
| REAL32 | 47.52% | 95.44% | 4.76% | 1.1551 |
| GradTop | 50.01% | 92.05% | 13.18% | 0.7860 |

REAL32的102–120窗口entropy约1.0172，131–150升至1.1551，而reward从51.19%降到47.52%。GT同窗口reward从51.75%到50.01%，entropy从0.8371到0.7860。题目不同，不能据此认定过拟合或因果趋势；但REAL回落不能直接解释成entropy塌缩。GT训练还有约13%的回答在8192终止，而eval允许32768且150端点触顶仅0.87%。训练与评测的长度cap、题库和轨迹分布均不同，不能用两比例之差估计截断对收益的因果影响；可以确定训练reward与长推理评测使用不同生成视野。梯度probe可能学到格式/终止偏好，不能自动等同于难题推理方向。

**新增结果三：REAL筛选会改变RL信息分布。** 在独立二元回答、单题成功率q的参照下，K8接收概率是 `Pr(2≤Binomial(8,q)≤6)`：q=0.05时5.72%，q=0.1时18.69%，q=0.5时92.97%。REAL因而更重视当前策略的中间难度区域；它随策略改变而移动。GT保留全对/全错的零advantage组及其分母，但也不因此自动获得这些题的非零学习信号。噪声、真实异质性、成功概率与长度相关，构成RL-specific问题，而不是普通矩阵近似误差。

**优先研究问题：在固定rank与总采样预算下，同题独立rollout之间可复现的batch更新，是否比单次逐题/批次能量更能预测实际LoRA-Adam训练后的策略改善？**

把当前GT作为保留的强基线。候选的中心变化只在统计目标：从一次rollout的能量，转向新rollout上可重复的更新效用；不加入sign、token mask、fixed-A组合。

对同一m题集合B生成两个独立K8组a、b，用相同loss构成 `Ḡ_Ba, Ḡ_Bb`，则候选交叉矩为

\[
C_{\rm rep}(m)=\mathbb E\operatorname{sym}(\bar G_{B,a}^\top\bar G_{B,b})
=M+H/m.
\]

它去掉了rollout噪声N/m，保留真实题间异质性的batch尺度H/m。若对同一有限P题库无放回采m题，则H系数改为 `(P−m)/(m(P−1))`。如果两次连题目集合也独立重采，交叉矩变成M，不再保留H/m。若按某次rollout是否通过REAL筛选再选题，就破坏了原分布解释；两次都通过会按接收概率平方重新加权。

一个完整且可检验的候选选基定义是 `max_(QᵀQ=I_r) tr(Qᵀ C_rep Q)`。在同题、新rollout、投影SGD和小步长的surrogate解释下，它对应跨重复的一阶转移效用。这里不能跳过三条边界：有限样本交叉矩可能非PSD；截负不再是无偏估计；换成独立新题的总体一阶目标时对应M而不是M+H/m。可重复的格式偏好和题型冲突也可能无助于新的数学能力。

此外elementwise Adam对一般正交基旋转不不变：相同rowspace、相同初始函数的A/B参数化也可能产生不同BA更新。真正权重增量应测 `ΔW=s(BΔA+ΔBA+ΔBΔA)`，不能对denseG做虚拟Adam来代替。故C_rep是有明确含义的研究候选，尚不是已证明的“最优RL子空间”。独立rollout交叉矩本身也是通用统计原语；方法贡献必须来自RL中的识别结果、受预算约束的估计及真实效用，不能仅换一个名字。

**4. 最小证伪实验与逐级执行方案。**

| 阶段 | 固定设计 | 回答的问题与预算 |
| --- | --- | --- |
| D0：目标诊断 | base策略；预先固定128题，64 discovery/64 audit；每题两组独立K8；0/13/27层的q_proj/down_proj六模块；两套独立宽32输出投影，套内跨题/重复共用 | 2048 responses；完整输入侧H约1.92GiB，另计cache/正式权重重放。分清N、H、M；这六模块诊断不冒充全196模块方法结果 |
| D1：条件漂移诊断 | D0通过后，在现有连续LoRA step50 adapter上重复同128题；step100为事先保留的扩展 | 每checkpoint再2048 responses。只有新策略下可复现更新落出旧空间，且变化超过同策略重复误差，才推进漂移主线 |
| D2：真实更新桥接 | 对通过D0的目标扩展到全部196模块；相同初始化尺度、rank32、A+B、LR、真实actor/Adam；配对训练数据；独立audit题上重新生成回答 | 测实际ΔBA、KL、strict reward、格式敏感性、parse及短程收益；完整配置与预算另登记，六层probe不能直接用作全方法替代品 |
| D3：正式方法比较 | 修复采样同步后，LoRA/当前GT/当前correctness-only REAL/通过验证的候选，固定50/100/150端点 | 每方法至少3训练seed；另立seed扩展协议。不能把现有seed42协议默默改写，也不能把旧轨迹与新干净轨迹混成同协议主证据 |

D0主要比较预定m16，m∈{1,4,16,64}只检查缩放关系。必须包含：逐题能量空间、m16 batch能量空间、m16交叉重复空间、全池平均梯度空间；最后一项用于回应REAL Appendix G。独立随机正交A是初始化尺度的机制对照，普通LoRA仍作为原方法baseline保留。相同有限样本下m64只有很少batch，不能用其训练内高覆盖宣称稳定。

第一轮先固定统计目标再统一求解精度，不同时改变sketch宽度、capacity、Nyström/SVD和A是否冻结。等题分支用于识别公式；GT全局token权重分支用于核对当前probe；正式actor分支需额外重放真实DP分配、length balance、microbatch packing及partial-prompt贡献。若只收了完整题组H与长度，则正式权重分支明确未完成。

通过同cache的有限样本恒等式估计 `U_hat=S_hat+N_hat`：`U_hat=(1/(2P))Σ(G_paᵀG_pa+G_pbᵀG_pb)`，`N_hat=(1/(2P))Σ(G_pa−G_pb)ᵀ(G_pa−G_pb)`，`S_hat=(1/P)Σsym(G_paᵀG_pb)`。U必须平均两个重复的自矩，不能只用a重复却声称有限样本恒等式。保留负特征值与区间，不能先截为PSD再宣称无偏分解。只在discovery拟合，在audit+独立投影上打分，交换投影角色作为预定敏感性；bootstrap按prompt成组，保留两个重复。这是给定已拟合候选的audit不确定性，不覆盖重新拟合/训练seed波动。分母接近零的交叉能量不报告不稳定百分比。

D2使用执行前锁定、未参与D0拟合及候选筛选、且不进入D2训练的新prompt审计集。D0 audit用于阶段决策，不再作为D2主要效用终点。针对本轮确实发现的Adam坐标依赖，只加入一个必要归因检查：对同一个Q用预定正交O构成QO，保持rowspace、A范数与初始函数相同；若短程效用差与不同统计目标间差异同量级，结论限定为特定基坐标下的LoRA-Adam初始化效应，不能归因于rowspace本身。这是机制验证，不是新方法组成部分。

**预先建议的推进/停止条件**（是待执行前冻结的研究决策阈值，不是当前实验结果）：

1. 工程关：第一次恢复生成确实使用checkpoint adapter；probe前后函数不变；组完整性/DP权重/投影坐标核验通过。任一失败先修执行链，不能拿失败产物评机制。
2. D0主判别量：在独立audit上，C_rep选出的空间相对当前batch空间，能否保留更多可重复更新，同时减少不可重复能量；以预定模块等权汇总并报告全部六模块。主差异的题目bootstrap区间跨0时，不能宣称支持；仅训练内能量升高判为失败。
3. 先区分prompt重采样误差和输出投影误差。对前者只允许一次预先登记的128→256题扩样；仍不稳定就停止该机制主张。若两套Ψ改变主结论，应判定当前宽32测量分辨率不足，停止当前判别或另行登记扩宽/新增投影验证；不能用增加题数声称已经消除固定投影误差，也不继续搜m、rank或更多局部开关。
4. D2需要真实actor/Adam短程收益与几何预测同向。建议先固定10个trainer step、40次逻辑更新，所有候选相同预算；一次局部改善只支持桥接，不替代正式150步实验。若几何更好但实际效用不改善，否定“该几何足以指导初始化”的故事。
5. strict reward与预定义格式敏感性同时报告。若新增收益只来自parser接受率，贡献限定为该reward协议下的学习效率，不写成数学推理增强；若在难题或新题audit上没有转移，不能用可复现H掩盖这一失败。
6. 漂移只有在D1独立重复变化小、跨策略效用损失大且恢复空间覆盖确实改善训练时才进入方法。当前不默认加入在线更新、遗忘系数或optimizer迁移；这些需要一个另行完整定义且有证据支撑的机制。

公平性需要两种不同问题各自有表：同cache/同目标数值精度的机制比较，以及相同总rollout/backward预算下各完整方法的成本比较。使用两次独立rollout不能称为免费去噪。正式候选若用256题×2×K8=4096条回答，GT/REAL应同时获得4096条回答的预算对照，并保留原2048-budget GT。重复题与更多独立题的预算分配也应作为明确比较，不能让新方法独占额外数据。

四方法×三seed×150步的核心正式包共1800 trainer steps、7200次逻辑Adam更新、115,200个正式训练prompts、921,600条训练responses，probe另计；若都评50/100/150则36次完整六榜评测、260,928条评测responses。先通过D0/D2再投入该预算。学习率在协议允许的共同范围内等预算选择；rank16/64与跨模型用于检验最终成立的机制，按阶段追加，不能先堆满表格。本文没有启动这些计划。

当前不建议把继续到200作为优先研究动作：它既不能修复既有采样偏差，也不能区分噪声/异质性/格式效应。更有信息量的下一步是修复并验收首轮adapter同步，以及D0的独立重复诊断。若诊断表明当前batch聚合已经保留了最有用方向而C_rep没有额外收益，应保留当前GT，把后续投入转为公平成本曲线、干净多seed验证和对其适用边界的解释。

证据索引：[实现审计](../../runs/ops/gradtop_research_review_20261016/implementation_audit.md)、[结果审计](../../runs/ops/gradtop_research_review_20261016/results_audit.md)、[论文与几何](../../runs/ops/gradtop_research_review_20261016/paper_geometry.md)、[原始记录重聚合](../../runs/ops/gradtop_research_review_20261016/results_raw_reaggregation.json)、[训练窗口](../../runs/ops/gradtop_research_review_20261016/training_comparison.json)、[保存空间测量](../../runs/ops/gradtop_research_review_20261016/saved_space_geometry.json)。
