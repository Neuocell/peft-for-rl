# REAL / GradTop 已发现的实现问题、修复和证据边界

日期：2026-10-18，Asia/Shanghai。本说明汇总10月16–18日已有审计及修复后的运行验收，供[近期实验总表](../docs/experiments/real_gradtop_summary_20261018_zh.md)共同引用。原始材料的逐字节副本和SHA256见[归档索引](../docs/experiments/evidence/real_gradtop_20261018/README.md)。不把旧错误从历史轨迹中抹去，也不将统计目标差异全部称为代码bug。

## 1. 恢复后的第一次rollout遗漏adapter：已修复，历史影响保留

**受影响范围：当前GradTop和旧REAL32的step51/101。** 当时 `grad_subspace` 走原生LoRA路径，`lora_adapter_path=None`、`load_format=dummy`、`val_before_train=False`。恢复只加载actor；首次wakeup中 `base_sync_done=False`，`collect_lora_params(False)` 过滤 `lora_` 参数，vLLM只加载base，尚未注册adapter，生成请求为 `lora_request=None`。第二次wakeup才安装adapter。

因此第一次恢复采样的行为策略与恢复actor不同；old log-prob仍由恢复actor重算。model/optimizer/RNG/data正确恢复并不能证明策略采样连续。LoRA对照1–150为单进程连续训练，没有51/101这两个恢复边界。GradTop和REAL32都出现相同缺陷，也不能推断偏差相消。

历史证据来自实际配置、冻结源码对应和日志旁证；旧REAL32审计还重建过冻结worker源码SHA。没有当时engine tensor/hash实测，不能把后来的回读证据追溯用于51/101。端点adapter的六榜评测仍是实际结果，但这条训练轨迹不能称为修复后从头的干净对照；累计性能影响尚未量化。

修复基于提交 `04182bb98c493355c86b5980c967531d8f6e5a33` 并接入本地GradTop/REAL：首次同步先加载base，再安装当前完整A/B adapter；sleep level2后同样重新安装。缺PEFT配置、空adapter、安装失败、hybrid请求无活动adapter或恢复缺model时阻断。进一步增加源权重→actor→vLLM注册→实际CUDA槽位→首次请求的验收，绑定源SHA、token和全392个A/B张量。

| 后续运行 | 实际验收 | 仍需保留的限制 |
| --- | --- | --- |
| GradTop150→200，首轮151 | 四GPU的actor/engine/ready/request各4份回执，已通过 | 51/101已有影响仍在 |
| GradTop200→250，首轮201 | 绑定实际200源文件，重新验收通过 | 不复用151回执 |
| GradTop250→270，首轮251 | 绑定实际250源文件，重新验收通过 | 不复用201回执 |
| 新REAL-P256，从0开始 | 初次零B及第一训练step后非零B两个阶段，均完成四GPU实际回读和请求验收 | 第二阶段source是实时actor导出，并非独立完整checkpoint；没有执行新的中断恢复实验 |

新REAL从0连续训练至210，避开旧51/101恢复边界。这不构成整条不中断/中断轨迹等价性的额外证明。原始证据见归档中的 `gradtop_r32_continue_review_20261016/training_audit.md`、`gradtop_research_review_20261016/implementation_audit.md`、三段GradTop的 `first_rollout_verification.json` 以及REAL的 `initial_sync_verification.json` / `after_step1_sync_verification.json`。

## 2. Probe归约与正式actor的token-mean不同：已澄清，未改历史训练语义

正式动态microbatch路径先对每个microbatch取token mean，再按response样本比例累计并跨DP平均。改变长度和动态打包会改变token权重；一般不等于整个逻辑minibatch的全局token mean。

当前GradTop显式probe在动态拆分前固定全局有效token总数，跨DP求和并乘DP补偿，microbatch后续缩放设为1；完整prompt groups不能被trainer长度balance打散。相应实现、contract与测试已接入。旧GradTop的历史归约不能自动解释成当前显式归约。REAL另按每response自身长度归一化，重算population-std reward contrast，先整组累计再平方。

所以GradTop、REAL probe和正式actor是三个不同的加权梯度目标。现有比较是端到端方法比较，不能把差异全部归因于“先聚合还是先平方”，也不能称probe梯度就是Adam实际接收的梯度。本次文档没有更改正式训练loss或重评分。

## 3. 梯度主路径审计：未发现此前怀疑的转置、投影混加或跨batch相消

已有代码与数值检查确认零函数 `A=0,B=Ψ` 满足 `grad(A).T/scaling=G.T Ψ`；完整observation后才刷新Ψ；收集路径不调用Adam、不做梯度裁剪。GradTop拼接原幅值因子，保留 `UΣ` 后再压缩，跨observation的G与−G不会因一次总平均而相消；同一minibatch内部仍有跨题交叉项。REAL先累计完整K8题组梯度再做二阶统计，保留组内正确/错误回答的交叉项。

`target_energy=0.95` 不能解释成每层保证95%捕获。当前uniform-r32实际模块均值约71.44%、最低约44.23%；`raw_energy_rank=65` 是capacity64下未达阈值的哨兵，不是完整真实谱的95%秩。当前产物 `completed_basis_directions=0`。REAL采用秩不足即失败的策略，没有任意补齐。

FSDP1同一flat leaf中可同时有冻结base/B和可训练probe A。冻结张量可能出现全零gradient view，占用梯度存储；“不计算dense base dW”不能升级为“绝不分配base大小的gradient storage”，CPU sketch buffer也不是整卡显存。现有验证范围为FSDP1、SP=1；不据此声称FSDP2已经获得同等验收。

冻结probe日志中仍有约2.30e−5至3.90e−5的clip fraction。BF16及不同重算路径是待核原因，尚无证据认定policy被更新；正确表述是运行了PPO分支，而非逐token精确ratio=1的理想恒等实现。

## 4. 参数化、尺度及成本混杂：保留为解释限制

GradTop/LoRA训练A+B，共36,929,536参数；REAL固定A、仅训练B，共20,643,840参数。普通LoRA的Kaiming A与正交A的初始化尺度也不同。相同rank/alpha/LR不等于相同有效更新尺度或相同可训练参数；LoRA因子上的Adam有坐标依赖，不能把dense-gradient投影几何直接称为实际Adam最优性。

旧REAL32/64的probe题数分别为GradTop的8/16倍；17.81倍旧墙钟比不能作为同预算算法加速。新REAL-P256与GradTop同为256题，墙钟约42.01/37.51分钟，但仍有筛选、sketch宽度和反传调度差异。诊断中REAL初始化的共享覆盖更高，也不能排除其原probe预算优势。

## 5. 评测parser与答案格式：分数核对通过，能力归因受限

10月16日审计重新聚合11组共79,728回答，与原summary一致，重放原parser也一致；没有发现主表抄算错误。`no_answer_rate` 实际表示未被当前parser解析，不能一概解释成回答没有写答案。训练/评测的A–D字母答案细节仅影响10条记录且原来均判错，不能解释主要差距。

事后保守补抽取Minerva末行 `Answer:` 后，LoRA150从22.70%变为28.40%，REAL32-150从23.07%变为30.06%，GradTop150保持33.64%。GT150−REAL150的Minerva差距从10.57pp缩至3.58pp；只替换这一任务后的六榜Macro差由+2.989pp缩至+1.825pp，探索性配对CI为[+0.523,+3.200]pp。这是格式敏感性分析，不能替换预登记主指标，也不能证明剩余差距的因果机制。

保留原评分、原records和原主表。抽取规则及探索性配对结果归档于 `gradtop_research_review_20261016/results_audit.md`、`minerva_parse_diagnostics.json` 和 `paired_sensitivity.json`。

## 6. 运行与归档中遇到的其他问题

| 项目 | 已知事实 / 状态 | 影响 |
| --- | --- | --- |
| 诊断PEFT键名前缀 | 首个CPU汇总失败；改为仓库canonical命名，核对五个策略A键和形状后恢复 | 原采样/反传文件SHA复核通过，复用原材料，没有重采样 |
| JSONL Unicode分隔符 | 合法回答内U+2028被`str.splitlines()`误拆；改为逐物理行读取，原5,120回答完整且唯一 | 修复解析，不修改科学定义；20份原回答SHA保持一致 |
| Epoch边界 | 17,917行、batch64、drop_last对应279批/epoch；270仍在第一epoch。超过279需额外epoch，旧累计游标断言也要调整 | 本轮270/210未触及该边界；300步只有CPU索引重放，不是完成训练 |
| REAL-P256实际停止点 | 原270步计划后改停210；210为840次逻辑更新的adapter snapshot，最近完整可恢复点200 | 没有210的optimizer/RNG/data完整恢复点；不能宣称训练完成270 |
| REAL-P256评测变慢 | 210 adapter/config与导出核对通过，A固定、B非零；manifest只变身份和adapter路径；回答明显更长 | 当前证据更符合实际输出行为变化；终点较弱的原因尚未确定；没有早期完整六榜，不能证明本轨迹早晚退化 |
| REAL-P256早期snapshot审计 | step0及10–210共22份，392张量齐全且有限、A固定，B及有效增量平滑增长；早期训练指标与GradTop接近 | 未发现初始化覆盖或数值爆炸；不能以这些检查证明87有效组的统计质量，或证明每轮rollout同步 |
| 本次CPU缓存测试 | 原live-hook测试依赖测试主机空闲空间≥15GiB，归档时主机不足，单项失败 | 仅对该测试注入确定性的disk_usage；生产15GiB空间保护保留，最终结果见归档validation |

## 7. 验证范围和后续使用

本次发布验证记录在[validation.json](../docs/experiments/evidence/real_gradtop_20261018/validation.json)，包含最终发布工作树的CPU回归、静态检查和归档完整性检查。双GPU REAL FSDP测试为显式启用项；本次CPU组合跳过，没有启动GPU任务。此前文档中的CPU测试、微型双GPU验证、三段实际四GPU恢复验收是不同范围的证据，不能相加为同一批独立测试。

已有修复让后续新进程使用正确同步路径，但不追溯改变51/101的训练更新。当前研究结果可支持单seed端点描述、预算比较和已测模块中的诊断现象；用于严格方法归因仍需修复后从头的GradTop轨迹及匹配对照。本文只汇总已有证据，没有启动这类新训练。
