# 当前 GradTop / 同协议 REAL、LoRA：独立实现审计

日期：2026-10-16，Asia/Shanghai。唯一 GradTop 对象为 `rlvr_v1_gradtop_r32_a64_lr1em5_b64m16n8_seed42_20261015`，包括其 16-observation probe 和 1–150 正式轨迹；REAL32/REAL64、LoRA 仅取交接文档所指定的同协议实验。本次未修改运行源码、未运行 probe/训练/生成评测。新增文件仅位于本审查目录。

**结论：当前 GradTop 的 random-B → 聚合梯度 sketch → 增量能量压缩 → A 初始化 → A+B 正式训练主链，与它登记的实现逻辑一致；未发现转置错误、跨 observation 错误相消、额外 sign、误做 optimizer 更新或导出 rank 错误。不能因此宣布整个实验“没有实现问题”：恢复首轮采样遗漏 adapter 的缺陷确实存在；正式 actor 的 token-mean 语义与显式 probe 不同；参数化尺度及 Adam 坐标会混杂“子空间更好”的解释。**

证据等级：A 为本次读取的原始日志/artifact/冻结哈希等直接证据；B 为实际配置与冻结源码的执行路径推导；C 为本次 CPU 数学/接入测试；D 为尚未定位或尚未执行的假说。A/B 不等同于 vLLM engine 的运行时 tensor/hash 证据。

## 1. 最高优先级：恢复首轮遗漏 adapter，且 REAL32 同样受影响

**严重性 P0；GradTop 为已知缺陷的独立复核，REAL32 的历史源码闭合为本次新增。证据 A+B。**

链路如下：

1. [fsdp_workers.py:237](/root/peft-for-rl/verl/workers/fsdp_workers.py:237) 将 `lora`、`grad_subspace` 均列入 `_is_lora`；REAL 正式训练使用后者。
2. [fsdp_workers.py:1083](/root/peft-for-rl/verl/workers/fsdp_workers.py:1083) 在 `load_format=dummy` 时初始化 `base_sync_done=False`。
3. [fsdp_workers.py:1728](/root/peft-for-rl/verl/workers/fsdp_workers.py:1728) 的 checkpoint 加载恢复 actor/optimizer 等，但没有首次生成前的双阶段 base+adapter 同步。
4. [fsdp_workers.py:1107](/root/peft-for-rl/verl/workers/fsdp_workers.py:1107) 首次使用 `collect_lora_params(..., base_sync_done=False)`；[fsdp_utils.py:680](/root/peft-for-rl/verl/utils/fsdp_utils.py:680) 对应分支收集 base，排除 LoRA 参数。
5. [vllm_rollout.py:240](/root/peft-for-rl/verl/workers/rollout/vllm_rollout/vllm_rollout.py:240) 只有 `peft_config and base_sync_done` 才 `add_lora`，否则走 `load_weights`。
6. [vllm_async_server.py:508](/root/peft-for-rl/verl/workers/rollout/vllm_rollout/vllm_async_server.py:508) 只有固定 adapter ID 已在 `list_loras()` 中才为请求设置 `lora_request`。
7. [fsdp_workers.py:1199](/root/peft-for-rl/verl/workers/fsdp_workers.py:1199) 在本轮同步末尾才设置 `base_sync_done=True`，因此下次 wakeup 才带入 adapter。

本次实际配置都满足上述路径触发条件：GradTop 两段日志的 `lora_adapter_path=None`、`load_format=dummy`、`val_before_train=False`；REAL32 两段对应字段也完全相同。可直接查 [GradTop train_to100.log:197](/root/peft-for-rl/runs/ops/gradtop_r32_continue_step150_20261015/train_to100.log:197)、[train_to150.log:197](/root/peft-for-rl/runs/ops/gradtop_r32_continue_step150_20261015/train_to150.log:197)、[REAL32 train_metrics.log:152](/root/peft-for-rl/runs/ops/real_probe32_continue_step150/train_metrics.log:152)、[REAL32 train.log:197](/root/peft-for-rl/runs/ops/real_probe32_early_step100/train.log:197)。

REAL32 step101 所用 `fsdp_workers.py` 与当前 SHA 相同。step51 的冻结 SHA 不同，但本次做了**不写源码的精确重建**：只从当前文件内存字符串删除第508–513行、专属于显式 GradTop 的 `lora_adapter_path` 六行 guard，得到 SHA256 `a2fd651564d4a2177075b651b34e0cfb3af4c147d4efab9014ac084d029ae60c`，与 [REAL32 原续训 plan](../real_probe32_continue_step150/plan.json) 的冻结值完全一致。恢复/rollout 代码未发生变化，历史链路因此可闭合到该 worker 的实际冻结版本。重建过程和结果见 [cpu_audit_checks.py](cpu_audit_checks.py) 与 [cpu_audit_checks.json](cpu_audit_checks.json)。部分下游文件未纳入当时冻结清单，运行时 engine 实测仍缺失，不能扩大证据范围。

| 实验 | 首次训练是否函数等于 base | 50/100 后恢复 | 本次结论 |
| --- | --- | --- | --- |
| GradTop | 是，B=0 | 两次，step51/101 | 两次首轮走 base-only，不能称严格连续采样恢复 |
| REAL32 | 是，B=0 | 两次，step51/101 | 同一缺陷，不能将其后期表现单独归因于 fixed-A/漂移 |
| LoRA | 是，B=0 | 指定轨迹连续1–150 | 原日志 `resume_mode=disable`，150个训练step；未受这两次恢复事件影响 |

LoRA 原始证据为 [LoRA train log:650](/root/peft-for-rl/runs/logs/rlvr_protocol/rlvr_v1_lora_r32_a64_lr1em5_b64m16n8_seed42.log:650)。新训练的首轮 base-only 因 B=0 与预期策略相同，所以缺陷影响有非零 adapter 的恢复场景，不应把新训练首轮也列为同等故障。

恢复时 `_compute_old_log_prob` 仍由恢复 actor 重算，见 [ray_trainer.py:1408](/root/peft-for-rl/verl/trainer/ppo/ray_trainer.py:1408)。实际行为是 base 生成轨迹、当前 actor 计算 old log-prob；它不是已正确计入行为策略概率的 off-policy 学习。该次数据进入 Adam 状态后，后续恢复正确也不能追溯消除影响。受影响共两批、128 prompts/1024 responses，不代表影响可以按128/9600线性忽略。

下一次任何训练前应做运行时验收：在恢复后的第一次 generate 前确认 base、adapter 权重均已加载、ID已注册、请求实际携带对应 ID；用多层 A/B tensor/hash 对照 checkpoint，且测试非零 B 恢复，不仅测试初始零函数 adapter。用于论文主表的干净轨迹应在修复后重跑。当前端点评测仍是有效 adapter 的实际分数，但对轨迹机制与方法因果归因存在此边界。

## 2. probe 与正式训练的归约不同，且正式行为依赖 packing

**严重性 P1；已知差异的独立核验，下面精确权重及单response极限为新增解释。证据 A+B+C。**

当前显式 GradTop 在 [gradtop_objectives.py:62](/root/peft-for-rl/verl/utils/gradtop_objectives.py:62) 聚合完整 observation 的 DP 总 token 数，loss 在 [gradtop_objectives.py:158](/root/peft-for-rl/verl/utils/gradtop_objectives.py:158) 除全局 token 数并乘 DP size；[dp_actor.py:1572](/root/peft-for-rl/verl/workers/actor/dp_actor.py:1572) 取消额外 microbatch 缩放。该逻辑正确实现了其登记的 `global_token_mean`，CPU 检查中改变不均等 microbatch 和模拟 DP 平均后梯度误差为0。

正式训练 `gradtop_probe_variant()` 返回 legacy 普通路径；[dp_actor.py:1566](/root/peft-for-rl/verl/workers/actor/dp_actor.py:1566) 乘 `n_micro/N_local`；[core_algos.py:988](/root/peft-for-rl/verl/trainer/ppo/core_algos.py:988) 调用 `agg_loss`，当前 actor 的 `global_batch_info` 是空字典，`token-mean` 在 [core_algos.py:802](/root/peft-for-rl/verl/trainer/ppo/core_algos.py:802) 以**本 microbatch**有效 token 总数作分母。FSDP 再跨DP平均。因此一个 token 的系数为

\[
w_{d,b}=\frac{n_{d,b}}{D N_{\mathrm{local}} T_{d,b}},\qquad
w_{\mathrm{global-token}}=\frac{1}{\sum_{d,b}T_{d,b}}.
\]

两者通常不同。若一个 microbatch 恰好只有一条 response，前者退化为 `1/(N_global L_i)`，即 response mean，而非 global token mean。当前长度上限8192、packing预算12288，长回答时单条或少量response的microbatch完全可能出现；**本次没有保存实际 packing 直方图，不声称已量化其频率或实际梯度夹角。**

三个目标须分开：GradTop 显式 probe 是全局 token mean；REAL 是逐 response 长度归一化后 K 平均；正式训练是依赖 packing 的 response 数加权 microbatch token mean。共同训练代码不保证共同的隐含重权，因为不同方法的回答长度改变 packing，进而改变权重。若协议里的“token-mean”本意是逻辑 minibatch 的全局 token mean，这就是语义不符；若以交接已经明示的实际执行规则为准，代码是按该规则执行，但不能将其概括成标准全局 DAPO token mean。

CPU 反例中，同一 token 张量的这两种 scalar 为 −0.09783 与 −0.04784；这是数学非等价的见证，不是当前实验损失差值估计。机制诊断须在同 cache 上明确控制实际 loss 权重；否则比较 m=1/16 时同时改变聚合和归约，不能归因到 batch-vs-prompt。

## 3. 参数化尺度与 Adam 坐标：不是“信号强弱”的公平度量

**严重性 P1（方法解释混杂，不是违规实现）；本次新增/深化。证据 A+B+C。**

LoRA 初始化审计明确使用 Kaiming A、零B，见 [initialization.json](/root/peft-for-rl/runs/protocol_contracts/rlvr_v1_lora_r32_a64_lr1em5_b64m16n8_seed42.initialization.json) 和 [peft_lora_init.py:238](/root/peft-for-rl/verl/utils/peft_lora_init.py:238)。每行 `E||A_i||²=1/3`。GradTop/REAL 导入单位正交行，见 [peft_gradient_subspace.py:1745](/root/peft-for-rl/verl/utils/peft_gradient_subspace.py:1745)，每行范数约1。因此同 rank/alpha/LR 仍包含 A 尺度差异。

在 B=0 时，`∇_B L=s G Aᵀ`、`∇_A L=0`。参数梯度范数同时反映投影捕获、A 尺度、参数化和轨迹，并不能直接称 reward-gradient “更强”。SGD 下有效第一步含 `AᵀA`；Adam 下即使梯度缩放被二阶矩部分消去，`ΔW=sΔBA` 仍包含 A 本身的尺度。本次 CPU 例将 A 乘 √3，Adam 首步有效权重更新范数比为1.73205083。

更进一步，**即使同一子空间、相同奇异值，一般正交旋转基底也会改变逐元素 Adam 的有效更新。** 用 `A₂=R A₁`，初始 B 均为0，两个输入投影矩阵差异只有2.78e−16，但同一 dense G 的 Adam 首步 `BA` 更新相对差异为0.73650，cosine为0.72879。它是非不变性的构造见证，不是当前训练差异大小。单纯 gauge 的符号/置换是特殊等变情况，不能据此认为任意旋转也等变。

所以“probe 子空间覆盖好 → 当前 LoRA-Adam 效用好”还缺一段真实优化证据。后续应记录实际 `Δ(sBA)`、独立题集 loss/reward 改善、策略KL；方向目标的机制对照需控制初始化尺度/优化预算，而不是把尺度归一化当作新的方法贡献。现有同LR主结果可报告端到端方法比较，不能称只改变统计目标的受控消融。

GradTop A 确实可训练，REAL A 确实冻结，见 [fsdp_workers.py:809](/root/peft-for-rl/verl/workers/fsdp_workers.py:809) 及 [peft_gradient_subspace.py:1682](/root/peft-for-rl/verl/utils/peft_gradient_subspace.py:1682)。根审查的新增 [saved_space_geometry.json](saved_space_geometry.json) 同时发现当前 step150 的 GradTop learned-A 相对变化均值0.8825%、初始/末端 rowspace overlap均值0.9999245、按BA能量加权初始空间外BA仅0.01030%。因此可以说“在当前LR和轨迹下，初始化空间约束实际上仍很强”；不能写成代码冻结了A，也不能预先断言其已有效在线跟踪策略漂移。

## 4. 主算法路径逐项核验

| 对象 | 直接实现与证据 | 审计判断 |
| --- | --- | --- |
| 零函数随机投影 | [peft_gradient_subspace.py:70](/root/peft-for-rl/verl/utils/peft_gradient_subspace.py:70) A=0，B为N(0,1/width)，按模块名/observation确定性刷新；[fsdp_workers.py:623](/root/peft-for-rl/verl/workers/fsdp_workers.py:623) probe alpha=width | `grad_Aᵀ/scaling=GᵀB`正确；本次CPU误差0；无额外prompt sign |
| 完整组与DP | [ray_trainer.py:1433](/root/peft-for-rl/verl/trainer/ppo/ray_trainer.py:1433) 产生稳定ID、校验分组/DP可整除；[ray_trainer.py:1930](/root/peft-for-rl/verl/trainer/ppo/ray_trainer.py:1930) 显式probe禁用batch balance；normalization检查每local group有完整K | 当前每observation全局16 prompts×K8；内部微批次可拆分；本次gloo CPU测试通过 |
| 不更新policy | [dp_actor.py:1758](/root/peft-for-rl/verl/workers/actor/dp_actor.py:1758) 直接capture，不调用optimizer/clipping | 与artifact `optimizer_updates=0`一致；B梯度在精确零A时为0；collector每次重置A=0 |
| 能量累计 | [peft_gradient_subspace.py:228](/root/peft-for-rl/verl/utils/peft_gradient_subspace.py:228) 拼接sketch；[peft_gradient_subspace.py:136](/root/peft-for-rl/verl/utils/peft_gradient_subspace.py:136) 薄QR+SVD保留UΣ；[peft_gradient_subspace.py:1494](/root/peft-for-rl/verl/utils/peft_gradient_subspace.py:1494) 每trainer step压缩 | 当前每4 observations压缩一次，capacity64。跨observation相反梯度不相消；同observation内可抵消。本次CPU保留±S总能量与2||S||²一致到float32误差 |
| rank/导出 | [peft_gradient_subspace.py:1529](/root/peft-for-rl/verl/utils/peft_gradient_subspace.py:1529) 输出前32个单位方向的转置；[peft_gradient_subspace.py:1745](/root/peft-for-rl/verl/utils/peft_gradient_subspace.py:1745) A加载、B=0 | 196模块uniform32，artifact补齐方向0。target_energy=.95只是诊断，不是保留95%的保证 |
| REAL逐题梯度 | [peft_real.py:49](/root/peft-for-rl/verl/utils/peft_real.py:49) binary/population std/每response长度/K；[dp_actor.py:517](/root/peft-for-rl/verl/workers/actor/dp_actor.py:517) 一题完整backward累计后capture | 保留题内正负回答交叉项；并非逐response平方。每题输出投影刷新，DP同题复制保证先合题而非跨题平均 |
| REAL二阶统计 | [peft_real.py:215](/root/peft-for-rl/verl/utils/peft_real.py:215) `Hᵀ(HΩ)`；[peft_real.py:140](/root/peft-for-rl/verl/utils/peft_real.py:140) 对称核心、伪逆平方根、最终SVD | Nyström代数正确；目标是被筛选题组条件下的未中心化二阶矩；本次显式矩阵oracle/尺度测试通过 |
| 筛选差异 | [ray_trainer.py:1655](/root/peft-for-rl/verl/trainer/ppo/ray_trainer.py:1655) REAL至少两正两负；GradTop不作此筛选；[core_algos.py:318](/root/peft-for-rl/verl/trainer/ppo/core_algos.py:318) GRPO默认sample std加ε | 都与各自登记一致。batch-vs-prompt并非现有两方法的唯一变化；REAL排除1成功/7失败等rare-success组，不能把选样后统计差异全叫题间噪声 |

GradTop 当前 [summary.json](/root/peft-for-rl/runs/analysis/gradtop_r32_o16_w8_c64_seed42_20261015/summary.json) 的 rank32 retained sketch energy 模块均值0.71439、最小0.44231；最后相邻累计空间 overlap均值0.91509。前者不是独立梯度能量覆盖率，后者不是两个独立probe重复的稳定性，均不能作为“已达到95%或已证明可重复”的证据。

## 5. 新发现的数值边界：frozen probe 并非逐token精确 ratio=1

**严重性 P2；本次新增日志观测，原因及效果未定位（观测A、解释D）。**

不做optimizer更新时，精确数学上 old/current policy一致，PPO ratio应为1，clipping不活跃。但本次4个probe step记录的 `actor/pg_clipfrac` 约2.30e−5、2.47e−5、3.90e−5、2.31e−5；`actor/ppo_kl` 约2.28e−6至7.93e−6，见 [probe.log:1112](/root/peft-for-rl/runs/ops/gradtop_r32_clean_20261015/probe.log:1112)。这说明运行中的重算不是数值上逐token完全一致。

old log-prob使用 `eval()` 且以整批动态packing重算，[dp_actor.py:1381](/root/peft-for-rl/verl/workers/actor/dp_actor.py:1381)；更新侧使用 `train()`、按mini-batch重新packing，[dp_actor.py:1438](/root/peft-for-rl/verl/workers/actor/dp_actor.py:1438)。BF16/packing/不同执行路径都是待检原因。当前数据只能证明小但非零的clip统计；不能断言存在实质policy漂移，不能量化其对最终子空间的影响。无需因该小统计推翻主结果，但将probe写成“严格无clipping的reward梯度”不准确；机制研究应统一执行路径或同时记录未clip score-gradient参照，以检验差异是否可忽略。

## 6. 冻结代码、验证范围与未闭合项

本次重新SHA256核验：GradTop首轮plan锁定29个文件，全部一致；续训plan锁定32个文件，全部一致。plan SHA分别为 `2b4ac58ed7899efcee302f59057527d18e945176c8a6f8a94c61b1edc5dc6cd6`、`a98a83cb423040b55122855c5622491b437bab6820e6b03d688d94db65f3bdb6`。不以git HEAD替代实际冻结源码。

使用实验环境 `/home/node/anaconda3/envs/peft-for-rl/bin/python`（torch2.8.0+cu128）运行20项已有CPU必要检查，**20 passed，14.53s**，仅Ray import deprecation warning。覆盖 token_mean双clip梯度、不均等microbatch/DP平均、完整prompt分配、2-rank CPU gloo、普通训练不误入probe、共享energy collector预算/导出、REAL逐题归一化/sketch/Nyström/actor接入以及Qwen2 probe checkpointing。精确命令见 [cpu_existing_tests_command.txt](cpu_existing_tests_command.txt)，输出见 [cpu_existing_tests.log](cpu_existing_tests.log)。共享collector的个别fixture使用非当前variant标签，但测试的是当前同一路energy累计/导出代码，不作为该variant实验结果。

另在同一runtime Python重新运行本次CPU数学例，输出见 [cpu_audit_checks.json](cpu_audit_checks.json)。这些测试不是4-GPU BF16 end-to-end验收，不能覆盖运行时rollout adapter加载。两次恢复缺陷、正式packing语义、真实Adam更新与probe几何之间的关联仍需专门验收/诊断；本报告没有启动这些额外GPU实验。
