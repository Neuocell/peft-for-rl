# 当前协议 GradTop-r32：从 step50 续训至 step150

登记日期：2026-10-15；完成更新：2026-10-16 09:57（Asia/Shanghai）。本轮继承[当前GradTop首轮实验](gradtop_r32_clean_baseline_20261015_zh.md)，同一experiment的100/150训练与六榜评测已全部完成，原probe复用。最终[续训 status.json](../../runs/ops/gradtop_r32_continue_step150_20261015/status.json)为`complete`、`completed_evaluations=[10,50,100,150]`、`auto_resume=false`；本轮执行止于150，是否追加到200尚待讨论。

## 完成结果与恢复点

完整结果见[原续训results.md](../../runs/ops/gradtop_r32_continue_step150_20261015/results.md)及[results.json](../../runs/ops/gradtop_r32_continue_step150_20261015/results.json)。GradTop step100/150的response-weighted Avg.为47.4%/48.4%，Macro Avg@k为41.3425%/42.3971%。150相对同step LoRA与REAL32的Macro差值为+3.5830/+2.9892 pp，题目配对95%区间分别为[+2.2411,+4.9986]/[+1.6527,+4.4124] pp。

自身轨迹上，150相对50为+1.0689 pp，区间[+0.0105,+2.1266] pp，下界很接近0；150相对100为+1.0545 pp，新补充区间[-0.2529,+2.3554] pp跨0。因此不能把150点估计更高直接写成后续训练必然继续改善。150−100来源为[独立补充配对审计](../../runs/ops/gradtop_r32_continue_review_20261016/paired_step150_vs_step100.json)，不改写原runtime结果。

[step100完整checkpoint验收](../../runs/ops/gradtop_r32_continue_step150_20261015/checkpoint_step100_verification.json)于10月16日02:18通过：400次逻辑Adam更新、dataloader100批/6,400题、四rank RNG/scheduler；[step150验收](../../runs/ops/gradtop_r32_continue_step150_20261015/checkpoint_step150_verification.json)于08:51通过：600次更新、150批/9,600题及四rank状态。两次新评测均通过四shard及整体验证，每次902题/7,248回答。最终status中的`latest_logged_step=149`是最后一次训练轮询记录，完整150checkpoint与完成状态才是终点证据。

上述验收确认模型、optimizer和数据等恢复状态。**实际配置与源码审计已确认首轮rollout同步缺陷：** `grad_subspace`属于`_is_lora`，配合`lora_adapter_path=None`、`load_format=dummy`、`val_before_train=False`，恢复只加载actor，初始化没有向rollout同步当前adapter。首次generate的wakeup因`base_sync_done=False`只收集并加载base权重，`lora_`参数被过滤；此时无adapter ID、`lora_request=None`。同步末尾才置True，第二次wakeup才`add_lora`。

因此按本次配置与代码路径，恢复后的step51/101首轮采样走base-only路径，日志首轮异常与之对应；**尚未对运行时engine中的tensor/hash直接实测**。不能把完整actor/optimizer/RNG/data恢复解释为严格连续的策略采样。端点评测分数仍有效，但同协议方法比较必须披露该轨迹偏差。下次续训前必须修复并验收首轮adapter已加载、注册且被首次生成实际使用，当前不启动200。下文保留原计划与启动过程，注明当时时点；原计划中的“完整恢复”仅代表当时目标，不能覆盖本次发现。

即使修复后从150继续，也不能追溯消除step51/101采样对既有optimizer更新的影响，只能改善后续恢复路径。这类接续可作为探索；论文级干净轨迹需要修复后重跑，并审计LoRA/REAL对照的恢复路径。本轮只判断，不启动新训练。

证据详见[训练与恢复审计](../../runs/ops/gradtop_r32_continue_review_20261016/training_audit.md)及[JSON](../../runs/ops/gradtop_r32_continue_review_20261016/training_audit.json)。部分下游`fsdp_utils`/vLLM源码不在原32项哈希清单内，审计保存当前SHA256及早于本run的mtime；结论是配置与代码推导、日志旁证，不能冒充原engine权重的直接测量。两次首轮采样偏差的累计性能影响尚未量化。

## 研究问题和预先固定的比较（2026-10-15启动前）

当前 GradTop 的六任务等权 Macro Avg@k 从 step10 的 37.4610% 上升至 step50 的 41.3282%；按 response 数加权的 Avg. 从 43.3% 上升至 47.2%。这支持继续观察训练轨迹，但不能外推出后续必然改善。已有同配置 REAL32 的 Macro 在 step100 为 42.0021%，step150 为 39.4078%，也说明延长预算可能出现退化。

本轮固定执行 **完整恢复50 → 训练100 → 评测100 → 完整恢复100 → 训练150 → 评测150**，到150及评测结束后停止。完整报告 GradTop 的10/50/100/150四个点，不按评测结果只挑最好一步。分别回答：

1. 额外50/100个训练step能否提高当前GradTop的泛化分数，提升是否集中在少数任务？
2. 在相同训练step下，相对LoRA和REAL32的差距是否持续？
3. 分数变化是否伴随parse、达到长度上限比例、平均输出长度或Pass@k变化？

这轮沿用单训练seed42。配对bootstrap的区间仅度量固定训练结果下的题目重采样不确定性，不代表不同训练seed下的方差，也不能据此单独证明共享更新或增量SVD的机制。

## 配置和恢复条件

| 项目 | 固定设置 |
| --- | --- |
| 实验名 | `rlvr_v1_gradtop_r32_a64_lr1em5_b64m16n8_seed42_20261015` |
| 科学协议 | 原 `gradtop_r32_clean_20261015.json` 原字节保留；本文件和续训plan扩展评测网格 |
| 模型、硬件 | DeepSeek-R1-Distill-Qwen-1.5B，4×L40S，bf16 |
| Adapter | 全部196个linear模块，uniform rank32 / alpha64 / dropout0，A/B均可训练 |
| 学习率 | 1e-5，constant，warmup0 |
| 训练采样 | 64 prompts/step × 每题8条response；PPO minibatch16，每step4次逻辑Adam更新 |
| 数据和随机种子 | 原boxed数学数据、原sampler顺序、所有seed42 |
| Probe | 复用原16 observations、width8、capacity64增量SVD artifact；没有新增probe预算 |
| Loss、奖励、长度 | 原协议全部保持；不加入sign、mask、KL、entropy或overlong shaping |
| 恢复起点 | 原 `global_step_50` 的四rank model、optimizer、RNG、scheduler及data.pt |
| step100累计预算 | 6,400 prompts、51,200 responses、400次逻辑Adam更新 |
| step150累计预算 | 9,600 prompts、76,800 responses、600次逻辑Adam更新 |
| 本次额外预算 | 100 trainer steps、6,400 prompts、51,200 responses、400次逻辑Adam更新 |
| 保存 | 每10步独立adapter snapshot；每50步完整checkpoint；完整checkpoint保留最近2份 |

起点已实际核验四rank均392个optimizer state、Adam step=200、LR=1e-5，scheduler last_epoch=50，DataLoader已产出50批/3,200题；step50 adapter SHA256为 `5d1aaa7cbb7704eab50bf2d97db25a78354db4bb11b9b42027c5a16cccea59f7`。

统一launcher使用`RESUME_MODE=auto`。控制器持有原实验和共享GPU资源锁，并在每段启动前强制latest marker等于50/100、验证完整恢复点，再从日志确认加载原路径的model/optimizer/RNG/scheduler。新的第一个训练step应分别为51/101。模型构建时读取原GradTop基底以建立相同结构，随后加载训练后的完整checkpoint；这不等于重新probe或把A/B重置后训练。

## 评测与表格

每个新端点都使用原固定六榜、7,248 responses/902题，eval seed42、temperature0.6、top-p0.95、max_new_tokens32768。现有LoRA100/150与REAL32_100/150均已重新验证同snapshot、同题目与response索引集合和采样配置，详见[对照验收](../../runs/ops/gradtop_r32_continue_step150_20261015/comparison_preflight_audit.json)。

主表沿用用户规定格式，百分数保留一位小数；任务单元格为 `Avg@k / Pass@k`：

| Method | Step | Avg. | AIME24@32 | AIME25@32 | AMC@32 | HMMT@32 | MATH500@4 | Minerva@4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GradTop | 10 | 43.3 | 29.8 / 76.7 | 22.3 / 56.7 | 70.8 / 92.5 | 9.4 / 33.3 | 70.1 / 79.4 | 22.4 / 35.7 |
| GradTop | 50 | 47.2 | 30.8 / 70.0 | 25.3 / 56.7 | 76.0 / 95.0 | 10.5 / 40.0 | 72.8 / 78.8 | 32.5 / 43.4 |
| GradTop | 100 | 47.4 | 29.9 / 70.0 | 23.9 / 50.0 | 78.0 / 97.5 | 9.9 / 33.3 | 73.6 / 79.8 | 32.8 / 43.4 |
| LoRA | 100 | 44.5 | 27.7 / 76.7 | 23.1 / 50.0 | 74.2 / 95.0 | 9.9 / 40.0 | 72.2 / 80.2 | 22.9 / 34.6 |
| REAL32 | 100 | 47.8 | 35.1 / 73.3 | 26.2 / 53.3 | 76.2 / 95.0 | 10.1 / 40.0 | 73.5 / 79.4 | 30.9 / 43.4 |
| GradTop | 150 | 48.4 | 33.0 / 73.3 | 25.6 / 56.7 | 78.0 / 100.0 | 10.0 / 40.0 | 74.1 / 80.4 | 33.6 / 45.2 |
| LoRA | 150 | 44.9 | 30.5 / 73.3 | 22.8 / 56.7 | 74.1 / 100.0 | 10.2 / 36.7 | 72.5 / 80.8 | 22.7 / 36.8 |
| REAL32 | 150 | 45.5 | 28.3 / 73.3 | 25.8 / 56.7 | 75.8 / 95.0 | 10.7 / 30.0 | 72.7 / 80.2 | 23.1 / 35.3 |

`Avg.` 是按response数加权，另表给出六榜等权Macro Avg@k、Macro Pass@k、Parse、Hit max和平均长度。每个新端点对同step LoRA、REAL32以及自身step50做题目配对bootstrap，10,000次、seed42，报告Macro差值与95%区间。自身step50比较属于训练轨迹，不是同预算方法比较。

| GradTop step | 对照 | Macro差值（pp） | 95%题目配对区间（pp） |
| --- | --- | ---: | --- |
| 100 | LoRA100 | +3.0035 | [+1.7492, +4.3414] |
| 100 | REAL32_100 | -0.6596 | [-2.1011, +0.8246] |
| 100 | 自身50 | +0.0144 | [-1.0081, +1.0496] |
| 150 | LoRA150 | +3.5830 | [+2.2411, +4.9986] |
| 150 | REAL32_150 | +2.9892 | [+1.6527, +4.4124] |
| 150 | 自身50 | +1.0689 | [+0.0105, +2.1266] |
| 150 | 自身100（补充审计） | +1.0545 | [-0.2529, +2.3554] |

原六项paired来源在本续训目录；150−100来源在独立review目录。区间只适用于Macro，不适用于response-weighted Avg.，也不覆盖训练seed波动。GradTop100→150的Macro Pass@k为62.3359%→65.9368%，parse为97.13%→99.03%，hit-max为0.80%→0.87%，平均长度为5394.2→5486.8；全部逐任务与行为指标见[最新交接主表](gradtop_real_handoff_20261016_zh.md)。

## 资源、溯源和文件位置

控制器：[continue_gradtop_r32.py](../../scripts/local/continue_gradtop_r32.py)。独立工作目录：[runs/ops/gradtop_r32_continue_step150_20261015](../../runs/ops/gradtop_r32_continue_step150_20261015)。原首轮plan、status、contract与验收记录另存`original_run/`，原probe三文件及训练/评测源码均按SHA256锁定。统一launcher会更新目标step contract，所以保留50原件及100/150阶段副本，检查科学配置仅`target_step`发生变化。

Contract比较仅排除一个派生诊断浮点数`orthogonality_error_max`：同一份artifact在不同CPU线程数下重算可得到2.563e-5或2.694e-5。三个artifact文件的完整SHA256仍必须一致，诊断值本身仍须有限且不超过原协议1e-4阈值；不豁免任何训练配置、artifact内容或其余contract字段。

| 文件 | 用途 |
| --- | --- |
| `plan.json`、`registration_preflight.json` | 冻结续训范围、源码/资产哈希与只读预检 |
| `controller.log`、`status.json` | 各阶段记录及09:57最终完成状态 |
| `train_to100.log`、`train_to150.log` | 各段训练日志，含恢复路径和step指标 |
| `resume_step50_verification.json`、`resume_step100_verification.json` | 各段完整恢复点验收 |
| `checkpoint_step100_verification.json`、`checkpoint_step150_verification.json` | 新完整checkpoint验收 |
| `results.md`、`results.json` | 已完成的10/50/100/150轨迹与同step对照结果 |
| `paired_step*_vs_*.json` | 配对比较及置信区间 |

启动前把字节相同的重复base文件改为指向canonical模型的symlink，逐文件SHA256不变，释放3.3167GiB。另将历史r8实验step100的四个完整model shard无损压缩为`.pt.zst`，逐文件验证解压SHA256，实际释放3.5957GiB；adapter、optimizer、RNG和数据状态保留。该历史checkpoint若需完整续训，先执行归档清单中的恢复命令。该操作没有改变本次GradTop或当前LoRA/REAL评测模型。记录见[base去重清单](../../runs/ops/gradtop_r32_continue_step150_20261015/base_model_deduplication.json)和[历史checkpoint归档清单](../../runs/ops/gradtop_r32_continue_step150_20261015/historical_checkpoint_archive.json)。

续训按实测checkpoint/snapshot大小检查空间，计入10%大小裕量及2GiB余量；100→150阶段仅在确认原retention=2会在写入新checkpoint前删除旧50 actor后计入可释放空间。控制器未额外清理其他实验。原step50 adapter事前单独备份，保存150时retention已删除50 actor完整状态，控制器随后恢复评测adapter引用；当前step50仅保留adapter和data.pt，100/150为完整恢复点。

## 启动验收过程记录（2026-10-15当时状态）

当时续训控制器登记PID `1248678`；冻结plan SHA256为 `a98a83cb423040b55122855c5622491b437bab6820e6b03d688d94db65f3bdb6`。只读实际预检通过，启动前空闲13.96GiB，首段预算10.661GiB。CPU回归61项通过，日志见[CPU checks](../../runs/ops/gradtop_r32_continue_step150_20261015/cpu_checks.log)。以下保留启动时确认，当前完成状态见本文开头。

2026-10-15 20:49实际启动确认：训练日志明确从原`global_step_50`载入model、optimizer、RNG和scheduler，进度条从`50/100`恢复；runtime contract与登记的target100匹配。控制器状态为`train_to100`，首个新增step尚未完成，新评测尚未开始。见[实际恢复确认](../../runs/ops/gradtop_r32_continue_step150_20261015/live_resume_verification.json)。
