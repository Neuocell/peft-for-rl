# 当前协议 GradTop-r32：干净对照实验

日期：2026-10-15（Asia/Shanghai）。状态：训练及step10/50评测已全部完成，19:53队列完成；[完整结果表](../../runs/ops/gradtop_r32_clean_20261015/results.md)采用近期REAL表格格式，另列Macro、行为指标和配对区间。

本轮只执行 **原 batch 梯度聚合 + random-B sketch + 增量 SVD + uniform rank32** 的 GradTop。目的首先是建立可与当前 LoRA/REAL 比较的有效实验结果；不预设 REAL 的逐题二阶矩更好，不把 prompt-sign、token mask、fixed-A 或新排序加入这一基线。

科学配置见 [机器可读协议](../../configs/experiments/gradtop_r32_clean_20261015.json)。研究动机和后续探测设计见 [共享更新方向研究路线](gradtop_update_geometry_research_20261015_zh.md)。本文件是本轮执行依据；10月14日候选文档中的“先跑 Prompt-sign”不再是当前优先级。

## 1. 唯一首轮配置

| 项目 | 冻结值 |
| --- | --- |
| 实验名 | `rlvr_v1_gradtop_r32_a64_lr1em5_b64m16n8_seed42_20261015` |
| METHOD | `gradtop`，显式指定，不能使用候选launcher默认的prompt-sign |
| 模型 | DeepSeek-R1-Distill-Qwen-1.5B |
| GPU / precision | 4×L40S / bf16 |
| 正式rank / alpha / scaling | 所有196模块rank32 / alpha64 / scaling2 |
| 正式adapter | `A=Q.T`正交初始化，`B=0`，A/B均可训练 |
| 参数量 | 36,929,536个可训练参数 |
| Dropout / target | 0 / all-linear |
| LR / scheduler / warmup | 1e-5 / constant / 0 |
| 采样 / PPO minibatch | 64 prompts/step / 16 prompts |
| 每题responses / PPO epochs | 8 / 1 |
| prompt / response上限 | 1024 / 8192 |
| 训练loss | 原token-mean DAPO-style GRPO，clip low0.20/high0.28/C10 |
| 其他训练项 | 无KL reward、无actor KL、无entropy bonus、无overlong shaping、weight decay0 |
| 种子 | data、rollout、FSDP、trainer、初始化均42 |
| 正式训练预算 | step50 = 3,200 prompts、25,600 responses、200次逻辑optimizer update |
| Snapshot / checkpoint | 每10步adapter snapshot；每50步及计划终点完整恢复点；完整点保留最近2份 |
| 评测 | 固定六榜，每次7,248 records，eval seed42、temperature0.6、top-p0.95、32768输出上限 |
| 评测网格与主指标 | step10/50；预先固定step50为主比较，不做best-checkpoint选优 |

正式训练设置与当前LoRA基线的共享`training`、`evaluation`字段逐项相同。REAL使用B-only，因而与GradTop是同rank/同正式更新预算的方法比较，不是等可训练参数比较；probe和总成本也单列。

这里的正式训练`token-mean`沿用当前actor实现：每个microbatch内部取token mean，再按microbatch样本占比累积并跨DP平均；一般不等于整个逻辑minibatch一次性全局token mean。本轮不改变各方法共同使用的正式训练实现。下文显式GradTop probe采用全局token分母，这个差别必须在机制诊断中单独控制，不能把probe梯度直接称为真实Adam输入梯度。

## 2. Probe的确切定义

| 项目 | 冻结值 |
| --- | --- |
| 初始policy | 与base相同，`A_probe=0` |
| 输出随机投影 | Gaussian B，每元素方差1/8，每个observation刷新 |
| Probe宽度 | 8；worker实际probe LoRA r8/alpha8/scaling1，与正式rank32不同 |
| Loss | 保留原GRPO advantage和PPO loss分支，真正的全局有效response-token mean |
| 统计单位 | 全局16个完整prompt组×8条回答 = 128 responses/observation |
| 预算 | 16 observations = 4个64-prompt采样step = 256 prompts / 2,048 responses |
| 内部更新 | optimizer update恒为0，无gradient clipping |
| 题组筛选 | 不采用REAL的至少两正两负筛选；零advantage组保留分母 |
| 题组与长度 | probe禁用trainer级长度balance，完整组先归入observation；microbatch可动态拆分 |
| 额外符号 / mask | 无 / 无 |
| CPU累计 | 原幅值sketch，增量QR/SVD，capacity64 |
| 压缩时点 | observations 4、8、12、16之后，亦即每个rollout step末 |
| 终止 | 固定4个step；不使用稳定性提前停止 |
| 最终rank | 每模块32，`rank_bins=[32]` |
| 信号秩不足 | 与现有实现一致，用确定性坐标QR补齐；逐模块及总数完整披露 |

每个observation的梯度是原batch loss的聚合梯度G，不再给每题乘随机符号。收集

\[
S_t=\nabla_A L_t^\top/s=G_t^TB_t.
\]

每个采样step末，对前一次压缩因子和本step的4份sketch拼接：

\[
F_k=\operatorname{truncate}_{64}
\left([F_{k-1},S_{4k-3},S_{4k-2},S_{4k-1},S_{4k}]\right).
\]

`truncate`先做薄QR，再对小R做SVD，保存列`u_j*sigma_j`以保留累计能量。最终取前32个单位输入方向，转置后导出A；A不携带奇异值幅值。`target_energy=0.95`仍保留为诊断配置，但`rank_bins=[32]`意味着它不再决定实际rank，不能声称每层都捕获95%能量。

这不是一次性总平均梯度SVD，也不是对每题单独平方。增量SVD近似保存跨observation的二阶能量：没有指数遗忘、没有时间衰减、没有单位范数重权；截断产生的顺序依赖不等于主动适应policy漂移。

## 3. 与历史GradTop的关系

本实验保留了历史方法的核心计算路径，正式命名为“当前协议GradTop-r32”，不称为旧run逐位复现。

| 项目 | 历史记录 | 本轮 |
| --- | --- | --- |
| probe预算 | 12 observations，192 prompts | 16 observations，256 prompts，多33.3% |
| 每次采样 | 16 prompts，通常每step一次observation | 64 prompts，每step四次observation |
| SVD压缩频率 | 通常每observation | 每4个observation |
| rank | 能量阈值分档，mean约31.63 | uniform32 |
| loss归约 | microbatch-token-mean再按样本数加权 | 分母在microbatch前固定的全局token mean |
| probe分组 | 旧长度balance可拆散题组 | 完整题组归入同observation |
| 停止条件 | 有稳定性提前停止 | 固定预算 |
| 正式训练 | 历史常见dropout0.05，LR依run | 本轮协议dropout0、LR1e-5 |

历史probe本身已经关闭dropout；不能把这一点写成本轮新增。选择16 observations是遵守现有注册预算、对齐64-prompt采样，不是通过本轮结果搜出的最佳值。若以后比较12 observations，应另注册3个采样step的预算消融。

## 4. 数据、资产与干净比较

沿用当前LoRA/REAL已登记的本地资产绑定：原始17,917行boxed数据、本地parquet SHA256前缀`8e3c9314`、sampler-prefix前缀`9c926a98`，所有base weights及benchmark snapshot保持相同。新协议同时记录canonical协议和LoRA本地协议的完整SHA256；不改写父协议。

Probe结束后新建正式actor/optimizer/dataloader，从seed42 sampler第一个prompt开始。probe消费的256题不推进正式训练的数据位置，不携带probe优化器状态。正式训练长度balance保持共享协议的原行为。

Step50主对照是：

| 对照 | Macro Avg@k（当前已完成） | 比较含义 |
| --- | ---: | --- |
| LoRA r32、LR1e-5、step50 | 38.7224% | 同共享训练配置，比较初始化及额外probe |
| REAL64、r32、LR1e-5、step50 | 40.0388% | 方法比较；披露冻结方式与probe成本不同 |
| REAL32、r32、LR1e-5、step50 | 40.3361% | 包含较小REAL probe预算的强对照 |
| 本轮改动版GradTop、r32、LR1e-5、step50 | 41.3282% | 当前协议适配版；非历史原实现严格复现 |

本轮GradTop的step10/50六任务等权Macro Avg@k分别为37.4610%/41.3282%，对应按response数加权的Avg.为43.3%/47.2%；不将旧39.12%/40.46%填入新主表。完整表同时报告逐任务Avg@k/Pass@k、Macro Pass@k、parse、hit-max和平均长度；以problem为cluster做配对bootstrap，10,000次、seed42。仅单训练seed，CI不覆盖training-seed不确定性。

评测执行顺序固定为step50后step10，以先取得主要对照。step10是预声明轨迹点，不与LoRA50作matched比较、不用它选优；若未来改为best-checkpoint比较，所有对照必须补同一候选网格。

## 5. 本轮不混入机制诊断

“共享更新vs逐题能量”的诊断需要同loss、同rollout cache和多组独立rollout，单纯把本轮GradTop与REAL最终分数相减无法归因于聚合。该研究按独立文档实施，不在本次256-prompt初始化中额外采样、换目标或改SVD容量。

首轮完成后可以回答当前协议GradTop是否有效、probe成本和训练轨迹如何。它不能单独回答逐题能量与共享更新谁更好、增量SVD是否比Nyström好、或fixed-A是否有益。

增量SVD应作为一个独立的估计器轴验证：同一组原始sketch分别做一次性SVD、当前分块增量SVD、不同顺序重放，并在必要时用同一目标的Nyström作对照。不能将统计目标、投影宽度、分解器、A是否冻结同时改变后，把全部收益归给SVD。

## 6. 执行、验证和成本记录

本轮复用已有GradTop launcher和统一正式训练入口。先独立`PHASE=probe`，完整验证artifact，再`PHASE=train`；不使用默认`METHOD`。

```bash
PYTHON_BIN=/home/node/anaconda3/envs/peft-for-rl/bin/python \
PROTOCOL_FILE=/root/peft-for-rl/configs/experiments/gradtop_r32_clean_20261015.json \
METHOD=gradtop PHASE=all LORA_RANK=32 LR=1e-5 TARGET_STEP=50 DRY_RUN=1 \
EXP_NAME=rlvr_v1_gradtop_r32_a64_lr1em5_b64m16n8_seed42_20261015 \
GRADTOP_RUN_DIR=/root/peft-for-rl/runs/gt32_1015 \
GRADTOP_ARTIFACT_DIR=/root/peft-for-rl/runs/analysis/gradtop_r32_o16_w8_c64_seed42_20261015 \
RAY_TEMP_BASE=/tmp/gt32_1015 RAY_NUM_CPUS=20 \
bash scripts/local/run_gradtop_candidates_4gpu.sh
```

短run目录用于避免Ray UNIX socket路径长度限制。`DRY_RUN=1`可以在资源忙时做配置/资产检查；没有artifact时训练阶段只是显式preview，不能声称artifact已验收。

真实probe验收覆盖196模块、rank/alpha、正交性、实际16次observations/4steps、256题/2048responses、无更新、补齐数和三件artifact哈希。正式训练需完整step50恢复点、step10/50快照，并核对Adam计数200。step10只有snapshot；评测时从本轮step50标准adapter复制配置，配上step10原权重并记录导出哈希，不冒充可恢复checkpoint。

成本记录至少包括probe与训练各自wall time、采样/old-logp/actor/CPU收集分解时间、rollout/backward tokens、observation与实际microbatch次数、峰显存和初始化参数量。若现有日志没有某项精确计数，标记未测，不从16 observations推断只有16次反传。已有REAL32为统计前缀导出，离线导出时间不能当完整采集成本。

运行目录：[runs/ops/gradtop_r32_clean_20261015](../../runs/ops/gradtop_r32_clean_20261015)。其中`plan.json`固定执行计划和源码哈希，`status.json`记录阶段，`probe.log`、`train.log`记录实际运行，六榜验证和配对报告独立保存。本轮自动执行止于step50及10/50评测，不自动追加sign、mask、其他LR或续训150。

当前四卡由REAL32续训与其100/150评测占用，GradTop排在其后；短暂GPU空闲不代表前驱已完成。队列冻结`real_probe32_early_step100`控制器的计划哈希与进程身份，等待其完整完成step150训练及step100/150评测、退出并释放资源锁，再进入GradTop；不自动跟随未登记的替代控制器。磁盘不足时等待，运行状态必须区分排队、probe、训练、评测和完成。

## 7. 本次登记与实际状态

2026-10-15 00:41（Asia/Shanghai）已启动等待控制器，PID `971428`。登记状态为`waiting_for_real150_and_evaluations`；GradTop probe、正式训练和评测尚未开始，没有本轮效果数字。

- [冻结执行计划](../../runs/ops/gradtop_r32_clean_20261015/plan.json)：SHA256 `2b4ac58ed7899efcee302f59057527d18e945176c8a6f8a94c61b1edc5dc6cd6`，包含29个运行源码/协议文件的哈希。
- [当前状态](../../runs/ops/gradtop_r32_clean_20261015/status.json)和[控制器日志](../../runs/ops/gradtop_r32_clean_20261015/controller.log)：后续阶段以它们为准。计划或源码漂移时停止，不静默换设置。
- [队列入口](../../scripts/local/run_gradtop_clean_after_real.py)：执行范围是probe→train50→eval50→eval10→三条step50配对比较。
- [预检结果](../../runs/ops/gradtop_r32_clean_20261015/queue_preflight.log)：共享资产和三条完整step50对照均通过；当前artifact状态明确为`pending_probe`。首次预检发现dry-run创建的两个空目录，已用仅删除空目录的`rmdir`清除后通过，不曾覆盖实验产物。
- [GradTop相关CPU检查](../../runs/ops/gradtop_r32_clean_20261015/cpu_checks.log)：102 passed，只有一条既有Ray API弃用警告。[队列与评测回归](../../runs/ops/gradtop_r32_clean_20261015/queue_checks.log)：另41 passed，包含21项新队列测试。以上均不替代真实四卡执行验证。

为容纳这次实验，已把闲置LoRA step150的完整22文件恢复点迁往另一块本地ext4磁盘`/data`，逐文件SHA256验证后在原路径保留symlink，完整内容保留。迁移后根盘约24.1GiB可用；[迁移清单与回迁说明](../../runs/ops/gradtop_r32_clean_20261015/checkpoint_relocation.json)记录全部文件哈希。probe/训练前要求15GiB空闲，评测前要求2GiB；没有自动删除旧checkpoint或扩展其他实验。

2026-10-15 10:28更新：REAL前驱已在08:37完成全部训练与评测；GradTop自08:38起因根盘空闲不足15GiB而等待，期间没有开始probe或正式训练。检查时根盘约14.64GiB，随后将已完成、无活动引用的REAL64 probe统计缓存完整迁到`/data/peft-for-rl-preserved-analysis/`，17个文件全部SHA256核验一致，原路径保留symlink，释放约1.275GiB。详见[缓存迁移记录](../../runs/ops/gradtop_r32_clean_20261015/probe_cache_relocation.json)。根盘恢复至约15.92GiB后，原队列于10:28:44自动进入`probe`；冻结计划与运行源码哈希均未改变。此时正式step50训练和评测尚未开始。

2026-10-15 19:53更新：完整管线已完成。Probe耗时37.51分钟；正式50步训练耗时5.98小时并于17:04完成，完整checkpoint验证200次optimizer更新、四卡RNG/scheduler及3,200题dataloader进度。step50/10分别通过完整7,248条评测验证。Step50相对LoRA/REAL64/REAL32的Macro差值分别为+2.6057/+1.2894/+0.9920个百分点，95%题目配对bootstrap区间分别为[+1.4295,+3.8130]/[+0.2312,+2.3687]/[-0.1483,+2.1325]个百分点；相对REAL32的区间跨0。结果表与原始来源哈希分别保存于[results.md](../../runs/ops/gradtop_r32_clean_20261015/results.md)和[results_table.json](../../runs/ops/gradtop_r32_clean_20261015/results_table.json)。自动执行已停止于预声明范围，未续训100/150。

2026-10-15 用户追加续训要求：原首轮结果保留，新增[50→100→150续训执行计划](gradtop_r32_continue_step150_20261015_zh.md)，复用本轮完整step50和原probe，分别完成100/150六榜评测及同step LoRA/REAL32对照。续训状态见新任务目录，不改写本文件上面的首轮完成记录。
