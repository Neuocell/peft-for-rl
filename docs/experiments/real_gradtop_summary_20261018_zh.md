# REAL 与 GradTop：近期实验、成功运行版本及配置总表

更新日期：2026-10-18，Asia/Shanghai。证据快照时间为 **15:50:48**，另补入15:51完成的早期checkpoint CPU审计；本次整理不重新训练、生成回答或修改评分。当前版本以本文、[版本登记 JSON](../../configs/experiments/real_gradtop_versions_20261018.json)、[完整结果快照](evidence/real_gradtop_20261018/results.md)和[实现审计说明](../../audits/real_gradtop_implementation_audit_20261018_zh.md)为入口。

当前 GradTop 已完成 270 步训练及六榜评测，受检点中 step150 的 Macro Avg@k 最高，为 **42.3971%**，step270 为 **41.6845%**。旧 REAL32 的 step100 为 **42.0021%**，到 step150 降为 **39.4078%**。新 REAL-P256 已连续训练至 210，step210 的完整六榜 Macro 为 **35.8401%**；15:48已按用户要求中断 step200 评测并取消 step150 队列，未排入新的GPU任务。因此，新 REAL 长跑不能再写成“尚未启动”，也不能记作“270 已完成”。

这里的“成功运行版本”指 probe 产物可验证、正式训练实际完成且至少一个端点完成六榜评测。它不等于所有版本都优于 LoRA，也不等于现有单 seed 结果已经证明稳定的方法优势。GradTop 与旧 REAL32 的历史恢复缺陷必须和分数同时披露。

## 1. 方法版本和实际预算

| 版本 ID / 简称 | Probe 的实际配置 | 筛选前 prompts / responses | 纳入统计的题组 | 正式 adapter | 已完成训练 / 六榜评测 |
| --- | --- | ---: | ---: | --- | --- |
| `gradtop-r32-o16-v20261015` / 当前 GradTop | 4 rollout steps；16 个全局 minibatch observations；每个 16 题×K8；width8、capacity64 增量能量 SVD | 256 / 2,048 | 256 | uniform r32 / alpha64；A+B 训练 | 训练270；评测10/50/100/110/120/130/140/150/200/250/270 |
| `real64-r32-v20261013` / REAL64 | 64 probe steps；逐题 reward contrast；双侧 sketch128；PSD Nyström | 4,096 / 32,768 | 1,499 | uniform r32 / alpha64；固定 A，仅训练 B | 训练50；评测50 |
| `real32-r32-prefix-v20261014` / REAL32 | REAL64 在 step032 保存的累计统计精确前缀，离线导出；同一 REAL 定义 | 2,048 / 16,384 | 735 | 同上 | 训练150；评测50/100/150 |
| `real-p256-r32-v20261017` / REAL-P256 | **重新采样**4 probe steps；同一 REAL 定义，不复用旧缓存 | 256 / 2,048 | 87 | 同上 | 连续训练0→210；仅210评测完整，200中断、150取消 |

REAL32/64 的后缀表示 **probe rollout steps**，四个版本正式 rank 都是32。GradTop 的 width8 是随机输出投影宽度，也不是正式 rank。REAL 的“87/735/1499”是满足至少2正2负等条件的有效题组数，预算应报告筛选前尝试的题数。

历史 2026-09 GradTop v1 使用12 observations、192题、能量分档异构 rank（均值约31.63）、历史归约和稳定性停止规则。它与当前 uniform-r32 版本共享 random-B sketch / 增量 SVD 路径，但不是同一个实验版本。其轨迹及 Compact/chord 后处理保留在[历史完整结果表](local_a6000_full_benchmark_comparison_2026_10_zh.md)，不混入本轮匹配表。`gradtop_short`、`gradtop_prompt_mean`、`gradtop_prompt_sign`、`gradtop_masked` 是已登记候选；本次没有为它们确认完整训练及六榜结果。REAL-P2048-repaired 也仍是方案。

## 2. 已实际完成的公共训练与评测配置

以下配置在四个已运行版本的冻结协议中逐字段核对了 `assets`、`training`、`evaluation`，三部分完全相同。方法特有的观察预算与可训练因子另列。

| 项目 | 实际配置 |
| --- | --- |
| 模型 / 训练设备 | DeepSeek-R1-Distill-Qwen-1.5B；4 GPU，bf16；当前记录为4×L40S |
| 模块 | 196 个 attention / MLP linear；`target_modules=all-linear` |
| 正式 rank / alpha / scaling / dropout | 32 / 64 / 2 / 0 |
| 数据 | 原始 DAPO Math 转 boxed，去重后17,917行；同一 source/sampler/benchmark 指纹 |
| 数据顺序 | train shuffle=True；data seed42；固定并验证 sampler 前缀；actor shuffle=False |
| 训练目标 | DAPO-style GRPO；PPO epochs=1；正式 loss 名为 `token-mean`，具体归约见审计 |
| 每 trainer step | 64 prompts×8 responses；全局 PPO minibatch16；**4次逻辑 Adam 更新** |
| 长度 | prompt1024，左截断；response8192；rollout max_model_len9216 |
| 学习率 / scheduler | 1e-5；constant；warmup0；weight decay0 |
| Clip | low0.20，high0.28，C10 |
| 正则 | actor KL=False；KL reward beta0；entropy coefficient0；overlong shaping=False |
| 种子 | trainer/data/method/FSDP/rollout 均42 |
| 动态打包 | actor max tokens/GPU12288；rollout max_num_seqs256；batched tokens9216 |
| 保存 | 每10步完整 adapter snapshot；每50步及原计划终点完整 checkpoint；完整恢复点保留最近2份 |
| 评测集 | AIME24、AIME25、AMC23、HMMT Feb 各k32；MATH500、Minerva各k4；902题、7,248回答 |
| 评测采样 | seed42；temperature0.6；top-p0.95；max_new_tokens32768；max_model_len34816 |

当前环境的包版本和完整公共字段已登记在版本 JSON。资产 SHA256 的权威值为：

| 资产 | SHA256 |
| --- | --- |
| 基座 `model.safetensors` | `58858233513d76b8703e72eed6ce16807b523328188e13329257fb9594462945` |
| 本轮 boxed 数据 | `8e3c9314db8b83c61ab62a3dc85e0a704dcc5f3a9d404d236943f719095ce82f` |
| 本轮 seed42 sampler 前缀 | `9c926a9896ac0f766a9acc640f26252bc9a6fcf8857d9d3511941003b5d97fda` |
| benchmark snapshot | `3416ba93286b324a9663777fa472d0b568af5446319f27f876a63b59b3e863da` |

不要用当前通用协议的默认数据指纹覆盖这些历史冻结协议。本地 parquet 的文件字节与最初 canonical parquet 不同，既有审计验证了数据行和 schema 的语义对应，并用 `local_asset_binding` 显式绑定本轮字节与 sampler；这里保留这种区别。

## 3. 四个版本的精确配置和初始化

| 版本 | 冻结协议 | 正式 experiment 名 |
| --- | --- | --- |
| 当前 GradTop | [gradtop_r32_clean_20261015.json](../../configs/experiments/gradtop_r32_clean_20261015.json) | `rlvr_v1_gradtop_r32_a64_lr1em5_b64m16n8_seed42_20261015` |
| REAL64 | [real64_r32_frozen_20261013.json](../../configs/experiments/real64_r32_frozen_20261013.json) | `rlvr_v1_real_r32_a64_lr1em5_b64m16n8_seed42` |
| REAL32 | [real32_r32_frozen_20261014.json](../../configs/experiments/real32_r32_frozen_20261014.json) | `rlvr_v1_real_probe32_r32_a64_lr1em5_b64m16n8_seed42` |
| REAL-P256 | [real_p256_r32_longrun270_20261017.json](../../configs/experiments/real_p256_r32_longrun270_20261017.json) | `rlvr_v1_real_p256_r32_a64_lr1em5_b64m16n8_seed42_20261017` |

两份旧 REAL 配置从原运行目录逐字节复制，保留原始字段及父协议绑定。REAL-P256 文件名保留最初270步计划；实际停止于210以及改为评测210/200/150的后续决定，单列于[执行计划快照](evidence/real_gradtop_20261018/sources/runs/ops/real_p256_stop210_eval_20261018/plan.json)，没有倒改原始预登记。

所有 probe 都保持初始 policy 不更新；采用 `A_probe=0, B_probe=Ψ` 的零函数 adapter，读取 `grad(A).T/scaling=G.T Ψ`。Probe 结束后重新构建正式 adapter、optimizer 和 dataloader，从正式 seed42 数据起点训练；probe 消耗不计入正式 trainer step。

**GradTop** 对完整逻辑 minibatch 使用显式全局有效 token 分母，跨 DP 固定分母并补偿 FSDP 平均；每个 observation 收集后刷新随机输出投影，不裁剪、不调用 optimizer。CPU 保留原幅值 `UΣ`，每个 rollout step 拼接4份 sketch 后压缩到64，最终导出 top32 正交输入方向。正式 `A=Q.T, B=0`，A/B均可训练，36,929,536参数，`use_orig_params=False`。当前产物196层均无需坐标补齐；累计 sketch energy retained 的模块均值71.4392%，不是95%保证。当前没有额外 prompt sign、token mask 或遗忘系数。

**REAL** 对完整K8题组重算二元 reward 的 population-std advantage，每个 response 先按自身有效长度归一化，再组内平均。仅接收至少2正2负、response非空且未重复的有效组。先累积整组梯度，再做未中心化逐题二阶统计；输入/输出 sketch 宽均128，CPU float64 PSD Nyström，relative spectral tolerance1e-6。正式 `A=P.T, B=0`、ν=1，固定A，仅训练B，20,643,840参数，`use_orig_params=True`。数值秩不足32时报错，不补造方向。

三个初始化文件 `rank_map.json`、`summary.json`、`subspaces.safetensors` 的 SHA256、大小及本地路径全部见[产物指纹](evidence/real_gradtop_20261018/artifact_fingerprints.json)；summary 和 rank map 随仓库归档，权重文件保留在本地 runtime。精确方法定义参见[实现对照](../gradtop_real_implementation_comparison_zh.md)。本地 REAL 使用本地 GRPO/r32/a64/LR1e-5/无KL设置，与论文主实验的模型、GSPO及rank/LR等配置不同，不能称为完整论文结果复现。

## 4. 主要结果和长跑变化

Avg. 是回答数加权正确率；Macro Avg@k 是六任务等权均值，二者分母不同。主表单位为%，所有完整精度数值、任务 Avg@k/Pass@k 和细粒度 GradTop110–140 都在[19个完整评测点](evidence/real_gradtop_20261018/results.md)。

| 方法 | 正式 step | Avg. | Macro Avg@k | 答案解析率 | 使用说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| LoRA | 50 / 100 / 150 | 44.4674 / 44.5088 / 44.9227 | 38.7224 / 38.3390 / 38.8141 | 88.5072 / 91.6391 / 91.4459 | 同协议，连续运行1–150 |
| REAL64 | 50 | 46.1507 | 40.0388 | 92.3703 | 4,096题probe；只完成50端点 |
| REAL32 | 50 / 100 / 150 | 46.4266 / 47.8339 / 45.5022 | 40.3361 / 42.0021 / 39.4078 | 92.1772 / 96.3162 / 91.9840 | 51/101恢复首轮问题影响100/150轨迹 |
| GradTop | 50 | 47.2130 | 41.3282 | 97.8201 | 本轮预先指定的首个主比较端点 |
| GradTop | 100 | 47.4338 | 41.3425 | 97.1302 | 受51恢复首轮问题影响 |
| GradTop | 150 | 48.3582 | 42.3971 | 99.0342 | 已评测点中最高；受51/101历史影响 |
| GradTop | 200 | 47.9857 | 42.0933 | 98.6755 | 151首轮实际GPU同步验收通过 |
| GradTop | 250 | 47.5855 | 41.6034 | 96.9647 | 201首轮重新验收通过 |
| GradTop | 270 | 47.6545 | 41.6845 | 95.5022 | 251首轮重新验收通过 |
| REAL-P256 | 210 | 41.1010 | 35.8401 | 82.0778 | 新probe，连续0→210；仅210完整评测，不作同step因果对照 |

当前 GradTop150 相对同step LoRA 的 Macro 为 +3.5830pp，题目配对95%CI为[+2.2411,+4.9986]pp；相对旧 REAL32-150 为 +2.9892pp，[+1.6527,+4.4124]pp。GradTop100 对 REAL32-100 为 −0.6596pp，[−2.1011,+0.8246]pp；step50 对 REAL32 为 +0.9920pp，[−0.1483,+2.1325]pp。不能把150的差异写成每个阶段都优胜。

GradTop270 相对150为 −0.7125pp，[−1.7373,+0.2848]pp；相对250为 +0.0811pp，[−1.0473,+1.2553]pp，后段未显示明确提升。解析率从150的99.0342%降至270的95.5022%。旧 REAL32 从100到150的 Macro下降2.5943pp，并伴随解析率下降4.3322pp；恢复、格式、策略变化均混杂，尚未确定单一原因。

REAL-P256-210 的 hit-max 为10.7340%，平均输出10,656.21 tokens；当前 GradTop270分别为1.0624%和4,760.94 tokens。210端点显示较长输出和较低解析率；由于本轨迹尚无早期完整六榜，不能据此证明相对其早期发生退化。已有配置及adapter核对确认210权重导出一致、全部392张量有限、A等于probe且B非零，200 snapshot与完整checkpoint逐张量一致，评测manifest的差异仅为实验身份和adapter路径。因此不能仅凭评测变慢认定加载错误；这些检查也没有确定终点较弱的原因。200评测中断、150队列取消，不能推断问题从哪一步开始。

15:51完成的[早期checkpoint CPU审计](evidence/real_gradtop_20261018/sources/runs/ops/real_p256_early_checkpoint_audit_20261018/audit.md)逐个读取step0及10–210的全部22个snapshot：392张量齐全有限，A始终等于probe，step0增量为0、其后B非零，有效权重增量范数平滑增长。未发现初始化覆盖、A意外训练或数值爆炸。前50步训练reward、解析和长度控制与当前GradTop接近，不能用训练指标替代六榜；87个有效probe组的各层数值秩充足，也不能证明其泛化估计充分。该审计只提出后续先评50、再按结果评10或100的顺序，没有启动或排队。

上述区间使用单条训练轨迹上的题目配对bootstrap，不能覆盖训练seed波动；GradTop与旧REAL32还包含已知恢复偏差。

## 5. Probe成本与已完成的机制诊断

| 方法 | 筛选前probe prompts | Probe墙钟 | 解释 |
| --- | ---: | ---: | --- |
| 当前 GradTop | 256 | 2,250.33秒，37.51分钟 | width8/capacity64、16 observations |
| REAL32 | 2,048 | 无独立采集计时 | 从REAL64的step032统计前缀导出 |
| REAL64 | 4,096 | 40,085.41秒，11.135小时 | 预算为GradTop的16倍 |
| REAL-P256 | 256 | 2,520.49秒，42.01分钟 | 新采样，接收87组；同prompt预算，仍非同backward计算量 |

旧 REAL64/GradTop 的17.81倍墙钟差包含16倍prompt预算差异。新 P256 与 GradTop 的单次墙钟比约1.12；这也不是跨重复的速度定律。不同方法的accepted groups、投影宽度、反传调度及正式可训练参数都不同；同prompt数不等于同GPU计算量或同估计质量。

10月17日完成的固定策略诊断覆盖 GradTop100/150/270、旧REAL100/150，64题×2组K8/策略，共5,120回答，预选6个模块；20份engine检查及20份HF检查通过。完整[诊断表](evidence/real_gradtop_20261018/sources/runs/ops/gradtop_real_gradient_diagnostics_20261017/results.md)、[跨阶段统计](evidence/real_gradtop_20261018/sources/runs/ops/gradtop_real_gradient_diagnostics_20261017/trajectory.json)已归档。

| 冻结policy | 正式归约下GradTop初始化空间共享能量保留率 | REAL初始化空间共享能量保留率 |
| --- | ---: | ---: |
| GradTop100 | 41.6873% | 47.2466% |
| REAL100 | 43.0857% | 49.5672% |
| GradTop150 | 43.4537% | 48.8018% |
| REAL150 | 39.2990% | 45.1432% |
| GradTop270 | 45.1482% | 51.6421% |

在同policy/cache/loss内，REAL初始化空间的逐题及共享能量覆盖都较高；这不支持“GradTop共享覆盖更优”的原先简单猜想。两份初始化的probe预算不同，不能单独归因于统计目标。REAL100→150在两个初始化空间中的覆盖都下降，支持这条策略轨迹的梯度分布发生变化；GradTop150→270覆盖上升而六榜无同步收益，说明覆盖率不能直接当性能代理。同流width32诊断中capacity64增量SVD与一次性SVD的共享覆盖差最大约0.0444pp；这限定于本诊断流，不证明width8正式probe无误差或分解器带来准确率收益。

## 6. 复查、预览与归档边界

先在训练环境中激活依赖，并准备与冻结协议指纹一致的基座、数据和benchmark。下面只生成独立预览contract，不启动训练；`EXP_NAME`与现有运行分开，避免改写其contract。

```bash
# 使用已完成的GradTop产物预览正式训练配置
PYTHON_BIN=python METHOD=gradtop LORA_RANK=32 LR=1e-5 TARGET_STEP=50 \
  DRY_RUN=1 EXP_NAME=archive_preview_gradtop_20261018 \
  PROTOCOL_FILE=configs/experiments/gradtop_r32_clean_20261015.json \
  GRADTOP_ARTIFACT_DIR=runs/analysis/gradtop_r32_o16_w8_c64_seed42_20261015 \
  bash scripts/local/run_rlvr_protocol_1p5b_4gpu.sh

# 使用已完成的REAL-P256产物预览；目标210来自后续停止决定
PYTHON_BIN=python METHOD=real LORA_RANK=32 LR=1e-5 TARGET_STEP=210 \
  DRY_RUN=1 EXP_NAME=archive_preview_real_p256_20261018 \
  PROTOCOL_FILE=configs/experiments/real_p256_r32_longrun270_20261017.json \
  REAL_ARTIFACT_DIR=runs/analysis/real_probe_r32_s4_p0_seed42_20261017 \
  bash scripts/local/run_rlvr_protocol_1p5b_4gpu.sh
```

重新probe当前GradTop应显式设置 `METHOD=gradtop` 使用 `run_gradtop_candidates_4gpu.sh`；该脚本默认是 `gradtop_prompt_sign`。重新probe REAL-P256应显式设 `REAL_PROBE_RANK=32 REAL_PROBE_STEPS=4 REAL_PROBE_PROMPTS=0`，默认REAL probe步数是64。旧REAL32的精确初始化依赖已保存的step032统计和随机投影，不能只用64步最终A“截取”出来。

正式续训必须使用完整model/optimizer/extra/RNG/scheduler/dataloader恢复点及真实同步验收。REAL-P256的step210只保存评测adapter，`resumable=false`，最近完整点为200；不能把210标为完整训练恢复点。

本次[证据归档](evidence/real_gradtop_20261018/README.md)让GitHub直接读取结果、配置和审计，不依赖被忽略的本机 `runs/` 链接。原始运行plan的逐文件SHA描述当时实际执行的代码，发布分支另外包含后来远端初始化修复的集成；不要将发布时的最新源码自动等同于所有历史执行源码。
