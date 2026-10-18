# 当前 GradTop-r32：验证首轮同步后从150续到200

登记：2026-10-16，Asia/Shanghai。用户明确授权当前评测结束后继续训练，并确认允许超过17:00，执行150→200及step200完整六榜评测。step110评测于15:26完成，细粒度队列按要求停止；新增完成点为140/130/120/110，90及以下未启动。

完成更新：本段于20:54完成step200训练，21:51完成六榜评测，21:52完成独立结果汇总验收。Step200 Avg.47.9857%、Macro Avg@k42.0933%；200−150的Macro差值为−0.3038个百分点，题目配对95%区间[-1.3862,+0.7508]个百分点。此后用户要求继续，最新执行另见[200→250续训记录](gradtop_r32_resume250_20261016_zh.md)。下文启动时刻及预检预算保留为当时记录。

当前唯一experiment仍为 `rlvr_v1_gradtop_r32_a64_lr1em5_b64m16n8_seed42_20261015`。本次续训在15:35启动，原step150完整checkpoint是恢复起点。训练配置和原probe保持一致；本次显式登记恢复同步修复及其运行时验收这一代码边界。已有step51/101错误采样历史仍存在，因此这是带已知历史的探索性接续，不是修复后从初始化开始的干净重跑。

截至15:43，首个续训step151已完成，用时380.36秒，全部已记录数值有限，LR仍为1e-5；答案解析率92.97%、命中8192长度上限13.67%、梯度范数0.07651。首次实际权重和请求验收已通过，首步也未重现原恢复边界的解析率骤降。单批训练指标不作为泛化改善结论；200六榜评测仍在等待训练完成。

## 本次执行

| 项目 | 设置 |
| --- | --- |
| 模型 / 设备 | DeepSeek-R1-Distill-Qwen-1.5B，4×L40S |
| Adapter | rank32 / alpha64，196模块，A/B均训练 |
| 学习率 | 1e-5，constant，warmup0 |
| Probe | 原16 observations、width8/capacity64增量SVD artifact，原文件SHA不变 |
| 训练采样 | 64 prompts/step × 8 responses；minibatch16；每step4次逻辑Adam更新 |
| 恢复 | 完整model、Adam、scheduler、四rank RNG、dataloader；150→200 |
| 累计预算 | 600→800次逻辑Adam更新；9,600→12,800训练prompts |
| 保存 | 每10步adapter snapshot，200保存完整checkpoint，保留最近2个完整actor |
| 评测 | 到200后运行原六榜902题/7248回答，seed42、temperature0.6、top-p0.95、max_new_tokens32768 |
| 停止边界 | 200训练及评测后停止；本次用户已取消17:00对这段续训的限制 |

原150起点已重新核验：四rank各392个Adam state，step均600、LR1e-5，scheduler/RNG齐全，dataloader150批/9600题。全部22个checkpoint文件的SHA已固定。Adapter SHA为 `1e7c3d825fd1775d101468d60fcbd917894d664a9890988936740b789a4c8051`。

## 恢复首轮的强制验收

本地先前同步修复使首次rollout先加载base、再加载当前adapter，并在训练请求缺adapter时直接抛错。本次再增加默认关闭、仅在指定恢复任务启用的验收，避免只根据源码或注册ID推断加载正确。

1. **Actor→checkpoint：** 首轮导出的全部392个A/B tensor与独立指定的step150 safetensors逐一比对规范名称、形状、有限值和数值；显式处理snapshot到actor dtype的转换，并要求训练后的B非零。
2. **vLLM注册及实际GPU槽位：** `add_lora`后读取注册的adapter和当前active device slot，检查全部A/B、QKV与gate/up packing、B的alpha/r缩放以及整个slot的padding；逐元素精确比较，覆盖392个源tensor。当前验收只支持实际使用的非量化TP=PP=1；未知布局直接停止。
3. **首次请求：** 每个server在调用`engine.generate`前核对四rank的ready回执，检查本次token、源SHA、共同payload指纹及正确adapter ID，然后记录首次request ID。验收缺失或不一致会在生成前抛错，不等第一次optimizer更新后再判断。
4. **控制器：** 独立复核actor、engine、ready、request四类回执共16份，绑定本次token、源checkpoint路径及固定SHA；若观察到step151更新却没有完整验收记录，停止本任务。

六项 `VERL_LORA_RESUME_VERIFY_*` 设置显式进入Ray `runtime_env.env_vars`。启动日志已确认实际注入；44个源码/协议文件哈希被冻结。新验收及相关同步回归CPU检查为67 passed。

**15:37的实际GPU首轮验收已通过。** 四rank的actor、vLLM注册adapter和CUDA活动槽位均通过全部392个tensor的核对，16/16份回执齐全。独立从源snapshot构造BF16权重并将B乘以alpha/r=2，所得摘要与四rank的注册权重及CUDA回读均一致。全部ready回执最迟在15:37:15.935完成，首次生成请求在15:37:16.873通过检查；四replica均使用adapter ID 123，引擎已加载ID列表为`[123]`。控制器已写入实际验收记录，`sync_verification=verified`。

首次异步并发burst产生19条成功的请求验收日志，各replica回执文件保留该burst最后一条，最早请求及时间保存在训练日志中；所有请求验收均晚于全部ready回执，没有提前放行。

这项验收核对恢复源、导出payload、实际安装/激活的A/B及请求adapter ID，没有进行两独立引擎的logits逐项对照。

## 存储与溯源

新版launcher要求runtime输入溯源。此次根据已冻结旧contract、数据/模型/原probe内容验证原150与目标200的identity一致后，登记legacy迁移manifest；`trajectory_origin`明确保留修复前历史。原plan/status/contract和源代码差异证据保存在本次独立ops目录，不改写早期实验已完成记录。

100/150完整checkpoint的22个文件逐一名称和大小一致。由此将本次checkpoint空间裕量从10%收窄到1%，snapshot裕量仍10%，另保留2GiB余量；这是资源预算调整，未改变训练设置。启动所需空闲约2.966GiB，实测约3.045GiB。

50/100/150的adapter已用hardlink无损保留到本次ops。200保存时，原retention先移除50/100的旧actor，保留完整150；被保留adapter不计入可回收空间。保存完成后恢复原50/100评测adapter路径，已有评测仍可复核。本次不删除其他实验数据。

2026-10-16 20:26存储更新：用户追加授权清理旧合并权重及已评测中间checkpoint。已删除两份旧SPAR合并模型，以及GradTop/REAL probe32各step100的model、optimizer和extra state，共释放20.57GiB；可用空间从2.39GiB增至22.97GiB。两个step100的adapter及评测证据保留，现不再支持精确续训；两组完整step150和全部snapshot均保留。当前训练已到195，controller和自动汇总进程正常，44项冻结代码及8点评测证据复核通过。逐文件记录见[磁盘清理记录](../../runs/ops/disk_cleanup_20261016_2025/README.md)。上述启动空间预算保留为当时记录，step100训练状态已提前清理，不能再次计入后续可回收空间。

## 当前入口

- [控制器状态](../../runs/ops/gradtop_r32_resume200_syncfix_20261016/status.json)
- [训练日志](../../runs/ops/gradtop_r32_resume200_syncfix_20261016/train_to200.log)
- [首轮实际GPU验收](../../runs/ops/gradtop_r32_resume200_syncfix_20261016/first_rollout_verification.json)
- [独立首轮审计](../../runs/ops/gradtop_r32_resume200_syncfix_20261016/independent_first_rollout_audit.md)
- [step151首步观测](../../runs/ops/gradtop_r32_resume200_syncfix_20261016/first_training_step_observation.json)
- [控制器日志](../../runs/ops/gradtop_r32_resume200_syncfix_20261016/controller.log)
- [冻结plan](../../runs/ops/gradtop_r32_resume200_syncfix_20261016/plan.json)
- [源step150完整审计](../../runs/ops/gradtop_r32_resume200_syncfix_20261016/source_step150_verification.json)
- [实际runtime迁移记录](../../runs/ops/gradtop_r32_resume200_syncfix_20261016/runtime_migration.json)
- [存储与溯源审计](../../runs/ops/gradtop_r32_resume200_syncfix_20261016/storage_and_provenance_audit.md)
- [已有细粒度结果表](../../runs/ops/gradtop_r32_fine_eval_20261016/results.md)
- [当前实验汇总表，200完成后自动补入](../../runs/ops/gradtop_r32_resume200_syncfix_20261016/results.md)
- [自动汇总进程状态](../../runs/ops/gradtop_r32_resume200_syncfix_20261016/reporter_status.json)
- [汇总程序及输入登记](../../runs/ops/gradtop_r32_resume200_syncfix_20261016/reporter_registration.json)

首轮完整验收已通过，`first_rollout_verification.json`已生成。到200后控制器将写入 `terminal_checkpoint_verification.json`，并运行独立六榜评测。实际结果未产生前不填入预测分数。

独立CPU汇总进程已于15:42启动，绑定本次controller的PID及进程起始时间；当前汇总表已重新核验并列出10/50/100/110/120/130/140/150共8点。200评测完整验收后，将重新核验四rank Adam各800次更新、首轮同步回执和9点评测，按既定表格格式补入200，并写出200−150的题目配对bootstrap比较（10,000次，seed42）。程序只读取本次实验，既有细粒度输出不改写，报错或控制器异常退出不记作完成。配对区间反映题目重采样不确定性，不覆盖训练seed波动。
