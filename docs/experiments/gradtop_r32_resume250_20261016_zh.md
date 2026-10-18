# 当前GradTop-r32：从200续到250并重新核验adapter同步

登记于2026-10-16，Asia/Shanghai。用户在查看step200结果后要求继续续训，并再次强调检查adapter同步。本段沿用此前每次增加50步、终点完整评测的执行方式，从完整step200恢复到250。唯一实验为`rlvr_v1_gradtop_r32_a64_lr1em5_b64m16n8_seed42_20261015`，不创建新的方法版本或probe。

2026-10-17更新：用户将最终终点定为270。250控制器及其冻结配置保持原样，先完成本段训练和评测；独立后继队列再绑定实际250源文件并重新验收同步，续到270评测后结束。详见[270执行安排及REAL预算研究设计](gradtop_real_longrun270_probe_budget_20261017_zh.md)。

上一段200训练和六榜评测均已完成：Avg.47.9857%，Macro Avg@k42.0933%；相对150分别下降0.3725和0.3038个百分点，Macro题目配对95%区间为[-1.3862,+0.7508]个百分点。此次按用户要求继续观察长跑；250完成前不填入预测结果。

## 执行设置

| 项目 | 设置 |
| --- | --- |
| 模型及设备 | DeepSeek-R1-Distill-Qwen-1.5B，4×L40S |
| 方法 | 当前GradTop，rank32/alpha64，196个模块，A/B均训练 |
| Probe | 原16 observations、width8/capacity64增量SVD artifact，三个文件哈希不变 |
| 训练 | LR1e-5 constant；64 prompts×8 responses；minibatch16；seed42 |
| 恢复 | 完整step200 model、Adam、scheduler、四rank RNG及dataloader |
| 本段预算 | 200→250；Adam累计800→1000；训练prompts累计12,800→16,000 |
| 保存 | 每10步adapter snapshot；250完整checkpoint；最近两个完整actor为200/250 |
| 评测 | 原六榜902题/7248回答，seed42、temperature0.6、top-p0.95、max_new_tokens32768 |
| 自动比较 | 250−200及250−150；题目配对bootstrap10000次、seed42 |
| 停止条件 | 250训练、评测和结果汇总完成后停止，不自动续到300 |

## 恢复与同步证据

源step200的22个恢复文件全部固定SHA256。四rank各392个Adam state，更新数均800，LR1e-5；scheduler/RNG和dataloader200批/12,800题验收通过。源adapter SHA256为`9c55282324b594f897d65a4787029cb82862c196743f316eec199da9e892901d`。

训练实现与上一段实际GPU同步验收通过的版本一致。本段单独登记controller和report程序，共冻结47个代码/协议文件。原runtime identity完全一致，launcher contract仅将target_step从200扩展为250。

首轮继续启用原来的强制验收：Actor导出的全部392个A/B张量对照源step200；vLLM加载后读取注册adapter和实际CUDA活动槽位，检查packing、B缩放和padding；所有rank ready后，首次生成请求必须携带正确adapter ID。新token、新回执目录和新的源权重SHA共同绑定本段，不能复用150→200的回执。任何不匹配会在生成前抛错；控制器另行拒绝没有完整验收回执的step201更新。

22:14已启动训练；**22:16首次实际GPU验收通过**，16/16回执齐全。四rank各392个tensor均对照step200源文件通过，独立从源文件计算BF16及B×2后的指纹，与所有rank的注册adapter及CUDA活动槽位回读一致。全部ready最迟22:16:01.868577，首次生成请求22:16:02.880969；四replica均携带adapter ID123，且已加载ID列表为`[123]`。日志同时确认四rank均恢复model、optimizer、rng及lr_scheduler。

截至22:17首个续训批次正在运行，尚无step201完成指标；控制器`sync_verification=verified`。运行时首次请求并发burst写入25条验收事件，均晚于全部ready；各replica回执保留该burst最后一条，最早请求时间取自训练日志。本次核验是实际权重及请求ID检查，没有进行独立引擎logits逐项对照。

## 存储与结果保留

预检可用约15.4GiB；新增一个完整checkpoint、五份snapshot均按10%增长裕量计，另留2GiB，需约10.7GiB。预算不预支旧checkpoint删除可回收的空间。

50/100/150/200的adapter在启动时以hardlink保留，并校验哈希。250终点retention会清理较旧的完整actor，保留完整200/250；控制器随后恢复旧评测adapter路径，确保现有9点评测能再次核验。150的adapter和评测保留，但完整150训练状态可能按原retention被清理。原始records、summary、verification和已完成运行的冻结记录保持原样。

原step51/101错误采样历史仍属于此轨迹。151及201分别按各自恢复源验收同步，不能追溯消除早期异常；这是同一当前实验的探索性长跑。题目配对区间不覆盖训练seed波动。

## 运行入口

- [控制器状态](../../runs/ops/gradtop_r32_resume250_syncverified_20261016/status.json)
- [控制器日志](../../runs/ops/gradtop_r32_resume250_syncverified_20261016/controller.log)
- [训练日志](../../runs/ops/gradtop_r32_resume250_syncverified_20261016/train_to250.log)
- [首轮实际GPU验收](../../runs/ops/gradtop_r32_resume250_syncverified_20261016/first_rollout_verification.json)
- [源权重摘要及请求时序复核](../../runs/ops/gradtop_r32_resume250_syncverified_20261016/first_rollout_audit.md)
- [冻结执行计划](../../runs/ops/gradtop_r32_resume250_syncverified_20261016/plan.json)
- [源step200恢复验收](../../runs/ops/gradtop_r32_resume250_syncverified_20261016/source_checkpoint200_verification.json)
- [预检及磁盘预算](../../runs/ops/gradtop_r32_resume250_syncverified_20261016/preflight.json)
- [结果表，250完成后自动补入](../../runs/ops/gradtop_r32_resume250_syncverified_20261016/results.md)

首轮`first_rollout_verification.json`已生成，终点将写入`terminal_checkpoint_verification.json`和`report_verification.json`。自动汇总由同一个控制器在评测完成后调用CPU报告程序，无需额外手动启动。

2026-10-17跨epoch核查：当前250以内不跨epoch。若后续到300，现有270步协议上限、launcher的total_epochs=1及单epoch数据游标验收必须先适配。真实数据为17,917行、batch64、drop_last=True，279批/epoch，step280进入第二epoch。生产sampler/loader从真实step200状态进行纯CPU回放，在total_epochs=2下模拟250→300及300→310的索引顺序均与不间断运行一致；当前训练和冻结配置未修改。详见[跨epoch审计](../../runs/ops/gradtop_epoch_boundary_audit_20261017/README.md)。
