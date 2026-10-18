# 当前 GradTop-r32：从 step150 向前补齐每10步评测

登记时间：2026-10-16（Asia/Shanghai）。用户要求利用剩余卡时，对当前新跑的 GradTop 做细粒度 checkpoint 评测，从150步往前；用户给出的卡时截止约为当天17:00。

15:26更新：140/130/120/110四个新增点评测均已完成。按用户新指令，在110完成后停止队列并交接GPU给[150→200续训](gradtop_r32_resume200_syncfix_20261016_zh.md)；用户随后明确允许该续训和200评测超过17:00。下方保留细粒度评测原预算，90及以下尚未执行。

唯一实验：`rlvr_v1_gradtop_r32_a64_lr1em5_b64m16n8_seed42_20261015`。本任务独立加载已保存的 adapter 做推理评测，不启动训练。有关训练恢复首轮 LoRA 同步的审计继续保留在[训练审计记录](../../runs/ops/gradtop_r32_continue_review_20261016/training_audit.md)；本次评测描述既有轨迹，不构成修复后的干净重跑。

## 顺序与时间预算

150、100、50、10已经完成同协议评测，核对原adapter/config哈希、records与manifest后复用。每10步保存的其他快照依次评测：

**140 → 130 → 120 → 110 → 90 → 80 → 70 → 60 → 40 → 30 → 20。**

以最近端点约1～1.2小时/点估算，当天剩余时间优先完成140/130/120/110，再争取90；实际数量以通过完整验证的结果为准。15:45后不启动下一个checkpoint，为最后一轮预留75分钟。17:00后若仍在生成，控制器在下一次评测心跳检查时停止本任务拥有的评测进程，保留已经产生的文件，未完整验证的点不计入结果表。心跳约60秒，进程退出清理会另占短暂时间。

边界控制保存在[control.json](../../runs/ops/gradtop_r32_fine_eval_20261016/control.json)，`stop_after_current`可控制当前点完成后停止；`do_not_start_after`和`finish_by`包含明确的北京时间时区。未评完的点留在status的`pending_steps_descending`中。

## 协议、快照与资源

| 项目 | 设置 |
| --- | --- |
| 模型与adapter | 原DeepSeek-R1-Distill-Qwen-1.5B；当前GradTop rank32 / alpha64 / 196 modules / 392 tensors |
| 评测题目 | 与已有150/100/50/10相同的902题快照 |
| 采样数 | AIME24、AIME25、AMC23、HMMT Feb各32次；MATH500、Minerva各4次；共7248回答 |
| 解码 | seed42、temperature0.6、top-p0.95、max_new_tokens32768、max_model_len34816 |
| 执行 | 4×L40S，每卡一个shard，每shard1812回答；每次只评一个checkpoint |
| 导出 | 独立评测adapter目录；权重链接至对应原snapshot，复制step150标准PEFT配置；不复制147.8MB权重、不改原快照 |
| 一致性 | 固定snapshot/config哈希，校验step及4×step更新计数、rank32、196模块和392张量；step150 snapshot与完整adapter逐张量相等 |
| 验收 | 四shard样本键/数量/配置检查，整体records与summary复核，完成后才更新结果表 |
| 资源保护 | 持有原共享锁及GradTop任务锁；GPU须空闲；每点开始前至少2GiB磁盘空闲，空间不足则记录停止及剩余点 |

只固定实际评测调用文件和评分依赖的哈希，不要求另一窗口审计中的训练源码保持不变。每个新点的正式输出使用其唯一step名称；已有输出会拒绝覆盖。当前控制器不自动续接不完整生成；如发生中断，应先审查残留结果与所属进程，再另行恢复。

## 日志和结果入口

- [队列status](../../runs/ops/gradtop_r32_fine_eval_20261016/status.json)：当前step、已完成点、待完成点、截止/资源停止原因。
- [控制器日志](../../runs/ops/gradtop_r32_fine_eval_20261016/controller.log)：各点启动、验证与退出记录。
- [结果表](../../runs/ops/gradtop_r32_fine_eval_20261016/results.md)及[JSON](../../runs/ops/gradtop_r32_fine_eval_20261016/results.json)：每完成一点评测即更新，主表沿用用户指定的Method/Step/Avg./六榜Avg@k与Pass@k格式，另表给Macro及行为指标。
- [冻结plan](../../runs/ops/gradtop_r32_fine_eval_20261016/plan.json)、[CPU预检](../../runs/ops/gradtop_r32_fine_eval_20261016/preflight.json)、[控制器](../../runs/ops/gradtop_r32_fine_eval_20261016/controller.py)。
- [step140推理日志目录](../../runs/outputs/full_bench_eval/rlvr_v1_gradtop_r32_a64_lr1em5_b64m16n8_seed42_20261015_step140_fullbench_32768_seed42/logs)。其余点使用相同路径命名规则替换step数。

结果反映同一个训练seed下的轨迹。细粒度扫描能够定位端点波动和峰值区间；从多点中选出的最好分数应标为探索性选择，不能直接替代预先指定step的对照结论。
