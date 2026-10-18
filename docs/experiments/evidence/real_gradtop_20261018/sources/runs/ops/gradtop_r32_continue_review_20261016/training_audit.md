# 当前 GradTop rank32：step1–150 训练与恢复审计

结论：step150 完整结束，四个 rank 的 Adam step 均为 600；没有非有限数值、跳过更新、OOM 或 traceback 的日志证据。暂不建议原样继续到 step200。恢复后第一次 rollout 未同步 checkpoint LoRA 的代码路径需要先修复和验证，且最近训练奖励没有持续上升的信号。

本审计仅分析本轮 GradTop。只读取日志、代码、状态和 CPU checkpoint；没有启动 GPU、修改源码、改写控制器状态或恢复训练。完整逐步指标见 `training_steps_1_150.csv`，机器可读窗口与审计证据见 `training_audit.json`。

## 固定窗口

| step窗口 | reward/acc | parse成功 | 平均response tokens | 命中8192上限 | entropy | grad norm | PPO clip比例 | sampled PPO KL | 秒/step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 31-50 | 47.90% | 87.47% | 5341.0 | 17.73% | 0.9147 | 0.07291 | 0.00492% | 0.00002394 | 390.8 |
| 81-100 | 50.56% | 89.15% | 5449.7 | 16.37% | 0.8505 | 0.06717 | 0.00491% | 0.00001078 | 396.6 |
| 101-120 | 51.06% | 91.37% | 5388.0 | 13.19% | 0.8408 | 0.08114 | 0.00477% | -0.00002816 | 389.3 |
| 121-140 | 50.44% | 94.53% | 5307.0 | 9.99% | 0.8197 | 0.07032 | 0.00442% | 0.00001026 | 381.8 |
| 131-150 | 50.01% | 92.05% | 5482.7 | 13.18% | 0.7860 | 0.06897 | 0.00457% | 0.00001083 | 394.3 |

81–100 到 131–150 的平均 reward 为 50.56%→50.01%，parse 为 89.15%→92.05%，命中长度上限为 16.37%→13.18%；但平均长度 5449.7→5482.7 tokens，没有持续缩短。Entropy 0.8505→0.7860，训练表现为格式改善后正确率进入平台。不同窗口使用不同 prompt，不能据此计算泛化显著性或声称过拟合已证实。101–120 包含恢复异常步101，JSON另列102–120作为敏感性检查。

所有150步的 LR 恒为 1e-5；aborted_ratio=0、reference kl_loss=0。pg_clipfrac_lower 仅step110和135非零（最大5.27e-7），其余148步为0。reward 与 accuracy 逐步完全相同。长度 clip 是达到8192的比例，PPO clip 是另一个指标；PPO KL 是可能为负的采样估计。日志中的 active_parameter_count=9,232,384 是单个 FSDP shard 数量，四卡总计36,929,536，不表示模型突然冻结。

## 恢复首轮生成策略问题

| step | reward | parse | response tokens | 命中8192上限 | entropy | grad norm |
|---|---:|---:|---:|---:|---:|---:|
| 50 | 45.90% | 89.26% | 5363.5 | 16.02% | 0.9104 | 0.07163 |
| 51 | 43.55% | 58.40% | 5869.7 | 42.97% | 0.9188 | 0.18011 |
| 52 | 44.53% | 91.21% | 5256.5 | 13.67% | 0.9313 | 0.07061 |
| 100 | 54.49% | 90.23% | 5384.3 | 14.06% | 0.7725 | 0.06947 |
| 101 | 38.09% | 55.27% | 6379.6 | 50.98% | 0.9101 | 0.26432 |
| 102 | 58.20% | 90.04% | 5369.3 | 16.41% | 0.7965 | 0.07655 |

代码与实际配置共同支持明确的控制流结论：使用 dummy 加载的新进程在恢复后第一次生成时仅加载 base 权重，第二次生成才加载恢复的 LoRA。两次恢复首步51/101的异常是与这一机制一致的观测，但尚未进行引擎权重 hash 或张量的运行时复现。续训计划固定了 fsdp_workers、ray_trainer、agent_loop 的源码哈希；部分下游fsdp_utils/vLLM文件不在原计划的哈希清单中，本审计记录了它们当前的SHA256和mtime。因此结论严格表述为当前实现与运行配置的代码推导，并以实际边界异常为旁证。

1. 实际配置为 peft_type=grad_subspace、lora_rank=32、lora_adapter_path=None、rollout.load_format=dummy、val_before_train=False；grad_subspace 明确属于 _is_lora。
2. `ray_trainer.py:1792` 在训练前恢复 actor checkpoint；`fsdp_workers.py:1728` 的 load_checkpoint 只恢复 actor 并 offload，没有同步 rollout。
3. `fsdp_workers.py:1083` 将新进程的 base_sync_done 设为 False。AgentLoopManager 初始化仅启动并 sleep，没有预先 wake。
4. 首次 `generate_sequences` 经 AgentLoopManager→Replica→vLLM server→worker.wake_up 调用 rollout_mode 一次。`collect_lora_params(..., base_sync_done=False)` 的 `fsdp_utils.py:704` 分支过滤全部 lora_ 权重，导出未合并 adapter 的 base。
5. `vllm_rollout.py:240` 仅在 peft_config 且 base_sync_done=True 时 add_lora；false 时只 model.load_weights。`vllm_async_server.py:509` 在 list_loras 无 adapter ID 时传 lora_request=None。
6. `fsdp_workers.py:1199` 在这次更新后才把 base_sync_done 设为 True，下一次 wake 才导出/安装 LoRA。首轮 old_log_probs 则由已恢复的 actor 重算，因此生成策略与 proximal actor 存在不一致。

该问题不否定 checkpoint 的模型/优化器恢复，也不直接否定离线加载导出 adapter 的评估结果。两次异常更新对后续轨迹的累计影响尚未量化。下一次恢复前应验证第一轮生成已同时具备正确 base 与 checkpoint LoRA；要量化修复对最终成绩的影响，需要另外记录干净的训练轨迹。审计没有修改此路径。

## 完整性、异常和成本

- 三段日志各50步，合计1–150全部覆盖，无重复或缺失；training/global_step 与 step 一致。所有解析出的数值均有限。
- step150 checkpoint：四个 rank 各392个 Adam state，step集合均为{600}、LR=1e-5；scheduler last_epoch=150；四rank CPU/NumPy/Python/CUDA RNG齐全。Dataloader为150批、9600条prompt，marker文本为150。
- Adapter共392个tensor，SHA256：`1e7c3d825fd1775d101468d60fcbd917894d664a9890988936740b789a4c8051`。完整文件大小和校验记录保存在JSON。
- 控制器status=complete、auto_resume=false，完成eval10/50/100/150；train_to150退出码0。latest_logged_step=149是30秒心跳留下的陈旧字段：只在child仍运行时刷新，完成路径没有重新写它。真实最终日志与checkpoint均为150。
- 实际存在磁盘告警：initial/to100/to150日志分别出现1/1/1953次“over 95% full”。未见实际spill失败或训练异常退出；后续运行前需重新核算真实空间与checkpoint预算。不能概括成“没有警告”。
- 其余主要为启动或依赖弃用警告；未匹配到非有限梯度/损失、跳过优化器更新、CUDA错误、OOM和Python traceback。日志未出现不代表内部每项状态都被监测。
- 按末20步平均394.3秒/步，再50步仅训练约5.48小时，四卡约21.9 GPU小时，另加启动/保存/评估。

目前适合先修复并验证恢复首轮同步，结合独立全量评估审计判断收益。仅凭这些训练日志，不足以支持继续投入50步会提高泛化准确率。如果评估存在可信的后段收益且step200能回答明确的平台问题，再做有边界的续训；否则rank32方法对照或种子重复的优先级更高。

## 输入日志

- `/root/peft-for-rl/runs/ops/gradtop_r32_clean_20261015/train.log`：50步，SHA256 `e7b4e82ea7a41a0390eb65b4250760438cc91fc76f5f442737c1d0a2dfa47dba`。
- `/root/peft-for-rl/runs/ops/gradtop_r32_continue_step150_20261015/train_to100.log`：50步，SHA256 `6b7773bc85cbb65f59860cf0309db2a572ee28f796f0be609ad26b25d0a0db5a`。
- `/root/peft-for-rl/runs/ops/gradtop_r32_continue_step150_20261015/train_to150.log`：50步，SHA256 `f1fa7f975fbaf195de312c8ca45b61d22a0d7d2b7dcabe7775a719c74875f1ad`。
