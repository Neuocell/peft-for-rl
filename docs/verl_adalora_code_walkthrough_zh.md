# verl 训练链路与官方 AdaLoRA 接入说明

本文对应本仓库的经典 verl PPO/FSDP 训练路径，以及 `peft==0.19.1` 的官方 AdaLoRA 实现。仓库没有复制或改写 Hugging Face 的 AdaLoRA 算法；这里只实现 verl 与官方 PEFT API 之间缺失的训练、FSDP 和 rollout 桥接。

## 1. 一次 GRPO 训练迭代经过哪些代码

```text
examples/verl_train/*.sh
  -> python -m verl.trainer.main_ppo
  -> TaskRunner.run
  -> RayPPOTrainer.init_workers
  -> ActorRolloutRefWorker.init_model
  -> 构造 HF actor + PEFT adapter + FSDP + optimizer
  -> RayPPOTrainer.fit
       -> rollout.generate_sequences
       -> actor.compute_log_prob / ref.compute_log_prob
       -> reward + GRPO advantage
       -> ActorRolloutRefWorker.update_actor
       -> DataParallelPPOActor.update_policy
            -> mini-batch / micro-batch
            -> model forward 得到新 log_prob
            -> PPO/GRPO policy loss
            -> backward
            -> optimizer.step
            -> AdaLoraModel.update_and_allocate
       -> actor 权重同步到 vLLM rollout
       -> 按 save_freq 保存 checkpoint
```

### 1.1 启动与 Ray 编排

- `examples/verl_train/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh` 组装 Hydra 参数并启动 `verl.trainer.main_ppo`。
- `verl/trainer/main_ppo.py:main` 初始化 Ray；`TaskRunner.run` 根据 role 创建远程 worker，构造 `RayPPOTrainer`，再调用 `init_workers()` 和 `fit()`。
- `verl/trainer/ppo/ray_trainer.py:RayPPOTrainer.init_workers` 建立 actor/rollout/ref worker group 和资源池。

Ray driver 负责调度，真正持有模型、optimizer 和 GPU 状态的是 Ray worker。排查问题时不能只看 driver 日志，还要看 TaskRunner/worker 日志。

### 1.2 actor、rollout 和 ref 的职责

- actor：当前正在优化的策略，使用 FSDP 持有训练参数。
- rollout：通常是 vLLM，负责高速生成 response；它不是同一个 PyTorch module，所以每轮需要从 actor 同步权重。
- ref：冻结参考策略，仅在启用 KL 项时提供 reference log probability。

经典混合 worker 位于 `verl/workers/fsdp_workers.py:ActorRolloutRefWorker`。`init_model()` 调用 `_build_model_optimizer()` 构造 HF 模型、PEFT adapter、FSDP、optimizer 和 scheduler，然后创建 `DataParallelPPOActor`。

### 1.3 rollout 到 policy update

`verl/trainer/ppo/ray_trainer.py:RayPPOTrainer.fit` 是外层训练循环，关键顺序为：

1. `generate_sequences` 让 rollout 模型对一批 prompt 采样。
2. 计算 reward，并按 group 计算 GRPO advantage。
3. 获取 old/ref log probability。
4. `_update_actor` 调 Ray worker 的 `update_actor`。
5. worker 最终进入 `verl/workers/actor/dp_actor.py:DataParallelPPOActor.update_policy`。

`update_policy` 先按 `ppo_mini_batch_size` 切 mini-batch，再按显存预算切 micro-batch。每个 mini-batch 只执行一次 optimizer step；micro-batch 只是梯度累积。因此 AdaLoRA 的 step 单位必须是 optimizer step，而不是外层 trainer step，也不是 micro-batch forward 次数。

例如当前配置：

```text
train batch = 64
PPO mini-batch = 16
PPO epochs = 1
```

每个 trainer step 有 `64 / 16 = 4` 次 optimizer update，所以 270 个 trainer steps 对应 AdaLoRA `total_step=1080`。

## 2. 官方 AdaLoRA 的参数和调度

官方线性增量可写为：

```text
Delta W = (alpha / ranknum) * B * diag(E) * A
```

- `lora_A`：右奇异向量方向。
- `lora_B`：左奇异向量方向。
- `lora_E`：每个 rank triplet 的幅值参数，可学习，也会被 allocator 置零。
- `ranknum`：PEFT 用于保持缩放语义的秩计数。

这里的初始秩由 `AdaLoraConfig.init_r` 控制。继承自 `LoraConfig` 的 `r` 对 AdaLoRA 无效，官方源码也会在非默认 `r` 时给出 warning。因此本仓库的 `build_adalora_config()` 不向官方配置传 `r`。

官方 RankAllocator 对每个 A/E/B 参数计算：

```text
instant importance = abs(parameter * gradient)
sensitivity        = EMA(instant importance, beta1)
uncertainty        = EMA(abs(instant - sensitivity), beta2)
triplet score      = sensitivity * uncertainty，再聚合 A/E/B
```

然后在所有目标模块之间进行全局预算分配。这是标准 AdaLoRA 的行为，不是“每个模块独立按阈值决定 rank”。

调度分三段：

- `step <= tinit`：初始 rank warmup，不裁剪。
- `tinit < step <= total_step - tfinal`：按 cubic schedule 降低全局 rank budget，每隔 `deltaT` 真正更新一次 mask。
- `step > total_step - tfinal`：固定最终 rank pattern，继续微调。

所有这些 step 都是 optimizer-step 编号。

## 3. AdaLoRA 的主要接入点

| 接入点 | 文件 | 作用 |
| --- | --- | --- |
| 配置映射 | `verl/utils/peft_adalora.py:build_adalora_config` | 将 verl snake_case 字段映射到官方 `AdaLoraConfig`，不传无效的 `r` |
| 模型识别 | `verl/utils/peft_adalora.py:find_adalora_model` | 按真实 `AdaLoraModel` 类型查找，避免命中 PeftModel/FSDP 的属性代理 |
| adapter 构造 | `verl/workers/fsdp_workers.py:_build_model_optimizer` | `get_peft_model(base_model, official_config)` |
| 官方正交 loss | `verl/workers/actor/dp_actor.py:_forward_micro_batch` | 用全 ignore labels 让 HF 模型返回零 CE，同时触发官方 AdaLoRA forward 的正交项 |
| loss 合并 | `verl/workers/actor/dp_actor.py:update_policy` | 将官方 `output.loss` 加到 PPO policy loss，并记录正交指标 |
| rank allocation | `verl/workers/actor/dp_actor.py:update_policy` | 每次成功的 `optimizer.step()` 后调用官方 `update_and_allocate()` |
| FSDP 完整参数 | `verl/utils/peft_adalora.py:update_and_allocate` | 在 `summon_full_params(writeback=True, with_grads=True)` 内执行 allocator |
| rollout 配置 | `verl/utils/peft_adalora.py:adalora_config_for_vllm` | 创建独立配置副本，使 vLLM 的 `r` 与 AdaLoRA A/B 的 `init_r` 维度一致 |
| rollout 同步 | `verl/utils/fsdp_utils.py:collect_lora_params` | 把 E/ranknum 折进 A，将官方 AdaLoRA 参数转换成 vLLM 可接收的普通 A/B |
| checkpoint | `verl/workers/fsdp_workers.py:save_checkpoint` | 保存 PEFT/FSDP actor 权重和 trainer extra state |

### 3.1 为什么 PPO 要额外处理正交 loss

PEFT 官方 `AdaLoraModel.forward()` 只在底层模型返回 `outputs.loss` 时计算并加入正交正则。SFT forward 本来就传 labels，所以自然有 loss；verl PPO 通常只请求 logits，再在模型外计算 policy loss，因此官方正交项原本不会执行。

本仓库在需要正交项时传入：

```python
labels = torch.full_like(input_ids, -100)
num_items_in_batch = 1
```

全 `-100` labels 使监督 CE 为零，PEFT 随后仍会在同一个官方 forward 中加入 A/B 的 Frobenius 正交正则。actor 直接使用返回的 `output.loss`，没有在 verl 中复制官方公式。

### 3.2 为什么 allocator 必须紧跟 optimizer.step

官方 allocator 用当前 A/B/E 参数和仍然存在的梯度更新 importance EMA。正确顺序是：

```text
forward -> PPO loss + official orth loss -> backward
-> optimizer.step
-> AdaLoraModel.update_and_allocate(optimizer_step)
-> zero_grad
```

原来的早期移植在整个 trainer step 后只调用一次，并传 trainer step。对 `batch=64, mini-batch=16` 的配置，这会把调度慢 4 倍，并丢失中间 3 次 optimizer update 的 importance 更新。

### 3.3 为什么需要 FSDP 特殊处理

官方 RankAllocator 并不知道 FSDP shard。它会遍历所有 A/B/E，读取 `parameter * gradient`，再对所有模块的 triplet score 做全局排序。如果直接对 shard 调用：

- 不同 rank 只看到局部参数；
- mask 排序不再是全模型排序；
- 某些参数的 grad 可能不可见；
- 各 rank 可能写回不同 mask。

因此当前支持路径限定为 FSDP1，并强制 `use_orig_params=True`。allocator 在每个 rank 的完整参数/梯度上下文内运行并写回。启动配置还设置 `reshard_after_forward=False`，保证 PEFT 官方 forward 在底层模型 forward 返回后遍历 A/B 时参数仍然可用。

FSDP2/DTensor 尚未声明支持，代码会明确报错，而不是静默运行错误的 allocation。

### 3.4 为什么 rollout 同步要折叠 E

训练 actor 使用官方 SVDLinear 的 A/E/B 三因子，而 vLLM LoRA loader 只认识普通 A/B。如果只发送 A/B，rollout 得到的策略与 actor 不一致，old log probability、采样分布和 PPO ratio 都会被污染。

同步时临时执行：

```text
A_for_vllm = A * E * configured_rank / (ranknum + 1e-5)
```

vLLM 自己再应用普通 LoRA 的 `alpha / configured_rank`，最终与 PEFT 的 `alpha / ranknum` 等价。同步函数会 clone 导出张量，然后恢复 actor 原始 A，不会修改训练参数。

官方 AdaLoRA config 继承的 `r` 仍是默认值 8，但实际 A/B 宽度是 `init_r`。传给 vLLM 前会 deep-copy config 并令 `r=init_r`，否则本实验的 rank-32 张量会错误使用 `alpha/8` 缩放。该修改只作用于 rollout config 副本，不改变训练 actor 的官方配置。

此外，PEFT AdaLoRA 导出的键为 `...lora_A` / `...lora_B`，同步边界会转换成 vLLM 需要的 `...lora_A.weight` / `...lora_B.weight`，并拒绝发送 `lora_E`、`ranknum` 等 vLLM 不支持的权重。

## 4. 当前标准实验配置

入口：

```bash
bash examples/verl_train/run_dapo_math_boxed_official_adalora_1p5b_4gpu_8k.sh
```

默认配置：

```text
trainer steps       270
train batch          64
PPO mini-batch       16
optimizer updates     4 / trainer step
AdaLoRA total_step  1080
init_r               32
target_r              8
tinit                100 optimizer steps
tfinal               200 optimizer steps
deltaT                20 optimizer steps
alpha                 64
dropout             0.05
orth_reg_weight    1e-3
learning rate       1e-6
weight decay           0
scheduler        constant
save frequency        50 trainer steps
```

launcher 会根据 `TOTAL_TRAINING_STEPS * TRAIN_PROMPT_BSZ / TRAIN_PROMPT_MINI_BSZ * PPO_EPOCHS` 自动推导 `ADALORA_TOTAL_STEP`；当前标准配置的 `PPO_EPOCHS=1`。

## 5. 应重点监控的指标

- `actor/pg_loss`：原始 policy-gradient loss。
- `adalora/orthogonal_regularization`：一次 optimizer update 内按 micro-batch loss 权重聚合的未加权官方正交正则。
- `adalora/orthogonal_loss`：同一 update 内实际加入 policy loss 的加权正交项。
- `adalora/pg_loss_abs`：同一 update 内 PG loss 绝对值的加权聚合值。
- `adalora/orthogonal_to_pg_abs_ratio`：先分别聚合正交项和 PG loss 绝对值，再计算二者之比。不能对 micro-batch 比值取平均，因为零优势或优势抵消会令单个 micro-batch 的 PG loss 接近零，产生虚假的千万级尖峰。
- `adalora/optimizer_step`：allocator 使用的真实 optimizer-step 编号。
- `adalora/scheduled_rank_mean`：当前全局预算除以目标模块数。
- `adalora/current_nonzero_rank_mean/min/max`：当前 E 中非零分量数。
- `adalora/mask_applied`：本次 update 是否处于实际 mask 边界。
- `adalora/masked_rank_mean/min/max`：mask 当下的 rank，避免与随后 Adam 可能重新长出的非零 E 混淆。
- `adalora/importance_ema_mean`、`uncertainty_ema_mean`、`importance_uncertainty_score_mean`：官方 allocator 状态的聚合诊断。
- `adalora/update_skipped_nonfinite`：梯度非有限时 optimizer 和 allocator 均未推进。

## 6. 已知边界

- 当前官方 AdaLoRA 路径只支持 FSDP1，不支持 FSDP2/DTensor allocation。
- 标准 launcher 使用 `trainer.resume_mode=disable`。PEFT 的 RankAllocator importance/uncertainty EMA 不是普通 model state_dict 的一部分；只恢复模型权重而重置 EMA 会改变裁剪轨迹。因此代码会拒绝从 `global_steps > 1` 的经典 PPO checkpoint 静默续训，直到 allocator state 被显式纳入 checkpoint。
- `target_modules=all-linear` 会对大量模块做全局 rank 排序，allocator 边界需要一次完整参数/梯度 materialization；这会带来显著但可解释的额外耗时。
- AdaLoRA 的 `target_r=8` 是所有目标模块的最终平均预算，不保证每个模块最终都为 rank 8。

### 6.1 RL 中的正交系数

PEFT 官方公式保持不变：`outputs.loss += orth_reg_weight * regu_loss`。本仓库只把实验默认值从 PEFT 面向监督训练的 `0.5` 调整为 `1e-3`。此前日志中未加权正则约为 `5.26`，使用 `0.5` 后实际正交 loss 约为 `2.63`，而 PG loss 通常只有 `0.01` 到 `0.09`，优化方向会主要由正交项决定。`1e-3` 对应约 `0.00526` 的正交 loss，仍保留约束，但与当前 RL 信号处于可比较量级。该值是本实验的起点，应继续结合 update 级 ratio、reward 和 rank 轨迹判断，而不是视为通用 AdaLoRA 常数。

### 6.2 分布式失败处理

Ray worker group 会按完成顺序逐个解析结果；任何 rank 失败都会立即抛给 driver，而不会等待其余已阻塞在 collective 的 rank。运行环境同时启用 PyTorch NCCL async error handling、heartbeat monitoring 和 timeout dump，并按 PyTorch 2.8 的变量名使用 `TORCH_FR_BUFFER_SIZE`。这样不能消除外部 `SIGKILL`、驱动故障或系统 OOM，但可以避免单 rank 消失后其余 rank 无限刷 `TCPStore Broken pipe`，并为后续故障留下更完整的 NCCL trace。

当前 `/home` 是约 15 TiB 的共享卷，使用率超过 Ray 默认的 95% 阈值时，即使仍有数百 GiB 可用空间，Ray 也会拒绝 object spilling 并持续告警。AdaLoRA 专用 launcher 将 `RAY_local_fs_capacity_threshold` 设为 `0.98`，继续使用 `/home` 下的短临时路径，同时保留约 2%（当前约 293 GiB）的磁盘保护余量。这个设置不是对磁盘不足的掩盖；达到 98% 后 Ray 仍会停止 spilling，启动前仍应检查实际可用空间。

## 7. 推荐阅读顺序

1. `examples/verl_train/run_dapo_math_boxed_official_adalora_1p5b_4gpu_8k.sh`
2. `verl/trainer/main_ppo.py`
3. `verl/trainer/ppo/ray_trainer.py:RayPPOTrainer.fit`
4. `verl/workers/fsdp_workers.py:ActorRolloutRefWorker`
5. `verl/workers/actor/dp_actor.py:DataParallelPPOActor.update_policy`
6. `verl/utils/peft_adalora.py`
7. `verl/utils/fsdp_utils.py:collect_lora_params`
8. PEFT `peft/tuners/adalora/config.py`、`layer.py`、`model.py`

测试位于 `tests/test_verl_adalora.py`，覆盖官方正交 forward、真实 AdaLoraModel 定位、optimizer-step allocation、rank 指标、vLLM A/E/B 折叠和 270 -> 1080 调度映射。
