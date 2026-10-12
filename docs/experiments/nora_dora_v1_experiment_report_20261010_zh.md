# NoRA + DoRA v1 实验记录

更新时间：2026-10-10 UTC

实验状态：50-step 正式训练运行中
Run ID：`nora_dora_r8a8_mag_init_lr1e6_b64m16n8_50_4gpu_boxed`

## 1. 研究问题

NoRA + DoRA + OFT v1 的六任务评测未达到晋级门槛。本轮移除 OFT/Cayley 正交旋转，只保留 NoRA 初始化和 DoRA magnitude，用来判断上一轮的主要限制是否来自正交旋转参数化。

这是结构消融，不是独立超参搜索。相对上一轮保持 rank、alpha、初始化、训练数据、GRPO 配置和 50-step 预算一致；唯一核心结构变化是：

```text
NoRA-DOFT: blockwise skew generator -> Cayley rotation -> magnitude
NoRA-DoRA: additive B @ A direction              -> magnitude
```

## 2. 方法定义

对每个目标线性层的冻结权重 `W in R^(out x in)`，学习：

```text
A in R^(r x in)
B in R^(out x r)
m in R^out
scaling = alpha / r
D = W + scaling * B @ A
W' = diag(m / stop_gradient(||D||_row)) @ D
```

其中 `r=8`、`alpha=8`、`scaling=1`。bias 不经过 magnitude 缩放。

初始化为：

```text
A ~ Normal(0, 1)
A = normalize(A, dim=0)   # 每个 input column 沿 rank 维归一化
B = 0
m = ||W||_row
```

因此 step 0 与 base model 同函数。`normalization=init` 表示归一化只发生在初始化；这与上一轮一致，避免同时改变 OFT 结构和持续归一化两个变量。每层使用 `seed + 1000003 * layer_index` 的确定性派生 seed。

方向范数使用 `detach`，与 PEFT 0.19.1 的 DoRA 梯度定义一致。首个反向步中 A 的梯度为零，B 和 magnitude 的梯度非零。

## 3. 参数量与实现

目标模块为 28 层中的 `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`，共 196 个线性层。

| 参数组 | 参数量 |
| --- | ---: |
| additive low-rank A + B | `9,232,384` |
| DoRA magnitude | `645,120` |
| 总计 | **`9,877,504`** |

真实模型启动日志已核对：`layers=196`、`trainable_params=9877504`、初始化列范数最大误差 `2.384e-07`。该参数量约为上一轮 NoRA-DOFT 的 `112.40%`，约为 matched LoRA r32 的 `26.75%`。

实现入口：

- `verl/utils/peft_nora_dora.py`
- `examples/verl_train/run_dapo_math_boxed_nora_dora_1p5b_4gpu_8k.sh`
- `scripts/local/start_nora_dora_4gpu.sh`
- `tina_run/scripts/local/eval/merge_nora_dora_wrapped_hf.py`

## 4. 正式训练配置

| 项目 | 设置 |
| --- | --- |
| Base model | DeepSeek-R1-Distill-Qwen-1.5B |
| 数据 | DAPO-Math-17k boxed，17,917 rows |
| 算法 | GRPO；KL reward/loss 均关闭 |
| GPU | RTX A6000 4–7，共 4 卡 |
| trainer steps | 50 |
| actor updates / trainer step | 4 |
| prompt batch / mini batch / responses | 64 / 16 / 8 |
| LR / scheduler / warmup / weight decay | `1e-6` / constant / 0 / 0 |
| PPO epochs | 1 |
| clip low / high | 0.2 / 0.28 |
| max prompt / response | 1,024 / 8,192 |
| actor/infer token cap | 12,288 / 12,288 |
| data / PPO loader seed | 42 / 42 |
| rollout seed | `null`，与上一轮实际协议一致 |
| checkpoint | 每 10 步，保留 model + optimizer + extra |
| resume | 首次运行 `disable`；checkpoint 具备续训所需 optimizer state |

运行服务：`peft-nora-dora-v1.service`

训练日志：`runs/nora-dora-v1/logs/nora_dora_training.log`

checkpoint 根目录：`runs/nora-dora-v1/ckpts/verl/DAPO-Math-17k/nora_dora_r8a8_mag_init_lr1e6_b64m16n8_50_4gpu_boxed`

## 5. 验证与评测计划

实现验证已通过：

```text
pytest tests/test_nora_dora.py tests/test_nora_doft.py tests/test_full_bench_eval.py
34 passed
```

测试覆盖初始函数等价、NoRA 列归一化、首步梯度、DoRA reference 公式、direction norm detach、continuous 模式、adapter disable、state-dict round trip、merged full-weight 等价及两条 FSDP 接线。

训练完成后沿用上一轮同一六任务 full-bench：AIME24、AIME25、AMC23、MATH500、Minerva Math、OlympiadBench，共 902 problems、每题 8 samples、temperature 0.7、max tokens 32,768、seed 42。主要指标为 Macro Avg@k，并与 no-RL base、matched LoRA r32 seed42/43、standard OFT、RLPO、random-B 和 NoRA-DOFT 做协议分层比较。

本轮最关键的配对解释是 NoRA-DoRA 对 NoRA-DOFT。若显著提升，说明 additive direction 比本次 blockwise Cayley 约束更适合该 RL 设置；若仍接近 base，则应优先检验 `continuous` NoRA、magnitude/方向学习率分组或信息化 B 初始化，而不是继续堆叠三种参数化。
