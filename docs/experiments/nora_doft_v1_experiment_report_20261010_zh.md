# NoRA + DoRA + OFT v1 实验记录

更新时间：2026-10-10 UTC

实验状态：训练完成，六任务 full-bench 完成并通过 postverify，当前决策为 `no-go`
Run ID：`nora_doft_r8b32_mag_init_lr1e6_b64m16n8_50_4gpu_boxed`

## 1. 执行摘要

本实验实现并测试了一个低参数量的正交微调变体：使用 NoRA 风格的归一化低秩因子构造反对称生成器，通过精确 Cayley transform 得到 blockwise 正交旋转，再使用 DoRA 风格的可训练行幅值恢复正交旋转不能表达的权重尺度变化。

核心结果如下：

| 项目 | 结果 |
| --- | --- |
| 训练模型 | DeepSeek-R1-Distill-Qwen-1.5B |
| 训练预算 | 50 trainer steps，约 200 次 actor optimizer update |
| 可训练参数 | `8,787,968` |
| 正式评测 | 6 tasks，902 problems，7,248 generations |
| Macro Avg@k | `0.3261753472` |
| Macro Pass@k | `0.5616503268` |
| Overall parse rate | `0.8163631347` |
| Overall hit-max rate | `0.0420805740` |
| 平均生成长度 | `9205.14` tokens |
| 相对 no-RL base | `+0.001827`，即 `+0.183 pp` |
| 相对 matched LoRA r32 seed42 | `-0.003878`，即 `-0.388 pp` |
| 相对 matched LoRA r32 两 seed 均值 | `-0.008232`，即 `-0.823 pp` |
| 最终决策 | 不续训、不补 seed43；作为负结果保留 |

当前结果不支持“将 NoRA、DoRA 和 OFT 直接组合即可涨点”。可以保留的积极信号是参数效率：本方法只使用 matched LoRA r32 的 `23.80%` 可训练参数，在 training seed42 上仅低 `0.388 pp`。但它相对 no-RL base 的绝对增益只有 `0.183 pp`，相对历史参数量相近的 standard LoRA r8 也只高 `0.306 pp`，且明显低于 random-B/full-gradient 等有信息初始化方法，因此不足以晋级。

## 2. 研究问题与实验假设

本实验来自如下经验性假设：

1. NoRA 的归一化低秩因子可能改善低秩参数化的尺度和条件数。
2. OFT 的正交旋转能够在保持权重几何结构的同时改变输入方向。
3. DoRA 式 magnitude 分支能够补充纯正交旋转无法改变行范数的表达自由度。
4. 三者结合后，可能在接近 uniform-r8 的参数预算下达到或接近 rank32 LoRA 的效果。

本轮只测试一个首版组合点，不是完整消融：

```text
generator rank = 8
block size = 32
normalization = init
magnitude = on
alpha = 8
initialization seed = 42
training seed = 42
50 trainer steps
```

因此本轮能回答“这个具体组合是否值得晋级”，不能分别估计 NoRA、DoRA 和 OFT 的独立贡献，也不能支持多 training-seed 稳定性结论。

## 3. 方法定义

### 3.1 Blockwise 低秩反对称生成器

对每个被替换的线性层，将输入维度按 `block_size=32` 划分。对每个输入块 `j` 学习：

```text
A_j in R^(r x b)
B_j in R^(b x r)
r = 8
b = 32
```

先构造低秩方阵：

```text
G_j = B_j A_j
```

然后取反对称部分并应用缩放：

```text
K_j = 0.5 * (alpha / r) * (G_j - G_j^T)
```

本实验 `alpha=r=8`，所以 `alpha/r=1`。代码中的 `0.5` 用于抵消 Cayley transform 在恒等点的一阶系数 2，使初始一阶 adapter scale 等于 `alpha/r`。

### 3.2 精确 Cayley transform

旋转矩阵由以下方程精确求解：

```text
R_j = (I + K_j)^(-1) (I - K_j)
```

实现使用 `torch.linalg.solve`，不是标准 OFT 配置中可选的有限项 Neumann 近似。只要求解数值有效，`R_j` 是正交矩阵。

代码先对输入块执行右乘旋转，再调用冻结的 base linear：

```text
x_j' = x_j R_j
y = x' W^T + bias
```

等价的 merged weight block 为 `W_j R_j^T`。

### 3.3 NoRA 风格初始化

每个 wrapper 独立创建 CPU generator，并固定 `seed=42`：

```text
A_j ~ Normal(0, 1)
A_j = normalize(A_j, dim=-2)
B_j = 0
```

这里的 `dim=-2` 表示对 A 的每一列沿 rank 维归一化，使列范数为 1；它不是正交化，也不保证不同列互相正交。

需要特别记录：当前实现在每个线性层中重新以同一个 seed 创建 generator。因此形状相同的模块会得到相同的初始随机 A，而不是 layer-specific 独立随机样本。这保证确定性，但可能降低跨层初始化多样性。

因为 `B=0`，所以初始 `G=K=0`、`R=I`，旋转分支不会改变初始模型函数。

本次设置为 `normalization=init`。初始化完成后，forward 直接使用可训练 A；只有设置为 `continuous` 时才会在每次 forward 中重新归一化 A。因此本轮没有在训练过程中持续维持 NoRA 列归一化约束。

### 3.4 DoRA 风格 magnitude

对 base weight 的每一个输出行保存初始范数 `m0`，并创建可训练参数 `m`：

```text
m0_i = ||W_i||_2
m_i  = m0_i at initialization
scale_i = m_i / max(m0_i, eps)
```

旋转后的线性输出按输出行乘以 `scale_i`。bias 在缩放后单独相加，因此 magnitude 不缩放 bias。由于 `m=m0`，magnitude 分支在初始化时同样保持恒等。

### 3.5 初始函数与训练自由度

初始状态同时满足：

```text
B = 0
R = I
m / m0 = 1
```

因此 adapter 初始函数严格等价于 base model。训练时 A、B 和 magnitude 都可训练，base linear 的 weight/bias 均被冻结。

## 4. 实现范围与参数预算

### 4.1 Target modules

`target_modules=all-linear` 在当前实现中解析为以下 7 类投影：

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

模型共 28 层，因此实际替换 `28 x 7 = 196` 个 `nn.Linear` 模块。

### 4.2 参数量分解

每个输入块的 A/B 参数量为：

```text
r*b + b*r = 2*r*b = 512
```

按所有 196 个模块汇总：

| 参数组 | 参数量 | 占比 |
| --- | ---: | ---: |
| A + B 低秩生成器 | `8,142,848` | `92.66%` |
| magnitude | `645,120` | `7.34%` |
| 总计 | **`8,787,968`** | `100%` |

训练日志在 FSDP 每 rank 上报告 `2,196,992` active parameters，4 个 rank 合计正好为 `8,787,968`。

参数预算对照：

| 方法 | 可训练参数 | NoRA-DOFT / 对照 |
| --- | ---: | ---: |
| NoRA-DOFT v1 | `8,787,968` | `1.000` |
| Historical LoRA r8 | `9,232,384` | `0.952` |
| Matched LoRA r32 | `36,929,536` | `0.238` |

因此本方法应主要按“约 uniform-r8 参数预算”解释；直接和 LoRA r32 比最终精度时，必须同时披露其参数量只有后者约四分之一。

## 5. 训练配置

### 5.1 输入与哈希

| 项目 | 实际值 |
| --- | --- |
| Repository | `/home/wangls/peft-for-rl` |
| Base model | `runs/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base` |
| Base weights SHA-256 | `58858233513d76b8703e72eed6ce16807b523328188e13329257fb9594462945` |
| Train dataset | `runs/datasets/verl_dapo/data/dapo-math-17k-boxed.parquet` |
| Dataset SHA-256 | `dc653393b0b8065a782e51745c8ee1c0a80aad64c1a36ac5cff0362b9e3d5f1e` |
| Reward | `verl/utils/reward_score/boxed_math_accuracy.py` |
| Launcher | `examples/verl_train/run_dapo_math_boxed_nora_doft_1p5b_4gpu_8k.sh` |
| Parent launcher | `examples/verl_train/run_dapo_math_boxed_stable_lora_1p5b_4gpu_8k.sh` |

本 run 启动前没有生成独立 training contract、代码文件聚合哈希或干净 worktree 快照。当前仓库 commit 为 `f6bad923dcdc01f851aeb2f3b8c8df7d2e3fd247`，但训练时工作树包含未提交修改，因此该 commit 不能单独代表实际训练代码。这是本实验相对 Phase-1 A-grade 证据的主要 provenance 缺口。

### 5.2 Adapter 配置

| 配置 | 实际值 |
| --- | --- |
| `peft_type` | `nora_doft` |
| generator rank | `8` |
| block size | `32` |
| alpha | `8` |
| generator scaling | `1.0` |
| normalization | `init` |
| magnitude | enabled |
| epsilon | `1e-6` |
| initialization seed | `42` |
| target modules | `all-linear` / 196 modules |
| ordinary LoRA rank | `0`，关闭 |

继承的 resolved config 中仍能看到 `lora_dropout=0.05` 和 standard OFT 参数，但 `peft_type=nora_doft`、`lora_rank=0`，这些 LoRA/OFT 字段不参与本次 adapter forward，不能将本实验描述为使用了 LoRA dropout。

### 5.3 GRPO 与优化器

| 配置 | 实际值 |
| --- | --- |
| Advantage estimator | GRPO |
| Trainer steps | `50` |
| Prompt batch / step | `64` |
| Responses / prompt | `8` |
| Generations / step | `512` |
| PPO mini-batch | `16` prompts |
| PPO epochs | `1` |
| Actor optimizer updates / trainer step | `4` |
| 估算总 optimizer updates | `200` |
| Actor learning rate | `1e-6` |
| Scheduler | constant |
| Warmup | `0` |
| Weight decay | `0` |
| Gradient clipping | `1.0` |
| Loss aggregation | token mean |
| Entropy coefficient | `0` |
| KL in reward | disabled |
| Actor KL loss | disabled |
| PPO clip low / high / c | `0.2 / 0.28 / 10.0` |
| Overlong buffer penalty | disabled |

### 5.4 数据与随机性

| 配置 | 实际值 |
| --- | --- |
| Data seed | `42` |
| PPO data-loader seed | `42` |
| Train shuffle | enabled |
| Actor mini-batch shuffle | disabled |
| Rollout seed | `null` |
| Rollout temperature | `1.0` |
| Rollout top-p / top-k | `1.0 / -1` |
| Max prompt length | `1024` |
| Max response length | `8192` |

虽然 data/PPO 和 adapter 初始化 seed 固定为 42，vLLM rollout seed 没有显式固定，异步 rollout 也不保证 request-level bitwise determinism。因此该 run 只能视为一个 training seed42 轨迹，不能视为完全可重复的确定性运行。

### 5.5 并行、显存与吞吐

| 配置 | 实际值 |
| --- | --- |
| Node / GPU | `1 / 4` |
| CUDA devices | `4,5,6,7` |
| FSDP size | `-1`，使用全部 4 卡 |
| FSDP original params | enabled |
| Parameter offload | enabled |
| Optimizer offload | enabled |
| Reshard after forward | enabled |
| Gradient checkpointing | enabled |
| Remove padding | enabled |
| vLLM tensor parallel | `1` |
| vLLM max model length | `9216` |
| vLLM max batched tokens | `9216` |
| vLLM max sequences | `256` |
| vLLM GPU memory utilization | `0.70` |
| Dynamic batch size | enabled |
| Actor/inference max tokens per GPU | `12288 / 12288` |

日志中观测到的 actor 最大 allocated/reserved 显存约为 `56.00 / 57.48 GiB`，最终 step 吞吐约 `1218.9 tokens/s`。本 run 没有保存 GPU 型号、UUID、driver 和 package lock，因此报告不推断具体硬件型号。

## 6. 训练执行与 checkpoint

### 6.1 时间线

| 事件 | UTC | Asia/Shanghai |
| --- | --- | --- |
| 训练日志创建 | 2026-10-10 00:45:58 | 2026-10-10 08:45:58 |
| Step 50 checkpoint 完成 | 2026-10-10 10:28:11 | 2026-10-10 18:28:11 |
| 开始 merge/eval controller | 2026-10-10 11:35:12 | 2026-10-10 19:35:12 |
| 正式评测启动 | 2026-10-10 11:36:47 | 2026-10-10 19:36:47 |
| 评测 postverify 完成 | 2026-10-10 14:31:10 | 2026-10-10 22:31:10 |

训练 progress 记录的 50-step 墙钟时间为 `9:40:48`，平均约 `696.97 s/step`。训练从 step0 到 step50 连续完成，没有发生已记录的中断恢复。

### 6.2 保存策略与恢复边界

| 配置 | 实际值 |
| --- | --- |
| Save frequency | 每 10 steps |
| 已保存 checkpoint | step10、20、30、40、50 |
| Max actor checkpoints | `10` |
| Resume mode | `disable` |
| Save contents | `model`, `extra` |
| Optimizer state | **未保存** |

Step50 checkpoint 包含 4 份 FSDP model shard、4 份 extra-state shard、HF config/tokenizer 和 `data.pt`，但没有 optimizer state。技术上可以从模型权重继续启动训练，但那会重建 optimizer，不是严格连续的 step51 轨迹。后续若需要连续训练，必须重新训练一个保存 optimizer state 的 run，或把 optimizer reset 明确列为实验变量。

### 6.3 训练动态

按 5-step 窗口汇总的 on-policy batch 指标如下：

| 窗口中心 step | Reward accuracy | Parse success |
| ---: | ---: | ---: |
| 3 | 0.317188 | 0.445312 |
| 8 | 0.307031 | 0.479687 |
| 13 | 0.306641 | 0.455469 |
| 18 | 0.358984 | 0.490625 |
| 23 | 0.312500 | 0.476562 |
| 28 | 0.342969 | 0.494531 |
| 33 | 0.333594 | 0.477734 |
| 38 | 0.354297 | 0.512891 |
| 43 | 0.346875 | 0.491406 |
| 48 | 0.328906 | 0.480469 |

训练过程数值稳定，没有 non-finite、显著 PPO ratio 爆炸或梯度爆炸；但 reward accuracy 高方差且没有形成单调上升趋势。单独的 step50 batch accuracy 为 `0.324219`，parse success 为 `0.468750`，不能作为 held-out 能力指标。

## 7. 正式评测设置

### 7.1 Merge 路径

评测前执行两步转换：

1. 将 step50 的 4 个 FSDP shard 合并为带 `NoRADoftLinear` wrapper 的 HF 模型。
2. 加载冻结 base model和 wrapper 参数，将旋转与 magnitude 物化进普通 HF weight，得到无 adapter wrapper 的 plain CausalLM。

评测 manifest 中 `lora_adapter=null`，vLLM 直接加载物化后的完整模型：

```text
runs/nora-doft-v1/merged/step50/actor_hf_vllm
```

### 7.2 冻结 full-bench 协议

| 配置 | 实际值 |
| --- | --- |
| Benchmarks | AIME24、AIME25、AMC23、HMMT Feb、MATH500、Minerva |
| Problems | `902` |
| Total generations | `7,248` |
| Eval seed | `42` |
| Temperature | `0.6` |
| Top-p | `0.95` |
| Max new tokens | `32768` |
| Max model length | `34816` |
| Samples for AIME/AMC/HMMT | `32/problem` |
| Samples for MATH500/Minerva | `4/problem` |
| Shards | `4` |
| Eval GPUs | `4,5,6,7` |
| Eval GPU memory utilization | `0.85` |
| Snapshot SHA-256 | `3416ba93286b324a9663777fa472d0b568af5446319f27f876a63b59b3e863da` |

逐任务请求数：

| Benchmark | Problems | Samples/problem | Records |
| --- | ---: | ---: | ---: |
| AIME24 | 30 | 32 | 960 |
| AIME25 | 30 | 32 | 960 |
| AMC23 | 40 | 32 | 1280 |
| HMMT Feb | 30 | 32 | 960 |
| MATH500 | 500 | 4 | 2000 |
| Minerva | 272 | 4 | 1088 |

### 7.3 Postverify

独立验证检查了：

- summary 和 records 存在且可重算；
- 6 个 benchmark 集合及各任务记录数正确；
- 7,248 个 request key 完整且唯一；
- 4 个 shard manifest 完整；
- seed、temperature、top-p、max-new-tokens 与冻结协议一致；
- benchmark snapshot 路径与 SHA-256 一致；
- merged full model 按无 adapter 模式评测。

最终状态为 `verified`。

## 8. Full-bench 结果

### 8.1 总体指标

| 指标 | 数值 |
| --- | ---: |
| Macro Avg@k | `0.3261753472` |
| Macro Pass@k | `0.5616503268` |
| Overall parse rate | `0.8163631347` |
| Overall hit-max rate | `0.0420805740` |
| Overall mean length | `9205.1433` |

### 8.2 逐任务指标

| Benchmark | Avg@k | Pass@k | Parse | Hit max | Mean length |
| --- | ---: | ---: | ---: | ---: | ---: |
| AIME24 | `0.216667` | `0.500000` | `0.778125` | `0.071875` | `14446.9` |
| AIME25 | `0.191667` | `0.533333` | `0.842708` | `0.084375` | `13979.7` |
| AMC23 | `0.608594` | `0.975000` | `0.853906` | `0.035156` | `7950.1` |
| HMMT Feb | `0.087500` | `0.233333` | `0.890625` | `0.062500` | `15060.0` |
| MATH500 | `0.649500` | `0.790000` | `0.896500` | `0.013000` | `4134.5` |
| Minerva | `0.203125` | `0.338235` | `0.569853` | `0.022059` | `5998.8` |

最明显的行为问题出现在长推理任务：AIME24、AIME25 和 HMMT 的平均长度约 `14k--15k` tokens，hit-max 为 `6.25%--8.44%`。整体 parse rate 不低，但更长输出和更高触顶率没有转换成更高正确率。

## 9. 与 matched LoRA 的配对比较

NoRA-DOFT records 与 matched LoRA 的 problem/sample keys 完全一致。统计采用 problem-cluster paired bootstrap：先在每道题内部计算样本均值差，再按 benchmark 内的题目重采样，最后对 6 个 benchmark 等权平均。每项使用 10,000 次 bootstrap、seed42。

### 9.1 对 matched LoRA seed42

| Benchmark | LoRA | NoRA-DOFT | Delta | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| AIME24 | 0.246875 | 0.216667 | -0.030208 | `[-0.100000, +0.035417]` |
| AIME25 | 0.197917 | 0.191667 | -0.006250 | `[-0.058333, +0.043750]` |
| AMC23 | 0.610156 | 0.608594 | -0.001562 | `[-0.051582, +0.047656]` |
| HMMT Feb | 0.092708 | 0.087500 | -0.005208 | `[-0.020833, +0.009375]` |
| MATH500 | 0.647000 | 0.649500 | +0.002500 | `[-0.014500, +0.020500]` |
| Minerva | 0.185662 | 0.203125 | +0.017463 | `[-0.004596, +0.039522]` |
| **Macro** | **0.330053** | **0.326175** | **-0.003878** | **`[-0.021511, +0.013277]`** |

Macro 的 bootstrap `P(delta > 0)=0.3259`。区间跨 0，因此不能声称显著劣于 seed42 LoRA；但 point estimate 为负，也没有优于 LoRA 的证据。

### 9.2 对 matched LoRA seed43

| Benchmark | LoRA | NoRA-DOFT | Delta | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| AIME24 | 0.242708 | 0.216667 | -0.026042 | `[-0.085417, +0.029167]` |
| AIME25 | 0.248958 | 0.191667 | -0.057292 | `[-0.113542, -0.008333]` |
| AMC23 | 0.599219 | 0.608594 | +0.009375 | `[-0.034375, +0.055469]` |
| HMMT Feb | 0.101042 | 0.087500 | -0.013542 | `[-0.025000, -0.003125]` |
| MATH500 | 0.632000 | 0.649500 | +0.017500 | `[-0.000500, +0.035000]` |
| Minerva | 0.208640 | 0.203125 | -0.005515 | `[-0.024816, +0.013787]` |
| **Macro** | **0.338761** | **0.326175** | **-0.012586** | **`[-0.028182, +0.002508]`** |

Macro 的 bootstrap `P(delta > 0)=0.0530`。AIME25 和 HMMT 的 95% CI 完全低于 0；MATH500 的正 point estimate 仍未达到 95% 区间排除 0。

### 9.3 生成行为对照

| 指标 | NoRA-DOFT | LoRA seed42 | LoRA seed43 |
| --- | ---: | ---: | ---: |
| Macro Avg@k | 0.326175 | 0.330053 | 0.338761 |
| Macro Pass@k | 0.561650 | 0.567031 | 0.613206 |
| Parse rate | 0.816363 | 0.801600 | 0.815259 |
| Hit max | 0.042081 | 0.036010 | 0.034630 |
| Mean length | 9205.1 | 8974.4 | 8940.4 |

NoRA-DOFT 相对 seed42 LoRA 的 parse rate 高 `1.48 pp`，但平均输出多约 231 tokens、hit-max 高 `0.61 pp`，Avg@k 反而低 `0.39 pp`。相对 seed43 LoRA，parse 基本相同，输出多约 265 tokens、hit-max 高 `0.75 pp`，Avg@k 低 `1.26 pp`。因此 parse 改善不能解释为推理能力增益。

## 10. 其他基线与证据边界

### 10.1 当前协议内或接近协议的定位

| 方法 | Macro Avg@k | NoRA-DOFT delta | 说明 |
| --- | ---: | ---: | --- |
| no-RL base | 0.324348 | +0.001827 | A-reference；绝对参考，不是 paired training control |
| matched LoRA r32 seed42 | 0.330053 | -0.003878 | 最接近的单 training-seed 对照，但参数量约 4.2 倍 |
| matched LoRA r32 seed43 | 0.338761 | -0.012586 | 同协议、不同 training seed |
| matched LoRA r32 两 seed均值 | 0.334407 | -0.008232 | 描述量，不是统计 gate |
| I0 两 seed均值 | 0.328955 | -0.002779 | Phase-1 matched random baseline |
| I8 两 seed均值 | 0.353928 | -0.027752 | Phase-1 signal + random |
| I32 两 seed均值 | 0.354840 | -0.028665 | Phase-1 signal-only endpoint |

NoRA-DOFT 相对 no-RL base 仅高 `0.183 pp`，不能据此声称产生了可靠 RL 增益。与 I0/I8/I32 两 seed均值的差异也只能作位置描述，因为 NoRA-DOFT 只有一个 training seed。

### 10.2 历史跨协议参照

| 方法 | Step | Macro Avg@k | NoRA-DOFT delta | 主要不匹配项 |
| --- | ---: | ---: | ---: | --- |
| Standard LoRA uniform-r8 | 50 | 0.323111 | +0.003064 | dropout0.05、历史训练链 |
| Standard OFT | 100 | 0.323712 | +0.002463 | 不同 step 和训练协议 |
| Random-B uniform-r8 | 50 | 0.358323 | -0.032148 | dropout0.05、历史 probe 链 |
| Full-gradient uniform-r8 | 50 | 0.358565 | -0.032389 | 不同初始化构造 |
| Original random-B mean-r31.61 | 50 | 0.367304 | -0.041129 | 不同容量、初始化和证据等级 |
| GradTop mean-r31.63 | 50 | 0.391198 | -0.065023 | request identity 不完全兼容，D-grade |
| RLPO r32 corrected | 270 | 0.398649 | -0.072474 | 不同步数，HMMT 为修复拼接 |

这些数值只能说明量级，不能支持严格方法优劣。尤其不能用 NoRA-DOFT 高于 standard OFT step100 的 `0.246 pp` 声称胜过 OFT；该差值既很小，又混合了训练协议和步数差异。

## 11. 结果解释

### 11.1 没有单一 benchmark 崩溃

相对 LoRA seed42，NoRA-DOFT 在 MATH500 和 Minerva 为正，其余四项为负；相对 seed43，仅 AMC23 和 MATH500 为正。负结果不是一个异常任务造成，而是多个竞赛任务共同持平或偏弱。

### 11.2 当前实现只使用一次性 NoRA 归一化

首版设置 `normalization=init`，A 在训练开始后可以任意漂移。因而该 run 不能回答“持续 NoRA normalization 与 OFT 是否互补”，只回答“一次归一化随机初始化 + OFT + magnitude”这个组合。

### 11.3 初始化方向仍是主要缺口

当前 A 来自 seeded Gaussian，没有使用 base-weight principal subspace、activation subspace 或 RL gradient probe。历史低预算结果中，random-B/full-gradient uniform-r8 比 standard LoRA r8 高约 `3.5 pp`，而 NoRA-DOFT 只比 historical standard LoRA r8 高 `0.3 pp`。这更支持“低秩方向选择不足”，而不是“rank8 容量必然不足”。

### 11.4 magnitude 是否有效目前不可识别

本 run 没有 magnitude-off control，也没有记录 magnitude drift、A/B 分组梯度或旋转距离。无法判断 magnitude 分支是否提供了有用尺度自由度、吸收了主要梯度，或者与旋转分支产生了优化竞争。

### 11.5 参数效率信号不足以晋级

使用 LoRA r32 约四分之一的参数而接近 seed42 LoRA，是本轮最有利的观察。但参数量接近的历史 random-B/full-gradient r8 明显更强，而 NoRA-DOFT 相对 base 几乎没有净增益。因此当前版本不应作为正结果继续扩大训练预算。

## 12. 证据等级与允许表述

本 run 的 full-bench 评测协议、records 和 postverify 是严格且可审计的；但训练没有预注册 contract、独立代码聚合哈希、optimizer state 和第二 training seed。因此建议登记为：

```text
Evaluation: verified, frozen-snapshot compatible
Training evidence: single-seed experimental candidate
Overall evidence role: B / negative ablation
```

允许表述：

- 在固定 50-step、单 training seed42 设置下，该 NoRA-DOFT v1 的 Macro Avg@k 为 `0.326175`。
- 它使用 `8.79M` 可训练参数，约为 matched LoRA r32 的 `23.8%`。
- 它没有显示相对 matched LoRA 或 no-RL base 的可信 held-out 增益。
- 当前结果支持继续研究 informed initialization 和 normalization/magnitude 消融，而不支持直接扩大当前配置。

不允许表述：

- “NoRA + DoRA + OFT 显著优于 LoRA/OFT”；
- “三种方法合并后稳定涨点”；
- “training-seed robust”或“跨任务普遍有效”；
- 将历史 OFT、RLPO、random-B 的跨协议点差写成严格因果比较；
- 将 `normalization=init` 描述为训练期间持续归一化。

## 13. 决策与下一轮设计

当前决策：

```text
不从 step50 继续当前配置
不补当前配置的 training seed43
不将当前配置列为正向主结果
保留 checkpoint、records、postverify 和本报告作为负结果
```

下一轮优先级：

1. 增加同协议、同 step50 的标准 OFT control，隔离三合一相对纯 OFT 的增量。
2. 做 `normalization={init,continuous} x magnitude={off,on}` 的 2x2 消融。
3. 用 gradient-probe 或 RLPO 风格输入子空间初始化 blockwise A，替换 seeded Gaussian。
4. 对比 `block16/rank8` 与当前 `block32/rank8`。
5. 记录 A/B/magnitude 分组梯度、A norm drift、Cayley rotation distance 和 magnitude drift。
6. 保存 optimizer state，并为每个正式 run 生成 training contract、代码聚合哈希和环境 manifest。
7. 先用较小 screen gate；只有 point estimate 和 paired evidence 均超过 matched low-budget baseline 的候选才进入第二 seed 和完整 full-bench。

## 14. 产物索引

### 14.1 代码与启动入口

- 方法实现：`verl/utils/peft_nora_doft.py`
- 模型接入：`verl/workers/config/model.py`
- FSDP 参数收集：`verl/utils/fsdp_utils.py`
- FSDP worker 接入：`verl/workers/fsdp_workers.py`
- 训练 launcher：`examples/verl_train/run_dapo_math_boxed_nora_doft_1p5b_4gpu_8k.sh`
- 物化 merge：`tina_run/scripts/local/eval/merge_nora_doft_wrapped_hf.py`
- 正式评测 controller：`scripts/local/run_nora_doft_step50_fullbench.sh`

### 14.2 训练与 checkpoint

- 训练日志：`runs/nora-doft-v1/logs/nora_doft_training.log`
- 训练曲线：`runs/nora-doft-v1/nora-doft-training-comparison.html`
- Step50 checkpoint：`runs/ckpts/verl/DAPO-Math-17k/nora_doft_r8b32_mag_init_lr1e6_b64m16n8_50_4gpu_boxed/global_step_50`
- 物化模型：`runs/nora-doft-v1/merged/step50/actor_hf_vllm`

### 14.3 评测与分析

- Summary：`runs/nora-doft-v1/outputs/full_bench_eval/nora_doft_r8b32_mag_init_lr1e6_b64m16n8_50_4gpu_boxed_step50_fullbench_32768_seed42/summary/nora_doft_r8b32_mag_init_lr1e6_b64m16n8_50_4gpu_boxed_step50_fullbench_32768_seed42.json`
- Records：`runs/nora-doft-v1/outputs/full_bench_eval/nora_doft_r8b32_mag_init_lr1e6_b64m16n8_50_4gpu_boxed_step50_fullbench_32768_seed42/records/nora_doft_r8b32_mag_init_lr1e6_b64m16n8_50_4gpu_boxed_step50_fullbench_32768_seed42.jsonl`
- Postverify：`runs/nora-doft-v1/analysis/evaluation_verification/nora_doft_r8b32_mag_init_lr1e6_b64m16n8_50_4gpu_boxed_step50_fullbench_32768_seed42.json`
- 对 LoRA seed42 的 paired bootstrap：`runs/nora-doft-v1/analysis/paired_comparisons/nora_doft_vs_matched_lora_seed42.json`
- 对 LoRA seed43 的 paired bootstrap：`runs/nora-doft-v1/analysis/paired_comparisons/nora_doft_vs_matched_lora_seed43.json`

## 15. 验证记录

2026-10-10 使用训练环境运行：

```bash
/home/wangls/miniconda3/envs/peft-for-rl/bin/python -m pytest -q \
  tests/test_nora_doft.py \
  tests/test_full_bench_eval.py
```

结果：

```text
23 passed in 6.78s
```

paired bootstrap 使用：

```bash
python scripts/analysis/compare_paired_full_bench.py \
  --baseline-records <matched-lora-records> \
  --candidate-records <nora-doft-records> \
  --output-prefix <comparison-output-prefix> \
  --bootstrap-samples 10000 \
  --seed 42
```

两组比较均确认 `record_keys_exactly_matched=true`、`num_problems=902`、`num_samples=7248`。
