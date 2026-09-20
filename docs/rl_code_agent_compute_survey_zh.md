# 代码与 Agent RL 任务的计算量调研

更新时间：2026-09-12

## 1. 结论

如果目标是在 `DeepSeek-R1-Distill-Qwen-1.5B`、4 张 RTX A6000 上，用数小时而不是约两天完成一次 PEFT 方法训练，那么优先级应当是：

1. **代码正式实验：难度过滤后的 DeepCoder 单轮代码生成。** 不照搬 7B/14B 的长输出配方，而是使用 100--200 step、16 prompts/step、8 rollouts/prompt、2048--3072 response tokens。预计单次约 2--8 小时，取决于是否跑到 200 step、实际输出长度和代码超时率。
2. **Agent 正式实验：RAGEN CoordSokoban。** 它是真正的多轮环境交互，环境是本地 Python，执行开销远低于搜索、浏览器和仓库级 Agent。先跑 100 step，必要时续到 200 step；预计约 2--8 小时。
3. **流程冒烟测试：Countdown。** 200 step 的最大生成预算只有当前数学实验的约 `0.9%`，预计 1--3 小时，但任务过于简单且仍偏算术，不适合作为主要跨领域结论。
4. **第二阶段才考虑：WebShop。** 它能代表真实的多轮 Agent，但重复长上下文 prefill 和 HTTP 环境开销明显；100--200 step 也可能需要 6--20 小时。
5. **首轮排除：SearchQA/HotpotQA、ALFWorld 官方长配方、SWE-Gym/SWE-bench Agent。** 它们的数据行数可能不多，但检索服务、长轨迹、容器重置和测试套件会主导墙钟时间。

最重要的实验设计要求不是“找最小数据集”，而是找一个能持续产生 **组内奖励方差** 的任务。对 GRPO 而言，全 0 或全 1 的 rollout group 都几乎不提供学习信号。因此，代码任务应先做一次共享的 base-model 难度筛选；这项成本只发生一次，之后所有 PEFT 方法复用同一训练清单。

## 2. 计算口径

单看训练集条数会严重误判 RL 周期。本文使用以下上界比较：

```text
训练生成量 C_out
  = trainer steps
  x 每步 prompt/environment groups
  x 每组 rollouts
  x 最大 turns
  x 每 turn 最大生成 tokens

总墙钟时间还应加入：
  + 多轮中不断增长上下文的重复 prefill
  + old/ref log-prob 与 actor backward/update
  + 代码编译、执行、超时和进程调度
  + 环境 step、检索、HTTP、容器 reset 和测试套件
  + validation rollout、保存 checkpoint 和诊断任务
```

对追加完整历史的多轮 Agent，第 `t` 轮会重新 prefill 前 `t-1` 轮内容。即使输出 token 总量不大，prefill 仍近似随 turn 数呈二次增长。因此，`5 x 400` 不能简单当成一次 `2000` token 的单轮生成。

### 2.1 本项目实测基线

当前 launcher：[`scripts/local/run_dapo_boxed_paper_clean_1p5b_4gpu.sh`](../scripts/local/run_dapo_boxed_paper_clean_1p5b_4gpu.sh)

完成日志：[`runs/logs/verl/rlpo_init_r32a64_b64m16n8_270_4gpu_boxed_v2.log`](../runs/logs/verl/rlpo_init_r32a64_b64m16n8_270_4gpu_boxed_v2.log)

| 项目 | 当前 DAPO-Math-17k 配置/实测值 |
|---|---:|
| Trainer steps | 270 |
| Prompts/step | 64 |
| Rollouts/prompt | 8 |
| Trajectories | 138,240 |
| 最大 response | 8,192 |
| 最大 completion 预算 | 1,132.46M tokens |
| 实测平均 response | 6,247 tokens |
| 实测 hit-max 比例 | 46.7% |
| 实测 completion 总量 | 863.63M tokens |
| 生成时间合计 | 23.56h |
| Old-log-prob 时间合计 | 5.97h |
| Actor update 时间合计 | 18.41h |
| Step 时间合计 | **47.96h** |
| 4-GPU 消耗 | **约 191.8 GPU-hours** |

下文把该次 47.96h 训练记为 `1.0x`。一个便于做初筛的模型侧估计是：

```text
候选模型侧小时数 ~= 候选最大生成 tokens / 863.63M x 47.96h
```

这是估算而不是承诺：短序列的固定开销、多轮 prefill、critic、有无 reference policy、过滤后保留多少轨迹，以及执行器等待都会改变结果。本文给出的最终时间区间已经在这一线性值上增加了相应余量。

## 3. 统一比较

表中的“训练输出上界”只计算训练 rollout，不默认包含 validation。`相对当前` 的分母是当前实验的最大 completion 预算 1,132.46M。

| 任务/公开配方 | Steps | Groups x rollouts | Turns x tokens | 训练轨迹 | 训练输出上界 | 相对当前 | 4xA6000 估计 | 判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 当前 DAPO 数学 | 270 | 64 x 8 | 1 x 8,192 | 138,240 | 1,132.46M | 1.000x | 实测 47.96h | 基线 |
| TDRM MATH L3 本地短配方 | 40 | 64 x 8 | 1 x 2,048 | 20,480 | 41.94M | 0.037x | 2--5h | 数学参考 |
| RAGEN Countdown | 200 | 8 x 16 | 1 x 400 | 25,600 | 10.24M | 0.009x | 1--3h | 只做 smoke test |
| RAGEN CoordSokoban | 200 | 8 x 16 | 5 x 400 | 25,600 | 51.20M | 0.045x | 3--8h | **Agent 首选** |
| RAGEN CoordFrozenLake | 200 | 8 x 16 | 5 x 400 | 25,600 | 51.20M | 0.045x | 3--7h | 容易过简单/受随机性影响 |
| RAGEN DeepCoder 官方 | 200 | 16 x 8 | 1 x 4,000 | 25,600 | 102.40M | 0.090x | 5--12h | 可用，但需降难度/长度 |
| DeepCoder 建议 pilot | 100 | 16 x 8 | 1 x 2,048 | 12,800 | 26.21M | 0.023x | **2--5h** | **代码首选起点** |
| DeepCoder 建议扩展 | 200 | 16 x 8 | 1 x 2,048 | 25,600 | 52.43M | 0.046x | **4--8h** | pilot 未收敛再续跑 |
| RAGEN WebShop release | 100/400 | 16 x 8 | 9 x 400 | 12,800/51,200 | 46.08M/184.32M | 0.041x/0.163x | 6--30h | 第二阶段 |
| RAGEN SearchQA | 200 | 16 x 8 | 5 x 400 | 25,600 | 51.20M | 0.045x | 8--20h+ | 需要检索 GPU，不选 |
| rLLM DeepCoder 全量一轮 | 约 1,518 | 16 x 4 | 1 x 16,384 | 约 97,148 | 约 1,591.67M | 1.406x | 明显超过当前目标 | 不照搬 |

算法口径还需单独说明：RAGEN 的 1.5B model-size Sokoban main-table 配方是 PPO/GAE，GRPO 的公开 DeepCoder 配方则使用 7B coder model。本文推荐的 1.5B Code/Sokoban 实验是为了保持本项目研究变量而做的 **GRPO 受控改写**，不是声称原论文已经在完全相同的模型和算法上报告了墙钟时间。GRPO 不需要 critic，理论上会比同 token 数的 PPO/GAE 稍省，但不能在实际 pilot 前把这部分直接当作确定加速比。

### 3.1 Validation 不是免费开销

RAGEN 默认 `val_before_train=True`，并通常每 10 step 验证一次：

- CoordSokoban 的默认验证是 512 environments x 1 rollout。200-step 训练若按 step 10、20、...、200 再加训练前验证，会额外产生 `21 x 512 = 10,752` 条验证轨迹；按 5 turns x 400 tokens，上界再增加 21.50M tokens，约为训练生成量的 42%。
- DeepCoder 配方使用 256 x 1 验证。相同频率下会增加 5,376 次代码生成和执行，最多再增加 21.50M tokens，约为训练生成量的 21%。
- WebShop release 使用 256 x 1 验证。400 step 加训练前验证时，验证输出上界约 37.79M tokens，且同样承担 9 轮环境与 prefill 开销。

为了比较 PEFT 方法，不需要照搬论文的高频大验证。建议固定 128 个 held-out seeds/problems，关闭 train-before validation，每 20 step 验证一次，并让所有方法共享同一验证清单和生成 seed。100-step pilot 只产生 5 次验证，即 640 条验证轨迹。

同时应关闭与主问题无关的 RAGEN mutual-information/collapse cross-scoring 和 gradient-analysis。它们适合分析 Agent collapse，但会增加额外 forward/log-prob 计算，并污染 PEFT 方法的纯训练时间对比。

## 4. 代码 RL 候选

### 4.1 首选：DeepCoder-Preview 的难度过滤子集

公开数据 [`agentica-org/DeepCoder-Preview-Dataset`](https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset) 共 24,287 条训练题：

| 子集 | 训练题数 | 特点 |
|---|---:|---|
| PrimeIntellect SYNTHETIC-1 verified | 16,252 | 题量大，适合抽候选池 |
| TACO Verified | 7,436 | 常用竞赛代码 RL 数据，难度跨度大 |
| LiveCodeBench v5 train | 599 | 新题但测试 payload 很大 |

测试集为 408 条 Codeforces 和 279 条 LiveCodeBench v5。数据卡称每题至少有 5 个测试用例。

两个公开训练实现给出了非常不同的计算预算：

- [RAGEN DeepCoder 配方](https://github.com/mll-lab-nu/RAGEN/blob/main/docs/experiment_deepcoder.md)：Qwen2.5-Coder-7B、8 GPU、GRPO、200 step、16 groups x 8 rollouts、4000 response tokens、single turn、mini-batch 32、TP=4。
- [rLLM DeepCoder 配方](https://github.com/rllm-org/rllm/blob/main/cookbooks/deepcoder/train_verl.sh)：Qwen3-4B、8 GPU、LoRA rank 32、batch 16、rollout n=4、16384 response tokens、全数据 1 epoch。按 24,287 条数据计算约 1,518 step 和 1.59B 最大生成 tokens，并不是短周期方案。

RAGEN 的 reward 会在一个 Python 子进程里执行该 solution 的全部测试，默认整份 solution 超时 3 秒，并返回 `passed_tests / total_tests` 的连续分数。200-step 配方需要 25,600 次 solution execution；32 路并发且全部超时时，理想下界是：

```text
25,600 executions x 3s / 32 workers = 2,400s = 0.67h
```

这只是执行阶段的理想并发下界。进程启动、CPU 争用、长测试、失败重试和任务尾部效应会使实际时间更高。当前机器有 64 个物理 CPU 核，32 executor workers 是合理起点。

#### 为什么必须先筛难度

当前模型不是 Qwen2.5-Coder-7B，而是偏数学推理的 1.5B distilled model。直接在 TACO/Codeforces 难题上跑，很可能一个 group 的 8 条 rollout 全部失败，GRPO 得到零方差；反过来，太简单的题会全部通过，同样没有信号。

建议的共享筛选流程：

1. 从 PrimeIntellect 和 TACO 分层抽 1,500--2,000 题；首轮不下载/使用体积很大的 LCBv5 train payload。
2. 用 base model、2048 token、temperature 1.0 做 pass@4 低成本扫描。
3. 保留 4 次中通过 1--3 次的题，并补少量全错/全对题用于覆盖边界。
4. 对入选题在正式训练的前 5--10 step 记录 pass@8 分布；目标是相当一部分 group 位于 `1/8` 到 `7/8`，而不是只看 batch mean reward。
5. 筛选得到的 problem ID、测试版本、prompt 模板和 seed 固化，所有 PEFT 方法完全复用。

扫描 2,000 题 x 4 rollouts x 2,048 tokens 的最大输出量是 16.38M tokens。它是一笔约 1--3 小时的一次性共享成本，不能对每个 PEFT 方法重复执行。

#### 建议正式配置

```yaml
model: DeepSeek-R1-Distill-Qwen-1.5B
algorithm: GRPO
trainer_steps: 100              # 尚未进入平台期则从 checkpoint 续到 200
prompt_groups_per_step: 16
rollouts_per_group: 8
max_response_length: 2048       # truncation 偏高时升到 3072
turns: 1
optimizer_minibatches_per_step: 4  # 语义目标，实际 Hydra 字段依训练栈而定
reward: passed_tests / total_tests
executor_timeout_per_solution: 3s
executor_workers: 32
validation: 128 fixed problems every 20 steps
online_rollout_filter: disabled
```

上面的 `optimizer_minibatches_per_step` 表示实验语义，不是现有 Hydra 的字段名；落到本仓库时应设置相应的 prompt mini-batch，使每个 trainer step 约有 4 个 optimizer mini-batches，并以日志中的实际 update 次数为准。

这里不建议一开始就使用在线 reward-variance top-p filtering。我们的研究变量是 PEFT 方法，在线过滤会改变每一步实际参与 backward 的 token 数量。先通过离线难度筛选控制信号密度，能够让各方法的训练样本和 optimizer update 数更可比。

#### 安全与接入

本仓库已经识别 `codecontests/apps/codeforces/taco` 数据源，并包含 `prime_code` reward、SandboxFusion reward 和 `sandbox_fusion.max_concurrent=64` 配置。因此单轮代码 RL 的接入难度是中等，而不是从零实现。

但是本地 `prime_code/testing_util.py` 明确说明它**不是安全沙箱**；RAGEN 的 subprocess 实现也只是进程隔离。正式跑模型生成代码时，应部署 [SandboxFusion](https://github.com/bytedance/SandboxFusion) 或等价的容器级隔离，并限制 CPU、内存、网络和文件系统。还需要确认当前 SandboxFusion 路径是“每份 solution 一次执行全部 tests”还是“每个 test 单独请求”：后者的执行上界应再乘平均测试数，不能使用上面的 0.67h 估算。

### 4.2 APPS、CodeContests、TACO 单独使用

这些都是常见代码生成/代码 RL 数据源，但“数据集常用”并不等于“有一套适合本项目的短 GRPO 官方配方”。

- **APPS**：经典 train split 约 5K 题，难度标签清楚，适合构造 introductory/interview 子集；但较早的 CodeRL 工作不是当前的 group-relative online RL 配方。
- **CodeContests**：竞赛题和测试完善，但 1.5B 非 coder model 容易产生大量全 0 group。
- **TACO Verified**：已包含在 DeepCoder-Preview，直接复用 DeepCoder 的 schema/executor 比再造一条数据管线更合理。

因此首轮不建议分别建立三个 benchmark。使用 DeepCoder schema，训练池先取 PrimeIntellect + TACO，并以 base pass@4/pass@8 控制难度即可。

## 5. Agent RL 候选

### 5.1 首选：RAGEN CoordSokoban

[RAGEN](https://github.com/mll-lab-nu/RAGEN) 基于 verl，公开支持 Sokoban、FrozenLake、Countdown、WebShop、DeepCoder、SearchQA、Lean、Bandit、Sudoku 等环境。

基础配置的关键量是：

```yaml
model: Qwen2.5-3B-Instruct       # main-table 也覆盖 Qwen2.5-1.5B
trainer_steps: 200              # base default；main-table 常用 400
train_env_groups: 8
group_size: 16
max_turn: 5
max_actions_per_turn: 2
response_length_per_turn: 400
ppo_mini_batch_size: 32
```

CoordSokoban 是 6x6、1 box、最多 10 actions 的本地程序化环境。它没有“只有多少条训练数据”的约束，环境 seed 决定题面；训练周期由 step、group、rollout、turn 和 length 直接控制。相同环境 seed 在 group 内复用，适合计算 group-relative advantage。

它比 FrozenLake 更适合 PEFT 正式比较：

- Sokoban 需要状态理解、规划和失败后的多轮调整，任务不是单纯格式遵循。
- CoordFrozenLake 的 main-table 配置可设成确定性 `success_rate=1.0`，对 1.5B 可能过简单；若增加滑动随机性，组内 reward 方差又混入环境随机性，容易掩盖 PEFT 方法差异。
- Sokoban 的 `step()` 是本地 Python 状态更新，和代码编译、检索服务、浏览器或 Docker 相比几乎可以忽略。

建议先忠实保留 RAGEN 的 8 groups x 16 rollouts，因为 16 个同题样本能更稳定地估计组内方差；所有 PEFT 方法使用相同 seed 列表。为控制周期，先跑 100 step，观察 success、pass@16、组内 reward std 和 response length，再从同一 checkpoint 续到 200 step。

要注意，RAGEN 使用自己的 environment state manager/StarPO 数据路径。当前仓库虽已有 multi-turn `AgentLoop`、tool abstraction、MCP、reward loop，但没有现成 Sokoban environment adapter。接入方式有两种：

1. 把 CoordSokoban 环境和 action parser 接到当前 verl AgentLoop，保留本项目全部 PEFT 实现。这更符合当前研究主线。
2. 把本项目 PEFT 方法移入 RAGEN fork。它能更快复现环境，但会同时改变 verl 版本和训练栈，方法间结果更难和现有数学实验对齐。

建议采用第一种，并先只接标准 LoRA 跑 10-step smoke test。确认每轮 observation/action、response mask、reward placement 和 advantage grouping 后，再展开其他 PEFT 方法。

### 5.2 FrozenLake 与 Countdown

- **FrozenLake**：计算量与 Sokoban 相同，可作为多轮环境管线的第二个检查任务。正式比较应优先用确定性转移，避免不同方法碰到不同随机滑动结果；但确定性版本可能很快饱和。
- **Countdown**：RAGEN 配方是 single turn、single action、每次最多 400 tokens。200 step 只有 10.24M 最大输出 tokens，非常便宜。TinyZero 的原始公开配方并不短：默认数据处理取 327,680 条训练题，batch 256、1024 response、15 epochs，约 19,200 trainer steps。RAGEN 的固定 200-step 版本才适合作为 smoke test。

Countdown 不应成为跨领域主结果，因为它本质仍是算术，而且 distilled reasoning model 可能一开始就全对；TinyZero 也明确报告 Qwen2.5-0.5B base 未学出推理，而 3B 才表现出复杂 reasoning emergence，说明小模型上的信号并不稳健。

### 5.3 WebShop：真实但不是短周期首选

RAGEN WebShop release 配方使用 Qwen2.5-3B-Instruct、GRPO、16 groups x 8 rollouts、最多 9 turns；release runner 默认 100 step，而复现实验命令使用 400 step。每一轮还会把商品页面、可点击项和历史放入上下文。

100-step 的输出上界 46.08M 看起来与短任务相近，但它低估了：

- 九轮不断增长的 prompt prefill；
- 本地/HTTP 商品检索与页面 step；
- 失败轨迹通常走满 turn budget；
- 高频 validation 同样要跑完整购物轨迹。

所以 WebShop 更合理的定位是：Sokoban 证明 PEFT 方法在多轮 Agent 上有效之后，再用 100-step LoRA pilot 验证可迁移性。不要第一轮就把全部 PEFT 方法搬过去。

### 5.4 SearchQA/HotpotQA：GPU 和检索吞吐成为瓶颈

RAGEN SearchQA 的训练生成上界只有 51.20M tokens，但配套设施包括约 74GB 的 Wikipedia/FAISS index、约 21M passages 和 E5-base-v2 retrieval server。官方文档明确建议使用独立 GPU，并明确不建议 CPU retrieval，因为每步数百环境会发出大量并发请求。

在 4-GPU 训练预算下，分出一张卡给 retrieval 会把模型训练缩到 3 GPU；若与 vLLM 共卡又有 OOM 风险。因此它不是当前机器上的短周期任务。Agent-R1 的 HotpotQA 新配方也需要本地 BGE/FAISS，并使用 5 steps、每步 1024 response、32 prompts x 8 rollouts 和 5 epochs，预算同样不小。

### 5.5 ALFWorld：环境轻于 SWE，但公开训练 horizon 太长

Agent-R1 当前 ALFWorld GRPO 脚本使用 Qwen3-4B、4 GPU、16 prompts x 8 rollouts、最多 20 steps、每 step 最大 4096 response tokens、10 epochs。仅一个 trainer step 的生成上界就是：

```text
16 x 8 x 20 x 4096 = 10.49M tokens/step
```

实际动作通常远短于 4096，所以上界很松；但 20 次重复 prefill、TextWorld 环境和 10 个数据 epoch 仍使它不可能天然成为几小时实验。若以后采用，应把 action response 限制到 128--256 tokens、max steps 降到 10，并改用明确的 fixed trainer steps，而不是照搬 10 epochs。

OpenManus-RL 虽然提供了 Qwen2.5-1.5B WebShop 示例，但它是 PPO、batch 128、15 environment steps、512 response tokens、150 epochs、2 GPU，而且数据路径还是 TODO，无法从公开脚本推导可复现的总 trainer steps。它可参考环境封装，不应作为计算量基线。

### 5.6 SWE-Gym/SWE-bench：行数少但单条轨迹昂贵

[SWE-Gym](https://github.com/SWE-Gym/SWE-Gym) 有约 2.4K 个真实软件工程任务，Lite split 只有 234 条。论文基线提到少于 500 条成功 agent-environment trajectories，但那主要是 SFT/拒绝采样数据，不等于 500 次廉价在线 GRPO rollout。

它的每条在线轨迹需要：

- 创建或恢复任务专属 Docker/repository 状态；
- 最多约 50 个 Agent iteration；
- 浏览和编辑多文件、应用 patch；
- 运行项目测试并等待超时；
- 为同一问题生成多个 rollout 才能得到 group-relative signal。

因此 234 条 SWE-Gym Lite 也可能需要数十到数百小时。它适合后期做 50--100 题的 agent evaluation 或 rejection-sampling 研究，不适合当前“每种 PEFT 数小时”的在线 RL 主实验。

## 6. 推荐实施顺序

### 阶段 A：一次性的可行性校准

1. **代码**：抽 1,500--2,000 条 PrimeIntellect/TACO，做 pass@4 扫描并固定 mixed-success 清单。
2. **Agent**：实现 CoordSokoban adapter，用标准 LoRA 跑 10 step，8 x 16、5 x 400。
3. 两条线都记录生成、old-log-prob、update、reward/environment、validation 五类独立时间，不只记录总 step 时间。

### 阶段 B：标准 LoRA pilot

| 线 | 配置 | 停止/续跑判断 |
|---|---|---|
| Code | 100 step, 16 x 8, 2048, single-turn | 若 reward/pass@1 已进入平台期则停止；仍稳定上升则续到 200 |
| Agent | 100 step, 8 x 16, 5 x 400 | 若 success/pass@16 已进入平台期则停止；否则续到 200 |

还应设资源保护条件：代码 timeout ratio 持续过高、有效 mixed groups 低于约 20%、response hit-max 过高，或 Agent 无效 action 比例持续过高时，暂停并修正数据/长度/解析器，而不是盲目补 step。

### 阶段 C：展开 PEFT 方法

只有标准 LoRA pilot 同时满足“能学习”和“单次周期可接受”后，才把同一配置复制到 AdaLoRA、RLPO、GeoRA 等方法。所有方法必须固定：

- base checkpoint 与 tokenizer；
- train/validation problem IDs 或 environment seeds；
- group size、rollout seeds、temperature 和 length；
- reward、代码 tests、executor timeout 和环境版本；
- trainer steps、每 step optimizer updates、validation 频率；
- 是否过滤 rollout，以及过滤后实际保留的 tokens/trajectories。

数学 checkpoint 的最终分数不能直接和代码/Agent checkpoint 比较。新的领域实验必须至少重跑一个标准 LoRA 对照，并比较该领域内相同 wall-clock、token budget 或 optimizer-update budget 下的学习曲线。

## 7. 最终选择

若只新增一个跨领域任务，选择 **DeepCoder 难度过滤子集**：它最接近当前单轮 RLVR 数据流，本仓库已有代码 reward/SandboxFusion 接口，改造量最小，也能检验 PEFT 方法是否从数学泛化到程序合成。

若希望论文中有一个真正的 Agent 结论，再增加 **CoordSokoban**：它的环境成本最低、计算预算可控，并且确实需要多轮观察、动作和状态转移。WebShop、SearchQA 和 SWE-Gym 应放到这两条线稳定之后。

## 8. 主要来源

- RAGEN repository and base config: <https://github.com/mll-lab-nu/RAGEN>
- RAGEN DeepCoder recipe: <https://github.com/mll-lab-nu/RAGEN/blob/main/docs/experiment_deepcoder.md>
- RAGEN WebShop release recipe: <https://github.com/mll-lab-nu/RAGEN/blob/main/docs/experiment_webshop_release.md>
- RAGEN SearchQA recipe: <https://github.com/mll-lab-nu/RAGEN/blob/main/docs/experiment_search.md>
- DeepCoder-Preview dataset: <https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset>
- rLLM DeepCoder cookbook: <https://github.com/rllm-org/rllm/tree/main/cookbooks/deepcoder>
- TinyZero: <https://github.com/Jiayi-Pan/TinyZero>
- Agent-R1: <https://github.com/AgentR1/Agent-R1>
- OpenManus-RL: <https://github.com/OpenManus/OpenManus-RL>
- SWE-Gym repository: <https://github.com/SWE-Gym/SWE-Gym>
- SWE-Gym paper: <https://arxiv.org/abs/2412.21139>
