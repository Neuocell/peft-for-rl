# REAL reward-gradient 探针：实现与已有设计对照

日期：2026-10-13。来源：用户提供的 `/root/1683_Reward_Gradient_Aligned_L.pdf`，
《Reward-Gradient Aligned Low-Rank Adaptation for Post-Training》，ICLR 2027 匿名投稿版本。
本文对应论文 §3.2–3.4（pp.5–6）、Appendix B（p.16）和 E/G（pp.17–20）。
状态：工程实现与数值验证；不包含新的模型训练或 benchmark 结果。

## 1. 实现的数学定义

对同一个 prompt 的 K 条回答，使用原始二元正确性 reward `R_i∈{0,1}`，
只接收至少两条正确、两条错误且每条回答至少有一个有效 token 的组。
使用 population standard deviation（`correction=0`）：

```text
A_i = (R_i - mean(R)) / std_population(R)
J_p = (1/K) Σ_i A_i [Σ_t mask_it log π(o_it | q,o_i,<t) / L_i]
G_p = ∇W J_p
    = sqrt(n_positive * n_negative) / K * (S_positive - S_negative)
L_i = Σ_t mask_it
```

这里先对每条 response 按长度归一化，再按 K 平均。不能替换成
`Σ_i,t A_i logπ_it / Σ_i L_i`，也不能直接复用训练器已计算的 advantage：
本项目的 GRPO standard deviation、epsilon 或 reward shaping 配置可能与论文不同。
代码最小化 `-J_p`；负号不影响后续二阶能量。

REAL 选择共享输入子空间 P：

```text
C = mean_p G_p.T G_p                  # 未中心化二阶矩
maximize trace(P.T C P), P.T P = I
```

`mean(G).T mean(G)` 与这个目标不同；`mean((G-mean(G)).T(G-mean(G)))`
也不同。例如 `G_2=-G_1` 时，均值梯度消失，但 REAL 保留两组的方向能量。
这保留了各组可使用的输入方向，不保证它们在同一个共享 minibatch 更新中都获得改进。

每个模块、每个 prompt 独立抽样 `Ψ_p∈R^(d_out×s_out)`，元素方差为 `1/s_out`；
输入侧 `Ω∈R^(d_in×s_in)` 使用标准高斯并在全部 prompt 间保持不变：

```text
H_p = Ψ_p.T G_p
Y = mean_p H_p.T (H_p Ω)
M = sym(Ω.T Y)
F = Y M^(†/2)
P = top-r left singular vectors of F
```

`E[Ψ_p Ψ_p.T]=I` 使二阶矩估计无偏；由此得到的有限宽度 Nyström 子空间本身
不具有“无偏子空间”的保证。使用相对谱阈值处理伪逆，显式拒绝不足 r 维的数值秩，
避免把未观测到的随机方向混进声称由 reward 选出的子空间。

## 2. 代码与接入方式

- `verl/utils/peft_real.py`：reward 系数、流式 Nyström、探针初始化、FSDP 收集和 artifact 导出。
- `verl/workers/actor/dp_actor.py`：每个完整 prompt group 的无更新 backward。
- `verl/trainer/ppo/ray_trainer.py`：完整组筛选、跨 batch 去重、复制到各 data-parallel actor。
- `verl/workers/fsdp_workers.py`：`peft_type=real_probe` 和 rollout/actor 接线。
- `verl/utils/real_probe_mode.py`：确定性观察模式，保留当前 Qwen2 decoder 的梯度检查点。
- `tests/test_real_probe.py`、`tests/test_real_probe_integration.py`：公式和执行路径验证。

探针不对 base weight 求导或收集完整 G。探测阶段临时安装标准 LoRA：

```text
A_probe = 0, B_probe = Ψ_p
grad(A_probe) / scaling = -Ψ_p.T G_p
ΔW_probe = 0
```

冻结 base 和 `B_probe`，只对 `A_probe` 求导。一个组内各条 response 逐条 backward，
先完整累计 H，再计算 `H.T(HΩ)`，最后刷新 B。不能在每条 response 后平方或刷新 B，
否则会丢掉组内交叉项并改变论文目标。全程不调用 optimizer step、不做 gradient clipping。
导出的 A 为 `P.T`，ν=1；正式训练重建 adapter 并置 B=0。

所有 DP actor 处理相同 prompt、相同随机投影，FSDP 的平均归约不会混入其他组。
逐 FSDP leaf 收集 A 梯度并释放完整参数视图，rank 0 累计 CPU 统计、写入产物。
这优先复用已有安全收集路径，尚未实现不同 DP rank 独立处理不同 prompt 的吞吐优化。
当前接入针对 FSDP1、`use_orig_params=True`、sequence parallel size 1。

论文的四个 sketch buffer 理论大小为
`s_out*(d_out+d_in) + 2*s_in*d_in` 个元素；实际实现还包含 LoRA 参数、
梯度、CPU 统计与分解临时量。该公式不包含模型激活、FSDP all-gather、vLLM KV cache，
不能据此宣称整卡显存下降相同比例。观察模式关闭 attention/dropout 的随机性，并在当前
Transformers 的 Qwen2 decoder 上保留梯度检查点；用非零 attention dropout 的微型
Qwen2 测试验证 eval logits、梯度一致及 backward 重计算。其他模型和旧版 Transformers
回退为 eval forward，可能关闭检查点；迁移到 Qwen3 等架构时需单独适配并验证。

双卡测试还确认：当前 FSDP1 将冻结 base/B 和可训练 A 包在同一 flat leaf 时，
完整梯度视图可能为冻结参数提供全零 buffer。因此这里不计算或收集 dense base dW，
但 **FSDP 的总 gradient storage 仍可能包含冻结 base 的空间**；只有子空间估计器的
持久缓冲满足上述线性界。若进一步追求论文级整条路径的内存节省，需要按 trainability
拆分 FSDP wrapping，并相应改写跨 leaf 的 A 梯度收集和 B 投影刷新，再测实际峰值。

产物复用现有初始化接口：

```text
rank_map.json           # 每个模块的 rank/alpha
subspaces.safetensors  # 每个模块 [rank, d_in] 的正交 A
summary.json           # 观察预算、有效组、随机种子、数值诊断和参数成本
```

训练用 `peft_type=grad_subspace`、`lora_freeze_a=True`。固定 A 仍随标准 PEFT adapter
保存，推理必须同时加载 A 和 B。B-only 参数量为 `Σ_m r*d_out,m`，A+B 为
`Σ_m r*(d_in,m+d_out,m)`；矩形 attention/MLP 模块不能笼统按“一半”计算。

## 3. 启动与预算

先激活项目环境：

```bash
conda activate peft-for-rl
cd /root/peft-for-rl
bash scripts/local/start_real_gradient_probe_4gpu.sh
```

默认 `DRY_RUN=1` 只做资源预检。实际执行观察：

```bash
DRY_RUN=0 REAL_PROBE_RANK=32 REAL_PROBE_STEPS=64 \
  bash scripts/local/start_real_gradient_probe_4gpu.sh
```

默认 rank32、`s_out=s_in=128`、64 prompts/step、K=8、seed42；64 个观察 step
最多尝试 4,096 个 prompt，只有满足筛选条件的组进入二阶矩。
论文主实验列出 global batch128 和每题K=8；若128指每轮独立prompt数，64个观测step
折算为8,192个prompt。PDF未提供TRL的gradient accumulation、generation batch及
minibatch配置，也未明确global batch是否按展开后的response计数，因此这一数字是
条件推算，不能当作已核实的作者实际prompt数。其Qwen3 rank16、distilled模型rank64。
本项目默认配置对齐本地资源和rank32基线，**不是完整论文实验复现**。
`REAL_PROBE_OUTPUT_WIDTH` 和 `REAL_PROBE_INPUT_WIDTH` 可分别覆盖，默认都是 4r。

调试可显式设置有效组上限，达到上限提前导出：

```bash
DRY_RUN=0 REAL_PROBE_STEPS=4 REAL_PROBE_PROMPTS=16 \
  EXP_NAME=real_probe_debug_r32_p16 \
  bash scripts/local/start_real_gradient_probe_4gpu.sh
```

`REAL_PROBE_PROMPTS=0` 表示没有有效组上限；按观察 step 结束。如果指定了组上限，
它仍是上限而非“保证能采够”的要求；实际组数与结束条件应从 summary 读取。
小预算只适合工程验证，不足以判断论文效果。无有效组或数值秩不足时应直接报错。
论文称观察占最初 64 个 training step；本文两阶段的 64 个观察 step 加后续 optimizer
step 是分别计数的，不应直接当作论文 600/300 step 的等总预算复现。

产物生成后，固定 A 的工程预检入口为：

```bash
GRADIENT_PROBE_ARTIFACT_DIR=runs/analysis/real_probe_r32_s64_p0_seed42 \
  bash scripts/local/start_real_fixed_a_4gpu.sh
```

这两个入口用于观察和实现调试。2026-10-13 已将 `METHOD=real` 注册到冻结协议，
正式训练使用 `REAL_ARTIFACT_DIR=<完整产物目录>` 配合
`scripts/local/run_rlvr_protocol_1p5b_4gpu.sh`，固定 alpha64、训练 token-mean 和全链路
数据顺序，并验证、记录初始化文件哈希。低层调试入口不自动构成协议内对照。
这里保留本地 GRPO 训练设置，不复现论文 GSPO；观察阶段不改变正式训练 sampler 起点。
长度奖励属于论文另一条实验分支，本实现只实现二元正确性探针。

## 4. 与本项目已有方案的实质差异

| 方案 | 统计信号和单位 | 候选目标 | 与 REAL 的关系 |
| --- | --- | --- | --- |
| 原始 random-B / GradTop | actor mini-batch 的聚合梯度 sketch | 累计 sketch 能量、增量截断 | 已有零输出随机 B 技巧；REAL 改为完整 prompt 组再平方，且增加输入 Nyström |
| Stable-SNR v2/v3 | 固定 B 的多 batch 窗口，估计均值/残差 | 信号对噪声的广义特征方向 | REAL 不要求跨组符号一致；保留相反 output correction 共享的输入方向 |
| Dominant atoms | 窗口局部主方向、跨窗口复现 | recurrence 加权局部 projector | 复现权重是额外先验；REAL 使用原始组梯度二阶能量 |
| SPAR v0 | 正确 response 的 CE | 正样本候选和稳定性校准 | REAL 同时用正确/错误回答，是 reward contrast；SPAR 已有廉价 held-out 投影评分 |
| Full-gradient mean / gain-LCB | 每 prompt 完整 G，再算均值/校准增益 | 一致更新方向或一阶 gain | REAL 保留被平均抵消的组级方向，但未提供未来更新收益的 LCB |
| Full-gradient uncentered covariance | 每 prompt 完整 G、输入侧 Nyström | `E[G.T G]` | 与 REAL 最接近；主要差别是输出压缩、response 权重、筛选及 A 是否冻结 |
| Phase-1 P2/I8/I32 | token mask、centered covariance、signal/random 配比 | 去均值后的变化方向和混合子空间 | 同时改变多项定义，不能把旧成绩当 REAL 消融 |
| Windowed Adam consensus | 窗口梯度、虚拟 Adam、局部 atom | 动量/预条件更新的复现方向 | 更贴近优化器但并非 LoRA factor Adam 的等价更新；应独立验证 |

旧 full-gradient 代码支持不同 covariance estimator，比较时应读取实际 artifact/config，
不能仅凭“covariance”名称认定是同一统计量。历史方法定义见
`full_gradient_rl_probe_v1_zh.md`、`verl_rl_gradient_subspace_stable_snr_v2_zh.md`、
`verl_rl_dominant_gradient_atom_probe_zh.md`、`spar_lora_v0_static_zh.md`、
`windowed_adam_consensus_probe_zh.md`。

## 5. 优先改进方向与可证伪实验

1. **先统一 prompt 单位和 loss，再比较压缩。** 用相同缓存的完整 rollout groups，
   相同 Eq.6 loss，比较 full-gradient uncentered oracle 与 REAL 的 2r/4r/8r 输出宽度，
   记录 held-out capture、主角度、时间和实测显存。不能先换 loss 再把差异归因于 Nyström。
   反向计算可重放相同 rollout；独立 Ψ 的 H 不能直接跨组做均值来重建 signed mean gradient。
2. **把已有独立验证接到 REAL 上。** Discovery 只选 P，calibration 选超参数，audit
   只报告最终质量。固定候选 A、B=0 时 `grad_B/scaling=G P`，无需完整 G 就能计算
   投影能量；分母可用独立输出 sketch 估计或小规模 full-gradient audit。
   若根据 audit 选优，它就变成 calibration，必须另留最终 audit。
3. **不要把所有梯度方差称为 reward 信号。** 未中心化二阶矩同时包含稳定均值、
   prompt 异质性和 rollout 噪声。相反方向确实可能有用，但高能量也可能来自高方差噪声。
   可对同一 prompt 独立采两组 rollout，研究交叉二阶统计/收缩；交叉统计不一定 PSD，
   需要定义对称化和负谱处理，并用独立 audit 验证，不能直接送进 PSD Nyström。
4. **先做 fixed-A 对照，再引入自适应 rank。** 同一个 REAL artifact 比较 fixed-A 与
   trainable-A；另设 random orthonormal fixed-A 对照。保持 rank、有效 scaling、
   rollout 和 seed 一致，分别报告可训练参数与采样/观察成本。论文 G.3 的 fixed-A
   相对 trainable-A 只高 0.15 个百分点，缺多 seed 显著性，不能据此认定冻结必胜。
5. **区分论文目标和本地训练目标。** Eq.6 的 sequence-mean 与冻结协议的 token-mean
   对长答案权重不同。先保留论文定义作为 REAL baseline；再单独消融使用本地真实
   policy-gradient reduction 的探针，按模型最终准确率和输出长度判断目标失配的影响。
6. **最后再做 rank allocation 或 signal/random 混合。** 旧实验中 atom 排序变化解释了
   大部分 capture 提升；相同 `P*R` 排序下 uniform/adaptive 的 audit capture
   为 0.83815/0.83841，差异很小。应先固定 basis/atom 排序再比较 rank，并按 B-only
   的 `r*d_out` 成本计算 fixed-A 预算，不能沿用 A+B 成本。

样本预算必须同时报告 attempted prompts、accepted prompts、rollout tokens、backward
tokens 和 wall time。`2≤n_positive≤K-2` 倾向于中等难度题，选择出的空间对全对/全错组
不提供直接保证；若扩展到至少一正一负，这是独立方法消融。

论文 Theorem 1 针对普通梯度上升递推证明 A 的旋转二阶小，不能直接用作 Adam 下的严格
保证。项目已有 A 漂移观测支持开展 fixed-A 实验，但不能替代 matched 训练证据。

已有成绩和失败记录统一以 `experiments/method_and_results_registry_zh.md` 为准：
Stable-SNR v3 fixed-A 因 held-out capture gate 未进入正式训练；I8/I32 的排序跨 seed
反转。两者都支持保留独立审计与多 seed 复现，不支持宣称 REAL 或任何改进已优于旧方法。

## 6. 本次验证记录

在项目 `peft-for-rl` conda 环境运行，CPU 核心/执行路径及已有功能回归共 **79 passed**：

```bash
PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m pytest -q \
  tests/test_real_probe.py tests/test_real_probe_integration.py \
  tests/test_real_probe_mode.py tests/test_gradient_subspace_init.py \
  tests/test_full_gradient_rl_probe.py
```

另实际运行了双 GPU 的微型 FSDP1 + bf16 测试，**1 passed**：

```bash
RUN_REAL_FSDP_TEST=1 PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m pytest -q tests/test_real_probe_fsdp.py
```

该测试默认跳过，显式启用才使用 GPU。它检查每组梯度在各 rank 一致、手工计算 Y
与 collector 一致、随机投影刷新一致、冻结参数的全零梯度视图、基座权重保持不变、
实际 rollout merge/unmerge 及 rank 0 产物导出/完成状态广播。
微型测试的显存不代表 1.5B/8K 探测的显存需求。

观察 launcher 的资源预检通过，修改文件的 Python 编译、shell 语法和 diff 空白检查通过。
本次未运行 1.5B 模型的完整观察阶段、后续 RLVR 训练或 benchmark；上述验证证明算法
与工程路径的已测行为，不证明新方法的准确率收益。


## 6. 保存观测统计，离线复用更小预算

`scripts/local/real_probe_cache.py` 提供可选的 CPU 累计统计快照与离线子空间导出。
快照保存 `Y_sum=sum_p H_p.T(H_p Omega)`、共享 `Omega`、各层计数/能量、prompt IDs、
投影种子和宽度，并校验文件 SHA256。已保存的前缀可以直接离线分解，不再生成 rollout
或执行 backward；仅最终 A 不能恢复这些累计统计，也不能从总和拆出未保存的早期前缀。

2026-10-13 的运行在 rank0 现有 worker RPC 的任务边界保存了前596个有效组，并在
该 collector 实例的 `finish_trainer_step` 返回后安装32/48/64步可选快照。缓存路径：
`runs/analysis/real_probe_r32_s64_p0_seed42_20261013_cache/`。原队列的模型/算法/配置文件
没有改动；快照写盘失败单独记录，不中断原实验。该缓存不包含dataloader/RNG，不能用于
完整恢复观测进程。

已保存的预算点可这样重建（确认对应快照已存在）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m scripts.local.real_probe_cache export   --snapshot runs/analysis/real_probe_r32_s64_p0_seed42_20261013_cache/step_032   --output-dir runs/analysis/real_probe_r32_s32_seed42_replayed
```

更小预算必须使用独立消融定义；当前正式 REAL 协议仍要求64步，不能将32步产物改写
metadata伪装成64步。快照保留原reward、筛选和投影定义，不能用于事后重算其他reward
或任意prompt子集。当前rank32/128宽度，每份Y约248.5 MiB，共享Omega另占248.5 MiB；
保存当前及32/48/64四个预算点约1.21 GiB。逐题保存H则会增长到数百GiB。
