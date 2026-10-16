# NoRA / DoRA 初始化依据与版本

2026-10-16 核验。新训练默认 `peft_nora_reference_v2`，历史定义为 `legacy_v1`。
这次对齐初始化和 NoRA normalization 公式；rank、alpha、LR、数据及训练 seed 保持各实验登记值。

## 可靠来源

| 来源 | 固定版本 | 支持的实现决定 |
| --- | --- | --- |
| [DoRA，ICML 2024](https://proceedings.mlr.press/v235/liu24bn.html)；[原文 §4.1、§4.3](https://arxiv.org/html/2402.09353v2) | 论文 v2 | magnitude/direction 分解；初始化以基座范数作为 magnitude；§4.3 保留 detached denominator |
| [NVlabs/DoRA 原始实现](https://github.com/NVlabs/DoRA/blob/7e2f10abbe8efe212c8fca1d983ae1d04ef13a18/commonsense_reasoning/peft/src/peft/tuners/dora.py) | `7e2f10abbe8efe212c8fca1d983ae1d04ef13a18` | `reset_parameters`：Kaiming-uniform A、零 B；模块替换时 magnitude 为基座逐行范数 |
| [Normalized Low-Rank Adaptation](https://arxiv.org/html/2608.31036v1) | arXiv v1，2026 预印本 | §3.1 Eq.7/9/11：rank 维 column normalization，continuous 与 init-only；Table 3 讨论多种初始化分布 |
| [NoRA 作者实现 layer.py](https://github.com/Joluck/NoRA/blob/4669c35ce0ffd027456272ee5050c0f58ac8d22a/peft/src/peft/tuners/lora/layer.py#L249)；[variants.py](https://github.com/Joluck/NoRA/blob/4669c35ce0ffd027456272ee5050c0f58ac8d22a/peft/src/peft/tuners/lora/variants.py#L133) | `4669c35ce0ffd027456272ee5050c0f58ac8d22a` | 默认 Kaiming A、零 B；只有 init 模式初始化归一化；continuous 在前向归一化且保留梯度 |
| [PEFT layer.py](https://github.com/huggingface/peft/blob/ba6a19060d6ab54a87538a6e77e3e4d5a907375b/src/peft/tuners/lora/layer.py)；[dora.py](https://github.com/huggingface/peft/blob/ba6a19060d6ab54a87538a6e77e3e4d5a907375b/src/peft/tuners/lora/dora.py) | v0.19.1，`ba6a19060d6ab54a87538a6e77e3e4d5a907375b` | 默认 LoRA A/B reset 及 DoRA magnitude 初始化；NoRA fork 的 dora.py 与此版本逐字节相同 |

作者归属链为论文 → [项目页](https://spherelab.ai/NoRA/) → Joluck/NoRA。
逐文件 URL、commit 和 SHA-256 保存在
[sources.json](experiments/rlvr_dora_nora_analysis_20261015/evidence/initialization_reference/sources.json)。

NoRA 论文的 Eq.7 写 `max(norm, eps)`，作者代码用 `norm + 1e-6`。
本地新版按作者代码使用 additive epsilon。不能声称公式与论文逐元素相同。
官方也支持 Gaussian（`std=1/r`），论文研究了多种分布；Kaiming 是本次采用的官方默认，
并非其他初始化一律错误。当前协议 `r32/alpha64` 保留，未采用 README 的 `alpha=r` 建议来更改实验。

## 新默认定义

对 A 的形状 `r × d_in`，使用 `kaiming_uniform_(a=sqrt(5))`，即
`U(-1/sqrt(d_in), 1/sqrt(d_in))`。所有 B 精确置零。

| 方法 | 存储的初始 A | forward / merge 使用的 A |
| --- | --- | --- |
| LoRA、standard DoRA | seeded Kaiming A | raw A |
| standalone NoRA-init、NoRA-init + DoRA | Kaiming A / (column norm + 1e-6) | 存储 A，不重复归一化 |
| continuous NoRA + DoRA | seeded Kaiming A，保留原始范数 | 每次使用 A / (column norm + 1e-6)，归一化参与求导 |

DoRA 的 magnitude 从 `||W||_row` 初始化；`||W+sBA||_row` 保持 detach。
这些方法的低秩增量在 step 0 为零。当前 wrapper 用 FP32 算范数并将 magnitude 存成基座 dtype：
当基座/adapter 存储为 BF16 时，范数舍入可能使 step-0 scale 偏离 1，
本次没有改变这个精度策略。FP32 且权重行范数不小于 eps 时的 identity 测试不能替代 BF16 identity 保证。

论文与 API 不规定本项目不同方法之间应如何共享随机数流。本地约定为：
seed42（可配置），CPU FP32 生成，按 `named_modules` 遍历顺序消费一个 generator，
复现 PEFT 创建 A/B `nn.Linear` 的两次构造随机数及最终 A reset。
这样 standard DoRA 与 LoRA 的 A/B 逐元素配对，两个 init-only NoRA 的初始 A 也逐元素配对。
初始化不推进外层训练 RNG。此为本地确定性约定，不能说论文要求这个 RNG 轨迹，
也不能说与原始 NVlabs 全局 RNG 消费序列一致。逐元素对齐验证范围是 FP32 master weights，
不扩大为任意 dtype/device 的位级复现承诺。

共享逻辑位于 [adapter_initialization.py](../verl/utils/adapter_initialization.py)，
standalone NoRA-init 与自定义组合 wrapper 均调用同一 column normalizer。
standard LoRA 的既有初始化 audit 格式保持；NoRA-init audit 标记为
`nora_init_kaiming_additive_eps_v2`。

## 历史兼容和当前训练

当前 GPU0–3 的 `nora_dora ... syncfix1` 进程保留旧 Gaussian + clamp 初始化，
GPU4–7 的 `dora ... syncfix1` 进程保留已有 seeded Kaiming standard 初始化。
本次没有重启训练或写入其 checkpoint。两条轨迹是其各自登记定义下的同步修复实验；
初始化对齐不会让它们自动成为新版的配对消融，两个实验的 LR 也不同。

`legacy_v1` 保留历史 Gaussian/per-layer seed 以及 continuous 的 clamped denominator。
快照合并器和独立同步验收脚本读取 manifest 中的 `initialization_version`：
缺失字段一律按 legacy 解释，防止旧 continuous A/B/m 加载后使用新版公式。
合并器检查显式 CLI 与 manifest 是否冲突；对有 manifest 的旧快照拒绝强行指定新版。
没有 manifest 的旧 wrapped checkpoint 同样默认 legacy；新版无 manifest 工件应显式传入
`--initialization_version peft_nora_reference_v2` 并核验其原始配置。

新快照记录初始化版本。冻结协议启动器采用新版，并将共享 helper SHA-256、
方法代码 SHA-256 纳入 runtime identity；任何不同定义都必须使用新的实验名。
已有 runtime manifest 不自动迁移，旧进程若重建 worker 需先核验其原始版本；
不能直接用新默认在旧实验目录中续训。离线评测旧快照仍有明确兼容路径。

## 验证范围

`tests/test_adapter_initialization_reference.py` 使用真实安装的 PEFT，比较多层不同形状下的
raw/effective A、B、standalone/composed NoRA 配对与外层 RNG；另检验 continuous
前向/梯度/merge、旧版和新版快照往返、CLI 冲突和未知版本拒绝。
原有历史版本回归保留；嵌套 FSDP 保存/恢复/首次导出回归采用新默认。
`test_rlvr_runtime_provenance.py` 检查共享初始化 helper 改变时实验身份同步改变。

验证不启动新的 RLVR 训练，也不据此宣称任何初始化已改善训练稳定性。

2026-10-16 本机验收：上述新增检查连同 LoRA 初始化、NoRA/DoRA/DOFT、真实两进程 CPU/Gloo FSDP、首次同步及回读保护共 **102 项测试通过**；Ruff、shell 语法和 diff whitespace 检查通过。两组训练服务仍为 active。

复验命令（已有项目依赖及 pytest）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest -q \
  tests/test_adapter_initialization_reference.py tests/test_standard_lora_init.py \
  tests/test_nora_dora.py tests/test_nora_doft.py \
  tests/test_custom_peft_fsdp_collection.py tests/test_custom_peft_checkpoint_resume.py \
  tests/test_rollout_adapter_resume.py tests/test_rollout_weight_verification.py \
  tests/test_rlvr_runtime_provenance.py
```
