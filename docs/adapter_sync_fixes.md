# Adapter 同步与续训修复

本次修复使 rollout 在生成前使用当前 actor 的 adapter，覆盖两个独立问题：

- 自定义 DoRA／NoRA-DoRA／NoRA-DOFT：统一嵌套 FSDP 的模块名与 state dict 名称，
  每层合并一次；遗漏、名称冲突或不完整合并立即报错。导出的 CPU 张量独立持有存储。
- 原生 PEFT LoRA／AdaLoRA，以及通过 LoRA 接口实现的 NoRA：dummy 引擎首次同步、
  或 sleep level 2 后，先加载基座，再加载当前 adapter，完成后才能生成。
  hybrid 训练缺少 adapter 或引擎拒绝加载时立即报错。

自定义 DoRA／DOFT 在首次同步和每次 checkpoint 恢复后的首次同步，还会回读 vLLM
实际接收的适配层权重。回读不一致、缺层或缺少完整回执都会在生成前阻断。
恢复配置必须包含 model，不能仅恢复 optimizer／额外状态。

协议入口同时支持 standard DoRA、NoRA-init 和 continuous 三种 DoRA 模式。
standard DoRA 的 A/B 初始化与同 seed 的标准 LoRA 对齐；离线合并支持相同模式。
自定义方法的原生 `lora_rank` 设为零，走完整合并权重路径。

## 在另一台机器更新

在目标环境的仓库目录执行：

```bash
git fetch origin
git switch experiments/spar-lora-v0
git pull --ff-only origin experiments/spar-lora-v0
python -m pip install -e . --no-deps
python -c 'import verl; print(verl.__file__)'
```

确认输出的 `verl` 路径属于刚更新的 checkout。使用其他本地分支时，可以将修复 commit
cherry-pick 到对应分支；必须保留该提交包含的运行时模块、启动器和测试依赖。

Python／Ray worker 已导入的代码不会因 `git pull` 自动更新。修复在新启动的训练进程或
重新创建的续训 worker 中生效；应从完整 checkpoint 恢复 model、optimizer、scheduler、
RNG 和 dataloader 状态。

协议启动器记录输入、方法及 exporter 身份，拒绝覆盖旧轨迹或将无修复版本凭据的旧
自定义 PEFT checkpoint 直接接入修复实验。缺少 runtime manifest 的历史实验需先核验
原始配置与输入来源。同步代码不会修复已经发生的错误训练更新：旧自定义合并缺陷轨迹
需从基座重建；原生首轮缺陷轨迹可从首次受影响更新之前的完整 checkpoint 重建后续步骤。

## 接入验收

自定义 DoRA／DOFT 在支持的配置下，应在首次 rollout 和恢复后首次 rollout 之前看到：

```text
[rollout_weight_verification] {"status": "verified", ...}
```

检查每个 rank 的 `tensor_count` 与适配层数一致（当前 1.5B 配置为196），
恢复后的 `checkpoint_source` 指向实际 `global_step_*/actor`。
`loaded_weights_sha256` 记录实际接收权重的摘要。原生 LoRA 使用 adapter 安装与存在性检查，
不产生上述自定义合并权重回读回执。

自定义回读的当前验收范围为非量化 vLLM、TP=1、PP=1，支持直接参数以及常见
Qwen／Llama QKV、gate/up 打包布局。FP8 不支持该精确回读，会报错；TP/PP大于1时
未启用该新增回读。嵌套 collector 与 checkpoint 恢复的分布式回归覆盖 FSDP1，
不据此宣称 FSDP2 已获得同等验证。

## 回归验证

在已有训练依赖及 pytest 的环境中，以下检查不使用训练 GPU：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest -q \
  tests/test_nora_dora.py tests/test_nora_doft.py \
  tests/test_custom_peft_fsdp_collection.py tests/test_custom_peft_checkpoint_resume.py \
  tests/test_rollout_adapter_resume.py tests/test_rollout_weight_verification.py \
  tests/test_rlvr_runtime_provenance.py
```

提交前该组合通过82项测试，包含真实两进程 CPU/Gloo 嵌套 FSDP 保存、恢复与首次导出，
以及生产同步方法体的顺序和错误注入检查。生产方法体测试使用 CPU transport 替身，
不等同于完整 GPU 训练验证。

需要独立 GPU 验收时，可使用
[`scripts/analysis/verify_nora_dora_rollout_sync.py`](../scripts/analysis/verify_nora_dora_rollout_sync.py)，
先运行 `--phase fsdp`，再运行 `--phase vllm`，显式提供 `--base-model`、`--snapshot`
和 `--output-dir`。该脚本针对当前1.5B、196适配层、r32/a64配置；使用空闲GPU，输出
逐层导出、实际加载以及固定 token 概率比较。短输入验收不代表长 rollout 或训练稳定性。
