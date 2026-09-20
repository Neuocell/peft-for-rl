# RLPO-init 在 verl 中的接入

本实现对应论文 *Geometry-Preserving Orthonormal Initialization for
Low-Rank Adaptation in RLVR* 及其公开仓库：

- 论文：<https://arxiv.org/abs/2606.31813>
- 源码：<https://github.com/Richard-ZZZ/geometry-preserving-orthonormal-init-rlvr>
- 对照文件：`scripts/prepare_init_adapters.py`
- 核对版本：`949965f209e54ddc36862a091930ebfb3209e8b7`

## 初始化定义

对每个被 LoRA 包装的冻结线性层权重
`W in R^(d_out x d_in)` 做 float32 完整薄 SVD：

```text
W = U diag(s) Vh
A_0 = Vh[:r, :]
B_0 = 0
Delta W_0 = (alpha / r) B_0 A_0 = 0
```

这里选择的是最大 `r` 个奇异值对应的右奇异向量，即官方源码中 RLPO
使用的 `mode="max"`。奇异值不会乘进 `A` 或 `B`，冻结的基础权重也不会
被修改。`A_0 A_0^T = I_r`，而 `B_0=0` 保证初始化后的模型函数与基础模型
完全一致。

初始化结束后，它就是普通 PEFT `LoraConfig` 的两因子 LoRA：`A`、`B`
均可训练。不使用 AdaLoRA 的 `lora_E`、importance score、`RankAllocator`、
动态裁秩或正交 loss。

## verl 接入链路

1. `main_ppo` 读取 Hydra 参数并创建 Ray actor/rollout/ref worker。
2. `ActorRolloutRefWorker` 从预训练 checkpoint 创建基础模型。
3. `peft_type=rlpo` 时用普通 `LoraConfig` 注入所有目标线性层。
4. global rank 0 调用 `apply_rlpo_initialization`，逐层计算 top-r 右奇异向量。
5. FSDP1 以 `sync_module_states=True` 从 rank 0 同步初始化参数。
6. 训练、vLLM rollout 权重同步和 adapter checkpoint 保存均复用普通 LoRA
   路径。

RLPO 需要读取真实基础权重，不能在 meta tensor 上做 SVD。因此 worker 对
`peft_type=rlpo` 关闭 meta 初始化。SVD 默认由 `rlpo_svd_device=auto` 放到
当前 CUDA device；也可以设为 `cpu`，以降低初始化期间的显存峰值。

## 当前实验配置

专用入口：

```bash
cd /home/wangls/peft-for-rl
bash scripts/local/start_rlpo_init_4gpu.sh
```

默认配置为 rank 32、alpha 64、dropout 0.05、`all-linear`、学习率
`1e-6`、weight decay 0、warmup 0、constant scheduler、global batch 64、
mini-batch 16、每 prompt 8 个 rollout、prompt/response 长度 1024/8192、
270 trainer steps、每 50 步保存，数据与 PPO dataloader seed 都为 42。

运行数据、checkpoint 和日志都位于仓库自己的 `runs/` 下。Ray 为规避
Unix socket 的 107 字节路径上限，单独使用短临时目录
`/home/wangls/rrlpo`；它不读取旧实验目录。默认日志为：

```text
runs/logs/verl/rlpo_init_r32a64_b64m16n8_270_4gpu_boxed.log
```

本次只完成了配置与 dry-run 验证，尚未执行该入口；正在运行的 AdaLoRA
实验没有被停止或覆盖。
