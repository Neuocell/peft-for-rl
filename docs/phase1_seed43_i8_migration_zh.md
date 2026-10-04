# Phase-1 seed-43 I8 异地迁移与恢复

本文只覆盖当前固定模型/任务下的 seed-43 I8。它不重启历史 controller，不重跑 seed-42，不续训已经完成的 seed-43 I0，也不授权 seed 44。

## 迁移边界

GitHub 分支 `experiments/spar-lora-v0` 保存训练/评测代码、冻结规范、seed-43 I0 的小型证据、I8 初始化空间、迁移 manifest、SHA 清单和专用启动器。公开仓库不保存训练 parquet、benchmark records、I0 merged records 或 adapter：adapter 单文件超过普通 GitHub 的 100 MB 限制，且数据、题目和模型输出的再分发许可未确认。

离线包 `peft-for-rl-phase1-seed43-i8-handoff-20261004.tar.zst` 只包含五个必要文件：训练 parquet、冻结 benchmark snapshot、seed-43 I0 merged records，以及 I0 adapter 的 config 和 safetensors。它不包含可从上游恢复并按哈希验收的 base model，也不包含 I0 raw shards、rollout cache、日志或约 15.4 GB 的 step-25/step-50 恢复 checkpoint。I0 已完成并独立 verified，后续 paired comparison 只需要 merged records；这些 checkpoint 不是 I8 输入。

最终归档大小为 `182,837,537` bytes，SHA-256 为 `60bfd3c6c4291b8b5d1811f04655638321d038903ba82287c5958285e043960e`。`zstd -t` 已通过，包内五个成员也分别从压缩流重新计算 SHA-256 并与 payload 清单一致。

完整机器可读边界见 `runs/phase1-signal-random-v1/migration/phase1_seed43_i8_handoff_manifest.json`，离线文件哈希见同目录的 `phase1_seed43_i8_handoff_payload.sha256`。

## 目标路径

冻结 contract 和 verifier 使用 `Path.resolve()` 绑定绝对路径。目标服务器应保留：

```text
/root/peft-for-rl
/data/peft-for-rl-runtime
/home/node/anaconda3/envs/peft-for-rl
```

若真实磁盘位于其他位置，应把真实目录 bind mount 到上述路径。普通 symlink 会被 `Path.resolve()` 展开，不足以保持冻结 contract 的路径身份。此次迁移不修改历史 JSON，也不修改训练、评测或 verifier 的 provenance；任意 checkout relocation 应作为协议 v2 单独实现和复验。

## 恢复与验收

先克隆同一分支到规范路径，再从文件系统根目录解压离线包：

```bash
git clone --branch experiments/spar-lora-v0 \
  ssh://git@ssh.github.com:443/Neuocell/peft-for-rl.git \
  /root/peft-for-rl
cd /
tar --use-compress-program=/home/node/anaconda3/bin/unzstd -xf \
  /path/to/peft-for-rl-phase1-seed43-i8-handoff-20261004.tar.zst
cd /
sha256sum --strict -c \
  /root/peft-for-rl/runs/phase1-signal-random-v1/migration/phase1_seed43_i8_handoff_payload.sha256
```

base model 不在包内。将相同的 DeepSeek-R1-Distill-Qwen-1.5B base 放到 manifest 指定位置；专用 preflight 会逐文件验证 config、generation config、tokenizer 和 `model.safetensors` 哈希。

先只校验迁移内容和初始化空间：

```bash
cd /root/peft-for-rl
bash runs/phase1-signal-random-v1/ops/run_seed43_i8_migrated_guarded.sh \
  --payload-only
```

完整 dry-run 还会检查无 controller/evaluator/EngineCore、无 seed-44 和 I8 部分状态，并要求 `/root` 至少有 `16,500,000,000` bytes 可用空间：

```bash
bash runs/phase1-signal-random-v1/ops/run_seed43_i8_migrated_guarded.sh
```

只有上述命令返回 `SEED43_I8_MIGRATED_PREFLIGHT_READY` 后才执行：

```bash
bash runs/phase1-signal-random-v1/ops/run_seed43_i8_migrated_guarded.sh --execute
```

该入口只调用 `TRAIN_SEED=43 PHASE1_RUN_REVISION=3 ... I8`，训练和六 benchmark 评测完成后运行独立 postverify，然后退出。不要在仅恢复本离线包的机器上执行旧 `start_seed43_i8_next_card_guarded.sh --mode replacement --execute`：它会重启完整 controller，缺少 seed-42 历史 adapter 时可能回头重训旧方法。

## 故障边界

入口拒绝覆盖任何 seed-43 I8 部分状态。若训练或评测已启动后中断，应先保留所有现场文件，根据 checkpoint、contract、shard 和 summary 的实际状态制定恢复动作；不要删除目录后盲目重跑。I8 postverify 完成后，I0/I8 两份 seed-43 merged records 才具备实现冻结 two-seed analyzer 和计算四个 gate 的输入条件。
