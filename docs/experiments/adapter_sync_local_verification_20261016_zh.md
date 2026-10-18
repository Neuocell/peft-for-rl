**2026-10-18阅读说明：** 本文保留10月16日CPU集成完成时的快照。后来GradTop151/201/251及新REAL-P256零B/首次更新后均已完成实际四GPU验收，最新说明见[实现审计与修复状态](../../audits/real_gradtop_implementation_audit_20261018_zh.md)。本文后续“尚未GPU验收”“尚未提交”等表述只描述原时点。

本机已于 2026-10-16 合入 GitHub 提交 `04182bb98c493355c86b5980c967531d8f6e5a33`
（`Fix adapter rollout synchronization and checkpoint resume`）。
本地分支为 `experiments/spar-lora-v0`，HEAD 与该远端分支一致；本地 GradTop/REAL
未提交工作已保留，本文所述额外集成修改也保留在工作区，尚未提交或推送。

本次结论是：更新、冲突解决、CPU 回归和实际协议 dry-run 均已完成。
本机 GPU/vLLM 端到端验收尚未执行，不将 CPU transport 替身测试视为实际 GPU 验收。

修复行为和本地接入如下：

- 原生 LoRA 路径在首次同步时先加载基座、再加载当前 adapter；恢复后的非零 adapter
  也走这一路径。sleep level 2 重新加载基座后同样加载当前 adapter。
  本地 GradTop 和 REAL 正式训练均使用 `grad_subspace`，已确认归入 `_is_lora` 路由。
- 缺少 default PEFT 配置、未收集到 adapter、引擎拒绝安装 adapter、hybrid 生成时
  找不到活动 adapter，以及恢复未包含 model，都会阻断，而非静默使用基座。
- 自定义 DoRA/DOFT 的嵌套 FSDP 导出使用规范化模块名称，检查每层合并完整性。
  首次同步及 checkpoint 恢复后的首次同步新增实际接收权重回读。
  该回读的适用条件和限制见 [远端修复说明](../adapter_sync_fixes.md)。
- 启动脚本的唯一冲突已解决：保留本地 REAL/GradTop artifact 校验、REAL 的冻结 A
  和 use_orig_params 配置，接入远端原子写入 contract 和 runtime provenance 检查。
- 缺 artifact 的 GradTop 预览仅限 DRY_RUN=1，写入独立 preview contract，
  不生成正式 runtime manifest。正式启动必须经过 provenance 检查；其他方法收到
  同名预览环境变量、或 EXP_NAME 以 `.preview` 结尾，均不能绕过该检查。
- 补上远端运行身份记录与本地方法的兼容性缺口：REAL/GradTop 的运行身份绑定
  `rank_map.json`、`summary.json`、`subspaces.safetensors` 的内容哈希。
  已有 manifest 的运行若更换 artifact 会拒绝恢复；相同字节内容迁移目录不会改变身份。

验证使用 `/home/node/anaconda3/envs/peft-for-rl/bin/python`。
从 `/tmp` 导入 `verl` 的实际路径为 `/root/peft-for-rl/verl/__init__.py`；
已安装的 `peft-for-rl 0.7.0` 是指向本仓库的 editable install，因此无需重新安装依赖。

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| 远端指定的七个 adapter CPU 回归文件 | 82 passed，19.63 秒 | `runs/ops/adapter_sync_update_20261016/upstream_cpu_tests.log` |
| 本地 adapter/协议集成及 GradTop/REAL 回归 | 181 passed，1 skipped，16.69 秒 | `runs/ops/adapter_sync_update_20261016/local_integration_tests.log` |
| 最终 provenance 实现与启动脚本集成复核 | 16 passed，1.08 秒 | `runs/ops/adapter_sync_update_20261016/final_runtime_tests.log` |
| 使用本机实际协议和 artifact 的 LoRA、当前 GradTop、REAL dry-run | 三组退出码均为 0，均未创建 runtime manifest | `runs/ops/adapter_sync_update_20261016/actual_dry_runs.json` 及对应日志 |
| Python AST / shell / JSON 静态检查 | 26 个文件通过；git diff --check 通过 | `runs/ops/adapter_sync_update_20261016/workspace_verification.json` |
| 本地原始文件与评测源码保护 | 10 个无重叠的本地修改文件、41 个原未跟踪文件保持原样；12 个评测冻结源码哈希一致 | 同上 |

测试组合存在重叠，以上通过数不能相加当作独立测试总数。唯一跳过项是
`test_real_probe_fsdp_bf16_capture_refresh_and_rollout_sync`，该项要求显式设置
`RUN_REAL_FSDP_TEST=1` 并有两张可用 GPU。本轮设置 CUDA_VISIBLE_DEVICES 为空，
未启用该 GPU 测试。另有一条 Ray state API 弃用警告，无测试失败。

CPU 回归包含真实两进程 CPU/Gloo 嵌套 FSDP 的保存、恢复及导出，也包含直接执行
生产同步方法体的非零 adapter 数值对照。新增 GradTop/REAL 同步案例覆盖 A 可训练/
冻结两种情况，但使用小型 PEFT 模型与 CPU rollout transport 替身；它验证同步分支，
不能代替完整 GradTop/REAL GPU 训练。启动器集成测试隔离到临时仓库，只替换资产
校验端和最终训练命令，执行真实协议启动器与 provenance checker。另行执行的三组
实际 dry-run 使用真实资产校验器和实际本地文件。

实际 dry-run 全部使用独立 EXP_NAME `adapter_sync_validation_20261016_<method>`：

- LoRA 与当前 GradTop 使用 `configs/experiments/gradtop_r32_clean_20261015.json`；
  GradTop 使用 `runs/analysis/gradtop_r32_o16_w8_c64_seed42_20261015`。
- REAL 使用 `runs/ops/real_probe32_continue_step150/protocol.json` 和
  `runs/analysis/real_probe_r32_s32_seed42_replayed`。
- 三组均为 LR=1e-5、rank=32、TARGET_STEP=50、DRY_RUN=1。
  这些检查只验证启动配置与已存在 artifact，不启动训练、不重新评测方法效果。

检查时四张 GPU 均被 `gradtop_r32_fine_eval_20261016` 的既有评测占用，状态为
`evaluating_step130`，每卡约使用 40616 MiB。本次没有中断或重启该评测，
没有修改其冻结计划，也没有启动修复后训练。仍需在 GPU 可用时验证实际 vLLM
首次/恢复同步；自定义 DoRA 可使用远端随附的
`scripts/analysis/verify_nora_dora_rollout_sync.py` 分阶段验收。
原生 LoRA/GradTop/REAL 需额外验证非零 checkpoint 的真实 adapter 安装及策略输出，
不能用自定义 DoRA 的合并权重回读结果代替。

同步修复在新启动的进程/重新创建的 worker 中生效，不能改变已导入旧模块的进程，
也不能追溯修复已经发生的错误训练更新。历史训练轨迹的已知问题和已有评测结果
仍按原审计记录保留。远端新增的 guard 会拒绝直接恢复没有 runtime manifest 的
历史运行；应核验原始输入与配置后作明确的迁移或重建决定，不伪造旧运行凭据。

恢复备份位于 `runs/ops/adapter_sync_update_20261016/`：`before.json`、
`local_before.patch`、`untracked_backup/`。tracked 改动另保存在 stash
`6d1c470495b7bb0e490dad8a1e981ecd9ad4873e`，已恢复到工作区且仍保留该 stash。
合并产生的暂存状态已清除，本地改动仍为未暂存状态，无未解决冲突。
