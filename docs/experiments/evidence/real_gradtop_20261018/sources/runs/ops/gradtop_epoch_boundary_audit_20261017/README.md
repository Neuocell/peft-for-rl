# 当前GradTop：250→300跨epoch可行性核查

2026-10-17 00:03，Asia/Shanghai。当前250续训仍运行，未修改冻结源码、协议、plan或运行参数，也未启动300任务。

结论：底层StatefulDataLoader支持跨epoch与数据顺序恢复，但当前执行配置不能直接跑到300，存在三处单epoch限制。

| 位置 | 当前行为 | 300所需调整 |
| --- | --- | --- |
| RLVR协议/verify_rlvr_protocol.py | max_training_steps=270，直接拒绝TARGET_STEP=300 | 单独登记长跑预算扩展，并保持现有科学参数、数据和初始化身份可追溯 |
| run_dapo_boxed_paper_clean_1p5b_4gpu.sh:535 | trainer.total_epochs=1；显式total_training_steps不会扩展外层epoch循环 | 本数据300步至少需total_epochs=2；按目标步数及真实loader长度计算更稳妥 |
| runs/ops/rlvr_protocol/step150_control.py:67 | 把data.pt的epoch内批数/采样数直接等同累计step/step×64 | 区分累计训练步数和当前epoch的数据游标，并验证sampler RNG/状态 |

实际日志中数据17,917行，batch64，drop_last=True，loader长度279。第一轮训练279个完整batch，共17,856题，尾部61题丢弃；第二轮重新打乱。250→300的前29步为251–279，后21步为第二epoch的280–300。训练日志的epoch编号分别为0和1。

已用安装环境中的生产RandomSampler和StatefulDataLoader、真实step200/data.pt进行纯CPU索引回放，行数、seed42、batch64、workers0和drop_last完全对齐；将dataset内容替换为行索引，不运行模型/rollout。先回放200→250取得模拟250状态，再检查250→300：

- total_epochs=1：只发出29批，最后279，不能达到300。
- total_epochs=2：发出全部50批，逐题索引与不间断运行完全一致；280进入新epoch，新排列不同于第一轮。
- 模拟300状态为epoch内21批/1344题，旧验收的300批/19200题断言会失败。
- 从模拟300恢复至310，逐题索引也与不间断运行一致。

这证明的是数据顺序与状态恢复，不是第二epoch GPU训练实测；250完整checkpoint尚未生成。当前200→250完全在第一epoch内，不受该跨epoch限制影响。

- [可复现CPU检查](check.py)
- [机器可读结果](report.json)
