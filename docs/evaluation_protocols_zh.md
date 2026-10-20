# 数学评测的两版协议

生效说明：2026-10-20。保留历史本地协议和 PeRL 发布版协议；当前正式报告使用 PeRL 版。训练 reward 不随本次离线评测调整而改变。

本文件定义判分版本、生成设置和结果使用范围。已有原始 records、summary、日志及旧文档保留；新口径另写结果目录，不覆盖旧分数。[本机双协议结果报告](experiments/local_results_dual_protocol_20261020_zh.md)记录可复评实验与缺失原始回答的归档。

## 1. 协议身份与用途

| 项目 | 历史本地协议 | 当前 PeRL 协议 |
| --- | --- | --- |
| 协议名 | `legacy_local_fullbench_v1` | `perl_release_judge_v1` |
| 用途 | 重放历史本地判分、追溯旧表；已知解析缺陷也属于该冻结版本 | 当前正式结果、后续统一重报告 |
| 判分入口 | `tina.analysis.rollout_utils.evaluate_math_answer` | `tina.analysis.perl_public_judge.judge_math_answer`，直接调用未经修改的上游 `grade_answer_perl` |
| 源码身份 | 本仓库提交 `7ddb86a` 的 `rollout_utils.py`；SHA256 `5f7483e0c330980bd115550dd2f8f11c759566c9dc1151a346721c9493d61d3f` | PeRL 提交 `a5ec6c36089cb3def58952d8544a69b93b30db62` 的 `perl/eval/grader.py`；SHA256 `b8790bb7ec815e0338905f411d2e8fa7c3eb3e55cca7e852523b7ca34163461f` |
| 数学比较 | Math-Verify / latex2sympy2_extended，异常或 gold 空解析时走字符串回退 | mathd normalization 或上游 SymPy grader 判等，任一通过即正确 |
| 分数 | 正确 1，其他 0 | 正确 1，其他 0；格式分另作诊断，不加到正确性分数 |
| CPU 复评分入口 | `scripts/analysis/rescore_legacy_local.py` | `scripts/analysis/rescore_perl_public.py` |

发布版上游：[PeRL grader](https://github.com/MikaStars39/PeRL/blob/a5ec6c36089cb3def58952d8544a69b93b30db62/perl/eval/grader.py#L513)、[评测入口](https://github.com/MikaStars39/PeRL/blob/a5ec6c36089cb3def58952d8544a69b93b30db62/perl/eval.py)、[论文 §2.3–2.5 与 Table 2](https://arxiv.org/html/2512.23165v2)。当前仓库默认分支后来新增的其他 evaluator 不替代此冻结发布版。

`fullbench_math_v2` 和 `tina_lighteval081_v1` 是此前排查中产生的诊断版本，继续保留已有产物；它们均不是本文件定义的两个主协议。不能把带 v2 标签的 inline 分数称为 legacy v1，也不能将其混入 PeRL 正式表。

更早的文件可能没有 `judge_version`。报告将其标为 `legacy_unversioned`，保留原保存分数；只有通过冻结函数逐条重放的结果才标记为已核验的 `legacy_local_fullbench_v1`。依赖版本或历史实现未完整保存时，不把原分数自动认证为该冻结版本的精确重放。

## 2. 答案提取与标准答案

| 项目 | `legacy_local_fullbench_v1` | `perl_release_judge_v1` |
| --- | --- | --- |
| boxed | 按原函数提取最后一个字面 `\boxed{…}` | 按上游原函数提取；不另加本地宽松分支 |
| 非 boxed | 原规则含 `Answer: A–D` 的字母分支及不区分大小写的 `final answer is` 分支；提取后仍走原 Math-Verify/回退 | 最后一行必须以大小写精确的 `Answer:` 开头；按上游处理末尾句号与外层 `$` |
| 格式宽容 | 原规则原样保留 | 不额外接受 `ANSWER:`、`Final Answer:`、裸数字或带空格的 `\boxed {…}` |
| gold | 使用 records 中原始 `gold_answer`；不补 boxed、不修正裸 LaTeX | MATH500 用原 `answer`；Minerva 按题目全文精确匹配 `math-ai/minervamath` 的短 `answer`；其他任务沿用原 `gold_answer` |
| 未通过格式/数学判等 | 记 0 | 记 0；不额外尝试自定义 Unicode、科学计数、物理等式 RHS 等补偿 |

历史协议存在确定性缺陷：裸 LaTeX gold 可能被默认解析器截断为数字前缀。例如 `8\sqrt{10}` 被解析为 `8`，会同时造成正确表达式漏判和错误数字误判。历史版保留这个行为以重放旧结果；当前正式报告采用上游原始判分路径解决该类问题。

Minerva gold 资产：`math-ai/minervamath`，revision `ee46ddc498933b1977577953250ca5c66be64f96`，272 题，JSONL SHA256 `e04ec2bc1d8f80a0091b4b1989f1ffb366f3583f5bae381546a066d4ccc26e3d`。默认本机资产在 `configs/evaluation/assets/minerva_mathai.jsonl`，仓库已携带该固定快照；也可用 `--minerva-source` 指定同哈希文件。精确题目匹配失败时停止该记录的复评，不猜测答案映射。

## 3. 沿用的生成与指标设置

本轮重报告只重新判分已保存的回答；以下是标准六任务生成设置。历史实验若采用其他值，在清单中保留原值并单列，不伪装为统一生成协议。

| 项目 | 本地标准设置 |
| --- | --- |
| 任务 / 题数 | AIME24 30、AIME25 30、AMC23 40、HMMT Feb 30、MATH500 500、Minerva 272 |
| 每题采样 | 前四项 32；MATH500、Minerva 为 4 |
| 总问题 / 回答 | 902 / 7248 |
| temperature / top-p | 0.6 / 0.95 |
| max_new_tokens / max_model_len | 32768 / 34816 |
| evaluation seed | 42；与 training seed 分开记录 |
| 提示词 | 已保存的本地 Tina 长 boxed 模板；部分导入题目还含原快照自带指令。复评分不改写 |

对任务 b，N_b 道题，每题 k_b 个回答，正确性为 c：

- `Avg@k = sum(c) / (N_b × k_b)`。
- `Pass@k = 至少有一个正确回答的题数 / N_b`。
- 表格 `Avg.` 为全部回答加权：`sum_b(N_b × k_b × Avg_b) / sum_b(N_b × k_b)`，不是六任务简单平均。
- 表格任务单元格为 `Avg@k / Pass@k`，单位为百分比。
- `macro_avg_at_k_over_benchmarks` 单独保留，禁止与总体 `Avg.` 混用。
- `format_valid`/`parse_rate` 是对应 judge 的格式诊断；PeRL 的格式分不等于“完整数学解析成功率”，也不能与旧版同名字段当成同一事件比较。
- 输出长度、触顶比例和采样文本不随重判而变化。

当前设置与 PeRL 论文正文的解码参数一致；发布期部分脚本使用 0.7 / 0.9 / 31744，提示词也与本地不同。因此当前协议是 **本地生成 + PeRL 发布版 judge**，不称逐 token 完整复现 PeRL。

## 4. 环境与重放命令

当前 PeRL 封装固定 `sympy==1.14.0`、`pylatexenc==2.10`。这些是本地复现锁定值；PeRL 发布版 requirements 没有精确锁版本，不能据此声称恢复其原始运行环境。历史版运行时记录 `math-verify`、`latex2sympy2-extended`、`sympy` 的实际版本，并检查原函数文件 SHA256 与归档版本一致（不依赖本地拥有历史提交对象）。两版都不改变训练环境中的 reward 配置。

从仓库根目录运行，`PYTHON_BIN` 指向已具备相应依赖的环境：

另一台机器修复已有评测时，只需 CPU 环境和原始 records，无需重新生成回答。推荐单独安装两个固定依赖，避免改变训练环境：

```bash
python3 -m venv .venv-eval-perl
.venv-eval-perl/bin/python -m pip install -r requirements-eval-perl.txt
PYTHON_BIN="$PWD/.venv-eval-perl/bin/python"

"$PYTHON_BIN" scripts/analysis/rescore_perl_public.py \
  --records /path/to/records/checkpoint1.jsonl /path/to/records/checkpoint2.jsonl \
  --output-dir /path/to/evaluation/perl_release_judge \
  --workers 4
```

默认使用仓库自带的 Minerva 固定 gold。输出目录里的 `summary/*.json` 是当前正式汇总，`records/*.jsonl` 保留新旧判分与原始回答；`changes/` 和 `unextracted/` 供排查。已有同源同协议的结果会验证 metadata 后复用；来源或协议不一致时拒绝覆盖。若沿用过去手工指定的 gold 路径重跑，继续传相同的 `--minerva-source`，以保持已有 metadata 身份。

历史协议重放还需安装 `requirements-eval-legacy.txt`；当前 PeRL CPU 复评不依赖 vLLM、torch、datasets 或 Math-Verify。

```bash
PYTHON_BIN=/path/to/eval-env/bin/python
RECORDS=/path/to/original/records/checkpoint.jsonl

# 当前正式口径；输出到独立目录，可验证来源后复用。
"$PYTHON_BIN" scripts/analysis/rescore_perl_public.py \
  --records "$RECORDS" \
  --output-dir runs/eval_replay/perl_release_judge \
  --minerva-source configs/evaluation/assets/minerva_mathai.jsonl \
  --workers 4

# 历史本地口径；目标目录须尚不存在；该入口核验标准六任务完整性。
"$PYTHON_BIN" scripts/analysis/rescore_legacy_local.py \
  --records "$RECORDS" \
  --output-dir runs/eval_replay/legacy_local_fullbench_v1 \
  --workers 4
```

新版生成启动器的 `EVAL_PUBLIC_STANDARD=1` 会在生成后追加 PeRL CPU 评分。关闭该开关只是关闭附加评分，**不表示 inline judge 自动切回历史 v1**；需要历史重放时显式调用上面的 legacy 入口。正在运行的旧 worktree 启动器可能尚无该附加步骤，应对完整 records 调用 CPU 入口。

## 5. 结果保存与比较规则

1. 原 records/summary 不覆盖。新结果写独立目录，保留 `source_judgment`、原 gold、原生成和源文件 SHA256。
2. 当前正式 judge 在同一表中必须相同；gold 来源也必须明确。不能用新口径 S2 减去旧口径 baseline。
3. 同一 judge 仍需核对训练协议、checkpoint step、training seed、问题/gold 快照、提示词、采样和生成长度。统一重判不会修复历史训练实现或恢复过程的缺陷。
4. 完整六任务、单任务修复、screen 子集分别报告。缺少问题、重复回答、采样不齐或源文件仍在变化的记录不进入完整表。
5. 只有 summary、没有原始回答的实验，保留旧值，当前口径记“无法复评”；不估算转换、不按其他实验的平均涨幅修正。
6. 新旧分数的变化只解释为判分变化。已有历史分析中的方法结论需要按对应证据范围阅读，不自动由重判表升级为 matched comparison。

批量盘点、重判和报告：

```bash
"$PYTHON_BIN" scripts/analysis/report_local_eval_protocols.py \
  --roots runs ../peft-for-rl-gradtop-s2-20261018/runs ../peft-for-rl-gradtop-s3-20261018/runs ../Tina_orthres_run/outputs \
  --output-dir runs/eval_protocol_reaudit_20261020 \
  --report docs/experiments/local_results_dual_protocol_20261020_zh.md \
  --workers 4
```

`--inventory-only` 只盘点；`--reuse-inventory` 使用已完成的冻结盘点继续执行。产物包含 `inventory.json`、逐实验 `source_sha256`、问题/gold 与 prompt 指纹、`batch_status.json`、分协议 summaries/records，以及报告旁的精确值 CSV。

完成后独立核验所有原始字段、原文件哈希、gold 映射与 Avg/Pass 汇总：

```bash
"$PYTHON_BIN" scripts/analysis/verify_local_eval_reaudit.py \
  --output-dir runs/eval_protocol_reaudit_20261020 --workers 4
```

核验回执写入运行目录的 `verification.json`。历史目录中其他 schema 的训练 heldout 验证不强行转换为六任务 records，排除原因保留在清单中。
