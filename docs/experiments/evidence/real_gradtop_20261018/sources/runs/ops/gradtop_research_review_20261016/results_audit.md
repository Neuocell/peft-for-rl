# 当前 GradTop / REAL / LoRA：结果与训练轨迹独立审计

范围严格限于交接文档指定的 GradTop experiment `rlvr_v1_gradtop_r32_a64_lr1em5_b64m16n8_seed42_20261015`（10/50/100/150），同协议 LoRA（50/100/150）、REAL32（50/100/150）、REAL64（50）。未引入其他 GradTop；未启动训练、生成或修改原记录。新增 CPU 再聚合、抽取器重放和明示的事后格式敏感性诊断都写在本目录。

**最重要的新结论：当前 GradTop 的优势真实存在于原协议记录中，但不能直接解释为推理更新空间更好。Minerva 的格式可解析性贡献很大；对原来被计错的规范末行 `Answer:` 做保守、统一重判后，GradTop150 对 REAL150 的六任务 Macro 差值从 +2.989 pp 缩至 +1.825 pp。优势没有消失。REAL32 后期退步也未被格式因素完全解释。恢复缺陷对 GradTop 与 REAL32 均适用，LoRA 则是连续到150，因此三方恢复处理并不完全相同。**

## 1. 独立核验与证据边界

- 重新读取全部 11×7,248＝79,728 条原始回答；正确数、解析数、长度、hit-max 与六任务 summary 全部一致。每组恰有7,248个唯一 `(benchmark,id,sample_index)`、902题；所有组的题目、gold和实际prompt文本逐键 SHA 一致；每条生成seed/temperature/top-p/max-new-tokens/max-model-len均一致。原 records SHA 已保存。
- 重新读取44份 manifest及11份shard completion，均为4 shard完整。LoRA150状态字面为 `all_four_shards_verified_before_exit_cleanup`，其余为 `all_shards_verified`。原每组verification均verified。再聚合沿用原correct标签，不等于独立复算全部数学判分。
- 当前 GradTop首轮plan的29项、续训plan的35项（含32项code和3项artifact）、REAL32最新plan的42项 SHA 全部匹配当前文件。首轮与续训GradTop的artifact、4 rollout steps、16 observations、width8/capacity64、256 prompts/2048 responses、global_token_mean、无额外sign/mask及A+B训练均符合plan。
- `tina_run/tina/analysis/rollout_utils.py` 的抽取器对全部79,728条回答重放，得到的extracted_answer与原记录全部一致。训练抽取器与评测抽取器并非逐字相同：eval多一个`Answer:[A-D]`分支，全部组共10条抽取差异、原correct均False；其规模不能解释主要差异。

证据：[`results_raw_reaggregation.json`](results_raw_reaggregation.json)、[`manifest_recheck.json`](manifest_recheck.json)、[`plan_hash_recheck.json`](plan_hash_recheck.json)、[`parser_consistency.json`](parser_consistency.json)。可复现入口：[`audit_results.py`](audit_results.py)；辅助只读检查的结果与源文件SHA在对应JSON中。未对原vLLM engine tensor/hash作运行时验收，不能把清单匹配升级为engine实际使用策略的证明。

## 2. 恢复路径不是三方相同

| 方法 | 实际训练进程分段 | 首轮恢复观测 |
|---|---|---|
| LoRA | 同一TaskRunner pid11135，step1–150连续 | 50/100之后没有新进程恢复首轮 |
| REAL32 | 1–50；50恢复→100；100恢复→150 | step51/101分别由新pid884481/984574采样 |
| GradTop | 1–50；50恢复→100；100恢复→150 | step51/101分别由新pid1250210/1348228采样 |

LoRA原始日志：`runs/logs/rlvr_protocol/rlvr_v1_lora_r32_a64_lr1em5_b64m16n8_seed42.log:943`（step1）、`:1050`（50）、`:1052`（51）、`:1158`（100）、`:1160`（101）、`:1267`（150），全为同一pid。REAL的恢复配置均为`lora_adapter_path=None`、`load_format=dummy`、`val_before_train=False`，新进程日志显式`Load from checkpoint folder`；详见[`training_comparison.json`](training_comparison.json)的startup_config、segments和带行号rows。

REAL32 `runs/ops/real_probe32_continue_step150/train.log:983`的step51 parse59.96%、hit8192为42.38%、grad_norm0.1476，step52迅速回到parse93.55%、hit11.52%、grad_norm0.0754；`runs/ops/real_probe32_early_step100/train.log:989`的step101 parse55.66%、hit48.44%、grad_norm0.2189，step102 parse91.99%、hit12.89%、grad_norm0.0767。这与GradTop恢复首轮异常相符。

当前共同源码中，`grad_subspace`属于LoRA同步分支（`verl/workers/fsdp_workers.py:237`）；首次base_sync_done=False仅导出base、在加载后才设True（`:1103`、`:1191`）；`verl/utils/fsdp_utils.py:704`排除lora_，`verl/workers/rollout/vllm_rollout/vllm_rollout.py:240`只有True才add_lora，`vllm_async_server.py:509`未注册adapter时lora_request=None。REAL32 step101的冻结源码与当前版本SHA相符。REAL step51的历史worker版本也由本轮实现审计闭合：仅在内存删除当前fsdp_workers.py中后来新增、专属显式GradTop的六行guard，所得SHA256为`a2fd651564d4a2177075b651b34e0cfb3af4c147d4efab9014ac084d029ae60c`，与REAL32原续训plan冻结值完全一致，恢复和rollout同步代码未变化。因此step51/101均有冻结worker源码支持，而非只凭异常日志推测；下游未全部冻结、缺少当时engine tensor/hash实测的边界仍保留。见[implementation_audit.md](/root/peft-for-rl/runs/ops/gradtop_research_review_20261016/implementation_audit.md:25)及[cpu_audit_checks.json](cpu_audit_checks.json)。

这意味着：GradTop与REAL32的恢复次数和位置匹配，但缺陷的实际影响可能依赖各自policy、A是否训练等，不能认为相互抵消。LoRA连续，不能声称三方都只是同一种无害resume。比较仍可描述为已执行轨迹端点比较；论文的干净方法比较需要共同修复、同样的恢复验收。原端点评测的adapter分数不因此失效。

## 3. Minerva：`no_answer_rate`实际是未被指定抽取器识别

评测入口导入`tina_run/tina/analysis/rollout_utils.py`（`tina_run/scripts/local/eval/eval_full_bench_vllm.py:22`）。抽取器`:220`优先取最后boxed，再认少量固定短语；数值或表达式`ANSWER: ...`没有通用分支。汇总器`eval_full_bench_vllm.py:329`的parse_rate是format_valid均值，`:331`的no_answer_rate只是`not format_valid`均值。因此“无答案”并不是“模型没有产出最终答案”的语义观测。

| Minerva端点 | 未解析回答 | 含Answer: | hit32768 | 长度≤8192 |
|---|---:|---:|---:|---:|
| LoRA150 | 364 | 363 | 1 | 312 |
| REAL32 100 | 144 | 138 | 5 | 112 |
| REAL32 150 | 339 | 332 | 8 | 320 |
| GradTop100 | 170 | 165 | 4 | 134 |
| GradTop150 | 22 | 18 | 4 | 15 |

这些计数证明“未解析”大部分不是max-token截断，且绝大多数含显式Answer标签。这里只保存末行格式示例，不复制长推理过程。事实不等于所有未解析回答都是数学正确。后续保守重判给出其中确实能通过同checker的部分。

在原标签上，`accuracy = parse × accuracy_given_parsed`是精确恒等式；对其做对称乘积分解：GradTop150−LoRA150 Minerva +10.938pp＝parse项+10.758＋条件正确率项+0.179；GradTop150−REAL150 +10.570pp＝+9.884＋+0.686。**这是描述性分解，不是格式的因果贡献**：两个模型的已解析子集不同，不能据此声称98%的推理收益被格式解释。

## 4. 事后、统一的保守格式敏感性重判

发现格式现象后，先把规则写入[`minerva_answer_sensitivity.py`](minerva_answer_sensitivity.py)，再对全部11组Minerva应用：只处理原未解析回答；只接受整个completion的最后非空行、恰好单个`ANSWER:`前缀、可去掉外围Markdown强调/数学定界符、长度≤180；拒绝含or/and/assuming等歧义措辞或多候选±的尾行。将候选包装为boxed，使用原math_verify/LatexExtractionConfig的normalization与strict verify；parse/verify各1秒限制、总120秒、4CPU。拒绝或未验证的回答继续算错，所有原解析标签保持原样。

共2,815个符合规则候选全部在22.96秒内完成；无超时、无不可解析候选；这里“verified状态”表示checker成功执行且两侧可解析，不表示全正确。新增correct数如下。原记录和主指标没有改写。

| 方法/Step | 原Minerva Avg@4 | 扩展抽取敏感性Avg@4 | 新增正确/1088 |
|---|---:|---:|---:|
| LoRA50 | 22.335% | 28.033% | 62 |
| LoRA100 | 22.886% | 29.228% | 69 |
| LoRA150 | 22.702% | 28.401% | 62 |
| REAL64 50 | 28.309% | 30.974% | 29 |
| REAL32 50 | 25.827% | 32.077% | 68 |
| REAL32 100 | 30.882% | 33.272% | 26 |
| REAL32 150 | 23.070% | 30.055% | 76 |
| GradTop10 | 22.426% | 28.585% | 67 |
| GradTop50 | 32.537% | 34.007% | 16 |
| GradTop100 | 32.813% | 34.099% | 14 |
| GradTop150 | 33.640% | 33.640% | 0 |

六任务Macro只替换Minerva的这一个事后敏感性评分，其他五项沿用原评分。独立重算10,000次按题配对bootstrap、seed42，各任务内部题目有放回重采样，再等权任务平均；不是训练seed区间。由于生成bootstrap的任务顺序与原脚本可能不同，原口径重算CI的小幅Monte Carlo差异不覆盖原存档。

| 比较 | 原Macro Δ | 扩展抽取Macro Δ | 扩展抽取95%题目CI |
|---|---:|---:|---:|
| GradTop50−REAL32 50 | +0.992pp | +0.195pp | [−0.940,+1.329] |
| GradTop100−REAL32 100 | −0.660pp | −0.843pp | [−2.332,+0.598] |
| GradTop150−REAL32 150 | +2.989pp | +1.825pp | [+0.523,+3.200] |
| GradTop150−LoRA150 | +3.583pp | +2.633pp | [+1.318,+4.016] |
| REAL32 150−100 | −2.594pp | −1.828pp | [−3.028,−0.696] |
| GradTop150−100 | +1.055pp | +0.840pp | [−0.425,+2.174] |

去掉Minerva后的五任务Macro，GradTop150−REAL150为+1.473pp，探索性区间[+0.006,+3.040]，下界贴近0；GradTop50−REAL50为−0.151pp。不能用事后去掉任务来重新定义获胜主指标，只可说明任务集中度。扩展抽取后GradTop的Minerva50/100/150约34.01/34.10/33.64%，未表现出对应的后期语义能力持续增长；AIME24/25的后期改善仍存在，不能把整体方法收益一概归零。

证据：[`minerva_parse_diagnostics.json`](minerva_parse_diagnostics.json)、[`minerva_answer_sensitivity.json`](minerva_answer_sensitivity.json)、[`paired_sensitivity.json`](paired_sensitivity.json)；后者可由[`paired_sensitivity.py`](paired_sensitivity.py)重放。局限：这是发现现象后制定的保守诊断，既不是预注册benchmark，也未检查全部已解析错误或训练reward的反事实；候选包装可能仍对单位/等价表达敏感。结果支持“格式验收影响排名幅度”，不证明原协议不合法。

## 5. 训练日志反对简单的统一熵坍塌解释

下表排除step101，在102–120和131–150取等长近似窗口均值；题目批次不同，不能把窗口差值当配对因果效应。原始每步及全部固定窗口在training_comparison.json。

| 方法 | reward 102–120→131–150 | parse | hit8192 | entropy | mean length |
|---|---|---|---|---|---|
| LoRA | 46.44→46.56% | 74.18→78.72% | 30.71→26.49% | 0.770→0.732 | 5612→5499 |
| REAL32 | 51.19→47.52% | 95.24→95.44% | 6.46→4.76% | 1.017→1.155 | 4929→4609 |
| GradTop | 51.75→50.01% | 93.27→92.05% | 11.20→13.18% | 0.837→0.786 | 5336→5483 |

REAL32退步同时训练entropy升高、截断减少、训练parse高且稳定；所以“退步是熵耗尽/回答太长/格式没学会”均不是日志足以支持的统一解释。它在训练分布与Minerva上的格式行为不同，说明reward匹配、格式泛化、轨迹分布都需要区分。GradTop训练entropy降低，但测试AIME24/25后期单题平均成功率提高；entropy是actor在当前采样回答上的统计，不是独立、可比较的全空间策略熵估计。

hard-task覆盖也未统一提高。GradTop150与LoRA150的AIME24/25 Pass@32分别相等（73.3/56.7）；HMMT Avg@32为10.0% vs10.21%、Pass@32为40% vs36.67%。按32回答无放回子采样得到的HMMT Pass@4估计，GradTop15.85%、LoRA19.09%、REAL32 16.81%，与完整K覆盖排名不同；不能将Avg@k改善或Pass@32一个数字等同于广泛探索改善。所有低k数值是有限K样本的unbiased subsampling estimator，不是新增采样结果。

GradTop150共3505个正确回答，其中123条最终长度>8192；LoRA150 3256中152条，REAL150 3298中56条。训练cap8192与评测32768的差异需纳入预算解释，但该长度分组不是实际8k-cap评测（长轨迹也可能早已写出答案）。不能用截长度统计冒充重跑8k。

训练actor/grad_norm亦存在量级差：末20步LoRA0.0077、REAL32 0.0828、GradTop0.0690。不同A尺度/可训练参数化使该值不直接可比；仅凭同LR和相同rank不能断言更新强度相同。正式机制试验需要记录实际有效ΔW或函数变化，不能以因子梯度范数替代。

## 6. 对研究设计的具体约束

优先问题应问：**少量on-policy探测得到的共享更新方向，能否在独立rollout与新题目上预测可迁移的任务改善，并且这种预测在答案验收形式改变后仍成立？** 本报告不建议把新parser或format mask拼成新算法，而建议把它作为reward/评测可证伪性的必要审计。

对父任务的D0双重复rollout设计，至少需同时登记原协议reward、是否有明确最终答案、格式可解析性、保守语义验证的诊断标签及长度/截断。几何诊断依然以预定原reward主分支开展；在同一缓存上检查高共享方向是否只预测format_success、还是预测语义正确性与后续策略改善。若所谓共享更新优势完全依赖boxed合规、在独立语义审计下不预测任务能力，应否定“更好的推理更新空间”故事。若去噪几何预测实际更新收益且跨格式/题型稳定，才支撑比REAL逐题能量更强的RL-specific方法论。

现有日志没有逐题双独立rollout、完整训练更新或未来policy条件矩，无法从本报告直接判定rollout噪声、题间异质性或on-policy漂移哪一个主导。REAL32的端点退步是真实观察，且保守格式敏感性后仍存在；这只能支持优先调查动态失配，不能单凭退步证明fixed-A或Nyström有错。
