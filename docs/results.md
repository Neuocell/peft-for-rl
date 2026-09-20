# 当前结果

这里只记录当前仓库要保留的核心基线结果。早期 OpenRS3、OrthRes、critique、format/cos reward、overlong buffer 等探索保留在原 Tina 工作目录中，不作为这个复现实验仓库的主体。

## Stable LoRA Fullbench 32K

配置：

```text
Base model: DeepSeek-R1-Distill-Qwen-1.5B
Train data: DAPO-Math-17k boxed
Reward: boxed outcome accuracy only
Train max_response_length: 8192
Eval max_new_tokens: 32768
Eval benchmarks: aime24,aime25,amc23,hmmt_feb,math500,minerva
Small benchmark samples: 32
Large benchmark samples: 4
```

总体结果：

| ckpt | macro avg | macro pass | parse | hit max | mean len |
|---|---:|---:|---:|---:|---:|
| base | 0.315514 | 0.571642 | 0.804084 | 0.040977 | 9241.7 |
| step100 | 0.331179 | 0.588252 | 0.806567 | 0.036700 | 9137.2 |
| step200 | 0.340859 | 0.586039 | 0.816915 | 0.036010 | 9005.7 |
| step300 | 0.346207 | 0.611590 | 0.837610 | 0.041115 | 9118.6 |

step300 分 benchmark：

| benchmark | avg | pass | parse | hit max | mean len |
|---|---:|---:|---:|---:|---:|
| aime24 | 0.273958 | 0.733333 | 0.841667 | 0.078125 | 14027.8 |
| aime25 | 0.205208 | 0.533333 | 0.860417 | 0.081250 | 13789.1 |
| amc23 | 0.637500 | 0.925000 | 0.871094 | 0.033594 | 8224.7 |
| hmmt_feb | 0.105208 | 0.366667 | 0.890625 | 0.072917 | 15092.9 |
| math500 | 0.655000 | 0.784000 | 0.914000 | 0.010000 | 4122.8 |
| minerva | 0.200368 | 0.327206 | 0.587316 | 0.011029 | 5629.9 |

观察：

```text
base -> step300 macro avg: +0.030693
base -> step300 macro pass: +0.039948
base -> step300 parse: +0.033526
平均长度没有明显失控，保持在约 9K
```

这说明当前 stable LoRA recipe 至少能产生正向训练信号，并且没有复现 overlong buffer 版本中的严重长度 hacking。

## Stable OFT 状态

```text
exp_name=dapo_math_boxed_stable_oft_1p5b_4gpu_8k_v1_20260724
global_step_100 actor checkpoint: 已在原机器生成
step100 eval: 之前因 controller 路径/日志污染问题失败，当前脚本已修复
step200/step300: 待继续训练和评测
```

OFT 后续应使用和 LoRA 完全相同的 stable v1 协议。差异只允许来自 PEFT 参数化：

```text
PEFT_TYPE=oft
OFT_BLOCK_SIZE=32
OFT_RANK=0
OFT_DROPOUT=0.0
```

## 不再作为当前主线的内容

早期 OrthRes/OpenRS3 试验显示可以影响训练动态，但当时 reward、format、生成长度、数据难度和评测协议没有完全稳定，难以作为 clean baseline 的结论来源。本仓库现在只保留稳定基线和最小工具；OrthRes 后续若重新比较，应在本 stable recipe 上只改方法项。

