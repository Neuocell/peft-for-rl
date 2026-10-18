# GradTop / REAL 固定策略梯度诊断

更新时间：2026-10-17T12:42:33.305489+08:00。完成5/5个策略状态。

64个固定prompts、两组独立K8；采样temperature1/top-p1、8192输出上限。
模块：第0/13/27层q_proj与down_proj，共6个。两组独立输出sketch各宽32；正式训练optimizer更新数为0。

| Policy | Reward | Parse | 平均长度 | Hit-max | 零advantage组/128 |
| --- | ---: | ---: | ---: | ---: | ---: |
| gradtop100 | 55.08% | 91.02% | 5297.0 | 13.77% | 56 |
| real100 | 59.47% | 91.21% | 5147.5 | 13.09% | 63 |
| gradtop150 | 55.76% | 92.58% | 5190.4 | 12.01% | 62 |
| real150 | 53.22% | 92.97% | 4337.4 | 7.23% | 59 |
| gradtop270 | 58.79% | 99.12% | 4709.4 | 0.59% | 64 |

下表对6个模块、两个独立sketch取等权均值；百分比是各自梯度能量保留率，不是准确率。完整分层值及逐题证据见各policy的analysis.json/gram_evidence.npz。

| Policy | 归约 | 候选子空间 | 逐题/partial-prompt能量保留 | 共享更新能量保留 |
| --- | --- | --- | ---: | ---: |
| gradtop100 | equal | gradtop_init | 41.174% | 41.120% |
| gradtop100 | equal | real_init | 46.991% | 46.921% |
| gradtop100 | equal | current_A | 41.177% | 41.122% |
| gradtop100 | global_token | gradtop_init | 42.262% | 42.064% |
| gradtop100 | global_token | real_init | 47.922% | 47.719% |
| gradtop100 | global_token | current_A | 42.264% | 42.066% |
| gradtop100 | formal | gradtop_init | 41.373% | 41.687% |
| gradtop100 | formal | real_init | 47.085% | 47.247% |
| gradtop100 | formal | current_A | 41.375% | 41.690% |
| real100 | equal | gradtop_init | 42.561% | 42.823% |
| real100 | equal | real_init | 48.752% | 49.354% |
| real100 | equal | current_A | 48.752% | 49.354% |
| real100 | global_token | gradtop_init | 42.822% | 42.736% |
| real100 | global_token | real_init | 48.594% | 48.779% |
| real100 | global_token | current_A | 48.594% | 48.779% |
| real100 | formal | gradtop_init | 42.659% | 43.086% |
| real100 | formal | real_init | 48.954% | 49.567% |
| real100 | formal | current_A | 48.954% | 49.567% |
| gradtop150 | equal | gradtop_init | 42.376% | 42.962% |
| gradtop150 | equal | real_init | 47.915% | 48.481% |
| gradtop150 | equal | current_A | 42.374% | 42.960% |
| gradtop150 | global_token | gradtop_init | 43.341% | 43.840% |
| gradtop150 | global_token | real_init | 48.828% | 49.308% |
| gradtop150 | global_token | current_A | 43.341% | 43.840% |
| gradtop150 | formal | gradtop_init | 42.457% | 43.454% |
| gradtop150 | formal | real_init | 48.013% | 48.802% |
| gradtop150 | formal | current_A | 42.455% | 43.450% |
| real150 | equal | gradtop_init | 39.329% | 39.524% |
| real150 | equal | real_init | 44.952% | 45.317% |
| real150 | equal | current_A | 44.952% | 45.317% |
| real150 | global_token | gradtop_init | 39.882% | 39.997% |
| real150 | global_token | real_init | 45.385% | 45.689% |
| real150 | global_token | current_A | 45.385% | 45.689% |
| real150 | formal | gradtop_init | 39.915% | 39.299% |
| real150 | formal | real_init | 45.637% | 45.143% |
| real150 | formal | current_A | 45.637% | 45.143% |
| gradtop270 | equal | gradtop_init | 46.139% | 45.702% |
| gradtop270 | equal | real_init | 52.067% | 51.860% |
| gradtop270 | equal | current_A | 46.094% | 45.646% |
| gradtop270 | global_token | gradtop_init | 46.265% | 45.754% |
| gradtop270 | global_token | real_init | 52.192% | 51.906% |
| gradtop270 | global_token | current_A | 46.221% | 45.700% |
| gradtop270 | formal | gradtop_init | 46.187% | 45.148% |
| gradtop270 | formal | real_init | 52.097% | 51.642% |
| gradtop270 | formal | current_A | 46.128% | 45.068% |

| Policy | 全空间可重复逐题能量 / 总能量 | 双组非零梯度余弦 |
| --- | ---: | ---: |
| gradtop100 | -0.0013 | 0.0020 |
| real100 | 0.0093 | 0.0115 |
| gradtop150 | -0.0186 | -0.0242 |
| real150 | -0.0076 | -0.0085 |
| gradtop270 | -0.0073 | -0.0054 |

跨重复能量允许为负，不截断为正；噪声与异质性分解及区间保留在analysis.json。
增量SVD审计固定同一统计目标和同一sketch流，拟合repeat0/projectionA，评价repeat1/projectionB；诊断width32不同于正式GradTop probe width8，其耗时不能当作正式probe成本。
REAL原目标敏感性保留在前32题的Gram中；有效组按两次都满足至少2正2负限定。原token与系数保留，可重新反传。
formal分支复现长度均衡、四DP划分和动态microbatch权重，但所有四个逻辑update冻结在同一个policy，不能称为历史Adam更新的逐位复现。
当前轨迹51/101的旧恢复异常仍存在；新诊断加载采用源权重→vLLM注册及CUDA槽位、HF adapter的独立检查。
REAL270没有checkpoint，本轮缺失该点；诊断题来自训练源且排除probe前4096题，不是独立测试集。
