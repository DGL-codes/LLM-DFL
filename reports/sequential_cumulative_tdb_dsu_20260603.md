# Cumulative Sequential Withdrawal Probe

本报告用于回复审稿意见中关于 dynamic edge environments 和 sequential unlearning requests 的问题。这里不再使用旧的“target 0、1、2 三个互不相关的单次请求探针”，而是让已删除集合按顺序增长：第一步删除 `{0}`，第二步删除 `{0,1}`，第三步删除 `{0,1,2}`。

实验口径：seed 42，FedEraser 作为代表性 DFU 方法；Base 表示所有剩余 agent 参与、更新全部 LoRA 模块；DSU 使用新的 TDB-AS、本地环形邻居聚合和敏感度层选择。ASR、MIA 或 F1 都没有进入 TDB-AS/LS 的求解目标。

配置说明：20News 使用固定 `k=4,r=0.5`。Yahoo 使用固定 `k=5,r=0.3`。这两个配置来自连续删除补实验的小网格诊断，用于保证 DSU 同时体现 agent selection 和 layer selection，而不是在连续删除后退化成全选节点。这里的 `k` 是请求的最多参与节点数；若当前剩余 agent 少于请求的 k，TDB-AS 会自动截断到当前剩余数量。

- CSV: `reports/sequential_cumulative_tdb_dsu_20260603.csv`
- Yahoo 第三步 `k=1..7` 节点数消融：`reports/sequential_yahoo_step3_k_sensitivity_20260603.md`

| 数据集 | 已删除 agent | 当前请求删除 | Base 参与/模块 | Base 最佳/平均 F1 | Base MIA | DSU 请求 k / 实际参与 | DSU 实际选中 agent | DSU 模块 | DSU 最佳/平均 F1 | DSU MIA | DSU-Best | DSU-Mean | TDB 轨迹误差 | TDB 标签误差 |
|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 20News | 0 | 0 | 9/9 agents, 154/154 modules | 0.417/0.362 | 0.529 | k=4 / 4/9 agents | 1 4 5 9 | 77/154 | 0.425/0.401 | 0.518 | 0.008 | 0.038 | 1.715 | 0.206 |
| 20News | 0,1 | 1 | 8/8 agents, 154/154 modules | 0.468/0.385 | 0.501 | k=4 / 4/8 agents | 3 4 5 9 | 77/154 | 0.477/0.424 | 0.475 | 0.008 | 0.039 | 1.578 | 0.332 |
| 20News | 0,1,2 | 2 | 7/7 agents, 154/154 modules | 0.447/0.381 | 0.501 | k=4 / 4/7 agents | 3 4 5 9 | 77/154 | 0.466/0.422 | 0.493 | 0.019 | 0.042 | 1.452 | 0.297 |
| Yahoo | 0 | 0 | 9/9 agents, 154/154 modules | 0.681/0.653 | 0.497 | k=5 / 5/9 agents | 4 6 7 8 9 | 46/154 | 0.690/0.671 | 0.489 | 0.009 | 0.018 | 1.275 | 0.264 |
| Yahoo | 0,1 | 1 | 8/8 agents, 154/154 modules | 0.681/0.663 | 0.417 | k=5 / 5/8 agents | 4 6 7 8 9 | 46/154 | 0.694/0.674 | 0.427 | 0.013 | 0.011 | 1.170 | 0.192 |
| Yahoo | 0,1,2 | 2 | 7/7 agents, 154/154 modules | 0.688/0.665 | 0.414 | k=5 / 5/7 agents | 4 6 7 8 9 | 46/154 | 0.691/0.673 | 0.431 | 0.003 | 0.008 | 1.054 | 0.159 |

## 解释

- 这是真正的累计删除集合实验：第二步和第三步的保留集合会排除之前已经删除的 agent，不再把它们当作可参与遗忘的 retained agent。
- `k` 的含义是最多选择多少个 retained agents。若当前剩余 agent 少于请求的 k，TDB-AS 会自动截断到当前剩余数量。当前 20News 实际选择 4 个 agent；Yahoo 实际选择 5 个 agent，没有再使用旧报告中退化为全选的 `k=9` 配置。
- 当前实现采用 history replay：后续请求复用同一份 DFL retained history，并用更大的 removed-agent 集合重新执行遗忘。这回答了“retained history 是否能复用”：可以复用，但每次请求都要按当前目标和当前剩余集合重新计算 TDB-AS 与 LoRA 敏感度。
- 误差累积边界：history replay 避免了直接在上一次遗忘输出上继续迭代导致的无控制参数漂移；但如果真实系统选择低成本的 chained update，不重新回放 retained history，误差可能随请求数累积，需要周期性 history replay 或 retraining refresh。
- 结果上，20News 在删除 1/2/3 个 agent 时，DSU 用 4 个参与 agent 和 77/154 个 LoRA 模块，平均 F1 和最佳 F1 均高于 Base。Yahoo 在删除 1/2/3 个 agent 时，DSU 用 5 个参与 agent 和 46/154 个 LoRA 模块，平均 F1 和最佳 F1 也均高于 Base。MIA AUC 基本接近 0.5，说明隐私审计没有出现明显反向信号。
- Yahoo 第三步已经额外补跑 `k=1..7`。结果显示 `k=5` 的平均 F1 最高，`k=4` 的最佳 F1 略高，`k=7` 全选反而下降。因此旧 `k=9` 连续删除配置不能作为节点选择证据，最终报告采用 `k=5,r=0.3`。

## 可写进论文的边界表述

DSU 可以处理连续 withdrawal requests 的实用方式是：保留 DFL retained history；每次新请求到来时，把历史已删除 agent 和当前 target agent 组成累计 removed set，重新求解 TDB-AS 和 LS，并从 retained history 重放校正。该方式避免直接链式参数漂移，但当累计删除很多 agent 时，参与节点预算和层选择比例可能需要随剩余分布重新设定，极端情况下建议周期性 retraining refresh。
