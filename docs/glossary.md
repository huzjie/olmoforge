# 术语表

- **MoE** — Mixture-of-Experts，稀疏门控 FFN。
- **Top-K** — 每个 token 激活的专家数。
- **Expert parallelism** — 专家常驻、数据路由的并行策略。
- **FSDP** — Fully Sharded Data Parallel，权重分片 + 每步 gather。
- **MXFP8** — 分块共享 scale 的 FP8 微缩放格式。
- **Token gerrymandering** — 聚合路由指标与实际负载分布背离的现象。
- **1F1B** — 流水线调度（1 forward 1 backward）。
