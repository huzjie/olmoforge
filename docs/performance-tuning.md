# 性能调优

按顺序排查（收益从大到小、风险从低到高）：

1. 通信/计算比 — `trainer.comm_estimate()`
2. 分组 GEMM — 减少 kernel launch
3. GPU 常驻路由 — 消除元数据拷贝
4. 流水线气泡 — 加 micro-batch
5. MXFP8 — 最后上，精度风险最高
