# 提吞吐排查顺序（工作流模板）

1. 先看通信/计算比：`trainer.comm_estimate()`，通信占大头才值得动并行策略。
2. 分组 GEMM 是否生效（`grouped_gemm` 基准的 launches_saved）。
3. 路由元数据是否 GPU 常驻（`gpu_resident.sync_cost_units()`）。
4. 流水线气泡（`pipeline.bubble_ratio`）——加 micro-batch。
5. 最后才考虑 MXFP8（收益最大但引入精度风险，放最后）。
