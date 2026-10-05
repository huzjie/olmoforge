# olmoforge.routing.grouped_gemm

`GroupedGEMM` 建模把 N 个小专家 matmul 合成 1 个大 kernel 的 launch 开销节省。
`speedup` 和 `kernel_launches_saved` 两个指标。
