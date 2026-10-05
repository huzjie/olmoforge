# Benchmarks

| benchmark | what it shows |
|-----------|---------------|
| `throughput` | FSDP vs expert-parallel tokens/s and speedup vs expert count |
| `grouped_gemm` | kernel-launch savings from batching expert matmuls |
| `mxfp8` | BF16 -> MXFP8 throughput estimate (+21%) |
| `pipeline` | pipeline bubble ratio and utilization |
| `token_gerrymandering` | balance score vs real load imbalance |
| `scaling` | projection to a trillion-parameter MoE |

Run them via `python -m olmoforge bench`.
