# 配方：47B / 128 专家（对齐 OLMo-core 3）

```yaml
# configs/olmo47b.yaml
moe:   { n_experts: 128, top_k: 6 }
model: { dim: 8192, hidden: 3072, n_layers: 32 }
parallel: { expert: true, pipeline_stages: 8, micro_batches: 32 }
precision: { mxfp8: true, block_size: 32 }
```

预期：专家并行通信 << FSDP，吞吐降幅 <5%（论文口径），MXFP8 再 +21%。
验证：`python -m olmoforge throughput` + `python -m olmoforge bench mxfp8`。
