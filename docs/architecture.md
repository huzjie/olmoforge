# Architecture

olmoforge mirrors the OLMo-core 3 training stack in a zero-dependency,
runnable form. The key architectural decision is the shift from FSDP to
**DDP-based expert parallelism**.

## The core inversion

- **FSDP (baseline):** shard weights across GPUs, gather + reshard on every
  micro-batch. Communication scales with the *total parameter count*.
- **Expert parallelism (olmoforge):** experts stay resident on their home
  device; you route *data* to them. Communication scales with
  `seq_len x top_k x dim`, which is independent of the total expert count.

## Module map

```
olmoforge/
  tensor.py            zero-dependency tensor core (autodiff)
  core/                MoE layer, SwiGLU expert, top-k router
  parallel/            expert / pipeline / distributed optimizer
  routing/             rowwise EP, grouped GEMM, GPU-resident routing
  precision/           MXFP8 emulation
  model/               MoE transformer
  train/               trainer + FSDP-vs-EP comm estimate
  bench/               throughput / token-gerrymandering / scaling
  backends/            deterministic trainable mock
  serving/             stdlib HTTP status server
  cli/                 command-line interface
```

See `docs/parallelism.md` and `docs/mxfp8.md` for the deep dives.
