# Parallelism: expert + pipeline + distributed optimizer

OLMo-core 3 combines three distribution techniques. Each is implemented here
as a small, self-contained module with a countable cost model.

## 1. Expert parallelism (`parallel/expert_parallel.py`)

Experts are resident; data is routed. The communication per step is
`seq_len x top_k x dim`, independent of how many experts exist. This is why
throughput stays nearly flat as expert count grows.

## 2. Pipeline parallelism (`parallel/pipeline.py`)

Layers are split across device groups. The classic pipeline bubble is
`(p-1)/(p+m-1)`; micro-batching drives it toward zero. See the `pipeline`
benchmark.

## 3. Distributed optimizer (`parallel/dist_optimizer.py`)

Optimizer state is sharded alongside the weights instead of replicated, so
per-rank memory does not grow with device count.

## Cost model

`utils/dist.py` provides `MockWorld`, which counts "token-vectors shipped".
The throughput benchmark compares `comm_for_seq` (expert-parallel) against
`comm_for_step` (FSDP) and reports the speedup.
