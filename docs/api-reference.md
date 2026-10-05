# API 参考

## ExpertParallel

- `route_tokens(seq_len)` — 把序列路由到激活专家，返回 token-向量数
- `comm_for_seq(seq_len)` — 该序列的通信成本（bytes）

## FSDPLegacy

- `gather_weights(micro_batches)` — 每次 all-gather 的权重搬运量
- `comm_for_step(micro_batches)` — 单步通信成本

## MockBackend

- `score(question, candidate)` — 打分（skill × is_correct + noise）
- `predict(question, candidates)` — 取最高分候选
- `train_step(question, candidates, correct)` — 恒正 delta 更新 skill

## Trainer

- `comm_estimate(seq_len, micro_batches)` — FSDP vs 专家并行通信对比
- `run(epochs)` — 训练并返回 (epoch, skill, acc) 历史
