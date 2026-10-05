# olmoforge.parallel.expert_parallel

`ExpertParallel`：专家常驻 home device，token 数据路由过去。`comm_for_seq` =
`seq_len × top_k × dim × bytes`，与专家总数无关。

`FSDPLegacy` 是对比基线：每步 all-gather 全量权重，`comm_for_step` 随总参数
线性增长。
