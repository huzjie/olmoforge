# olmoforge.parallel.dist_optimizer

`DistributedOptimizer` 是分片 AdamW：优化器状态（m/v）随权重分片而非每 rank
复制，per-rank 内存不随设备数增长。
