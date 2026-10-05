# 配方：检查点

- 专家并行下每个 rank 只存自己常驻的专家分片。
- 保存时带 `n_experts / top_k / dim / capacity_factor` 元数据。
- 恢复时先校验分片数与专家 home 设备一致，再加载。
