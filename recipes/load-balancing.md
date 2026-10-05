# 配方：负载均衡

负载均衡辅助损失 `n_experts * sum(f * (f - mean))` 与主损失按 1e-2~1e-3 加权。

要点：用 per-expert 直方图 + Gini 系数验证真实均衡，别只看均值平衡分（防
token gerrymandering）。
