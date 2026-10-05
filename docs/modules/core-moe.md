# olmoforge.core.moe

`MoELayer` 组合路由器和专家 FFN：路由每个 token 到 top-k 专家，只跑被选中的
专家，输出 gated 加权。`active_params()` 估算每 token 实际触达的参数量，
`total_params()` 是全部参数量——两者的比就是稀疏比。
