# olmoforge.core.router

`TopKRouter` 对每个 token 打分（`x @ W`），softmax 后取 top-k 专家和门控权重。
`load_balance_loss` 是 Shazeer 式辅助损失，鼓励专家使用均匀。
