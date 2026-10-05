# olmoforge.tensor

零依赖张量核心。`Tensor` 用嵌套 list 存数据，带 `requires_grad` / `grad` /
`_ctx` 三件套。支持 add/sub/mul/div/relu/gelu/sum/mean/matmul 与反向传播。

关键实现点：md5 前 8 字节做种子（见 utils/stable.py 注释），避免低熵。
