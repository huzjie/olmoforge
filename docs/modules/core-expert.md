# olmoforge.core.expert

`ExpertFFN` 是 SwiGLU 结构：`out = W2 @ (gelu(W1@x) * W3@x)`。W1/W3=(hidden×dim)，
W2=(dim×hidden)。注意维度：gate 分支和 value 分支都是 (hidden×dim) 的线性层。
