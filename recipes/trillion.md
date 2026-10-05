# 配方：万亿参数 MoE 扩展投影

```bash
python -m olmoforge bench scaling   # 输入 total_params=1.2T active=58B
```

典型输出：~4096 专家、top_k=6、稀疏比 >20x。要点：活跃参数固定时，通信只随
`top_k` 走，不随专家数走——这正是专家并行能扩到万亿的根本原因。
