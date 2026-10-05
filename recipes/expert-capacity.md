# 配方：专家容量调优

capacity_factor 越小越省显存，但丢 token 越多。经验起点 1.25，观察丢 token 率：

```python
# 丢 token 率 = 1 - 实际处理 token / 路由请求 token
```

丢 token 率 >5% 就上调 capacity_factor；<1% 可尝试下调省显存。
