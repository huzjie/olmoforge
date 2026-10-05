# 配方：快速演示（单机，秒级）

```bash
python examples/01_train_mock.py   # skill 0.5 -> 1.0
python examples/02_throughput.py   # FSDP vs 专家并行
python examples/04_token_gerrymandering.py
```

无需 GPU、无需网络、零依赖，三条命令即可完整复现本项目全部核心结论。
