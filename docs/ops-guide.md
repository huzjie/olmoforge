# 运维指南

## 健康检查

```bash
curl http://127.0.0.1:8899/health    # -> {"status":"ok"}
curl http://127.0.0.1:8899/status    # -> 专家数/top-k 快照
```

## 监控要点

- per-expert 负载直方图（防 token gerrymandering）
- 通信/计算比（`trainer.comm_estimate()`）
- 分块 scale 稳定性（MXFP8 精度风险前兆）
