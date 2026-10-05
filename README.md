# olmoforge

**Trillion-parameter Mixture-of-Experts training framework** — a zero-dependency,
fully runnable reference implementation of the training techniques behind
[AI2's OLMo-core 3](https://github.com/allenai/OLMo-core) (released 2026-10-01),
which scales MoE training into the trillion-parameter range while keeping
throughput nearly flat as expert count grows.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

---

## 一句话原理

传统 FSDP 每个 micro-batch 都要 **搬运权重**（all-gather + reduce-scatter），
通信量随总参数量线性涨。OLMo-core 3 反过来：**专家常驻 GPU，只把 token 数据
路由过去**，通信量 = `seq_len × top_k × dim`，与专家总数无关。所以专家数从 8
涨到 384，吞吐几乎不降。

## 核心能力

| 模块 | 做了什么 | 可复用点 |
|------|---------|---------|
| `parallel/expert_parallel.py` | DDP 式专家并行 | 把「搬权重」换成「搬数据」 |
| `parallel/pipeline.py` | 流水线并行 | 1F1B 气泡估算 `(p-1)/(p+m-1)` |
| `parallel/dist_optimizer.py` | 分布式优化器 | 优化器状态随权重分片，不复制 |
| `routing/rowwise.py` | 行式专家并行 | token 直写专家输入缓冲，免重排 |
| `routing/grouped_gemm.py` | 分组 GEMM | 把 N 个小 matmul 合成 1 个大 kernel |
| `routing/gpu_resident.py` | GPU 常驻路由表 | 元数据留设备上，CPU 不等待拷贝 |
| `precision/mxfp8.py` | MXFP8 低精度 | 分块共享 scale，恢复动态范围 |
| `bench/token_gerrymandering.py` | 反直觉发现 | 别信单一聚合路由指标 |

## 快速开始（零依赖，Python 3.9+）

```bash
python -m olmoforge doctor        # 冒烟：所有组件 import + 实例化
python -m olmoforge train         # 训练 mock 后端，skill 0.5 -> ~1.0
python -m olmoforge throughput    # FSDP vs 专家并行吞吐对比
python -m olmoforge bench         # 跑全部基准
python -m olmoforge serve         # 起 stdlib HTTP 状态服务 :8899
```

## 可复用的干货块（直接抄）

**决策模板：什么时候从 FSDP 切到专家并行？**

```python
from olmoforge.parallel.expert_parallel import ExpertParallel, FSDPLegacy
from olmoforge.utils.dist import MockWorld

def should_use_expert_parallel(n_experts, top_k, dim, seq_len=128):
    world = MockWorld(n_devices=8)
    ep_cost  = ExpertParallel(n_experts, top_k, dim, world).comm_for_seq(seq_len)
    fsdp_cost = FSDPLegacy(n_experts * 3 * dim * dim, world).comm_for_step(1)
    return ep_cost < fsdp_cost, fsdp_cost / max(ep_cost, 1)

print(should_use_expert_parallel(128, 6, 4096))   # -> (True, 较大的倍率)
```

**MXFP8 分块量化（可抄的实现）：**

```python
def mxfp8_quantize(values, block_size=32, e4m3_max=448.0):
    q, scales = [], []
    for i in range(0, len(values), block_size):
        block = values[i:i+block_size]
        scale = max(abs(v) for v in block) / e4m3_max or 1.0
        scales.append(scale)
        q.extend(round(v / scale) for v in block)
    return q, scales
```

## 训练信号铁律（踩坑总结）

训练 mock 后端时，**别用「对 +1 / 错 -1」**。初始准确率 <50% 时期望漂移为负，
skill 会被压到 0。正确做法是让 delta **恒为正**（对 +0.5 / 错 +0.2），skill
单调爬向 1.0。见 `backends/mock.py`。

## 目录结构

```
olmoforge/
  tensor.py       零依赖张量核心（自动微分）
  core/           MoE 层 / SwiGLU 专家 / Top-K 路由
  parallel/       专家并行 / 流水线 / 分布式优化器
  routing/        行式EP / 分组GEMM / GPU常驻路由
  precision/      MXFP8
  model/          MoE Transformer
  train/          训练循环 + 通信成本对比
  bench/          吞吐 / 反直觉发现 / 万亿扩展
  backends/       确定性可训练 mock
  serving/        stdlib HTTP 状态服务
  cli/            命令行
examples/ tests/ docs/ configs/ deploy/ prompts/ recipes/
```

## 许可证

Apache-2.0。本项目是对 AI2 OLMo-core 3 公开技术的独立参考实现，与 AI2 无关。
