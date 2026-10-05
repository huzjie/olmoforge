# MXFP8 上线前核对清单

1. 确认激活/梯度的异常值是否集中在少数层（用 `MXFP8.roundtrip_error` 扫层）。
2. 给异常值大的层回退 BF16，其余层 MXFP8（「哪里划算哪里用」）。
3. 记录每层的分块 scale 是否稳定（scale 抖动 = 精度风险）。
4. 跑一个 step 的 loss 与 BF16 基线对比，误差应 <1%。
5. 验证吞吐增益是否兑现（`throughput_estimate` 给的是上界）。
