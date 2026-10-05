# MXFP8 low-precision training

OLMo-core 3 reports ~+21% training throughput over BF16 when MXFP8 is applied
where it helps most. MXFP8 is a *micro-scaling* FP8 format: it shares a
per-block scale across a small group of values, recovering dynamic range that
per-tensor FP8 loses.

## How the emulation works

`precision/mxfp8.py` quantizes block-by-block:

1. split values into blocks of `block_size`
2. compute the block max magnitude -> scale
3. round each value / scale to the FP8 grid
4. store the shared scale once per block

Round-trip error is measured with `MXFP8.roundtrip_error`; the throughput
estimate uses `throughput_estimate(bf16_tps, gain=0.21)`.
