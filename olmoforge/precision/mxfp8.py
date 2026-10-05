"""MXFP8 micro-scaling float emulation.

OLMo-core 3 reports ~+21% training throughput over BF16 when MXFP8 is applied
where it helps most. MXFP8 is a micro-scaling format: it shares a per-block
scale across a small group of FP8 values, which recovers much of the dynamic
range lost by per-tensor FP8.

This module *emulates* the format in pure Python (quantize -> dequantize) so
the precision benchmark can measure the quantization error and estimate the
throughput delta, without requiring MXFP8 hardware.
"""


class MXFP8:
    def __init__(self, block_size=32, e4m3_max=448.0):
        self.block_size = block_size
        self.e4m3_max = e4m3_max

    def quantize(self, values):
        """Quantize a list of floats block-by-block with a shared scale."""
        out = []
        scales = []
        for i in range(0, len(values), self.block_size):
            block = values[i:i + self.block_size]
            mx = max(abs(v) for v in block) or 1.0
            scale = mx / self.e4m3_max
            scales.append(scale)
            q = [round(v / scale) for v in block]
            out.extend(q)
        return out, scales

    def dequantize(self, q, scales):
        out = []
        bs = self.block_size
        for i, s in enumerate(scales):
            block = q[i * bs:(i + 1) * bs]
            out.extend(v * s for v in block)
        return out

    def roundtrip_error(self, values):
        q, s = self.quantize(values)
        d = self.dequantize(q, s)
        n = len(values)
        err = sum((a - b) ** 2 for a, b in zip(values, d)) / n
        return err


def quantize_mxfp8(values, block_size=32):
    return MXFP8(block_size).quantize(values)


def dequantize_mxfp8(q, scales):
    return MXFP8().dequantize(q, scales)


def throughput_estimate(bf16_baseline_tokens_per_s, gain=0.21):
    """Estimate MXFP8 throughput from a BF16 baseline (paper: ~+21%)."""
    return bf16_baseline_tokens_per_s * (1.0 + gain)
