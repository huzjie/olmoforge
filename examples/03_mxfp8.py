import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
"""MXFP8 quantization error + throughput estimate.

Run:  python examples/03_mxfp8.py
"""
from olmoforge.precision.mxfp8 import MXFP8, throughput_estimate
from olmoforge.utils.stable import stable_vector

vals = stable_vector("fp8-demo", 256, -1.0, 1.0)
fmt = MXFP8(block_size=32)
err = fmt.roundtrip_error(vals)
print(f"MXFP8 roundtrip MSE over 256 floats: {err:.2e}")
print(f"BF16 52400 tok/s -> MXFP8 ~{round(throughput_estimate(52400))} tok/s (+21%)")
