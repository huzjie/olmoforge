import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
"""Grouped GEMM kernel-launch savings.

Run:  python examples/05_grouped_gemm.py
"""
from olmoforge.routing.grouped_gemm import GroupedGEMM

gg = GroupedGEMM(384)
print(f"384 experts: unbatched {gg.unbatched_time_us():.0f}us "
      f"vs grouped {gg.grouped_time_us():.0f}us = {gg.speedup:.2f}x "
      f"({gg.kernel_launches_saved} launches saved)")
