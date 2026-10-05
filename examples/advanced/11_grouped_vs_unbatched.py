import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
"""Grouped GEMM: launch count vs expert count.

Run:  python examples/advanced/11_grouped_vs_unbatched.py
"""
from olmoforge.routing.grouped_gemm import GroupedGEMM

for n in [8, 32, 128, 384]:
    gg = GroupedGEMM(n)
    print(f"experts={n:>4} speedup={gg.speedup:.2f}x launches_saved={gg.kernel_launches_saved}")
