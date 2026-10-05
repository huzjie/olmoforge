import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
"""Pipeline bubble vs micro-batch count.

Run:  python examples/advanced/09_pipeline.py
"""
from olmoforge.parallel.pipeline import PipelineParallel

print("micro_batches -> bubble_ratio")
for m in [1, 2, 4, 8, 16, 32, 64]:
    pp = PipelineParallel(8, m)
    print(f"  {m:>3} -> {pp.bubble_ratio:>7.2%}")
