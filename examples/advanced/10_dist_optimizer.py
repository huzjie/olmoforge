import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
"""Distributed optimizer memory saving.

Run:  python examples/advanced/10_dist_optimizer.py
"""
from olmoforge.parallel.dist_optimizer import DistributedOptimizer

do = DistributedOptimizer(8)
n = 1_000_000
replicated = do.replicated_memory(n)
sharded = do.memory_per_rank(n)
print(f"replicated/rank: {replicated/1e6:.1f} MB")
print(f"sharded/rank   : {sharded/1e6:.1f} MB")
print(f"saving          : {(1 - sharded/replicated)*100:.0f}%")
