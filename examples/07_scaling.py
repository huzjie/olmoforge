import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
"""Project the scaling curve to a trillion-parameter MoE.

Run:  python examples/07_scaling.py
"""
from olmoforge.bench import run_benchmark

run_benchmark("scaling", verbose=True)
