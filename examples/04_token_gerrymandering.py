import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
"""The counterintuitive routing finding: balance score vs real imbalance.

Run:  python examples/04_token_gerrymandering.py
"""
from olmoforge.bench import run_benchmark

run_benchmark("token_gerrymandering", verbose=True)
