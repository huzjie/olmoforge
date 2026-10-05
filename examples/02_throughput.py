import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
"""FSDP vs DDP expert-parallel throughput (the OLMo-core 3 headline).

Run:  python examples/02_throughput.py
"""
from olmoforge.bench import run_benchmark

run_benchmark("throughput", verbose=True)
