import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
"""Start the stdlib status server (Ctrl-C to stop).

Run:  python examples/08_serve.py
"""
from olmoforge.serving.server import run_server

run_server(n_experts=384, top_k=6, port=8899)
