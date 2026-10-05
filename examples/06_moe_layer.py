import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
"""Build a real sparse MoE layer and run a forward pass.

Run:  python examples/06_moe_layer.py
"""
from olmoforge.core.moe import MoELayer
from olmoforge.tensor import Tensor

layer = MoELayer(n_experts=32, top_k=6, dim=64, hidden=32, seed=0)
x = Tensor([[0.1 * (i % 7 - 3) for i in range(64)]])
out, assignments = layer.forward(x)
print("output shape:", out.shape)
print("assignments (first token):", assignments[0][1][:3], "...")
print(f"active params ~ {layer.active_params():,} / total {layer.total_params():,}")
