"""Tests for the MoE layer and router."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

from olmoforge.core.moe import MoELayer
from olmoforge.core.router import load_balance_loss
from olmoforge.tensor import Tensor


class TestMoE(unittest.TestCase):
    def test_forward_shape(self):
        layer = MoELayer(n_experts=8, top_k=2, dim=16, hidden=8, seed=0)
        x = Tensor([[float(i) for i in range(16)]])
        out, assigns = layer.forward(x)
        self.assertEqual(out.shape, (1, 16))
        self.assertEqual(len(assigns), 1)

    def test_top_k_count(self):
        layer = MoELayer(n_experts=8, top_k=2, dim=16, hidden=8, seed=1)
        x = Tensor([[float(i) for i in range(16)]])
        _, assigns = layer.forward(x)
        _, picks = assigns[0]
        self.assertEqual(len(picks), 2)

    def test_load_balance_loss_nonnegative(self):
        loss = load_balance_loss([(0, [(0, 0.5), (1, 0.5)])], 8)
        self.assertGreaterEqual(loss, 0.0)

    def test_active_less_than_total(self):
        layer = MoELayer(n_experts=8, top_k=2, dim=16, hidden=8, seed=2)
        self.assertLess(layer.active_params(), layer.total_params())


if __name__ == "__main__":
    unittest.main()
