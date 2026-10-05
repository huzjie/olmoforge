"""Tests for the zero-dependency tensor core."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

from olmoforge.tensor import Tensor, softmax, layernorm


class TestTensor(unittest.TestCase):
    def test_shape(self):
        t = Tensor([[1, 2], [3, 4]])
        self.assertEqual(t.shape, (2, 2))

    def test_matmul(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        c = a.matmul(b)
        self.assertEqual(c.data, [[19, 22], [43, 50]])

    def test_add(self):
        a = Tensor([[1, 2]])
        c = a + Tensor([[3, 4]])
        self.assertEqual(c.data, [[4, 6]])

    def test_relu(self):
        t = Tensor([[-1, 2]]).relu()
        self.assertEqual(t.data, [[0, 2]])

    def test_backward_matmul(self):
        a = Tensor([[1.0, 2.0]], requires_grad=True)
        b = Tensor([[3.0], [4.0]], requires_grad=True)
        c = a.matmul(b)
        c.backward()
        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(b.grad)

    def test_softmax_sums_to_one(self):
        p = softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(p), 1.0)

    def test_layernorm_2d(self):
        out = layernorm([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(sum(out[0]), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
