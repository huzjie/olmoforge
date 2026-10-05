"""SwiGLU expert feed-forward network.

Each expert is a small FFN. In a sparse MoE only the top-k routed experts run
for a given token, so total compute is ~(active/total) of the dense cost.
"""
from ..tensor import Tensor


class ExpertFFN:
    """SwiGLU expert: out = W2 @ (gelu(W1@x) * (W3@x))."""

    def __init__(self, dim, hidden, seed=0):
        self.dim = dim
        self.hidden = hidden
        import random

        r = random.Random(seed)
        scale = 1.0 / (dim ** 0.5)

        def vv(rows, cols):
            return Tensor([[r.uniform(-scale, scale) for _ in range(cols)]
                           for _ in range(rows)], requires_grad=True)

        self.w1 = vv(hidden, dim)
        self.w3 = vv(hidden, dim)
        self.w2 = vv(dim, hidden)

    def forward(self, x):
        """x: Tensor of shape (dim,) or (seq, dim). Returns a Tensor (1, dim)."""
        gate = self.w1.matmul(x).gelu()
        act = self.w3.matmul(x)
        prod = gate * act
        return self.w2.matmul(prod)

    def parameters(self):
        return [self.w1, self.w3, self.w2]
