"""A compact MoE transformer: embedding -> N MoE blocks -> head.

Every block is a full MoE layer (matching OLMo/Kolibri-style all-MoE
architectures) rather than the common "every other layer is MoE" pattern.
"""
from ..tensor import Tensor, layernorm
from ..core.moe import MoELayer


class MoEBlock:
    def __init__(self, n_experts, top_k, dim, hidden, seed=0):
        self.moe = MoELayer(n_experts, top_k, dim, hidden, seed=seed)

    def forward(self, x):
        h, assignments = self.moe.forward(x)
        if h.data and isinstance(h.data[0], (list, tuple)):
            ln = Tensor(layernorm(h.data))
            return ln, assignments
        return h, assignments

    def parameters(self):
        return self.moe.parameters()


class MoETransformer:
    def __init__(self, n_layers, n_experts, top_k, dim, hidden, vocab, seed=0):
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.top_k = top_k
        self.dim = dim
        self.vocab = vocab
        self.blocks = [MoEBlock(n_experts, top_k, dim, hidden, seed=seed + i * 1000)
                       for i in range(n_layers)]

    def forward(self, x):
        assignments_all = []
        for b in self.blocks:
            x, a = b.forward(x)
            assignments_all.append(a)
        return x, assignments_all

    def total_params(self):
        return sum(b.moe.total_params() for b in self.blocks)

    def active_params(self):
        return sum(b.moe.active_params() for b in self.blocks)

    def parameters(self):
        out = []
        for b in self.blocks:
            out.extend(b.parameters())
        return out
