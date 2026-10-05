"""Sparse Mixture-of-Experts layer.

Combines the router and the expert FFNs: route each token to top-k experts,
run only those experts, and combine their gated outputs.
"""
from .expert import ExpertFFN
from .router import TopKRouter


class MoELayer:
    """One MoE block: x -> router -> top-k experts -> gated combination."""

    def __init__(self, n_experts, top_k, dim, hidden, seed=0, capacity_factor=1.25):
        self.n_experts = n_experts
        self.top_k = top_k
        self.dim = dim
        self.router = TopKRouter(n_experts, top_k, dim, seed=seed,
                                 capacity_factor=capacity_factor)
        self.experts = [ExpertFFN(dim, hidden, seed=seed + 1000 + i)
                        for i in range(n_experts)]

    def forward(self, x):
        """x: Tensor (dim,) or (seq, dim). Returns (out, assignments)."""
        assignments = self.router.forward(x)
        if x.data and isinstance(x.data[0], (list, tuple)):
            out = [[0.0] * self.dim for _ in range(len(x.data))]
            for idx, picks in assignments:
                for j, g in picks:
                    e = self.experts[j].forward(Tensor([x.data[idx]]))
                    row = e.data[0]
                    for k in range(self.dim):
                        out[idx][k] += g * row[k]
            from ..tensor import Tensor

            return Tensor(out, requires_grad=x.requires_grad), assignments
        out = [0.0] * self.dim
        for _idx, picks in assignments:
            for j, g in picks:
                e = self.experts[j].forward(x)
                for k in range(self.dim):
                    out[k] += g * e.data[0][k]
        from ..tensor import Tensor

        return Tensor([out], requires_grad=x.requires_grad), assignments

    def parameters(self):
        params = list(self.router.parameters())
        for e in self.experts:
            params.extend(e.parameters())
        return params

    def active_params(self):
        """Estimate params touched per token: router + top_k experts."""
        router = self.dim * self.n_experts
        per_expert = 3 * self.dim * self.hidden  # w1,w3,w2 rough
        return router + self.top_k * per_expert

    def total_params(self):
        router = self.dim * self.n_experts
        all_experts = self.n_experts * 3 * self.dim * self.hidden
        return router + all_experts
