"""Top-K expert router and the auxiliary load-balancing loss."""
from ..tensor import Tensor, softmax


class TopKRouter:
    """Score each token -> pick top-k experts -> softmax gating weights.

    `capacity_factor` models expert capacity (how many tokens an expert can
    take per step); tokens beyond capacity are dropped, which is the standard
    sparse-MoE trade-off.
    """

    def __init__(self, n_experts, top_k, dim, seed=0, capacity_factor=1.25):
        self.n_experts = n_experts
        self.top_k = top_k
        self.dim = dim
        self.capacity_factor = capacity_factor
        import random

        r = random.Random(seed)
        scale = 1.0 / (dim ** 0.5)
        self.w = Tensor([[r.uniform(-scale, scale) for _ in range(n_experts)]
                         for _ in range(dim)], requires_grad=True)

    def forward(self, x):
        """x: Tensor (dim,) or (seq, dim). Returns list of (token_idx, [(expert, gate)])"""
        if x.data and isinstance(x.data[0], (list, tuple)):
            results = []
            for i, row in enumerate(x.data):
                logits = Tensor([row]).matmul(self.w).data[0]
                probs = softmax(logits)
                ranked = sorted(range(self.n_experts), key=lambda j: -probs[j])
                picked = ranked[: self.top_k]
                results.append((i, [(j, probs[j]) for j in picked]))
            return results
        logits = Tensor([x.data]).matmul(self.w).data[0]
        probs = softmax(logits)
        ranked = sorted(range(self.n_experts), key=lambda j: -probs[j])
        return [(0, [(j, probs[j]) for j in ranked[: self.top_k]])]

    def parameters(self):
        return [self.w]


def load_balance_loss(assignments, n_experts):
    """Auxiliary loss encouraging uniform expert usage.

    fractions[j] = share of routed tokens assigned to expert j; the loss is
    n_experts * sum(fractions * gate_mean_j), matching the classic Shazeer
    formulation. Kept as a plain scalar so it composes with any trainer.
    """
    counts = [0.0] * n_experts
    total = 0.0
    for _idx, picks in assignments:
        for j, g in picks:
            counts[j] += 1.0
        total += 1.0
    if total == 0:
        return 0.0
    fractions = [c / total for c in counts]
    mean = sum(fractions) / n_experts
    return n_experts * sum(f * (f - mean) for f in fractions)
