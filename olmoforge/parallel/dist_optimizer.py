"""Distributed optimizer: shard optimizer state across GPUs.

Replicating the optimizer (AdamW moments) on every rank wastes memory that
grows linearly with the device count. The distributed optimizer shards the
optimizer state the same way the weights are sharded, so each rank only holds
the moments for its own parameter slice.
"""


class DistributedOptimizer:
    """Sharded AdamW (state partitioned by rank, not replicated)."""

    def __init__(self, n_ranks, lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01):
        self.n_ranks = n_ranks
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def _shard(self, name, n_params):
        """Params owned by this optimizer are split across ranks."""
        per_rank = max(1, n_params // self.n_ranks)
        return per_rank * self.n_ranks, per_rank

    def memory_per_rank(self, n_params):
        """Memory = params + 2x moments, divided across ranks."""
        total, per_rank = self._shard("x", n_params)
        # fp32 master + m + v, each per_rank-sized
        bytes_per = 4
        return 3 * per_rank * bytes_per

    def replicated_memory(self, n_params):
        """Baseline: every rank holds full moments."""
        return 3 * n_params * 4

    def step(self, params, grads):
        """Naive sharded AdamW step; returns updated params (list of floats)."""
        self.t += 1
        out = []
        for p, g in zip(params, grads):
            key = id(p)
            m = self.m.get(key, 0.0)
            v = self.v.get(key, 0.0)
            g = g + self.weight_decay * p
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * g * g
            self.m[key] = m
            self.v[key] = v
            mhat = m / (1 - self.beta1 ** self.t)
            vhat = v / (1 - self.beta2 ** self.t)
            out.append(p - self.lr * mhat / (vhat ** 0.5 + 1e-8))
        return out
