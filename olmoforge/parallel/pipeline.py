"""Pipeline parallelism: split layers across device groups.

Splitting the model across pipeline stages reduces per-device memory, but it
introduces a bubble unless micro-batching is used to overlap compute and
communication. This module exposes a small scheduler that computes the bubble
ratio for a given stage/micro-batch configuration.
"""


class PipelineParallel:
    """1F1B-style schedule bubble estimator."""

    def __init__(self, n_stages, n_micro_batches):
        self.n_stages = n_stages
        self.n_micro_batches = n_micro_batches

    @property
    def bubble_ratio(self):
        """Fraction of the schedule spent idle (the classic (p-1)/(p+m-1))."""
        p, m = self.n_stages, self.n_micro_batches
        if m <= 1:
            return (p - 1) / p
        return (p - 1) / (p + m - 1)

    @property
    def utilization(self):
        return 1.0 - self.bubble_ratio

    def schedule(self):
        """Return a simple list of (stage, micro_batch) execution order (1F1B)."""
        order = []
        # warmup
        for s in range(self.n_stages):
            order.append((s, s))
        # steady state
        for m in range(self.n_stages, self.n_micro_batches):
            order.append((self.n_stages - 1, m))
        # cooldown
        for s in range(self.n_stages - 2, -1, -1):
            order.append((s, self.n_micro_batches - 1))
        return order
