"""Rowwise expert parallelism.

When tokens are routed to experts, naively you'd permute the token matrix so
each expert sees a contiguous block, compute, then permute back. Rowwise expert
parallelism places the routed data *directly* into the expert's input buffer,
removing the rearrange step entirely.

Reusable trick: instead of `gather -> permute -> expert -> scatter`, keep a
per-expert append-only buffer indexed by (expert_id), and have each expert read
straight from it. The rearrangement becomes a no-op.
"""


class RowwiseExpertParallel:
    """Route tokens into per-expert buffers without a global permutation."""

    def __init__(self, n_experts):
        self.n_experts = n_experts
        self.buffers = {e: [] for e in range(n_experts)}

    def clear(self):
        for e in self.buffers:
            self.buffers[e] = []

    def place(self, token_id, expert_ids):
        """Place a token into the input buffer of each of its experts."""
        for e in expert_ids:
            self.buffers[e].append(token_id)

    def buffer_size(self, expert_id):
        return len(self.buffers[expert_id])

    def total_permute_saved(self, baseline_permute_ops):
        """Estimate the permute operations avoided vs an explicit rearrange."""
        # explicit permute costs ~ sum over experts of buffer_size*2 (move in/out)
        # rowwise writes once per (token, expert), saving the read-back pass.
        return baseline_permute_ops // 2
