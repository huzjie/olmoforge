"""Grouped GEMM: batch many small expert matmuls into fewer kernels.

A sparse MoE runs a *lot* of tiny matmuls (one per expert per token batch).
Launching a GEMM per expert wastes kernel-launch overhead. Grouped GEMM
concatenates the expert inputs, runs one large GEMM, then splits the output.

Reusable trick: when you have N small matrix multiplies that share a weight
layout, stack them into one (sum_k_i x in_dim) x out_dim GEMM and slice the
result. Kernel-launch count drops from N to 1.
"""


class GroupedGEMM:
    """Model the launch-overhead win of batching expert matmuls."""

    def __init__(self, n_experts, launch_overhead_us=5.0, useful_us=12.0):
        self.n_experts = n_experts
        self.launch_overhead_us = launch_overhead_us
        self.useful_us = useful_us

    def unbatched_time_us(self):
        """One kernel launch per expert."""
        return self.n_experts * (self.launch_overhead_us + self.useful_us)

    def grouped_time_us(self):
        """One launch for the whole stack."""
        return self.launch_overhead_us + self.n_experts * self.useful_us

    @property
    def speedup(self):
        u = self.unbatched_time_us()
        g = self.grouped_time_us()
        return u / g if g > 0 else 1.0

    @property
    def kernel_launches_saved(self):
        return self.n_experts - 1
