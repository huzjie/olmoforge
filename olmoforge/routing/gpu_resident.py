"""GPU-resident routing metadata.

If the router's routing tables live in CPU memory, the CPU must copy them to
the GPU before each step, serializing with computation. GPU-resident routing
keeps the metadata on the device so the CPU can enqueue work without waiting
for data copies.
"""


class GPUResidentRouting:
    """Track whether routing metadata is device-resident and its copy cost."""

    def __init__(self, n_experts, n_tokens, resident=True):
        self.n_experts = n_experts
        self.n_tokens = n_tokens
        self.resident = resident

    def sync_cost_units(self):
        """Cost of copying routing tables each step when NOT resident."""
        if self.resident:
            return 0
        return self.n_tokens * self.n_experts  # per-token routing decisions

    def copy_events_avoided(self):
        return 0 if self.resident else 1
