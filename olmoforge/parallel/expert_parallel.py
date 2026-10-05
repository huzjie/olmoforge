"""DDP-based expert parallelism.

This is the headline idea from OLMo-core 3. FSDP gathers and reshardes the
model weights for *every* micro-batch, so communication scales with the full
parameter count. Expert parallelism inverts that: each expert is resident on a
home device, and you ship only the *routed token vectors* to it. As the expert
count grows, the total weight count grows, but the number of token vectors
shipped stays proportional to (seq_len x top_k), which is why throughput stays
nearly flat.

Reusable mental model
---------------------
    FSDP:        ship weights every step  -> comm ~ total_params
    Expert-par:  ship token vectors       -> comm ~ seq_len * top_k * dim

We model both in the `MockWorld` cost counter so the throughput benchmark can
produce the FSDP-vs-expert-parallel comparison without real GPUs.
"""
from ..utils.dist import MockWorld, DistWorld


class ExpertParallel:
    """Route data to resident experts instead of gathering weights."""

    def __init__(self, n_experts, top_k, dim, world: MockWorld | None = None):
        self.n_experts = n_experts
        self.top_k = top_k
        self.dim = dim
        self.world = world or MockWorld()
        self.cluster = DistWorld(size=self.world.n_devices)
        self.expert_home = {e: self.cluster.resident(e) for e in range(n_experts)}

    def route_tokens(self, seq_len):
        """Ship one sequence of length seq_len to the experts it activates.

        Each token picks top_k experts; the number of token-vectors that must
        cross device boundaries is seq_len * top_k (a fixed factor independent
        of total expert count).
        """
        vecs = seq_len * self.top_k
        self.world.ship(vecs, self.dim, tag="expert-parallel:tokens")
        return vecs

    def comm_for_seq(self, seq_len):
        return seq_len * self.top_k * self.dim * self.world.bytes_per_weight


class FSDPLegacy:
    """Reference baseline: FSDP gathers+reshardes every micro-batch."""

    def __init__(self, total_params, world: MockWorld | None = None):
        self.total_params = total_params
        self.world = world or MockWorld()

    def gather_weights(self, micro_batches=1):
        # every batch re-ships the full weight set (all-gather + reduce-scatter)
        for _ in range(micro_batches):
            self.world.ship(self.total_params, 1, tag="fsdp:all-gather")
        return self.total_params * micro_batches

    def comm_for_step(self, micro_batches=1):
        return self.total_params * micro_batches * self.world.bytes_per_weight
