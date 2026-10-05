"""A tiny supervised training loop that drives the mock skill upward.

Also demonstrates the *core* OLMo-core 3 win: we compute the communication
cost of FSDP vs expert-parallel for the same step, so the training log shows
why expert parallelism scales better as expert count grows.
"""
from ..utils.dist import MockWorld
from ..parallel.expert_parallel import ExpertParallel, FSDPLegacy


class Trainer:
    def __init__(self, backend, questions, candidates, n_experts, top_k, dim,
                 total_params=None, lr=0.05, world=None):
        self.backend = backend
        self.questions = questions
        self.candidates = candidates
        self.n_experts = n_experts
        self.top_k = top_k
        self.dim = dim
        self.total_params = total_params or (n_experts * 3 * dim * dim)
        self.lr = lr
        self.world = world or MockWorld(n_devices=8)

    def comm_estimate(self, seq_len=128, micro_batches=1):
        ep = ExpertParallel(self.n_experts, self.top_k, self.dim, self.world)
        fsdp = FSDPLegacy(self.total_params, self.world)
        ep_cost = ep.comm_for_seq(seq_len) * micro_batches
        fsdp_cost = fsdp.comm_for_step(micro_batches)
        return {"expert_parallel": ep_cost, "fsdp": fsdp_cost,
                "ratio": fsdp_cost / max(ep_cost, 1.0)}

    def run(self, epochs=10, seq_len=128):
        history = []
        for epoch in range(epochs):
            for q in self.questions:
                self.backend.train_step(q, self.candidates, _truth(q))
            acc = self.backend.accuracy(self.questions, self.candidates)
            history.append((epoch, self.backend.skill, acc))
        return history


def _truth(q):
    from ..backends.mock import _ground_truth

    return _ground_truth(q)
