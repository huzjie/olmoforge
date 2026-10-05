"""Distributed-world emulation.

OLMo-core 3's core move is DDP-based expert parallelism: experts stay resident
on their home device, and *data* is routed to them. This module provides a
small device/rank model plus a MockWorld that can simulate N GPUs with
countable "token trips" so the throughput benchmark can compare strategies
without real GPUs.

Design idea (reusable): model the cost of a training strategy as
`compute_cost + communication_cost`, where communication is counted in
"token-vectors shipped". FSDP ships weights every batch; DDP expert-parallel
ships only the routed token vectors, which is why it wins as expert count
grows.
"""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Device:
    rank: int
    name: str = "gpu"

    @property
    def label(self):
        return f"{self.name}:{self.rank}"


class DistWorld:
    """A tiny virtual cluster of devices used by the parallel strategies."""

    def __init__(self, size: int = 8, name: str = "gpu"):
        self.size = size
        self.devices = [Device(i, name) for i in range(size)]

    def shard(self, items, key=None):
        """Round-robin / hash placement of items across devices."""
        out: Dict[int, list] = {d.rank: [] for d in self.devices}
        for i, it in enumerate(items):
            r = i % self.size if key is None else (key(it) % self.size)
            out[r].append(it)
        return out

    def resident(self, expert_id: int) -> int:
        """Which device an expert is resident on (simple modulo placement)."""
        return expert_id % self.size


@dataclass
class MockWorld:
    """A zero-dependency cluster simulator with countable communication."""

    n_devices: int = 8
    bytes_per_weight: int = 4
    comm_cost: int = 0
    compute_cost: int = 0
    log: List[str] = field(default_factory=list)

    def ship(self, n_vectors: int, vec_size: int, tag: str = ""):
        """Simulate shipping n_vectors of vec_size scalars across devices."""
        self.comm_cost += n_vectors * vec_size * self.bytes_per_weight
        self.log.append(f"ship {n_vectors}x{vec_size} {tag}")
        return self.comm_cost

    def compute(self, flops: int, tag: str = ""):
        self.compute_cost += flops
        self.log.append(f"compute {flops} {tag}")
        return self.compute_cost

    def reset(self):
        self.comm_cost = 0
        self.compute_cost = 0
        self.log = []
