"""Adapter sketch for vLLM-style expert-parallel serving (no hard dep)."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class VLLMServeSpec:
    """Parameters a vLLM deployment would use, derived from our config."""

    n_experts: int = 128
    top_k: int = 6
    tensor_parallel: int = 8
    expert_parallel: bool = True
    extra_flags: List[str] = field(default_factory=list)

    def to_cli(self):
        flags = [
            f"--tensor-parallel-size {self.tensor_parallel}",
            f"--num-experts {self.n_experts}",
            f"--moe-top-k {self.top_k}",
        ]
        if self.expert_parallel:
            flags.append("--enable-expert-parallel")
        flags.extend(self.extra_flags)
        return " ".join(flags)
