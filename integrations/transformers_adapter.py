"""Adapter sketch for swapping in HF Transformers experts (no hard dep).

Documents the contract a real backend must satisfy so the parallelism/routing
modules stay unchanged. Provided as reference; not executed by default.
"""


class HFExpertAdapter:
    """Maps a HF checkpoint's expert weights into ExpertFFN's interface."""

    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def load_expert(self, expert_id):
        # In a real integration this reads shard files. Kept as a stub with the
        # documented return contract: an object with .forward(x) and .parameters().
        raise NotImplementedError(
            "Load expert shards from "
            f"{self.checkpoint_dir}/expert-{expert_id:04d}.safetensors"
        )
