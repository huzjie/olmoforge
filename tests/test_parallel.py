"""Tests for parallelism strategies."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

from olmoforge.parallel.expert_parallel import ExpertParallel, FSDPLegacy
from olmoforge.parallel.pipeline import PipelineParallel
from olmoforge.parallel.dist_optimizer import DistributedOptimizer
from olmoforge.utils.dist import MockWorld


class TestParallel(unittest.TestCase):
    def test_expert_parallel_cheaper_than_fsdp(self):
        world = MockWorld(n_devices=8)
        ep = ExpertParallel(128, 6, 4096, world)
        fsdp = FSDPLegacy(128 * 3 * 4096 * 4096, world)
        self.assertLess(ep.comm_for_seq(128), fsdp.comm_for_step(1))

    def test_pipeline_bubble_in_range(self):
        pp = PipelineParallel(8, 32)
        self.assertGreaterEqual(pp.bubble_ratio, 0.0)
        self.assertLessEqual(pp.bubble_ratio, 1.0)
        self.assertLess(PipelineParallel(8, 32).bubble_ratio,
                        PipelineParallel(8, 1).bubble_ratio)

    def test_dist_optimizer_shards_memory(self):
        do = DistributedOptimizer(8)
        self.assertLess(do.memory_per_rank(100000), do.replicated_memory(100000))

    def test_dist_optimizer_step(self):
        do = DistributedOptimizer(4, lr=0.1)
        out = do.step([1.0, -1.0], [0.5, -0.5])
        self.assertEqual(len(out), 2)


if __name__ == "__main__":
    unittest.main()
