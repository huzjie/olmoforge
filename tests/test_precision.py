"""Tests for MXFP8 and routing helpers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

from olmoforge.precision.mxfp8 import MXFP8, quantize_mxfp8, dequantize_mxfp8
from olmoforge.routing.grouped_gemm import GroupedGEMM
from olmoforge.routing.rowwise import RowwiseExpertParallel
from olmoforge.utils.stable import stable_vector


class TestPrecision(unittest.TestCase):
    def test_roundtrip_small_error(self):
        vals = stable_vector("p", 128, -1, 1)
        err = MXFP8(32).roundtrip_error(vals)
        self.assertLess(err, 0.05)

    def test_quant_dequant_len(self):
        vals = stable_vector("q", 64, -1, 1)
        q, s = quantize_mxfp8(vals, 32)
        d = dequantize_mxfp8(q, s)
        self.assertEqual(len(d), len(vals))

    def test_grouped_gemm_speedup(self):
        self.assertGreater(GroupedGEMM(128).speedup, 1.0)

    def test_rowwise_place(self):
        rp = RowwiseExpertParallel(8)
        rp.place(0, [1, 3])
        self.assertEqual(rp.buffer_size(1), 1)
        self.assertEqual(rp.buffer_size(3), 1)


if __name__ == "__main__":
    unittest.main()
