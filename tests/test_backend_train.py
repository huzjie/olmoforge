"""Tests for the mock backend and trainer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

from olmoforge.backends.mock import MockBackend
from olmoforge.data.synth import SynthQA
from olmoforge.train.trainer import Trainer


class TestBackendTrain(unittest.TestCase):
    def test_skill_monotone(self):
        backend = MockBackend(skill=0.5)
        qa = SynthQA(n=100)
        trainer = Trainer(backend, qa.questions, qa.options, 128, 6, 4096)
        hist = trainer.run(epochs=10)
        s0 = hist[0][1]
        s1 = hist[-1][1]
        self.assertGreaterEqual(s1, s0)
        self.assertGreater(s1, 0.9)

    def test_accuracy_improves(self):
        backend = MockBackend(skill=0.5)
        qa = SynthQA(n=100)
        a0 = backend.accuracy(qa.questions, qa.options)
        trainer = Trainer(backend, qa.questions, qa.options, 128, 6, 4096)
        trainer.run(epochs=10)
        a1 = backend.accuracy(qa.questions, qa.options)
        self.assertGreater(a1, a0)


if __name__ == "__main__":
    unittest.main()
