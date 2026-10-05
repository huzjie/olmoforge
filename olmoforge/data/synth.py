"""Synthetic datasets: deterministic, no network needed."""
from ..utils.stable import stable_ints, stable_choice


class SynthQA:
    """Deterministic multiple-choice QA set for training the mock backend."""

    def __init__(self, n=200, n_options=4, seed=0):
        self.n = n
        self.n_options = n_options
        self.seed = seed
        self.questions = [f"q{i}" for i in range(n)]
        self.options = [chr(ord("A") + j) for j in range(n_options)]

    @property
    def candidates(self):
        return self.options


class SynthTokens:
    """Synthetic token sequences to drive the throughput benchmark."""

    def __init__(self, vocab=32000, seq_len=128, seed=0):
        self.vocab = vocab
        self.seq_len = seq_len
        self.seed = seed

    def batch(self, n_seqs):
        return [stable_ints(f"tok:{self.seed}:{i}", self.seq_len, 0, self.vocab - 1)
                for i in range(n_seqs)]

    def one(self):
        return self.batch(1)[0]
