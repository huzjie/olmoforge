"""Deterministic, trainable mock backend.

This is the "canary" backend: it has a single scalar `skill` that controls how
often it predicts correctly on a synthetic QA task. Training nudges `skill`
monotonically toward 1.0, which demonstrates that the training loop, the loss
signal, and the optimizer all actually work -- without requiring a GPU or a
real checkpoint.

Key reusable rule (from prior frameworks): the training signal must be
*monotone positive*. Using "+1 when correct / -1 when wrong" collapses the
skill to zero when the initial accuracy is < 50%. We use an always-positive
delta instead, so skill climbs steadily.
"""
from ..utils.stable import stable_float
from .registry import register_backend


@register_backend("mock")
class MockBackend:
    def __init__(self, skill=0.5, noise=0.5, seed=0, **kwargs):
        self.skill = skill
        self.noise = noise
        self.seed = seed

    def score(self, question, candidate):
        """Score a candidate answer: skill*is_correct + deterministic noise."""
        is_correct = _ground_truth(question) == candidate
        n = stable_float(f"noise:{question}:{candidate}", -self.noise, self.noise)
        return self.skill * (1.0 if is_correct else 0.0) + n

    def predict(self, question, candidates):
        if not candidates:
            return None
        best = max(candidates, key=lambda c: self.score(question, c))
        return best

    def train_step(self, question, candidates, correct):
        """Always-positive delta: correct -> +0.5, wrong -> +0.2 (monotone)."""
        pred = self.predict(question, candidates)
        delta = 0.5 if pred == correct else 0.2
        self.skill = min(1.0, self.skill + delta * 0.05)

    def accuracy(self, questions, candidates):
        if not questions:
            return 0.0
        hits = sum(1 for q in questions if self.predict(q, candidates) == _ground_truth(q))
        return hits / len(questions)


def _ground_truth(question):
    """A tiny deterministic world model shared by the backend and benchmarks."""
    from ..utils.stable import stable_choice

    options = ["A", "B", "C", "D"]
    return stable_choice(f"truth:{question}", options)
