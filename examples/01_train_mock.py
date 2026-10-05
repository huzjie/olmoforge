import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
"""Train the deterministic mock backend: skill 0.5 -> ~1.0.

Run:  python examples/01_train_mock.py
"""
from olmoforge.backends.mock import MockBackend
from olmoforge.data.synth import SynthQA
from olmoforge.train.trainer import Trainer

qa = SynthQA(n=200)
backend = MockBackend(skill=0.5, noise=0.5)
trainer = Trainer(backend, qa.questions, qa.options, n_experts=128, top_k=6, dim=4096)

print("before:", round(backend.skill, 3), "acc", round(backend.accuracy(qa.questions, qa.options), 3))
hist = trainer.run(epochs=10)
e, s, a = hist[-1]
print("after :", round(s, 3), "acc", round(a, 3))
print("comm  : FSDP vs expert-parallel ratio =", round(trainer.comm_estimate()["ratio"], 1), "x")
