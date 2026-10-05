# 训练指南

## 最小可复现训练

```python
from olmoforge.backends.mock import MockBackend
from olmoforge.data.synth import SynthQA
from olmoforge.train.trainer import Trainer

qa = SynthQA(n=200)
b = MockBackend(skill=0.5)
t = Trainer(b, qa.questions, qa.options, 128, 6, 4096)
hist = t.run(epochs=10)   # skill 0.5 -> ~1.0
```

## 信号设计（关键）

训练 delta 必须恒为正。见 `docs/training-signal.md`。
