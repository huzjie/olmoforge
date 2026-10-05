# Token gerrymandering (and other counterintuitive findings)

The OLMo-core 3 technical report documents findings where the obvious metric
moves the wrong way. The cleanest is **token gerrymandering**: a routing
balance score improved even as the *actual* workload distribution got worse.

## Why aggregate metrics lie

A mean balance score can be gamed by concentrating load on a few experts while
keeping the *average* flat. The fix is to track the **per-expert load
distribution** (e.g. Gini coefficient), not just the mean.

`bench/token_gerrymandering.py` builds a deterministic reproduction: strategy B
has a better balance score but a worse Gini.

Other documented findings (paraphrased, see the technical report):
- Lowering expert learning rates (they see fewer tokens) did NOT help.
- GPU compute time varied even with identical shapes when values changed.
- Overlapping comm/compute on separate streams sometimes *slowed* execution.
