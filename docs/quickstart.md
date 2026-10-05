# Quickstart

No third-party dependencies are required. Python 3.9+ is enough.

```bash
# sanity check (imports + instantiates everything)
python -m olmoforge doctor

# train the mock backend (skill 0.5 -> ~1.0)
python -m olmoforge train

# FSDP vs expert-parallel throughput
python -m olmoforge throughput

# run all benchmarks
python -m olmoforge bench

# start the stdlib status server
python -m olmoforge serve --experts 384 --top-k 6
```

Run a specific example:

```bash
python examples/01_train_mock.py
```
