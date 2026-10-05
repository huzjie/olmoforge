"""Benchmarks: throughput, token-gerrymandering, scaling."""
_BENCHMARKS = {}


def register_benchmark(name):
    def deco(fn):
        _BENCHMARKS[name] = fn
        return fn
    return deco


def list_benchmarks():
    return sorted(_BENCHMARKS)


def run_benchmark(name, **kwargs):
    if name not in _BENCHMARKS:
        raise KeyError(f"unknown benchmark {name!r}; available={list(_BENCHMARKS)}")
    return _BENCHMARKS[name](**kwargs)


from . import throughput, token_gerrymandering, scaling  # noqa: E402,F401
