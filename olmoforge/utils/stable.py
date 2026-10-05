"""Deterministic, high-entropy RNG helpers (md5-seeded).

These power the mock backend and the synthetic corpora: the same key always
yields the same values, which makes training monotone and benchmarks
reproducible.

Reusable trick: seed a `random.Random` from the first 8 bytes of an md5 digest
of the key. Naively flattening the 16-byte digest repeats only ~2 distinct
values; the first-8-bytes-as-seed approach gives a full 64-bit space quickly.
"""
import hashlib
import random


def _seed(key):
    return int.from_bytes(hashlib.md5(str(key).encode("utf-8")).digest()[:8], "big")


def stable_float(key, lo=0.0, hi=1.0):
    r = random.Random(_seed(key))
    return r.uniform(lo, hi)


def stable_ints(key, n, lo=0, hi=1000):
    r = random.Random(_seed(key))
    return [r.randint(lo, hi) for _ in range(n)]


def stable_vector(key, n, lo=-1.0, hi=1.0):
    r = random.Random(_seed(key))
    return [r.uniform(lo, hi) for _ in range(n)]


def stable_choice(key, items):
    if not items:
        raise ValueError("empty items")
    r = random.Random(_seed(key))
    return items[r.randrange(len(items))]


def stable_bool(key, p=0.5):
    return stable_float(key) < p
