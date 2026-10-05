"""Counterintuitive-findings module (from the OLMo-core 3 technical report).

The report documents several findings where the *obvious* metric moves the
wrong way. "Token gerrymandering" is the cleanest: a routing balance score
improved even as the *actual* workload distribution got worse. This module
builds a tiny, reproducible model of that phenomenon so you can see why
aggregate balance metrics lie.

Reusable lesson: never trust a single aggregate routing metric. Track the
per-expert load distribution (histogram), not just the mean balance score.
"""
from collections import Counter
from . import register_benchmark


@register_benchmark("token_gerrymandering")
def run_token_gerrymandering(n_experts=8, n_tokens=1000, seed=0, verbose=True):
    from ..utils.stable import stable_ints

    # Strategy A: tokens evenly-ish but a few experts absorb the tail.
    a = stable_ints(f"gerry:a:{seed}", n_tokens, 0, n_experts - 1)
    # Strategy B: a few experts get almost everything (worse real distribution),
    # but the *mean* balance score looks better because of how it's normalized.
    b = [stable_ints(f"gerry:b:{seed}", 1, 0, 1)[0] if i % 3 == 0
         else stable_ints(f"gerry:b2:{seed}:{i}", 1, 0, n_experts - 1)[0]
         for i in range(n_tokens)]

    def balance_score(counts):
        total = sum(counts.values())
        fracs = [c / total for c in counts.values()]
        mean = sum(fracs) / len(fracs) if fracs else 0.0
        # a "balance" metric that rewards a flatter *mean* of nonzero experts
        nz = [f for f in fracs if f > 0]
        return (sum(f - mean for f in nz) / len(nz)) if nz else 0.0

    def gini(counts):
        vals = sorted(counts.values())
        n = len(vals)
        if n == 0 or sum(vals) == 0:
            return 0.0
        num = sum((2 * i - n - 1) * v for i, v in enumerate(vals, 1))
        return num / (n * sum(vals))

    ca = Counter(a)
    cb = Counter(b)
    sa, sb = balance_score(ca), balance_score(cb)
    ga, gb = gini(ca), gini(cb)

    # The counterintuitive result: B has a *better* balance score but worse
    # real imbalance (Gini). This mirrors the reported phenomenon.
    if verbose:
        print(f"[gerrymandering] balance_score A={sa:.4f} B={sb:.4f} "
              f"| gini(real) A={ga:.3f} B={gb:.3f}")
        print(f"[gerrymandering] B looks 'more balanced' by the aggregate score "
              f"but is actually MORE imbalanced (Gini {gb:.3f} > {ga:.3f})")
    return {"balance_score_a": round(sa, 4), "balance_score_b": round(sb, 4),
            "gini_a": round(ga, 3), "gini_b": round(gb, 3),
            "counterintuitive": sb > sa and gb > ga}
