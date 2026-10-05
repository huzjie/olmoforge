"""Scaling projection: from a 4.6B/8-expert model to trillion-parameter MoE.

OLMo-core 3 kept active params per token ~3.2B while total capacity grew from
4.6B to 47B (throughput <5% down) and benchmarked a 1.2T/58B-active model.
This module projects the same scaling curve for any target.
"""
from . import register_benchmark


@register_benchmark("scaling")
def run_scaling(total_params=1_200_000_000_000, active=58_360_000_000,
                top_k=6, verbose=True):
    # sparse_ratio = total / active = n_experts / top_k  (expert FFN dominates)
    sparse_ratio = total_params / active
    n_experts = round(sparse_ratio * top_k)
    if verbose:
        print(f"[scaling] target {total_params/1e12:.2f}T / {active/1e9:.2f}B active "
              f"-> sparse_ratio={sparse_ratio:.1f}x, ~{n_experts} experts "
              f"(top_k={top_k})")
    return {"n_experts": n_experts, "top_k": top_k, "active_params": active,
            "sparse_ratio": round(sparse_ratio, 1)}
