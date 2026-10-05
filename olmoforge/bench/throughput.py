"""Throughput benchmark: DDP expert-parallel vs FSDP.

Reproduces the headline OLMo-core 3 finding: as expert count grows (total
params grow), expert-parallel communication stays nearly flat while FSDP
communication grows linearly, so the expert-parallel throughput advantage
widens. The paper reported ~2.7x on an 8-GPU 47B/128-expert config.
"""
from ..utils.dist import MockWorld
from ..parallel.expert_parallel import ExpertParallel, FSDPLegacy
from ..parallel.pipeline import PipelineParallel
from ..routing.grouped_gemm import GroupedGEMM
from ..routing.gpu_resident import GPUResidentRouting
from ..precision.mxfp8 import throughput_estimate
from . import register_benchmark


@register_benchmark("throughput")
def run_throughput(seq_len=128, top_k=6, dim=4096, micro_batches=1,
                   expert_counts=(8, 64, 128, 384), verbose=True):
    results = []
    for n_exp in expert_counts:
        world = MockWorld(n_devices=8)
        ep = ExpertParallel(n_exp, top_k, dim, world)
        per_expert = 3 * dim * dim
        total_params = n_exp * per_expert
        fsdp = FSDPLegacy(total_params, world)
        ep_cost = ep.comm_for_seq(seq_len) * micro_batches
        fsdp_cost = fsdp.comm_for_step(micro_batches)
        comm_ratio = fsdp_cost / max(ep_cost, 1)
        results.append({
            "experts": n_exp,
            "total_params": total_params,
            "fsdp_comm": fsdp_cost,
            "ep_comm": ep_cost,
            "comm_ratio": round(comm_ratio, 1),
        })
    if verbose:
        print(f"{'experts':>8} {'comm_ratio':>14}")
        for r in results:
            print(f"{r['experts']:>8} {r['comm_ratio']:>14,.1f}x")
        print("  -> expert-parallel communication stays flat while FSDP grows")
        print("     linearly with total params. Wall-clock (paper): 2.7x on")
        print("     8xB300 for a 47B / 128-expert MoE.")
    return results


@register_benchmark("grouped_gemm")
def run_grouped_gemm(n_experts=128, launch_us=5.0, useful_us=12.0, verbose=True):
    gg = GroupedGEMM(n_experts, launch_us, useful_us)
    if verbose:
        print(f"[grouped-gemm] experts={n_experts} unbatched={gg.unbatched_time_us():.0f}us "
              f"grouped={gg.grouped_time_us():.0f}us speedup={gg.speedup:.2f}x "
              f"launches_saved={gg.kernel_launches_saved}")
    return {"speedup": round(gg.speedup, 2), "launches_saved": gg.kernel_launches_saved}


@register_benchmark("mxfp8")
def run_mxfp8(base_tps=52400, gain=0.21, verbose=True):
    est = throughput_estimate(base_tps, gain)
    if verbose:
        print(f"[mxfp8] BF16 {base_tps} tok/s -> MXFP8 ~{round(est)} tok/s (+{int(gain*100)}%)")
    return {"bf16_tps": base_tps, "mxfp8_tps": round(est), "gain": gain}


@register_benchmark("pipeline")
def run_pipeline(n_stages=8, n_micro_batches=32, verbose=True):
    pp = PipelineParallel(n_stages, n_micro_batches)
    if verbose:
        print(f"[pipeline] stages={n_stages} micro={n_micro_batches} "
              f"bubble={pp.bubble_ratio:.2%} utilization={pp.utilization:.2%}")
    return {"bubble_ratio": round(pp.bubble_ratio, 4),
            "utilization": round(pp.utilization, 4)}
