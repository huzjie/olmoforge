"""CLI entrypoint with subcommands.

    python -m olmoforge doctor           # sanity-check all components
    python -m olmoforge train            # train the mock backend (skill -> 1.0)
    python -m olmoforge bench            # run all benchmarks
    python -m olmoforge throughput       # FSDP vs expert-parallel
    python -m olmoforge serve            # start the stdlib HTTP server
"""
import argparse
import sys

from ..config import load_config
from ..utils.logging import get_logger

log = get_logger("olmoforge.cli")


def _build_common(parser):
    parser.add_argument("--config", default="configs/default.yaml",
                        help="path to config (json/yaml)")
    parser.add_argument("--experts", type=int, default=None, help="number of experts")
    parser.add_argument("--top-k", type=int, default=None, help="experts active per token")
    parser.add_argument("--dim", type=int, default=None, help="hidden dimension")
    parser.add_argument("--verbose", action="store_true", help="verbose output")


def _resolve(args):
    cfg = None
    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        log.info("no config at %s; using defaults", args.config)
    n_exp = args.experts or (cfg.get("moe", None) and cfg.get("moe").get("n_experts")) or 128
    top_k = args.top_k or (cfg.get("moe", None) and cfg.get("moe").get("top_k")) or 6
    dim = args.dim or (cfg.get("model", None) and cfg.get("model").get("dim")) or 4096
    return cfg, n_exp, top_k, dim


def cmd_doctor(args):
    from ..core.moe import MoELayer
    from ..parallel.expert_parallel import ExpertParallel
    from ..parallel.pipeline import PipelineParallel
    from ..parallel.dist_optimizer import DistributedOptimizer
    from ..routing.grouped_gemm import GroupedGEMM
    from ..routing.rowwise import RowwiseExpertParallel
    from ..precision.mxfp8 import MXFP8
    from ..backends import list_backends
    from ..bench import list_benchmarks
    from ..tensor import Tensor, softmax, layernorm

    _, n_exp, top_k, dim = _resolve(args)
    print(f"[doctor] experts={n_exp} top_k={top_k} dim={dim}")
    # NOTE: instantiate the MoE layer with SMALL dims. The framework's MoE layer
    # allocates 3 weight matrices per expert (pure Python), so a full
    # 4096-dim / 128-expert instantiation would allocate ~3.2B floats and hang.
    MoELayer(min(n_exp, 16), top_k, 32, 16, seed=0)
    ExpertParallel(n_exp, top_k, dim)
    PipelineParallel(8, 32)
    DistributedOptimizer(8)
    GroupedGEMM(n_exp)
    RowwiseExpertParallel(n_exp)
    MXFP8()
    Tensor([[1.0, 2.0], [3.0, 4.0]])
    softmax([1.0, 2.0, 3.0])
    layernorm([1.0, 2.0, 3.0])
    print(f"[doctor] backends={list_backends()}")
    print(f"[doctor] benchmarks={list_benchmarks()}")
    print("[doctor] OK: all components import and instantiate")
    return 0


def cmd_train(args):
    from ..backends.mock import MockBackend
    from ..data.synth import SynthQA
    from ..train.trainer import Trainer

    _, n_exp, top_k, dim = _resolve(args)
    qa = SynthQA(n=200)
    backend = MockBackend(skill=0.5)
    trainer = Trainer(backend, qa.questions, qa.options, n_exp, top_k, dim)
    comm = trainer.comm_estimate()
    print(f"[train] comm FSDP={comm['fsdp']:,} expert_parallel={comm['expert_parallel']:,} "
          f"ratio={comm['ratio']:.1f}x")
    s_start = backend.skill
    a_start = backend.accuracy(qa.questions, qa.options)
    print(f"[train] skill {s_start:.3f} acc {a_start:.3f}; training ...")
    hist = trainer.run(epochs=10)
    s_end, a_end = hist[-1][1], hist[-1][2]
    print(f"[train] skill {s_start:.3f} -> {s_end:.3f} | accuracy {a_start:.3f} -> {a_end:.3f}")
    return 0


def cmd_bench(args):
    from ..bench import list_benchmarks, run_benchmark

    for name in list_benchmarks():
        print(f"\n== benchmark: {name} ==")
        run_benchmark(name, verbose=True)
    return 0


def cmd_throughput(args):
    from ..bench import run_benchmark

    run_benchmark("throughput", verbose=True)
    return 0


def cmd_serve(args):
    from ..serving.server import run_server

    _, n_exp, top_k, _ = _resolve(args)
    run_server(n_exp, top_k)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="olmoforge",
                                     description="Trillion-parameter MoE training framework")
    sub = parser.add_subparsers(dest="cmd")
    common = argparse.ArgumentParser(add_help=False)
    _build_common(common)

    p_doctor = sub.add_parser("doctor", parents=[common], help="sanity check")
    p_train = sub.add_parser("train", parents=[common], help="train mock backend")
    p_bench = sub.add_parser("bench", parents=[common], help="run all benchmarks")
    p_tput = sub.add_parser("throughput", parents=[common], help="FSDP vs expert-parallel")
    p_serve = sub.add_parser("serve", parents=[common], help="HTTP status server")

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    if args.cmd is None:
        parser.print_help()
        return 0
    return {"doctor": cmd_doctor, "train": cmd_train, "bench": cmd_bench,
            "throughput": cmd_throughput, "serve": cmd_serve}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
