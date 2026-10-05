"""olmoforge: trillion-parameter Mixture-of-Experts training framework.

A zero-dependency, fully runnable reference implementation of the training
techniques behind AI2's OLMo-core 3 (released 2026-10-01), an open-source
stack that scales MoE training into the trillion-parameter range while keeping
throughput nearly flat as expert count grows. This repo distills its most
reusable engineering ideas into a single, self-contained framework:

1. DDP-based expert parallelism -- experts stay resident on GPUs; you route
   *data* to experts instead of gathering weights (vs FSDP).
2. Three parallelism axes -- expert + pipeline + distributed optimizer.
3. Routing optimizations  -- rowwise expert parallelism, GPU-resident routing,
   grouped GEMM.
4. MXFP8 low-precision training -- ~+21% throughput vs BF16.

Highlights
----------
- Zero-dependency tensor core (pure Python) with autodiff over the op set
- Deterministic trainable mock backend (skill monotonically -> 1.0)
- Throughput benchmark: DDP expert-parallel vs legacy FSDP (2.7x in paper)
- Counterintuitive-findings module (token gerrymandering, etc.)
- CLI + Docker/K8s/Helm/CI + prompts + recipes
"""
from .version import __version__
from .config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
