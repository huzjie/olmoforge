"""Optional MCP-style tool exposure (stdout JSON-RPC, no hard deps).

Exposes `olmoforge_throughput` and `olmoforge_train` as tools that an MCP host
can call. The transport is the simplest possible: read JSON-RPC from stdin,
write JSON-RPC to stdout. Run with `python -m integrations.mcp_server`.
"""
import json
import sys


def _throughput(args):
    from olmoforge.bench import run_benchmark

    res = run_benchmark("throughput", verbose=False)
    return {"expert_counts": [r["experts"] for r in res],
            "speedups": [r["speedup"] for r in res]}


def _train(args):
    from olmoforge.backends.mock import MockBackend
    from olmoforge.data.synth import SynthQA
    from olmoforge.train.trainer import Trainer

    qa = SynthQA(n=200)
    b = MockBackend(skill=0.5)
    t = Trainer(b, qa.questions, qa.options, 128, 6, 4096)
    hist = t.run(epochs=10)
    return {"skill_start": hist[0][1], "skill_end": hist[-1][1],
            "acc_end": hist[-1][2]}


TOOLS = {
    "olmoforge_throughput": _throughput,
    "olmoforge_train": _train,
}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        name = req.get("method", "")
        tid = req.get("id")
        tool = TOOLS.get(name)
        if tool is None:
            resp = {"id": tid, "error": {"code": -32601, "message": f"unknown {name}"}}
        else:
            try:
                resp = {"id": tid, "result": tool(req.get("params") or {})}
            except Exception as e:  # noqa: BLE001
                resp = {"id": tid, "error": {"code": -32000, "message": str(e)}}
        print(json.dumps(resp), flush=True)


if __name__ == "__main__":
    main()
