"""Run the full benchmark sweep.

Iterates: models × workloads × runners.
Writes one JSON file per (model, batch_size) into benchmarks/results/<run_id>/.

Usage:
    python -m benchmarks.sweep                       # all models, b=1
    python -m benchmarks.sweep --batches 1,2,4,8     # batch sweep
    python -m benchmarks.sweep --models qwen3-30b-a3b,olmoe-1b-7b
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

from .models import ModelRegistry, MoEModelConfig
from .workloads import Workload, BenchmarkResult
from .runners import FlashMoERunner, VLLMRunner, PyTorchRunner


def parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def parse_str_list(s: str) -> list[str]:
    return [x for x in s.split(",") if x]


def make_run_id() -> str:
    name = torch.cuda.get_device_name(0).replace(" ", "_").replace("/", "_")
    return f"{datetime.now():%Y-%m-%d_%H%M}_{name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=parse_str_list, default=None,
                    help="comma-separated model names; default = all registered")
    ap.add_argument("--batches", type=parse_int_list, default=[1])
    ap.add_argument("--dtypes", type=parse_str_list, default=["fp16"])
    ap.add_argument("--runners", type=parse_str_list, default=["flashmoe", "vllm"],
                    help="subset of: flashmoe, vllm, pytorch-eager")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default="benchmarks/results")
    args = ap.parse_args()

    if args.models is None:
        models = ModelRegistry.all()
    else:
        models = [ModelRegistry.by_name(n) for n in args.models]

    runner_classes = {"flashmoe": FlashMoERunner,
                      "vllm": VLLMRunner,
                      "pytorch-eager": PyTorchRunner}
    selected_runners = [runner_classes[r] for r in args.runners]

    run_id = args.run_id or make_run_id()
    out_dir = Path(args.out_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "run_id": run_id,
        "device": torch.cuda.get_device_name(0),
        "cuda_arch": torch.cuda.get_device_capability(0),
        "models": [m.name for m in models],
        "batches": args.batches,
        "dtypes": args.dtypes,
        "runners": args.runners,
        "warmup": args.warmup,
        "iters": args.iters,
        "timestamp": datetime.now().isoformat(),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    all_results: list[BenchmarkResult] = []

    for model in models:
        for dtype in args.dtypes:
            for batch in args.batches:
                wl = Workload(batch_size=batch, dtype=dtype,
                              warmup_iters=args.warmup, bench_iters=args.iters)
                for runner_cls in selected_runners:
                    runner = runner_cls()
                    if not runner.supports(model, wl):
                        print(f"  [skip] {model.name} b={batch} {dtype} {runner.name}: unsupported")
                        continue

                    print(f"  [run]  {model.name} b={batch} {dtype} {runner.name} ...", flush=True)
                    t0 = time.time()
                    res = runner.benchmark(model, wl)
                    dt = time.time() - t0

                    if res.error:
                        print(f"         ERROR: {res.error}")
                    else:
                        print(f"         {res.mean_ms:8.4f} ms  (p50={res.p50_ms:.3f} p99={res.p99_ms:.3f})  [{dt:.1f}s wall]")
                    all_results.append(res)

                    # Persist after each result so a crash doesn't lose everything
                    (out_dir / "results.json").write_text(
                        json.dumps([r.model_dump() for r in all_results], indent=2)
                    )

    print(f"\nDone. Wrote {len(all_results)} results to {out_dir}")
    print(f"Generate report:  python -m benchmarks.report {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
