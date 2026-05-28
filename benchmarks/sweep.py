"""Run the full benchmark sweep WITH correctness validation.

For each (model, workload):
  1. Generate one shared random weight tensor (MoEWeights).
  2. Use the *reference runner* (vLLM if available, else pytorch-eager) to
     produce a callable that maps input → expected output.
  3. Run every other runner — each runs correctness first, then timing.

A runner that fails correctness has `correctness_ok=False` and `mean_ms=NaN`
in its result. A runner that times out / errors fills `error` and is reported
but its result is otherwise dropped.

Writes JSON results into benchmarks/results/<run_id>/.

Usage:
    python -m benchmarks.sweep                       # all models, b=1
    python -m benchmarks.sweep --batches 1,2,4,8     # batch sweep
    python -m benchmarks.sweep --models qwen3-30b-a3b
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

from .models import ModelRegistry
from .workloads import Workload, BenchmarkResult
from .runners import FlashMoERunner, VLLMRunner, PyTorchRunner
from .runners.weights import MoEWeights


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
    ap.add_argument("--reference", type=str, default="auto",
                    choices=["auto", "vllm", "pytorch-eager", "none"],
                    help="Which runner produces ground truth. 'auto' = vllm if "
                         "the model is supported, else pytorch-eager. 'none' "
                         "skips correctness checking.")
    ap.add_argument("--atol", type=float, default=0.05,
                    help="Absolute tolerance for correctness check")
    ap.add_argument("--correctness-inputs", type=int, default=5,
                    help="How many random inputs to validate per runner")
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

                # Shared random weights for this (model, dtype) pair.
                torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                               "fp32": torch.float32}[dtype]
                torch.manual_seed(wl.seed)
                shared_weights = MoEWeights.random(
                    model.hidden_size, model.intermediate_size,
                    model.num_experts, torch_dtype)

                # Set up the reference runner (produces ground truth callable).
                ref_callable = None
                ref_runner = None
                ref_name = None
                if args.reference != "none":
                    ref_candidates = []
                    if args.reference == "auto":
                        ref_candidates = [VLLMRunner, PyTorchRunner]
                    elif args.reference == "vllm":
                        ref_candidates = [VLLMRunner]
                    else:
                        ref_candidates = [PyTorchRunner]

                    for ref_cls in ref_candidates:
                        ref_runner = ref_cls()
                        if ref_runner.supports(model, wl):
                            ref_name = ref_runner.name
                            try:
                                ref_runner.setup(model, wl, weights=shared_weights)
                                # Build a thin closure for correctness checks
                                def _ref(x, _rr=ref_runner):
                                    with torch.no_grad():
                                        return _rr.forward(x)
                                ref_callable = _ref
                                print(f"\n=== {model.name} b={batch} {dtype} "
                                      f"(reference: {ref_name}) ===")
                                break
                            except Exception as e:
                                print(f"  [ref-setup] {ref_runner.name} failed: {e!r}")
                                ref_runner = None
                    if ref_callable is None:
                        print(f"  [warn] no reference runner available — "
                              f"correctness checks skipped")

                # Bench each non-reference runner
                for runner_cls in selected_runners:
                    runner = runner_cls()
                    # Skip the runner that we already used as reference
                    if ref_runner is not None and runner.name == ref_name:
                        # We still want to time it, but its correctness is
                        # tautological. Skip correctness; do timing.
                        print(f"  [run]  {runner.name} (ref — timing only) ...", flush=True)
                        t0 = time.time()
                        res = runner.benchmark(model, wl, weights=shared_weights)
                        dt = time.time() - t0
                        if res.error:
                            print(f"         ERROR: {res.error}")
                        else:
                            print(f"         {res.mean_ms:8.4f} ms  "
                                  f"(p50={res.p50_ms:.3f} p99={res.p99_ms:.3f})  "
                                  f"[{dt:.1f}s wall]")
                        all_results.append(res)
                        (out_dir / "results.json").write_text(
                            json.dumps([r.model_dump() for r in all_results], indent=2))
                        continue

                    if not runner.supports(model, wl):
                        print(f"  [skip] {runner.name}: unsupported")
                        continue

                    print(f"  [run]  {runner.name} ...", flush=True)
                    t0 = time.time()
                    res = runner.benchmark(
                        model, wl,
                        weights=shared_weights,
                        reference_output=ref_callable,
                        atol=args.atol,
                        num_correctness_inputs=args.correctness_inputs,
                    )
                    dt = time.time() - t0

                    if res.error:
                        if res.correctness_ok is False:
                            print(f"         CORRECTNESS FAIL: max_abs={res.max_abs_error:.4f} "
                                  f"(tol={args.atol})")
                        else:
                            print(f"         ERROR: {res.error}")
                    else:
                        corr = "✓" if res.correctness_ok else ("—" if res.correctness_ok is None else "✗")
                        err_str = (f" (max_abs={res.max_abs_error:.4f})"
                                   if res.max_abs_error is not None else "")
                        print(f"         [{corr}] {res.mean_ms:8.4f} ms  "
                              f"(p50={res.p50_ms:.3f} p99={res.p99_ms:.3f})"
                              f"{err_str}  [{dt:.1f}s wall]")
                    all_results.append(res)
                    (out_dir / "results.json").write_text(
                        json.dumps([r.model_dump() for r in all_results], indent=2)
                    )

                # Teardown the reference runner now that all benches done
                if ref_runner is not None:
                    ref_runner.teardown()

    print(f"\nDone. Wrote {len(all_results)} results to {out_dir}")
    print(f"Generate report:  python -m benchmarks.report {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
