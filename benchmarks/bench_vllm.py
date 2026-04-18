"""Benchmark vLLM's Qwen3 MoE block on RTX 4070.

Run:  .venv/bin/python -m benchmarks.bench_vllm
"""

from __future__ import annotations

import time
import torch
from benchmarks.vllm_utils import (
    H, I, E, K,
    setup_vllm, make_moe, run_vllm_forward, teardown_vllm,
)
from vllm.forward_context import set_forward_context


def time_moe(moe, x, vllm_config, use_graph, warmup=10, iters=100):
    def call(x_):
        return run_vllm_forward(moe, x_, vllm_config)

    for _ in range(warmup):
        call(x)
    torch.cuda.synchronize()

    if use_graph:
        static_in = torch.empty_like(x); static_in.copy_(x)
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3): call(static_in)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.no_grad():
            call(static_in)

        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            g.replay()
        end.record()
        torch.cuda.synchronize()
    else:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            call(x)
        end.record()
        torch.cuda.synchronize()

    return start.elapsed_time(end) / iters


def main():
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Config: H={H} I={I} E={E} K={K}")

    dtypes = [("fp16", torch.float16)]
    if torch.cuda.is_bf16_supported():
        dtypes.append(("bf16", torch.bfloat16))

    print()
    print(f"{'dtype':6s} {'T':>4s} {'graph':>6s} {'ms':>10s} {'per-token-us':>14s} {'GB/s':>9s}")
    print("-" * 60)

    for name, dtype in dtypes:
        vllm_config = setup_vllm(dtype)
        moe, _, _, _ = make_moe(vllm_config, dtype)
        dsz = torch.tensor([], dtype=dtype).element_size()

        for T in (1, 4, 16, 32):
            x = torch.randn(T, H, dtype=dtype, device="cuda")
            for use_graph in (False, True):
                try:
                    ms = time_moe(moe, x, vllm_config, use_graph)
                    bytes_per_iter = min(K * T, E) * (2 * H * I + H * I) * dsz
                    gbps = bytes_per_iter / (ms * 1e-3) / 1e9
                    print(f"{name:6s} {T:4d} {'yes' if use_graph else 'no':>6s} {ms:10.4f} {ms * 1000 / T:14.2f} {gbps:9.1f}")
                except Exception as e:
                    print(f"{name:6s} {T:4d} {'yes' if use_graph else 'no':>6s}   FAILED: {type(e).__name__}: {e}")

        teardown_vllm()
        time.sleep(0.2)

    print()


if __name__ == "__main__":
    main()
