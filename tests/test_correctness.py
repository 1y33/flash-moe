"""Compare FlashMoE kernel output against vLLM's Qwen3 MoE block.

Tests correctness on multiple random inputs, then benchmarks both.

Run:  .venv/bin/python tests/test_correctness.py
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.cpp_extension import load
from benchmarks.vllm_utils import (
    H, I, E, K,
    setup_vllm, make_moe, extract_weights_for_kernel,
    run_vllm_forward, teardown_vllm,
)

NUM_INPUTS = 20
WARMUP     = 10
BENCH_ITER = 200
ATOL       = 0.05


def build_kernel():
    print("JIT compiling FlashMoE kernel...")
    return load(
        name="flash_moe_ext",
        sources=["csrc/binding.cu", "csrc/kernel.cu"],
        extra_cuda_cflags=["-arch=sm_89", "-O3", "-Icsrc"],
        verbose=True,
    )


def run_ours(ext, x, router_weight, gate_projs, up_projs, down_projs):
    return ext.forward(x, router_weight, gate_projs, up_projs, down_projs)


def check_one(vllm_out, our_out):
    """Returns (max_abs, mean_abs, max_rel) for one input."""
    v = vllm_out.squeeze(0).float()
    o = our_out.float()
    abs_d = (v - o).abs()
    rel_d = abs_d / (v.abs() + 1e-8)
    return abs_d.max().item(), abs_d.mean().item(), rel_d.max().item()


def correctness_suite(ext, moe, vllm_config,
                      router_weight, gate_projs, up_projs, down_projs):
    print(f"\n{'=' * 60}")
    print(f" Correctness — {NUM_INPUTS} random inputs")
    print(f"{'=' * 60}")

    max_abs_all = 0.0
    results = []

    for i in range(NUM_INPUTS):
        torch.manual_seed(i)
        x = torch.randn(1, H, dtype=torch.float16, device="cuda")

        vllm_out = run_vllm_forward(moe, x, vllm_config)
        our_out  = run_ours(ext, x.squeeze(0), router_weight,
                            gate_projs, up_projs, down_projs)

        ma, mea, mr = check_one(vllm_out, our_out)
        results.append((i, ma, mea, mr))
        if ma > max_abs_all:
            max_abs_all = ma

    # Table
    print(f"\n  {'seed':>4s}  {'max_abs':>10s}  {'mean_abs':>10s}  {'max_rel':>10s}  {'status':>6s}")
    print(f"  {'─' * 4}  {'─' * 10}  {'─' * 10}  {'─' * 10}  {'─' * 6}")
    for seed, ma, mea, mr in results:
        ok = "OK" if ma < ATOL else "FAIL"
        print(f"  {seed:4d}  {ma:10.6f}  {mea:10.6f}  {mr:10.6f}  {ok:>6s}")

    passed = max_abs_all < ATOL
    print(f"\n  worst max_abs = {max_abs_all:.6f}  (tol = {ATOL})")
    print(f"  {'PASSED' if passed else 'FAILED'}")
    return passed


def bench_ms(fn, warmup=WARMUP, iters=BENCH_ITER):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters


def benchmark_suite(ext, moe, vllm_config,
                    router_weight, gate_projs, up_projs, down_projs):
    print(f"\n{'=' * 60}")
    print(f" Benchmark — {BENCH_ITER} iterations, {WARMUP} warmup")
    print(f"{'=' * 60}\n")

    seeds = [0, 1, 2, 3]
    inputs = []
    for s in seeds:
        torch.manual_seed(s)
        inputs.append(torch.randn(1, H, dtype=torch.float16, device="cuda"))

    idx = [0]

    def vllm_call():
        x = inputs[idx[0] % len(inputs)]
        idx[0] += 1
        return run_vllm_forward(moe, x, vllm_config)

    def ours_call():
        x = inputs[idx[0] % len(inputs)]
        idx[0] += 1
        return run_ours(ext, x.squeeze(0), router_weight,
                        gate_projs, up_projs, down_projs)

    idx[0] = 0
    vllm_ms = bench_ms(vllm_call)

    idx[0] = 0
    ours_ms = bench_ms(ours_call)

    # Weight memory (read per forward)
    dsz = 2  # fp16
    weight_bytes = E * (2 * H * I + H * I) * dsz  # gate+up+down for all experts
    # But only K experts are active: effective read
    active_bytes = K * (2 * H * I + H * I) * dsz
    ours_gbps = active_bytes / (ours_ms * 1e-3) / 1e9
    vllm_gbps = active_bytes / (vllm_ms * 1e-3) / 1e9

    print(f"  {'':20s} {'ms/token':>10s} {'us/token':>10s} {'GB/s':>8s}")
    print(f"  {'─' * 20} {'─' * 10} {'─' * 10} {'─' * 8}")
    print(f"  {'vLLM':20s} {vllm_ms:10.4f} {vllm_ms*1000:10.1f} {vllm_gbps:8.1f}")
    print(f"  {'FlashMoE (ours)':20s} {ours_ms:10.4f} {ours_ms*1000:10.1f} {ours_gbps:8.1f}")
    print(f"\n  Speedup: {vllm_ms / ours_ms:.2f}x")


# ── Main ─────────────────────────────────────────────────────

def main():
    dtype = torch.float16

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Config: H={H} I={I} E={E} K={K}  dtype=fp16")

    vllm_config = setup_vllm(dtype)
    moe, gate_w, w13, w2 = make_moe(vllm_config, dtype)
    router_weight, gate_projs, up_projs, down_projs = extract_weights_for_kernel(gate_w, w13, w2)

    ext = build_kernel()

    passed = correctness_suite(ext, moe, vllm_config,
                               router_weight, gate_projs, up_projs, down_projs)

    benchmark_suite(ext, moe, vllm_config,
                    router_weight, gate_projs, up_projs, down_projs)

    teardown_vllm()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
