"""Compare FlashMoE kernel output against vLLM's Qwen3 MoE block.

Runs correctness check (with tolerance) and benchmarks both implementations.

Run:  python tests/test_correctness.py
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


def build_kernel():
    print("JIT compiling FlashMoE kernel...")
    return load(
        name="flash_moe_ext",
        sources=["csrc/binding.cu", "csrc/kernel.cu"],
        extra_cuda_cflags=["-arch=sm_89", "-Icsrc"],
        verbose=True,
    )


def run_our_forward(flash_moe, x_f32, router_weight, gate_projs, up_projs, down_projs):
    return flash_moe.forward(x_f32, router_weight, gate_projs, up_projs, down_projs)


def check_correctness(vllm_out, our_out, atol=0.05):
    vllm_f32 = vllm_out.squeeze(0).float()
    our_f32 = our_out.float()

    abs_diff = (vllm_f32 - our_f32).abs()
    rel_diff = abs_diff / (vllm_f32.abs() + 1e-8)

    print(f"vLLM output norm:  {vllm_f32.norm().item():.6f}")
    print(f"Our output norm:   {our_f32.norm().item():.6f}")
    print(f"vLLM[:8]:  {vllm_f32[:8].tolist()}")
    print(f"Ours[:8]:  {our_f32[:8].tolist()}")
    print(f"Max abs diff:      {abs_diff.max().item():.6f}")
    print(f"Mean abs diff:     {abs_diff.mean().item():.6f}")
    print(f"Max rel diff:      {rel_diff.max().item():.6f}")
    print(f"Mean rel diff:     {rel_diff.mean().item():.6f}")

    passed = abs_diff.max().item() < atol
    if passed:
        print(f"\nPASSED (max abs diff < {atol})")
    else:
        print(f"\nFAILED (max abs diff >= {atol})")
        worst = abs_diff.topk(5)
        for val, idx in zip(worst.values, worst.indices):
            print(f"  [{idx.item()}] vllm={vllm_f32[idx].item():.6f} "
                  f"ours={our_f32[idx].item():.6f} diff={val.item():.6f}")

    return passed


def benchmark(flash_moe, moe, x, vllm_config, router_weight,
              gate_projs, up_projs, down_projs, warmup=10, iters=100):
    print("\n--- Benchmark ---")

    x_f32 = x.squeeze(0).float()

    # Warmup vLLM
    for _ in range(warmup):
        run_vllm_forward(moe, x, vllm_config)
    torch.cuda.synchronize()

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run_vllm_forward(moe, x, vllm_config)
    end.record()
    torch.cuda.synchronize()
    vllm_ms = start.elapsed_time(end) / iters

    # Warmup ours
    for _ in range(warmup):
        run_our_forward(flash_moe, x_f32, router_weight, gate_projs, up_projs, down_projs)
    torch.cuda.synchronize()

    start.record()
    for _ in range(iters):
        run_our_forward(flash_moe, x_f32, router_weight, gate_projs, up_projs, down_projs)
    end.record()
    torch.cuda.synchronize()
    ours_ms = start.elapsed_time(end) / iters

    print(f"vLLM:    {vllm_ms:.4f} ms/token")
    print(f"Ours:    {ours_ms:.4f} ms/token")
    print(f"Speedup: {vllm_ms / ours_ms:.2f}x")


def main():
    dtype = torch.float16

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Config: H={H} I={I} E={E} K={K}\n")

    # 1. Setup vLLM + weights
    print("Setting up vLLM...")
    vllm_config = setup_vllm(dtype)
    moe, gate_w, w13, w2 = make_moe(vllm_config, dtype)
    router_weight, gate_projs, up_projs, down_projs = extract_weights_for_kernel(gate_w, w13, w2)

    # 2. Build our kernel
    flash_moe = build_kernel()

    # 3. Run both on same input
    torch.manual_seed(42)
    x = torch.randn(1, H, dtype=dtype, device="cuda")

    vllm_out = run_vllm_forward(moe, x, vllm_config)
    our_out = run_our_forward(
        flash_moe, x.squeeze(0).float(), router_weight, gate_projs, up_projs, down_projs)

    # 4. Correctness check
    passed = check_correctness(vllm_out, our_out)

    # 5. Benchmark
    benchmark(flash_moe, moe, x, vllm_config, router_weight,
              gate_projs, up_projs, down_projs)

    teardown_vllm()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
