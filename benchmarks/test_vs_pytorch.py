"""Compare our kernel against a pure PyTorch reference (no vLLM).

This isolates our kernel from vLLM's implementation details.
If this fails, the bug is in our kernel.
If this passes but test_correctness fails, the bug is in weight extraction.

Run:  python benchmarks/test_vs_pytorch.py
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.cpp_extension import load

H = 2048
I = 768
E = 128
K = 8


def pytorch_moe_reference(x, router_w, gate_projs, up_projs, down_projs):
    """Pure PyTorch MoE forward — the ground truth."""
    # Router
    logits = x @ router_w.T  # [E]
    topk_weights, topk_ids = torch.topk(logits, K)
    topk_weights = torch.softmax(topk_weights, dim=-1)

    # Expert computation
    output = torch.zeros(H, device=x.device)
    for i in range(K):
        eid = topk_ids[i].item()
        w = topk_weights[i].item()

        gate_out = x @ gate_projs[eid].T  # [I]
        up_out = x @ up_projs[eid].T      # [I]
        act = torch.nn.functional.silu(gate_out) * up_out  # [I]
        expert_out = act @ down_projs[eid].T  # [H]
        output += w * expert_out

    return output, topk_ids, topk_weights


def main():
    device = "cuda"
    torch.manual_seed(42)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Config: H={H} I={I} E={E} K={K}\n")

    # Build kernel
    print("JIT compiling...")
    flash_moe = load(
        name="flash_moe",
        sources=["csrc/binding.cu", "csrc/kernel.cu"],
        extra_cuda_cflags=["-arch=sm_89", "-Icsrc"],
        verbose=False,
    )

    # Random weights (small scale for numerical stability)
    router_w = torch.randn(E, H, device=device) * 0.01
    gate_projs = [torch.randn(I, H, device=device) * 0.01 for _ in range(E)]
    up_projs = [torch.randn(I, H, device=device) * 0.01 for _ in range(E)]
    down_projs = [torch.randn(H, I, device=device) * 0.01 for _ in range(E)]

    x = torch.randn(H, device=device)

    # PyTorch reference
    ref_out, ref_ids, ref_weights = pytorch_moe_reference(
        x, router_w, gate_projs, up_projs, down_projs)

    print(f"PyTorch top-K experts: {ref_ids.tolist()}")
    print(f"PyTorch top-K weights: {[f'{w:.4f}' for w in ref_weights.tolist()]}")
    print(f"PyTorch output norm:   {ref_out.norm().item():.6f}")

    # Our kernel
    our_out = flash_moe.forward(x, router_w, gate_projs, up_projs, down_projs)

    print(f"Our output norm:       {our_out.norm().item():.6f}")

    # Compare
    abs_diff = (ref_out - our_out).abs()
    rel_diff = abs_diff / (ref_out.abs() + 1e-8)

    print(f"\nMax abs diff:  {abs_diff.max().item():.6f}")
    print(f"Mean abs diff: {abs_diff.mean().item():.6f}")
    print(f"Max rel diff:  {rel_diff.max().item():.6f}")
    print(f"Mean rel diff: {rel_diff.mean().item():.6f}")

    atol = 0.01
    if abs_diff.max().item() < atol:
        print(f"\nPASSED (max abs diff < {atol})")
    else:
        print(f"\nFAILED (max abs diff >= {atol})")
        worst = abs_diff.topk(5)
        for val, idx in zip(worst.values, worst.indices):
            print(f"  [{idx.item()}] ref={ref_out[idx].item():.6f} "
                  f"ours={our_out[idx].item():.6f} diff={val.item():.6f}")


if __name__ == "__main__":
    main()
