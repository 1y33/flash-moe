"""Standalone smoke test for the FlashMoE kernel (no vLLM dependency).

Run:  .venv/bin/python -m benchmarks.test_moe
"""

import torch
from torch.utils.cpp_extension import load

flash_moe = load(
    name="flash_moe",
    sources=["csrc/binding.cu", "csrc/kernel.cu"],
    extra_cuda_cflags=["-arch=sm_89", "-Icsrc"],
    verbose=True,
)

H = 2048
I = 768
E = 128

torch.manual_seed(42)
device = "cuda"

input_vec = torch.randn(H, device=device)
router_weight = torch.randn(E, H, device=device) * 0.01
gate_projs = [torch.randn(I, H, device=device) * 0.01 for _ in range(E)]
up_projs = [torch.randn(I, H, device=device) * 0.01 for _ in range(E)]
down_projs = [torch.randn(H, I, device=device) * 0.01 for _ in range(E)]

output = flash_moe.forward(input_vec, router_weight, gate_projs, up_projs, down_projs)

print(f"Output shape: {output.shape}")
print(f"Output norm:  {output.norm().item():.4f}")
print(f"Output[:5]:   {output[:5]}")
print("Done!")
