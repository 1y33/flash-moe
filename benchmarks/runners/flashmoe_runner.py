"""FlashMoE runner.

JIT-compiles a separate kernel per (model, dtype) by injecting compile-time
defines for hidden_size / intermediate_size / num_experts / top_k.

NOTE: This only works for batch_size == 1 with the current kernel. Higher
batches return error='unsupported' until the kernel grows a batch dim.
"""

from __future__ import annotations
import os
import hashlib
import torch
from torch.utils.cpp_extension import load

from ..models import MoEModelConfig
from ..workloads import Workload
from .base_runner import BaseRunner
from .weights import MoEWeights


def _build_ext(model: MoEModelConfig, dtype: str):
    """JIT-compile FlashMoE with model dims baked in via -D."""
    defines = [
        f"-DHIDDEN_SIZE_OVERRIDE={model.hidden_size}",
        f"-DMOE_INTERMEDIATE_SIZE_OVERRIDE={model.intermediate_size}",
        f"-DNUM_EXPERTS_OVERRIDE={model.num_experts}",
        f"-DTOP_K_OVERRIDE={model.top_k}",
    ]
    flags = "_".join(d.replace("-D", "").replace("=", "") for d in defines)
    name_hash = hashlib.sha1(flags.encode()).hexdigest()[:8]
    ext_name = f"flash_moe_{model.name.replace('-', '_').replace('.', '_')}_{dtype}_{name_hash}"

    return load(
        name=ext_name,
        sources=["csrc/binding.cu", "csrc/kernel.cu"],
        extra_cuda_cflags=["-arch=sm_89", "-O3", "-Icsrc"] + defines,
        verbose=False,
    )


class FlashMoERunner(BaseRunner):
    name = "flashmoe"

    def supports(self, model: MoEModelConfig, wl: Workload) -> bool:
        # Current kernel is single-token only.
        if wl.batch_size != 1:
            return False
        # Only fp16 path currently exposed via binding.cu.
        if wl.dtype != "fp16":
            return False
        return True

    def setup(self, model: MoEModelConfig, wl: Workload,
              weights: MoEWeights | None = None) -> None:
        self.ext = _build_ext(model, wl.dtype)
        H, I, E = model.hidden_size, model.intermediate_size, model.num_experts
        torch_dtype = torch.float16

        if weights is None:
            weights = MoEWeights.random(H, I, E, torch_dtype)

        self.router_weight = weights.router.contiguous()
        self.gate_projs = [weights.gate_w[e].contiguous() for e in range(E)]
        self.up_projs   = [weights.up_w[e].contiguous()   for e in range(E)]
        self.down_projs = [weights.down_w[e].contiguous() for e in range(E)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Kernel expects a single token vector [H].
        return self.ext.forward(x.squeeze(0), self.router_weight,
                                self.gate_projs, self.up_projs, self.down_projs)

    def teardown(self) -> None:
        self.router_weight = None
        self.gate_projs = None
        self.up_projs = None
        self.down_projs = None
        super().teardown()
