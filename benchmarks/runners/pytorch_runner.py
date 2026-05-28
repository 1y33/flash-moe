"""Reference PyTorch MoE — slow but model-agnostic baseline."""

from __future__ import annotations
import torch
import torch.nn.functional as F

from ..models import MoEModelConfig
from ..workloads import Workload
from .base_runner import BaseRunner
from .weights import MoEWeights


class PyTorchRunner(BaseRunner):
    name = "pytorch-eager"

    def supports(self, model: MoEModelConfig, wl: Workload) -> bool:
        return wl.dtype in ("fp16", "bf16", "fp32")

    def setup(self, model: MoEModelConfig, wl: Workload,
              weights: MoEWeights | None = None) -> None:
        self.model = model
        torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                       "fp32": torch.float32}[wl.dtype]
        H, I, E = model.hidden_size, model.intermediate_size, model.num_experts

        if weights is None:
            weights = MoEWeights.random(H, I, E, torch_dtype)
        self.router = weights.router
        self.gate_w = weights.gate_w
        self.up_w   = weights.up_w
        self.down_w = weights.down_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H]
        K = self.model.top_k
        # Router
        logits = x @ self.router.t()                  # [B, E]
        probs = F.softmax(logits, dim=-1)
        weights, ids = probs.topk(K, dim=-1)            # [B, K]
        # Renormalize (some MoEs do this; not all)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        B, H = x.shape
        out = torch.zeros(B, H, dtype=x.dtype, device=x.device)
        for k in range(K):
            eids = ids[:, k]                            # [B]
            ws   = weights[:, k]                        # [B]
            # Gather per-token expert weights
            g = self.gate_w[eids]                       # [B, I, H]
            u = self.up_w[eids]
            d = self.down_w[eids]
            gx = torch.einsum("bih,bh->bi", g, x)
            ux = torch.einsum("bih,bh->bi", u, x)
            act = F.silu(gx) * ux
            y = torch.einsum("bhi,bi->bh", d, act)
            out += ws.unsqueeze(-1) * y
        return out

    def teardown(self) -> None:
        self.router = None
        self.gate_w = None
        self.up_w = None
        self.down_w = None
        super().teardown()
