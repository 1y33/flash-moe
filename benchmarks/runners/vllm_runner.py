"""vLLM runner — uses the existing vllm_utils setup (Qwen3-30B-A3B), so it
currently only supports models with H=2048, I=768, E=128, K=8.

For other configs, supports() returns False. Generalizing this requires
calling Qwen3MoeSparseMoeBlock with a custom config, which we leave as
future work — the goal here is to have *at least one* external baseline
that runs across all our model entries via flashmoe_runner.
"""

from __future__ import annotations
import torch

from ..models import MoEModelConfig
from ..workloads import Workload
from .base_runner import BaseRunner


class VLLMRunner(BaseRunner):
    name = "vllm"

    def __init__(self):
        self._setup_done = False
        self.vllm_config = None
        self.moe = None

    def supports(self, model: MoEModelConfig, wl: Workload) -> bool:
        # vllm_utils.py is currently hardcoded to Qwen3-30B-A3B dims.
        if (model.hidden_size, model.intermediate_size,
                model.num_experts, model.top_k) != (2048, 768, 128, 8):
            return False
        if wl.dtype not in ("fp16", "bf16"):
            return False
        return True

    def setup(self, model: MoEModelConfig, wl: Workload) -> None:
        from benchmarks.vllm_utils import setup_vllm, make_moe
        dtype = torch.float16 if wl.dtype == "fp16" else torch.bfloat16
        if not self._setup_done:
            self.vllm_config = setup_vllm(dtype)
            self._setup_done = True
        self.moe, _, _, _ = make_moe(self.vllm_config, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from benchmarks.vllm_utils import run_vllm_forward
        # vLLM expects a leading dim
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return run_vllm_forward(self.moe, x, self.vllm_config)

    def teardown(self) -> None:
        self.moe = None
        super().teardown()
