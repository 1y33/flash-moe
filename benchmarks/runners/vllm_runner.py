"""vLLM runner — generalized to accept arbitrary MoE dims.

We bootstrap vLLM once with the Qwen3 config (cheap, just metadata — no
weights are loaded), then patch the in-memory hf_config to the dims of
whatever model we're benchmarking before constructing Qwen3MoeSparseMoeBlock.

This lets us compare FlashMoE vs vLLM on every MoE config — not just the
single hardcoded shape. vLLM still only supports models whose activation is
SiLU/SwiGLU and whose routing is plain top-K softmax (i.e. what
Qwen3MoeSparseMoeBlock implements). DeepSeek's shared-expert path is NOT
emulated here — for those models, what we time is the routed path only.
"""

from __future__ import annotations
import torch

from ..models import MoEModelConfig
from ..workloads import Workload
from .base_runner import BaseRunner
from .weights import MoEWeights


class VLLMRunner(BaseRunner):
    name = "vllm"

    # Class-level cache so we only pay the EngineArgs/distributed init once
    # per process, regardless of how many model shapes we bench.
    _vllm_config = None
    _setup_done = False
    _prefix_counter = 0

    def __init__(self):
        self.moe = None
        VLLMRunner._prefix_counter += 1
        self._prefix = f"bench{VLLMRunner._prefix_counter}"

    def supports(self, model: MoEModelConfig, wl: Workload) -> bool:
        if wl.dtype not in ("fp16", "bf16"):
            return False
        # Qwen3MoeSparseMoeBlock uses SiLU/SwiGLU. Everything else differs.
        if model.activation not in ("silu", "swiglu"):
            return False
        return True

    def setup(self, model: MoEModelConfig, wl: Workload,
              weights: MoEWeights | None = None) -> None:
        from benchmarks.vllm_utils import setup_vllm
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock

        dtype = torch.float16 if wl.dtype == "fp16" else torch.bfloat16
        if not VLLMRunner._setup_done:
            VLLMRunner._vllm_config = setup_vllm(dtype)
            VLLMRunner._setup_done = True
        vllm_config = VLLMRunner._vllm_config

        # Patch the hf_config to the dims we want to benchmark. The block
        # reads everything from this struct at construction time.
        hf = vllm_config.model_config.hf_config
        hf.hidden_size = model.hidden_size
        hf.moe_intermediate_size = model.intermediate_size
        hf.num_experts = model.num_experts
        hf.num_experts_per_tok = model.top_k

        with set_current_vllm_config(vllm_config):
            moe = Qwen3MoeSparseMoeBlock(vllm_config=vllm_config, prefix=self._prefix)
        moe.to("cuda").to(dtype)

        H, I, E = model.hidden_size, model.intermediate_size, model.num_experts
        if weights is None:
            weights = MoEWeights.random(H, I, E, dtype)

        # vLLM stores w13 as [E, 2*I, H] = [gate; up] stacked on dim 1.
        gate_w = weights.router.to(dtype).contiguous()
        w13 = torch.empty(E, 2 * I, H, dtype=dtype, device="cuda")
        w13[:, :I, :] = weights.gate_w.to(dtype)
        w13[:, I:, :] = weights.up_w.to(dtype)
        w2 = weights.down_w.to(dtype).contiguous()

        moe.gate.weight.data = gate_w
        moe.experts.w13_weight.data = w13
        moe.experts.w2_weight.data = w2
        qm = moe.experts.quant_method
        if hasattr(qm, "process_weights_after_loading"):
            qm.process_weights_after_loading(moe.experts)

        self.moe = moe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from benchmarks.vllm_utils import run_vllm_forward
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return run_vllm_forward(self.moe, x, VLLMRunner._vllm_config)

    def teardown(self) -> None:
        self.moe = None
        super().teardown()
