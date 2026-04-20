"""Shared vLLM setup helpers for benchmarks and correctness tests."""

from __future__ import annotations

import os

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12355")
os.environ.setdefault("VLLM_USE_V1", "1")

import torch

from vllm import EngineArgs
from vllm.config import set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.models.qwen3_moe import Qwen3MoeSparseMoeBlock
from vllm.v1.worker.workspace import init_workspace_manager, is_workspace_manager_initialized

H = 2048
I = 768
E = 128
K = 8


def setup_vllm(dtype: torch.dtype):
    """Initialize vLLM single-rank environment. Returns vllm_config."""
    engine_args = EngineArgs(model="Qwen/Qwen3-30B-A3B", dtype=dtype)
    vllm_config = engine_args.create_engine_config()
    torch.set_default_dtype(dtype)

    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1, rank=0,
            distributed_init_method="env://",
            local_rank=-1, backend="nccl",
        )
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

    if not is_workspace_manager_initialized():
        init_workspace_manager(torch.device("cuda"))

    return vllm_config


def make_moe(vllm_config, dtype):
    """Build Qwen3MoeSparseMoeBlock with random weights."""
    with set_current_vllm_config(vllm_config):
        moe = Qwen3MoeSparseMoeBlock(vllm_config=vllm_config, prefix="bench")
    moe.to("cuda").to(dtype)

    gate_w = torch.empty(E, H, dtype=dtype, device="cuda")
    w13    = torch.empty(E, 2 * I, H, dtype=dtype, device="cuda")
    w2     = torch.empty(E, H, I, dtype=dtype, device="cuda")
    for t in (gate_w, w13, w2):
        torch.nn.init.uniform_(t, -0.1, 0.1)

    moe.gate.weight.data = gate_w
    moe.experts.w13_weight.data = w13
    moe.experts.w2_weight.data = w2

    qm = moe.experts.quant_method
    if hasattr(qm, "process_weights_after_loading"):
        qm.process_weights_after_loading(moe.experts)

    return moe, gate_w, w13, w2


def extract_weights_for_kernel(gate_w, w13, w2):
    """Split vLLM's fused weights into per-expert tensors for our kernel.

    vLLM stores w13 as [E, 2*I, H] = [gate_proj; up_proj] stacked on dim 1.
    Our kernel expects separate gate_proj[I,H], up_proj[I,H], down_proj[H,I].
    """
    gate_projs = []
    up_projs = []
    down_projs = []
    for e in range(E):
        gate_projs.append(w13[e, :I, :].contiguous())
        up_projs.append(w13[e, I:, :].contiguous())
        down_projs.append(w2[e].contiguous())

    router_weight = gate_w.contiguous()
    return router_weight, gate_projs, up_projs, down_projs


def run_vllm_forward(moe, x, vllm_config):
    """Single forward pass through the vLLM MoE block."""
    with set_forward_context(None, vllm_config), torch.no_grad():
        return moe(x)


def teardown_vllm():
    """Cleanup vLLM distributed state."""
    destroy_model_parallel()
    destroy_distributed_environment()
