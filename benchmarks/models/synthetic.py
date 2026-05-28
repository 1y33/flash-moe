"""Synthetic MoE configs for dimension sweeps.

Each axis (H, I, E, K) varies around Qwen3-30B-A3B (H=2048, I=768, E=128, K=8)
holding the others fixed. Used to find the dimension regions where FlashMoE
beats pytorch-eager (and, where shape matches Qwen3, vLLM).

Constraints respected:
  - E must be a multiple of 32 (warp size, see softmax_topk.cuh:15)
  - H, I must be multiples of 8 (fp16 VEC width)
  - I must be a multiple of TILE_ROWS=128 (see flashmoe.cuh:85)
  - TOP_K*(FFN1_TILES + FFN2_TILES) <= CAPACITY=512
"""

from .base import MoEModelConfig, ModelRegistry


def _reg(name, H, I, E, K, family="synth"):
    ModelRegistry.register(MoEModelConfig(
        name=name, family=family,
        hidden_size=H, intermediate_size=I,
        num_experts=E, top_k=K,
        num_moe_layers=1, activation="silu",
        default_dtype="fp16",
        notes=f"synthetic axis sweep: H={H} I={I} E={E} K={K}",
    ))


# Baseline (already registered as qwen3-30b-a3b, skip)

# Axis: hidden_size (H), holding I=768 E=128 K=8
for H in (1024, 1536, 2560, 3072, 4096):
    _reg(f"synth-H{H}", H=H, I=768, E=128, K=8, family="synth-H")

# Axis: intermediate_size (I), holding H=2048 E=128 K=8
# I must be % 128 (TILE_ROWS) and the per-expert weight fits 8GB
# Each expert is 2*I*H + H*I = 3*I*H bytes (fp16 → *2)
# 128 experts * 3*I*2048*2 bytes = 1.5*I*MB. So I<=1024 → 1.5GB ok.
for I in (256, 512, 1024, 1536):  # 1536 → ~2.25GB experts, still ok
    _reg(f"synth-I{I}", H=2048, I=I, E=128, K=8, family="synth-I")

# Axis: num_experts (E), holding H=2048 I=768 K=8 (or K<=E)
# E must be % 32
for E in (32, 64, 96, 160, 192, 256):
    K = min(8, E)
    _reg(f"synth-E{E}", H=2048, I=768, E=E, K=K, family="synth-E")

# Axis: top_k (K), holding H=2048 I=768 E=128
# TOTAL_TASKS = K * (FFN1_TILES + FFN2_TILES) = K * (6 + 16) = 22K must be ≤512
# So K ≤ 23. Test K = 1, 2, 4, 16
for K in (1, 2, 4, 16):
    _reg(f"synth-K{K}", H=2048, I=768, E=128, K=K, family="synth-K")

# Cross-axis: many experts AND high top-k.
# This is the regime where vLLM's per-expert grouped-GEMM has to do more work
# and the dispatch/reduce becomes a bigger fraction. Hypothesis: FlashMoE's
# single-kernel design pulls ahead more as E*K grows.
#   TOTAL_TASKS = K * (I/128 + ceil(H/128)) = K * (6 + 16) = 22K  (H=2048, I=768)
#   Need 22K ≤ 512 → K ≤ 23
# Pairs (E, K):
for (E, K) in [(64, 16), (128, 16), (256, 8), (256, 16), (192, 12), (96, 16)]:
    _reg(f"synth-EK{E}x{K}", H=2048, I=768, E=E, K=K, family="synth-EK")

# Cross-axis: many experts AND wider FFN (I) AND high top-k.
# Trying to push the work-per-token even higher.
#   E=128, I=1024 → tasks = K*(8+16)=24K, K ≤ 21
for (I, K) in [(1024, 12), (1024, 16), (512, 16)]:
    _reg(f"synth-IK{I}x{K}", H=2048, I=I, E=128, K=K, family="synth-IK")
