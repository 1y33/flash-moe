from .base import MoEModelConfig, ModelRegistry


DeepSeek_V2_Lite = ModelRegistry.register(
    MoEModelConfig(
        name="deepseek-v2-lite",
        family="deepseek",
        hidden_size=2048,
        intermediate_size=1408,
        num_experts=64,
        top_k=6,
        num_moe_layers=27,
        has_shared_expert=True,
        activation="silu",
        default_dtype="bf16",
        huggingface_id="deepseek-ai/DeepSeek-V2-Lite",
        notes="2 shared experts + 64 routed (top-6). Smaller K than Qwen3.",
    )
)

DeepSeek_V3 = ModelRegistry.register(
    MoEModelConfig(
        name="deepseek-v3",
        family="deepseek",
        hidden_size=7168,
        intermediate_size=2048,
        num_experts=256,
        top_k=8,
        num_moe_layers=58,
        has_shared_expert=True,
        activation="silu",
        default_dtype="bf16",
        huggingface_id="deepseek-ai/DeepSeek-V3",
        notes="671B total, 37B active. Far too large for a 4070 — kept for spec only.",
    )
)
