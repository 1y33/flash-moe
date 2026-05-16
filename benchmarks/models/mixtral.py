from .base import MoEModelConfig, ModelRegistry


Mixtral_8x7B = ModelRegistry.register(
    MoEModelConfig(
        name="mixtral-8x7b",
        family="mixtral",
        hidden_size=4096,
        intermediate_size=14336,
        num_experts=8,
        top_k=2,
        num_moe_layers=32,
        has_shared_expert=False,
        activation="silu",
        default_dtype="fp16",
        huggingface_id="mistralai/Mixtral-8x7B-v0.1",
        notes="Wide-FFN, low-K regime. Each expert is ~3.5x larger than Qwen3.",
    )
)

Mixtral_8x22B = ModelRegistry.register(
    MoEModelConfig(
        name="mixtral-8x22b",
        family="mixtral",
        hidden_size=6144,
        intermediate_size=16384,
        num_experts=8,
        top_k=2,
        num_moe_layers=56,
        default_dtype="fp16",
        huggingface_id="mistralai/Mixtral-8x22B-v0.1",
    )
)
