from .base import MoEModelConfig, ModelRegistry


Qwen3_30B_A3B = ModelRegistry.register(
    MoEModelConfig(
        name="qwen3-30b-a3b",
        family="qwen3",
        hidden_size=2048,
        intermediate_size=768,
        num_experts=128,
        top_k=8,
        num_moe_layers=48,
        has_shared_expert=False,
        activation="silu",
        default_dtype="fp16",
        huggingface_id="Qwen/Qwen3-30B-A3B",
        notes="The current FlashMoE target. 30B total, 3B active.",
    )
)
