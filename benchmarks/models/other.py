from .base import MoEModelConfig, ModelRegistry


Phi35_MoE = ModelRegistry.register(
    MoEModelConfig(
        name="phi-3.5-moe",
        family="phi",
        hidden_size=4096,
        intermediate_size=6400,
        num_experts=16,
        top_k=2,
        num_moe_layers=32,
        activation="silu",
        default_dtype="bf16",
        huggingface_id="microsoft/Phi-3.5-MoE-instruct",
    )
)

OLMoE_1B_7B = ModelRegistry.register(
    MoEModelConfig(
        name="olmoe-1b-7b",
        family="olmoe",
        hidden_size=2048,
        intermediate_size=1024,
        num_experts=64,
        top_k=8,
        num_moe_layers=16,
        activation="silu",
        default_dtype="bf16",
        huggingface_id="allenai/OLMoE-1B-7B-0924",
    )
)

Qwen2_57B_A14B = ModelRegistry.register(
    MoEModelConfig(
        name="qwen2-57b-a14b",
        family="qwen2",
        hidden_size=3584,
        intermediate_size=2560,
        num_experts=64,
        top_k=8,
        num_moe_layers=28,
        has_shared_expert=True,
        default_dtype="bf16",
        huggingface_id="Qwen/Qwen2-57B-A14B",
    )
)
