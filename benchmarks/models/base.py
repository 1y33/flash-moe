from __future__ import annotations
from typing import Literal, ClassVar
from pydantic import BaseModel, Field

DType = Literal["fp16", "bf16", "fp32"]

_DTYPE_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4}


class MoEModelConfig(BaseModel):
    """One MoE layer's configuration.

    A "model" here is one MoE block, not a whole transformer — that's what
    FlashMoE actually fuses into a single kernel. Whole-model latency just
    multiplies by num_layers.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    name: str
    family: str = Field(description="Architecture family, for grouping plots")
    hidden_size: int = Field(gt=0, description="H — token embedding width")
    intermediate_size: int = Field(
        gt=0, description="I — per-expert FFN inner width"
    )
    num_experts: int = Field(gt=0, description="E — total experts in this layer")
    top_k: int = Field(gt=0, description="K — experts activated per token")
    num_moe_layers: int = Field(
        gt=0, description="How many MoE layers in the full model (informational)"
    )
    has_shared_expert: bool = Field(
        default=False, description="DeepSeek-style always-active shared expert"
    )
    activation: Literal["silu", "gelu", "swiglu"] = "silu"
    default_dtype: DType = "fp16"

    huggingface_id: str | None = None
    notes: str | None = None

    def expert_weight_bytes(self, dtype: DType | None = None) -> int:
        """Bytes per single expert (gate + up + down projections)."""
        b = _DTYPE_BYTES[dtype or self.default_dtype]
        return b * (2 * self.hidden_size * self.intermediate_size
                    + self.hidden_size * self.intermediate_size)

    def all_experts_weight_bytes(self, dtype: DType | None = None) -> int:
        """Bytes for all experts in one MoE layer."""
        return self.num_experts * self.expert_weight_bytes(dtype)

    def active_weight_bytes_per_token(self, dtype: DType | None = None) -> int:
        """Bytes actually read per token in one MoE layer (top-K experts)."""
        return self.top_k * self.expert_weight_bytes(dtype)

    def router_weight_bytes(self, dtype: DType | None = None) -> int:
        b = _DTYPE_BYTES[dtype or self.default_dtype]
        return b * self.hidden_size * self.num_experts

    def ffn1_tiles_per_expert(self, tile_rows: int) -> int:
        return (self.intermediate_size + tile_rows - 1) // tile_rows

    def ffn2_tiles_per_expert(self, tile_rows: int) -> int:
        return (self.hidden_size + tile_rows - 1) // tile_rows

    def total_tasks(self, tile_rows: int) -> int:
        return self.top_k * (
            self.ffn1_tiles_per_expert(tile_rows)
            + self.ffn2_tiles_per_expert(tile_rows)
        )

    def __str__(self) -> str:
        return (
            f"{self.name} (H={self.hidden_size}, I={self.intermediate_size}, "
            f"E={self.num_experts}, K={self.top_k})"
        )


class ModelRegistry:
    """Iterable registry of MoE model configs."""

    _models: ClassVar[list[MoEModelConfig]] = []

    @classmethod
    def register(cls, m: MoEModelConfig) -> MoEModelConfig:
        if any(x.name == m.name for x in cls._models):
            raise ValueError(f"Model {m.name} already registered")
        cls._models.append(m)
        return m

    @classmethod
    def all(cls) -> list[MoEModelConfig]:
        return list(cls._models)

    @classmethod
    def by_name(cls, name: str) -> MoEModelConfig:
        for m in cls._models:
            if m.name == name:
                return m
        raise KeyError(f"Unknown model {name!r}. Known: {[x.name for x in cls._models]}")

    @classmethod
    def filter(cls, family: str | None = None,
               max_expert_bytes: int | None = None) -> list[MoEModelConfig]:
        out = cls._models
        if family is not None:
            out = [m for m in out if m.family == family]
        if max_expert_bytes is not None:
            out = [m for m in out if m.expert_weight_bytes() <= max_expert_bytes]
        return out
