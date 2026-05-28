"""Shared MoE weight container.

Lets multiple runners use exactly the same numerical weights — important
for apples-to-apples correctness checks (runners with different random
inits would never agree).
"""

from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class MoEWeights:
    """Per-expert MoE weights in a runner-agnostic layout.

    Conventions:
      - router: [E, H]   (one row per expert)
      - gate_w: [E, I, H]
      - up_w:   [E, I, H]
      - down_w: [E, H, I]
    Runners convert into their preferred layout in their own setup().
    """

    router: torch.Tensor
    gate_w: torch.Tensor
    up_w: torch.Tensor
    down_w: torch.Tensor

    @classmethod
    def random(cls, H: int, I: int, E: int, dtype: torch.dtype,
               device: str = "cuda", scale: float = 0.1) -> "MoEWeights":
        def t(*shape):
            return torch.empty(*shape, dtype=dtype, device=device).uniform_(-scale, scale)
        return cls(
            router=t(E, H),
            gate_w=t(E, I, H),
            up_w=t(E, I, H),
            down_w=t(E, H, I),
        )

    @property
    def E(self) -> int:
        return self.router.shape[0]

    @property
    def H(self) -> int:
        return self.router.shape[1]

    @property
    def I(self) -> int:
        return self.gate_w.shape[1]
