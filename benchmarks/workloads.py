from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

from .models import DType


class Workload(BaseModel):
    """One benchmark workload: how to run, not what to run."""

    model_config = {"frozen": True, "extra": "forbid"}

    batch_size: int = Field(ge=1, default=1)
    dtype: DType = "fp16"
    warmup_iters: int = Field(ge=0, default=50)
    bench_iters: int = Field(ge=1, default=200)
    seed: int = 0

    @property
    def label(self) -> str:
        return f"b{self.batch_size}_{self.dtype}"


class BenchmarkResult(BaseModel):
    """One (model, workload, runner) measurement."""

    model_config = {"extra": "forbid"}

    model_name: str
    runner: str
    batch_size: int
    dtype: DType
    iters: int

    mean_ms: float = Field(description="Mean per-iteration latency, ms")
    p50_ms: float | None = None
    p99_ms: float | None = None
    min_ms: float | None = None
    max_ms: float | None = None

    achieved_gbps: float | None = None
    peak_gbps: float | None = None
    bandwidth_utilization: float | None = None

    correctness_ok: bool | None = None
    max_abs_error: float | None = None

    error: str | None = None
    notes: str | None = None
