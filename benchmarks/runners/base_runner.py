from __future__ import annotations
from abc import ABC, abstractmethod
import torch

from ..models import MoEModelConfig
from ..workloads import Workload, BenchmarkResult
from .weights import MoEWeights


class BaseRunner(ABC):
    """Interface every runner implements."""

    name: str = "unknown"

    @abstractmethod
    def supports(self, model: MoEModelConfig, wl: Workload) -> bool:
        """Return True if this runner can run this (model, workload) combo."""

    @abstractmethod
    def setup(self, model: MoEModelConfig, wl: Workload,
              weights: MoEWeights | None = None) -> None:
        """Allocate weights, build kernels. Called once per (model, wl).

        If `weights` is given, the runner MUST use it (verbatim) instead of
        generating its own random tensors. This enables apples-to-apples
        correctness checks across runners.
        """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one forward. x shape: [batch, hidden_size]."""

    def teardown(self) -> None:
        torch.cuda.empty_cache()

    def benchmark(self, model: MoEModelConfig, wl: Workload,
                  weights: MoEWeights | None = None) -> BenchmarkResult:
        if not self.supports(model, wl):
            return BenchmarkResult(
                model_name=model.name, runner=self.name,
                batch_size=wl.batch_size, dtype=wl.dtype,
                iters=0, mean_ms=float("nan"),
                error="unsupported",
            )

        torch.manual_seed(wl.seed)
        try:
            self.setup(model, wl, weights=weights)
        except Exception as e:
            return BenchmarkResult(
                model_name=model.name, runner=self.name,
                batch_size=wl.batch_size, dtype=wl.dtype,
                iters=0, mean_ms=float("nan"),
                error=f"setup failed: {e!r}",
            )

        torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                       "fp32": torch.float32}[wl.dtype]
        x = torch.randn(wl.batch_size, model.hidden_size,
                        dtype=torch_dtype, device="cuda")

        try:
            for _ in range(wl.warmup_iters):
                _ = self.forward(x)
            torch.cuda.synchronize()

            events_start = [torch.cuda.Event(enable_timing=True)
                            for _ in range(wl.bench_iters)]
            events_end   = [torch.cuda.Event(enable_timing=True)
                            for _ in range(wl.bench_iters)]
            for i in range(wl.bench_iters):
                events_start[i].record()
                _ = self.forward(x)
                events_end[i].record()
            torch.cuda.synchronize()

            ts = sorted([s.elapsed_time(e) for s, e in zip(events_start, events_end)])
            mean_ms = sum(ts) / len(ts)
            p50 = ts[len(ts) // 2]
            p99 = ts[min(len(ts) - 1, int(len(ts) * 0.99))]

            # Memory bandwidth: how many bytes were *required* per forward?
            #   active experts × (gate + up + down) × per-token (no reuse at b=1)
            # At batch>1, tokens can share an expert's weights → effective bytes
            # drop, but worst case the expert is only read once per batch.
            # Use: weights_read = top_k * per_expert_bytes, then divide by batch
            # to get "per-token bandwidth" (a useful normalization).
            per_token_bytes = model.active_weight_bytes_per_token(wl.dtype)
            achieved_gbps = (per_token_bytes / (mean_ms * 1e-3)) / 1e9
            achieved_gbps *= wl.batch_size  # batch shares weights ideally

            result = BenchmarkResult(
                model_name=model.name, runner=self.name,
                batch_size=wl.batch_size, dtype=wl.dtype,
                iters=wl.bench_iters,
                mean_ms=mean_ms,
                p50_ms=p50, p99_ms=p99,
                min_ms=ts[0], max_ms=ts[-1],
                achieved_gbps=achieved_gbps,
            )
        except Exception as e:
            result = BenchmarkResult(
                model_name=model.name, runner=self.name,
                batch_size=wl.batch_size, dtype=wl.dtype,
                iters=0, mean_ms=float("nan"),
                error=f"forward failed: {e!r}",
            )
        finally:
            self.teardown()
        return result
