"""Arithmetic intensity & roofline analysis.

Static analysis of kernel FLOP / byte ratio based on instruction counts.
Two views:
  - **Per-instruction analytical**: count FMA, ADD, MUL → FLOPs; count
    LDG/LDS widths → bytes. This is the *static-issue* AI — what the
    compiler emitted, before any runtime memory-coalescing effects.
  - **Workload-level**: derived from problem dimensions (H, I, K, batch).
    Both ratios should agree for a well-vectorized memory-bound kernel.

We *don't* attempt a true dynamic count (would need profiling), but the
static-issue AI is already useful to compare kernels and predict roofline
position.
"""

from __future__ import annotations
from dataclasses import dataclass
from .archs import ArchSpec
from .sass_parser import SassFunctionStats


@dataclass
class RooflineResult:
    flops: int
    bytes_loaded: int
    bytes_stored: int
    arithmetic_intensity: float  # FLOP / byte
    ridge_point_fp16: float
    ridge_point_tc: float | None
    peak_dram_gbps: float
    peak_fp16_tflops: float
    peak_tc_tflops: float | None
    # Roof-imposed limits for *this* kernel:
    bandwidth_bound_tflops: float
    is_memory_bound: bool
    is_memory_bound_vs_tc: bool


def flops_per_op(name: str, count: int) -> int:
    """1 FMA = 2 FLOPs, 1 add/mul = 1 FLOP. HFMA2 = 4 FLOPs (2 FMAs at once)."""
    factor = {
        "ffma": 2, "fadd": 1, "fmul": 1,
        "hfma2": 4, "hadd2": 2, "hmul2": 2,
        # HMMA throughput depends heavily on the shape — count tile FLOPs in
        # the workload-level path instead; we don't try to attribute FLOPs to
        # individual HMMA instructions here.
        "hmma": 0,
        "imad": 0,   # int — not a FLOP
        "iadd3": 0,
    }
    return factor.get(name, 0) * count


def bytes_from_widths(widths: dict[int, int]) -> int:
    """Sum bits_per_access * count / 8."""
    return sum((w_bits // 8) * count for w_bits, count in widths.items())


def analyze_kernel(fn: SassFunctionStats, arch: ArchSpec,
                   use_tc_for_ai: bool = False) -> RooflineResult:
    flops = (flops_per_op("ffma",  fn.ffma)
             + flops_per_op("fadd", fn.fadd)
             + flops_per_op("fmul", fn.fmul)
             + flops_per_op("hfma2", fn.hfma2)
             + flops_per_op("hadd2", fn.hadd2)
             + flops_per_op("hmul2", fn.hmul2))
    # Global loads only — shared is intra-block reuse, doesn't count as "fresh" bandwidth.
    bytes_loaded = bytes_from_widths(fn.ldg_widths)
    bytes_stored = bytes_from_widths(fn.stg_widths)
    total_bytes = bytes_loaded + bytes_stored

    ai = (flops / total_bytes) if total_bytes > 0 else 0.0

    ridge_fp16 = arch.ridge_point_fp16
    ridge_tc = arch.ridge_point_tc
    is_memory_bound_cuda = ai < ridge_fp16
    is_memory_bound_tc = (ai < ridge_tc) if ridge_tc else True

    # The bandwidth ceiling for *this* AI:
    bw_bound = arch.dram_bw_gbps * 1e9 * ai / 1e12  # TFLOPs

    return RooflineResult(
        flops=flops,
        bytes_loaded=bytes_loaded,
        bytes_stored=bytes_stored,
        arithmetic_intensity=ai,
        ridge_point_fp16=ridge_fp16,
        ridge_point_tc=ridge_tc,
        peak_dram_gbps=arch.dram_bw_gbps,
        peak_fp16_tflops=arch.fp16_tflops,
        peak_tc_tflops=arch.tc_fp16_tflops,
        bandwidth_bound_tflops=bw_bound,
        is_memory_bound=is_memory_bound_cuda,
        is_memory_bound_vs_tc=is_memory_bound_tc,
    )


@dataclass
class WorkloadRoofline:
    """Per-token MoE workload arithmetic intensity, derived from H, I, K."""
    H: int
    I: int
    K: int
    batch: int

    @property
    def flops_per_token(self) -> int:
        # gate: 2*H*I FMAs   → 2*2*H*I FLOPs
        # up:   2*H*I FMAs
        # silu+mul: ~3*I FLOPs (sigmoid + multiply)
        # down: 2*I*H FMAs   → 2*2*H*I FLOPs
        # ×K experts
        return self.K * (2 * (2 * self.H * self.I) + 2 * (2 * self.H * self.I) + 3 * self.I)

    @property
    def bytes_per_token_fp16(self) -> int:
        # Reads: K * (gate[I,H] + up[I,H] + down[H,I]) in fp16 = 2 bytes
        # Plus router weight read once: H*E negligible vs experts
        # Activations are tiny.
        return self.K * (2 * self.H * self.I + self.H * self.I) * 2

    @property
    def arithmetic_intensity_fp16(self) -> float:
        # Many tokens share an expert at batch>1, so weight bytes amortize.
        bytes_amortized = self.bytes_per_token_fp16
        return self.flops_per_token / bytes_amortized

    def for_batch(self, batch: int) -> "WorkloadRoofline":
        # At batch B, the same weights serve B tokens, increasing AI by B.
        return WorkloadRoofline(self.H, self.I, self.K, batch)

    def amortized_intensity_fp16(self, batch: int) -> float:
        """AI assuming each weight matrix is read once per batch (best case)."""
        return self.arithmetic_intensity_fp16 * batch
