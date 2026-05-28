"""GPU architecture specifications used by occupancy + roofline analysis.

Numbers from NVIDIA whitepapers and CUDA Programming Guide §C compute
capability tables. Bandwidth/TFLOPs entries reflect the boost-clock peak
each vendor publishes; real-world will be 10-20% lower.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ArchSpec:
    name: str
    sm_arch: str           # e.g. "sm_89"

    # Per-SM
    max_regs_per_sm: int
    max_regs_per_thread: int
    max_threads_per_sm: int
    max_blocks_per_sm: int
    max_smem_per_sm: int       # bytes
    max_smem_per_block: int    # bytes (dynamic + static)
    warp_size: int

    # Device-wide
    num_sms_typical: int       # one representative GPU per arch
    typical_gpu_name: str
    dram_bw_gbps: float        # peak DRAM bandwidth
    fp16_tflops: float
    fp32_tflops: float
    tc_fp16_tflops: float | None  # tensor-core peak; None on Pascal-and-earlier

    @property
    def ridge_point_fp16(self) -> float:
        """FLOP/byte where the roofline goes from memory- to compute-bound."""
        return self.fp16_tflops * 1e12 / (self.dram_bw_gbps * 1e9)

    @property
    def ridge_point_tc(self) -> float | None:
        if self.tc_fp16_tflops is None:
            return None
        return self.tc_fp16_tflops * 1e12 / (self.dram_bw_gbps * 1e9)


ARCH_DB: dict[str, ArchSpec] = {
    "sm_75": ArchSpec(  # Turing (T4, RTX 20-series)
        name="Turing", sm_arch="sm_75",
        max_regs_per_sm=65536, max_regs_per_thread=255,
        max_threads_per_sm=1024, max_blocks_per_sm=16,
        max_smem_per_sm=65536, max_smem_per_block=49152, warp_size=32,
        num_sms_typical=40, typical_gpu_name="T4",
        dram_bw_gbps=320.0, fp16_tflops=65.0, fp32_tflops=8.1,
        tc_fp16_tflops=130.0,
    ),
    "sm_80": ArchSpec(  # Ampere datacenter (A100)
        name="Ampere (A100)", sm_arch="sm_80",
        max_regs_per_sm=65536, max_regs_per_thread=255,
        max_threads_per_sm=2048, max_blocks_per_sm=32,
        max_smem_per_sm=167936, max_smem_per_block=163840, warp_size=32,
        num_sms_typical=108, typical_gpu_name="A100-80GB",
        dram_bw_gbps=2039.0, fp16_tflops=78.0, fp32_tflops=19.5,
        tc_fp16_tflops=312.0,
    ),
    "sm_86": ArchSpec(  # Ampere consumer (RTX 30-series)
        name="Ampere (RTX 30)", sm_arch="sm_86",
        max_regs_per_sm=65536, max_regs_per_thread=255,
        max_threads_per_sm=1536, max_blocks_per_sm=16,
        max_smem_per_sm=102400, max_smem_per_block=99328, warp_size=32,
        num_sms_typical=84, typical_gpu_name="RTX 3090",
        dram_bw_gbps=936.0, fp16_tflops=35.6, fp32_tflops=35.6,
        tc_fp16_tflops=142.0,
    ),
    "sm_89": ArchSpec(  # Ada Lovelace (RTX 40-series)
        name="Ada Lovelace", sm_arch="sm_89",
        max_regs_per_sm=65536, max_regs_per_thread=255,
        max_threads_per_sm=1536, max_blocks_per_sm=24,
        max_smem_per_sm=102400, max_smem_per_block=99328, warp_size=32,
        # RTX 4070 Laptop (the user's GPU)
        num_sms_typical=36, typical_gpu_name="RTX 4070 Laptop",
        dram_bw_gbps=272.0, fp16_tflops=22.6, fp32_tflops=22.6,
        tc_fp16_tflops=90.5,
    ),
    "sm_90": ArchSpec(  # Hopper (H100)
        name="Hopper", sm_arch="sm_90",
        max_regs_per_sm=65536, max_regs_per_thread=255,
        max_threads_per_sm=2048, max_blocks_per_sm=32,
        max_smem_per_sm=233472, max_smem_per_block=232448, warp_size=32,
        num_sms_typical=132, typical_gpu_name="H100 SXM",
        dram_bw_gbps=3350.0, fp16_tflops=133.8, fp32_tflops=66.9,
        tc_fp16_tflops=989.4,
    ),
}


def get_arch(name: str) -> ArchSpec:
    if name not in ARCH_DB:
        raise KeyError(
            f"Unknown arch {name!r}. Known: {sorted(ARCH_DB.keys())}"
        )
    return ARCH_DB[name]
