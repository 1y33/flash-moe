"""Register-pressure → occupancy analysis."""

from __future__ import annotations
from dataclasses import dataclass
from .archs import ArchSpec


@dataclass
class OccupancyResult:
    regs_per_thread: int
    threads_per_block: int
    smem_per_block: int

    max_threads_per_sm_by_regs: int
    max_threads_per_sm_by_smem: int
    max_threads_per_sm_by_blocks: int
    effective_max_threads_per_sm: int

    max_blocks_per_sm: int
    theoretical_occupancy: float   # 0..1
    spill_risk: bool
    smem_per_thread: float


def analyze(regs_per_thread: int, threads_per_block: int,
            smem_per_block: int, arch: ArchSpec) -> OccupancyResult:
    if regs_per_thread <= 0:
        regs_per_thread = 1

    threads_by_regs = arch.max_regs_per_sm // regs_per_thread
    threads_by_regs = min(threads_by_regs, arch.max_threads_per_sm)

    if smem_per_block > 0:
        blocks_by_smem = arch.max_smem_per_sm // smem_per_block
    else:
        blocks_by_smem = arch.max_blocks_per_sm
    threads_by_smem = blocks_by_smem * threads_per_block
    threads_by_smem = min(threads_by_smem, arch.max_threads_per_sm)

    blocks_by_count = arch.max_blocks_per_sm
    threads_by_blocks = blocks_by_count * threads_per_block
    threads_by_blocks = min(threads_by_blocks, arch.max_threads_per_sm)

    eff = min(threads_by_regs, threads_by_smem, threads_by_blocks)
    blocks = eff // threads_per_block
    occ = eff / arch.max_threads_per_sm

    return OccupancyResult(
        regs_per_thread=regs_per_thread,
        threads_per_block=threads_per_block,
        smem_per_block=smem_per_block,
        max_threads_per_sm_by_regs=threads_by_regs,
        max_threads_per_sm_by_smem=threads_by_smem,
        max_threads_per_sm_by_blocks=threads_by_blocks,
        effective_max_threads_per_sm=eff,
        max_blocks_per_sm=blocks,
        theoretical_occupancy=occ,
        spill_risk=regs_per_thread >= arch.max_regs_per_thread,
        smem_per_thread=(smem_per_block / threads_per_block) if threads_per_block else 0,
    )
