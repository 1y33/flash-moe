"""Categorize SASS opcodes into high-level buckets for instruction-mix charts."""

from __future__ import annotations
from dataclasses import dataclass
from .sass_parser import SassFunctionStats


CATEGORIES = [
    "Memory (global)",
    "Memory (shared)",
    "Memory (local/spill)",
    "Memory (atomic)",
    "Compute (FP)",
    "Compute (INT)",
    "Compute (Tensor)",
    "Control flow",
    "Synchronization",
    "Other",
]


@dataclass
class InstructionMix:
    counts: dict[str, int]
    total: int

    def percent(self, cat: str) -> float:
        return (self.counts.get(cat, 0) / self.total * 100) if self.total else 0.0


def categorize(fn: SassFunctionStats) -> InstructionMix:
    counts = {c: 0 for c in CATEGORIES}

    counts["Memory (global)"] = (sum(fn.ldg_widths.values())
                                  + sum(fn.stg_widths.values()))
    counts["Memory (shared)"] = (sum(fn.lds_widths.values())
                                  + sum(fn.sts_widths.values()))
    counts["Memory (local/spill)"] = (sum(fn.ldl_widths.values())
                                       + sum(fn.stl_widths.values()))
    counts["Memory (atomic)"] = sum(fn.red_atomic_counts.values())

    counts["Compute (FP)"] = (fn.ffma + fn.fadd + fn.fmul
                              + fn.hfma2 + fn.hadd2 + fn.hmul2)
    counts["Compute (INT)"] = fn.imad + fn.iadd3
    counts["Compute (Tensor)"] = fn.hmma + fn.imma

    counts["Control flow"] = fn.branches
    counts["Synchronization"] = fn.barriers + fn.membars + fn.shfl

    accounted = sum(counts.values())
    counts["Other"] = max(0, fn.total_insns - accounted)

    return InstructionMix(counts=counts, total=fn.total_insns)
