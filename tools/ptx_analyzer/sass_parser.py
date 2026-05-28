"""Parse cuobjdump --dump-sass output.

SASS is the actual ISA that runs on the GPU (after ptxas allocates registers
and lowers PTX). Counting SASS gives ground-truth per-instruction stats, not
just PTX virtual estimates.

Sample SASS line shape:
    /*0260*/    LDG.E.128 R32, [R102.64+0x200] ;
    /*0090*/   @!P0  BRA 0xdf0 ;
    /*01b0*/         CS2R R6, SRZ ;

Each line starts with /*offset*/ then optional predicate guard, then the
mnemonic (which may itself contain dots and qualifiers, like
`LDG.E.128.SYS` or `FFMA.SAT`).
"""

from __future__ import annotations
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


# Function header
_FN_RE = re.compile(r"Function\s*:\s*(\S+)")
# Instruction line: /*OFFSET*/ [predicate] MNEMONIC operands ;
_INSN_RE = re.compile(
    r"^\s*/\*0x[0-9a-fA-F]+\*/.*$"   # skip raw hex bytes lines (after instr)
)
_PREFIX_RE = re.compile(r"^\s*/\*[0-9a-fA-F]+\*/\s*")
_GUARD_RE = re.compile(r"^@!?P\d+\s+")
_MNEMONIC_RE = re.compile(r"^([A-Z0-9._]+)")


@dataclass
class SassFunctionStats:
    name: str
    total_insns: int = 0
    mnemonic_counts: Counter = field(default_factory=Counter)
    opcode_base_counts: Counter = field(default_factory=Counter)  # e.g. LDG, FFMA

    # Memory
    ldg_widths: Counter = field(default_factory=Counter)   # global load widths in bits
    stg_widths: Counter = field(default_factory=Counter)
    lds_widths: Counter = field(default_factory=Counter)   # shared load
    sts_widths: Counter = field(default_factory=Counter)   # shared store
    ldl_widths: Counter = field(default_factory=Counter)   # local load (spill)
    stl_widths: Counter = field(default_factory=Counter)   # local store (spill)
    red_atomic_counts: Counter = field(default_factory=Counter)  # REDG / ATOMG / etc.

    # Compute
    ffma: int = 0      # FP32 FMA on CUDA cores
    fadd: int = 0
    fmul: int = 0
    hfma2: int = 0     # FP16x2 FMA
    hadd2: int = 0
    hmul2: int = 0
    hmma: int = 0      # Tensor core MMA
    imma: int = 0      # Integer tensor cores
    imad: int = 0
    iadd3: int = 0

    # Control & sync
    branches: int = 0
    barriers: int = 0   # BAR.SYNC etc.
    membars: int = 0    # MEMBAR
    shfl: int = 0

    @property
    def memory_loads(self) -> int:
        return sum(self.ldg_widths.values()) + sum(self.lds_widths.values()) \
             + sum(self.ldl_widths.values())

    @property
    def memory_stores(self) -> int:
        return sum(self.stg_widths.values()) + sum(self.sts_widths.values()) \
             + sum(self.stl_widths.values())

    @property
    def atomics(self) -> int:
        return sum(self.red_atomic_counts.values())

    @property
    def tensor_core_insns(self) -> int:
        return self.hmma + self.imma

    @property
    def has_local_memory_spills(self) -> bool:
        return bool(sum(self.ldl_widths.values()) + sum(self.stl_widths.values()))


@dataclass
class SassStats:
    functions: list[SassFunctionStats] = field(default_factory=list)

    @property
    def aggregate(self) -> SassFunctionStats:
        agg = SassFunctionStats(name="(aggregate)")
        for fn in self.functions:
            agg.total_insns += fn.total_insns
            agg.mnemonic_counts.update(fn.mnemonic_counts)
            agg.opcode_base_counts.update(fn.opcode_base_counts)
            agg.ldg_widths.update(fn.ldg_widths)
            agg.stg_widths.update(fn.stg_widths)
            agg.lds_widths.update(fn.lds_widths)
            agg.sts_widths.update(fn.sts_widths)
            agg.ldl_widths.update(fn.ldl_widths)
            agg.stl_widths.update(fn.stl_widths)
            agg.red_atomic_counts.update(fn.red_atomic_counts)
            agg.ffma += fn.ffma; agg.fadd += fn.fadd; agg.fmul += fn.fmul
            agg.hfma2 += fn.hfma2; agg.hadd2 += fn.hadd2; agg.hmul2 += fn.hmul2
            agg.hmma += fn.hmma; agg.imma += fn.imma
            agg.imad += fn.imad; agg.iadd3 += fn.iadd3
            agg.branches += fn.branches; agg.barriers += fn.barriers
            agg.membars += fn.membars; agg.shfl += fn.shfl
        return agg


# Memory mnemonic widths come from suffixes:
#   LDG.E       -> 32 bit
#   LDG.E.64    -> 64 bit
#   LDG.E.128   -> 128 bit
#   LDS.U.128   -> 128 bit shared
# So we extract the trailing ".N" if present, else default to 32.
def _width_bits(mnemonic: str) -> int:
    for w in (128, 64, 16, 8):
        if f".{w}" in mnemonic:
            return w
    return 32


def _classify_and_count(fn: SassFunctionStats, mnemonic: str):
    fn.total_insns += 1
    fn.mnemonic_counts[mnemonic] += 1
    base = mnemonic.split(".", 1)[0]
    fn.opcode_base_counts[base] += 1

    # ── Memory ──────────────────────────────────────────────────
    if base == "LDG":
        fn.ldg_widths[_width_bits(mnemonic)] += 1
    elif base == "STG":
        fn.stg_widths[_width_bits(mnemonic)] += 1
    elif base in ("LDS", "LDSM"):
        fn.lds_widths[_width_bits(mnemonic)] += 1
    elif base == "STS":
        fn.sts_widths[_width_bits(mnemonic)] += 1
    elif base == "LDL":
        fn.ldl_widths[_width_bits(mnemonic)] += 1
    elif base == "STL":
        fn.stl_widths[_width_bits(mnemonic)] += 1
    elif base.startswith("ATOM") or base.startswith("RED"):
        fn.red_atomic_counts[mnemonic] += 1

    # ── Compute (FP) ─────────────────────────────────────────────
    elif base == "FFMA":
        fn.ffma += 1
    elif base == "FADD":
        fn.fadd += 1
    elif base == "FMUL":
        fn.fmul += 1
    elif base == "HFMA2":
        fn.hfma2 += 1
    elif base == "HADD2":
        fn.hadd2 += 1
    elif base == "HMUL2":
        fn.hmul2 += 1
    elif base in ("HMMA", "HMNMX2"):
        fn.hmma += 1
    elif base in ("IMMA",):
        fn.imma += 1

    # ── Compute (INT) ─────────────────────────────────────────────
    elif base == "IMAD":
        fn.imad += 1
    elif base == "IADD3":
        fn.iadd3 += 1

    # ── Control & sync ─────────────────────────────────────────────
    elif base in ("BRA", "BRX", "BSSY", "BSYNC", "JMP", "CALL", "RET"):
        fn.branches += 1
    elif base == "BAR":
        fn.barriers += 1
    elif base == "MEMBAR":
        fn.membars += 1
    elif base in ("SHFL", "WARPSYNC"):
        fn.shfl += 1


def parse_sass(sass_path: Path) -> SassStats:
    stats = SassStats()
    current: SassFunctionStats | None = None

    for raw in sass_path.read_text().splitlines():
        # Skip the second hex line each instruction has (its encoding bytes).
        # Those start with /*0xHEX*/ but have no mnemonic.
        # Function header
        m = _FN_RE.search(raw)
        if m:
            current = SassFunctionStats(name=m.group(1))
            stats.functions.append(current)
            continue

        if current is None:
            continue

        # Strip /*OFFSET*/ prefix
        s = _PREFIX_RE.sub("", raw).strip()
        if not s:
            continue
        # Skip the second-line hex encoding: looks like `0x000fc400078e00ff */`
        if s.startswith("/*0x") or s.startswith("0x"):
            continue
        # Strip predicate guard
        s = _GUARD_RE.sub("", s)
        m = _MNEMONIC_RE.match(s)
        if not m:
            continue
        mnemonic = m.group(1)
        # Filter junk: instructions must contain an uppercase letter
        if not any(c.isalpha() for c in mnemonic):
            continue
        # Skip pure label/data lines
        if mnemonic in ("EF_CUDA_TEXMODE_UNIFIED", ".headerflags"):
            continue
        _classify_and_count(current, mnemonic)

    return stats
