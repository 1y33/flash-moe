"""Parse PTX source. Returns per-entry-kernel + global statistics.

PTX line shape (after tab indent):
    [@%pred]   opcode  operands... ;
    label:
    .directive
    { } block delimiters
    // comment
    inline asm wrapped in { ... }  inside `// begin inline asm` markers
"""

from __future__ import annotations
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


_OPCODE_RE = re.compile(r"^(@!?%\w+\s+)?([a-z][a-z0-9_.]+)")
_ENTRY_RE = re.compile(r"\.(?:visible\s+)?\.entry\s+(\w+)")
# Sample: .reg .f32 %f<4272>;
_REG_DECL_RE = re.compile(r"\.reg\s+\.(\w+)\s+%\w+<(\d+)>")


@dataclass
class PtxKernelStats:
    name: str
    instructions: int = 0
    opcode_counts: Counter = field(default_factory=Counter)
    family_counts: Counter = field(default_factory=Counter)

    # Memory
    ld_global_widths: Counter = field(default_factory=Counter)
    ld_shared_widths: Counter = field(default_factory=Counter)
    ld_local_widths: Counter = field(default_factory=Counter)
    st_global_widths: Counter = field(default_factory=Counter)
    st_shared_widths: Counter = field(default_factory=Counter)
    st_local_widths: Counter = field(default_factory=Counter)
    atomic_ops: Counter = field(default_factory=Counter)

    # Compute
    fma_f16: int = 0
    fma_f32: int = 0
    fma_f64: int = 0
    add_f16: int = 0
    add_f32: int = 0
    mul_f16: int = 0
    mul_f32: int = 0
    mad: int = 0
    cvt_count: int = 0

    # Control & sync
    branches: int = 0
    bar_sync: int = 0
    fences: int = 0
    shfl: int = 0

    # Other
    inline_asm_blocks: int = 0
    predicates: int = 0

    @property
    def loads(self) -> int:
        return (sum(self.ld_global_widths.values())
                + sum(self.ld_shared_widths.values())
                + sum(self.ld_local_widths.values()))

    @property
    def stores(self) -> int:
        return (sum(self.st_global_widths.values())
                + sum(self.st_shared_widths.values())
                + sum(self.st_local_widths.values()))

    @property
    def atomics(self) -> int:
        return sum(self.atomic_ops.values())

    @property
    def fma_total(self) -> int:
        return self.fma_f16 + self.fma_f32 + self.fma_f64


@dataclass
class PtxStats:
    entries: list[str] = field(default_factory=list)
    register_decls: dict[str, int] = field(default_factory=dict)
    ptx_bytes: int = 0
    # We attribute everything to a single "global" kernel because PTX flattens
    # __device__ functions into the entry kernel after inlining.
    aggregate: PtxKernelStats = field(default_factory=lambda: PtxKernelStats(name="(merged)"))


def _is_instruction_line(s: str) -> bool:
    stripped = s.strip()
    if not stripped or stripped.startswith("//") or stripped.startswith("."):
        return False
    if stripped.startswith("$") or stripped.endswith(":"):
        return False
    if stripped in ("{", "}"):
        return False
    return True


def _family(op: str) -> str:
    parts = op.split(".")
    if not parts:
        return op
    base = parts[0]
    if base in ("ld", "st", "atom", "red", "cp"):
        return f"{base}.{parts[1]}" if len(parts) >= 2 else base
    return base


def _vector_width_string(op: str, state_space_marker: str) -> str:
    """For `ld.global.v4.u32` -> `v4.u32`. For `ld.global.u32` -> `u32`."""
    rest = op.split(".", 1)[1] if "." in op else ""
    parts = rest.split(".")
    if parts and parts[0] in ("global", "shared", "local", "param",
                              "const", "generic"):
        parts = parts[1:]
    return ".".join(parts) if parts else "?"


def _bump_compute(stats: PtxKernelStats, op: str):
    if op.startswith("fma."):
        if ".f16" in op:
            stats.fma_f16 += 1
        elif ".f32" in op:
            stats.fma_f32 += 1
        elif ".f64" in op:
            stats.fma_f64 += 1
    elif op.startswith("add."):
        if ".f16" in op:
            stats.add_f16 += 1
        elif ".f32" in op:
            stats.add_f32 += 1
    elif op.startswith("mul."):
        if ".f16" in op:
            stats.mul_f16 += 1
        elif ".f32" in op:
            stats.mul_f32 += 1
    elif op.startswith("mad."):
        stats.mad += 1
    elif op.startswith("cvt."):
        stats.cvt_count += 1


def parse_ptx(ptx_path: Path) -> PtxStats:
    text = ptx_path.read_text()
    stats = PtxStats(ptx_bytes=len(text.encode()))
    stats.entries = re.findall(r"\.(?:visible\s+)?(?:weak\s+)?\.entry\s+(\w+)", text)
    if not stats.entries:
        stats.entries = re.findall(r"\.entry\s+(\w+)", text)

    for kind, n in _REG_DECL_RE.findall(text):
        stats.register_decls[kind] = stats.register_decls.get(kind, 0) + int(n)

    inside_asm = False
    k = stats.aggregate
    for line in text.splitlines():
        if "begin inline asm" in line:
            inside_asm = True
            k.inline_asm_blocks += 1
            continue
        if "end inline asm" in line:
            inside_asm = False
            continue
        if inside_asm:
            continue
        if not _is_instruction_line(line):
            continue

        stripped = line.strip()
        if stripped.startswith("@"):
            k.predicates += 1

        m = _OPCODE_RE.match(stripped)
        if not m:
            continue
        op = m.group(2)

        k.instructions += 1
        k.opcode_counts[op] += 1
        k.family_counts[_family(op)] += 1
        _bump_compute(k, op)

        if op.startswith("ld."):
            w = _vector_width_string(op, "ld")
            if ".global" in op:
                k.ld_global_widths[w] += 1
            elif ".shared" in op:
                k.ld_shared_widths[w] += 1
            elif ".local" in op:
                k.ld_local_widths[w] += 1
            else:
                k.ld_global_widths[w] += 1  # default param/etc
        elif op.startswith("st."):
            w = _vector_width_string(op, "st")
            if ".global" in op:
                k.st_global_widths[w] += 1
            elif ".shared" in op:
                k.st_shared_widths[w] += 1
            elif ".local" in op:
                k.st_local_widths[w] += 1
            else:
                k.st_global_widths[w] += 1
        elif op.startswith("atom.") or op.startswith("red."):
            k.atomic_ops[op] += 1

        if op.startswith("bra"):
            k.branches += 1
        if op.startswith("bar.") or op == "bar":
            k.bar_sync += 1
        if op.startswith("membar") or op.startswith("fence"):
            k.fences += 1
        if op.startswith("shfl."):
            k.shfl += 1

    return stats


def parse_ptxas_verbose(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        m = re.search(r"Used (\d+) registers", line)
        if m:
            out.setdefault("registers", []).append(int(m.group(1)))
        m = re.search(r"(\d+) bytes (smem|gmem|cmem\[\d+\]|stack frame|spill stores|spill loads)",
                      line)
        if m:
            n, kind = int(m.group(1)), m.group(2)
            out.setdefault(kind, []).append(n)
    return out


# Type-width in bytes for memory ops (PTX uses these tokens in the type suffix)
PTX_TYPE_BYTES = {
    "u8": 1, "s8": 1, "b8": 1,
    "u16": 2, "s16": 2, "b16": 2, "f16": 2, "bf16": 2,
    "u32": 4, "s32": 4, "b32": 4, "f32": 4,
    "u64": 8, "s64": 8, "b64": 8, "f64": 8,
    "u128": 16, "s128": 16, "b128": 16,
}


def width_to_bytes(width_str: str) -> int:
    """Decode 'v4.u32' / 'u16' / 'f32' to bytes per access."""
    if not width_str or width_str == "?":
        return 0
    parts = width_str.split(".")
    n = 1
    if parts[0].startswith("v"):
        try:
            n = int(parts[0][1:])
            parts = parts[1:]
        except ValueError:
            pass
    if not parts:
        return 0
    return n * PTX_TYPE_BYTES.get(parts[0], 0)
