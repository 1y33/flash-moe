"""Compile the FlashMoE kernel and dump PTX + analysis.

Usage:
    python tools/dump_ptx.py                              # default config
    python tools/dump_ptx.py --model mixtral-8x7b
    python tools/dump_ptx.py --hidden 2048 --inter 768 \
                            --experts 128 --topk 8
    python tools/dump_ptx.py --keep-files                 # don't clean .ptx

Writes ./ptx_output/<config>/ with:
    kernel.ptx          — full PTX
    kernel.sass         — disassembled SASS (if cuobjdump available)
    summary.txt         — instruction histogram, register usage, memory ops
"""

from __future__ import annotations
import argparse
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.models import ModelRegistry  # noqa: E402


REPO = Path(__file__).resolve().parent.parent


def compile_ptx(out_dir: Path, hidden: int, inter: int, experts: int, topk: int,
                arch: str, dtype: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ptx_path = out_dir / "kernel.ptx"
    cmd = [
        "nvcc",
        f"-arch={arch}",
        "-O3",
        f"-DHIDDEN_SIZE_OVERRIDE={hidden}",
        f"-DMOE_INTERMEDIATE_SIZE_OVERRIDE={inter}",
        f"-DNUM_EXPERTS_OVERRIDE={experts}",
        f"-DTOP_K_OVERRIDE={topk}",
        "-ptx",
        "--ptxas-options=-v",   # also print register/smem usage to stderr
        str(REPO / "tests" / "test_kernel.cu"),
        "-o", str(ptx_path),
    ]
    if dtype == "fp32":
        cmd.append("-DMODEL_TYPE_FP32")

    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit(f"nvcc -ptx failed with code {result.returncode}")
    # ptxas verbose info goes to stderr
    (out_dir / "ptxas_verbose.txt").write_text(result.stderr)
    return ptx_path


def compile_sass(out_dir: Path, hidden: int, inter: int, experts: int, topk: int,
                 arch: str, dtype: str) -> Path | None:
    elf = out_dir / "kernel.cubin"
    cmd = [
        "nvcc",
        f"-arch={arch}",
        "-O3",
        f"-DHIDDEN_SIZE_OVERRIDE={hidden}",
        f"-DMOE_INTERMEDIATE_SIZE_OVERRIDE={inter}",
        f"-DNUM_EXPERTS_OVERRIDE={experts}",
        f"-DTOP_K_OVERRIDE={topk}",
        "--cubin",
        "--ptxas-options=-v",
        str(REPO / "tests" / "test_kernel.cu"),
        "-o", str(elf),
    ]
    if dtype == "fp32":
        cmd.append("-DMODEL_TYPE_FP32")
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if result.returncode != 0:
        print("cubin compile failed — skipping SASS dump")
        print(result.stderr)
        return None
    (out_dir / "ptxas_cubin_verbose.txt").write_text(result.stderr)

    sass_path = out_dir / "kernel.sass"
    with open(sass_path, "w") as f:
        try:
            subprocess.run(["cuobjdump", "--dump-sass", str(elf)],
                           stdout=f, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("cuobjdump unavailable — skipping SASS dump")
            return None
    return sass_path


# PTX opcode parsing —
# An instruction line looks like (tab-indented):
#   ld.global.v4.u32  {%r152, %r153, %r154, %r155}, [%rd15];
#   @%p83 bra $L__BB0_7;
#   add.s64 %rd60, %rd13, %rd59;
# We want to capture the whole opcode (everything up to first whitespace), after
# stripping a leading guard predicate `@%pXX` or `@!%pXX`.
_OPCODE_RE = re.compile(r"^(@!?%\w+\s+)?([a-z][a-z0-9_.]+)")


def is_instruction_line(s: str) -> bool:
    """Heuristic: instruction lines are tab-indented and contain a semicolon
    or end with a label/operand, but exclude directives (start with '.'),
    labels (end with ':'), block braces, and comments."""
    if not s or s.startswith("//"):
        return False
    stripped = s.strip()
    if not stripped:
        return False
    if stripped.startswith("//"):
        return False
    if stripped.startswith("."):       # directive
        return False
    if stripped.startswith("$") or stripped.endswith(":"):  # label
        return False
    if stripped in ("{", "}"):
        return False
    return True


def family(op: str) -> str:
    """Coarse opcode family: e.g. ld.global.v4.u32 -> ld.global"""
    parts = op.split(".")
    if not parts:
        return op
    base = parts[0]
    if base in ("ld", "st", "atom", "red", "cp"):
        if len(parts) >= 2:
            return f"{base}.{parts[1]}"
    elif base in ("mov", "cvt", "selp", "setp", "fma", "mad", "add", "sub",
                  "mul", "div", "min", "max", "abs", "neg", "and", "or",
                  "xor", "shl", "shr", "shf", "rem"):
        return base
    elif base in ("bar", "membar", "fence"):
        return base
    elif base in ("bra", "call", "ret", "exit"):
        return base
    elif base in ("shfl", "vote", "match", "activemask"):
        return base
    return base


def analyze_ptx(ptx_path: Path) -> dict:
    text = ptx_path.read_text()
    lines = text.splitlines()

    # Find every entry kernel
    kernels = re.findall(r"\.visible\s+\.entry\s+(\w+)", text)

    op_counter: Counter[str] = Counter()
    family_counter: Counter[str] = Counter()
    # Memory access detail
    ld_widths: Counter[str] = Counter()   # e.g. "v4.u32", "u16", "v2.b32"
    st_widths: Counter[str] = Counter()
    atomic_kinds: Counter[str] = Counter()  # e.g. "atom.global.add.f32"

    # Math op counters
    fma_count = 0
    fma_f16_count = 0
    fma_f32_count = 0
    mad_count = 0
    add_count = 0
    mul_count = 0

    # Control flow
    branches = 0
    barriers = 0
    fences = 0
    shfl_count = 0

    # Inline asm count (cvt.f32.f16 is wrapped in inline asm in CUDA fp16 path)
    inline_asm_blocks = 0

    inside_asm = False
    for line in lines:
        if "begin inline asm" in line:
            inside_asm = True
            inline_asm_blocks += 1
            continue
        if "end inline asm" in line:
            inside_asm = False
            continue
        if inside_asm:
            continue

        if not is_instruction_line(line):
            continue
        m = _OPCODE_RE.match(line.strip())
        if not m:
            continue
        op = m.group(2)
        op_counter[op] += 1
        fam = family(op)
        family_counter[fam] += 1

        if op.startswith("ld."):
            # ld.global.v4.u32 -> width = "v4.u32"
            rest = op.split(".", 1)[1] if "." in op else ""
            # drop the state space (global/shared/local)
            parts = rest.split(".")
            if parts and parts[0] in ("global", "shared", "local", "param",
                                       "const", "generic"):
                parts = parts[1:]
            ld_widths[".".join(parts)] += 1
        elif op.startswith("st."):
            rest = op.split(".", 1)[1] if "." in op else ""
            parts = rest.split(".")
            if parts and parts[0] in ("global", "shared", "local", "param",
                                       "generic"):
                parts = parts[1:]
            st_widths[".".join(parts)] += 1
        elif op.startswith("atom.") or op.startswith("red."):
            atomic_kinds[op] += 1

        if op.startswith("fma."):
            fma_count += 1
            if "f16" in op:
                fma_f16_count += 1
            elif "f32" in op:
                fma_f32_count += 1
        elif op.startswith("mad."):
            mad_count += 1
        elif op.startswith("add."):
            add_count += 1
        elif op.startswith("mul."):
            mul_count += 1
        if op.startswith("bra"):
            branches += 1
        if op.startswith("bar.") or op == "bar":
            barriers += 1
        if op.startswith("membar") or op.startswith("fence"):
            fences += 1
        if op.startswith("shfl."):
            shfl_count += 1

    # Register usage from .reg directives
    regs = {}
    for kind, count in re.findall(r"\.reg\s+\.(\w+)\s+%\w+<(\d+)>", text):
        regs[kind] = regs.get(kind, 0) + int(count)

    return {
        "ptx_bytes": ptx_path.stat().st_size,
        "kernels": kernels,
        "total_instructions": sum(op_counter.values()),
        "loads": sum(ld_widths.values()),
        "stores": sum(st_widths.values()),
        "atomics": sum(atomic_kinds.values()),
        "branches": branches,
        "barriers": barriers,
        "fences": fences,
        "shfl_ops": shfl_count,
        "fma_total": fma_count,
        "fma_f16": fma_f16_count,
        "fma_f32": fma_f32_count,
        "mad_total": mad_count,
        "add_total": add_count,
        "mul_total": mul_count,
        "inline_asm_blocks": inline_asm_blocks,
        "register_estimate": regs,
        "ld_widths": ld_widths.most_common(),
        "st_widths": st_widths.most_common(),
        "atomic_kinds": atomic_kinds.most_common(),
        "top20_opcodes": op_counter.most_common(20),
        "top20_families": family_counter.most_common(20),
    }


def parse_ptxas_verbose(text: str) -> dict:
    """Pull useful numbers from `ptxas -v` output.

    Sample lines:
        ptxas info    : Function properties for ...
        ptxas info    : Used 96 registers, 384 bytes smem, 376 bytes cmem[0]
    """
    out = {}
    for line in text.splitlines():
        m = re.search(r"Used (\d+) registers", line)
        if m:
            out.setdefault("registers", []).append(int(m.group(1)))
        m = re.search(r"(\d+) bytes (smem|gmem|cmem\[\d+\]|stack frame|spill stores|spill loads)", line)
        if m:
            n, kind = int(m.group(1)), m.group(2)
            out.setdefault(kind, []).append(n)
    return out


def write_summary(out_dir: Path, info: dict, ptxas: dict, config: dict) -> Path:
    p = out_dir / "summary.txt"
    with open(p, "w") as f:
        f.write("FlashMoE PTX Analysis\n")
        f.write("=" * 60 + "\n\n")
        f.write("Config:\n")
        for k, v in config.items():
            f.write(f"  {k:20s} = {v}\n")
        f.write("\nKernels found:\n")
        for k in info["kernels"]:
            f.write(f"  {k}\n")
        f.write(f"\nPTX size:                {info['ptx_bytes']:>10,} bytes\n")
        f.write(f"Total instructions:      {info['total_instructions']:>10,}\n")
        f.write(f"Inline-asm blocks:       {info['inline_asm_blocks']:>10,}\n")
        f.write(f"\nMemory:\n")
        f.write(f"  loads                  {info['loads']:>10,}\n")
        f.write(f"  stores                 {info['stores']:>10,}\n")
        f.write(f"  atomic / red           {info['atomics']:>10,}\n")
        f.write(f"\nMath:\n")
        f.write(f"  fma (total)            {info['fma_total']:>10,}\n")
        f.write(f"    fma.f16 / f16x2      {info['fma_f16']:>10,}\n")
        f.write(f"    fma.f32              {info['fma_f32']:>10,}\n")
        f.write(f"  mad                    {info['mad_total']:>10,}\n")
        f.write(f"  add                    {info['add_total']:>10,}\n")
        f.write(f"  mul                    {info['mul_total']:>10,}\n")
        f.write(f"\nControl & sync:\n")
        f.write(f"  branches               {info['branches']:>10,}\n")
        f.write(f"  bar.sync               {info['barriers']:>10,}\n")
        f.write(f"  membar / fence         {info['fences']:>10,}\n")
        f.write(f"  shfl                   {info['shfl_ops']:>10,}\n")

        if ptxas:
            f.write("\nptxas -v report:\n")
            for k, v in ptxas.items():
                f.write(f"  {k:20s} {v}\n")

        f.write("\nLoad widths (memory access patterns):\n")
        for w, n in info["ld_widths"]:
            f.write(f"  {w:30s} {n:8d}\n")
        f.write("\nStore widths:\n")
        for w, n in info["st_widths"]:
            f.write(f"  {w:30s} {n:8d}\n")
        if info["atomic_kinds"]:
            f.write("\nAtomic instructions:\n")
            for op, n in info["atomic_kinds"]:
                f.write(f"  {op:30s} {n:8d}\n")

        f.write("\nTop 20 PTX opcodes (full):\n")
        for op, n in info["top20_opcodes"]:
            f.write(f"  {op:30s} {n:8d}\n")
        f.write("\nTop 20 opcode families:\n")
        for fam, n in info["top20_families"]:
            f.write(f"  {fam:30s} {n:8d}\n")

        f.write("\nRegister counts (PTX virtuals, before SASS allocation):\n")
        for kind, n in info["register_estimate"].items():
            f.write(f"  .{kind:10s} {n}\n")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="registered model name")
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--inter", type=int, default=768)
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--arch", default="sm_89")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--out", default="ptx_output")
    ap.add_argument("--no-sass", action="store_true",
                    help="Skip the SASS dump step")
    args = ap.parse_args()

    if args.model:
        cfg = ModelRegistry.by_name(args.model)
        H, I, E, K = cfg.hidden_size, cfg.intermediate_size, cfg.num_experts, cfg.top_k
        tag = cfg.name
    else:
        H, I, E, K = args.hidden, args.inter, args.experts, args.topk
        tag = f"H{H}_I{I}_E{E}_K{K}"

    out_dir = Path(args.out) / tag
    config = dict(model=tag, hidden=H, intermediate=I, experts=E, topk=K,
                  arch=args.arch, dtype=args.dtype)

    ptx_path = compile_ptx(out_dir, H, I, E, K, args.arch, args.dtype)
    info = analyze_ptx(ptx_path)

    if not args.no_sass:
        compile_sass(out_dir, H, I, E, K, args.arch, args.dtype)

    ptxas_info = {}
    cubin_verbose = out_dir / "ptxas_cubin_verbose.txt"
    if cubin_verbose.exists():
        ptxas_info = parse_ptxas_verbose(cubin_verbose.read_text())

    summary = write_summary(out_dir, info, ptxas_info, config)
    print(f"\nPTX:     {ptx_path}")
    print(f"Summary: {summary}")
    print(f"\n--- summary.txt ---")
    print(summary.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
