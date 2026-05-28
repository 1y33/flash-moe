"""FlashMoE PTX/SASS analyzer.

Usage:
    # Analyze one model (compile + parse + report)
    python tools/dump_ptx.py analyze --model qwen3-30b-a3b
    python tools/dump_ptx.py analyze --hidden 4096 --inter 14336 \
        --experts 8 --topk 2

    # Compare multiple registered models
    python tools/dump_ptx.py compare --models qwen3-30b-a3b,olmoe-1b-7b

Outputs (per model) go under ptx_output/<tag>/:
    kernel.ptx, kernel.cubin, kernel.sass
    ptxas_ptx_verbose.txt, ptxas_cubin_verbose.txt
    analysis.md     <-- main human-readable report
    plots/instruction_mix.png
    plots/roofline.png
    plots/memory_widths.png
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarks.models import ModelRegistry, MoEModelConfig  # noqa: E402
from ptx_analyzer import (  # noqa: E402
    get_arch,
    compile_ptx, compile_cubin, dump_sass,
    parse_ptx, parse_ptxas_verbose, parse_sass,
    analyze_kernel, analyze_occupancy, WorkloadRoofline,
    plot_instruction_mix, plot_roofline, plot_memory_widths,
    render_markdown, ARCH_DB,
)


REPO = Path(__file__).resolve().parent.parent


def _resolve_model(args) -> tuple[str, int, int, int, int]:
    if args.model:
        cfg = ModelRegistry.by_name(args.model)
        return cfg.name, cfg.hidden_size, cfg.intermediate_size, cfg.num_experts, cfg.top_k
    H = args.hidden; I = args.inter; E = args.experts; K = args.topk
    return f"H{H}_I{I}_E{E}_K{K}", H, I, E, K


def analyze_one(tag: str, H: int, I: int, E: int, K: int,
                arch: str, dtype: str,
                threads_per_block: int,
                out_root: Path,
                skip_compile: bool = False) -> Path:
    arch_spec = get_arch(arch)
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    defines = {
        "HIDDEN_SIZE_OVERRIDE": H,
        "MOE_INTERMEDIATE_SIZE_OVERRIDE": I,
        "NUM_EXPERTS_OVERRIDE": E,
        "TOP_K_OVERRIDE": K,
    }
    if dtype == "fp32":
        defines["MODEL_TYPE_FP32"] = 1

    config = dict(model=tag, source="tests/test_kernel.cu",
                  hidden=H, intermediate=I, experts=E, topk=K,
                  arch=arch, dtype=dtype, threads_per_block=threads_per_block)

    src = REPO / "tests" / "test_kernel.cu"
    if not skip_compile:
        print(f"[compile] PTX  → {out_dir / 'kernel.ptx'}")
        ptx_path, _ = compile_ptx(out_dir, src, arch, defines, REPO)
        print(f"[compile] cubin → {out_dir / 'kernel.cubin'}")
        cubin_path, _ = compile_cubin(out_dir, src, arch, defines, REPO)
        if cubin_path:
            print(f"[dump-sass] → {out_dir / 'kernel.sass'}")
            dump_sass(cubin_path, out_dir / "kernel.sass")
        else:
            print("[warn] cubin compile failed; SASS unavailable.")
    else:
        ptx_path = out_dir / "kernel.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"--skip-compile but {ptx_path} missing")

    # Parse
    ptx_stats = parse_ptx(out_dir / "kernel.ptx")
    sass_path = out_dir / "kernel.sass"
    sass_stats = parse_sass(sass_path) if sass_path.exists() else None

    cubin_verbose_path = out_dir / "ptxas_cubin_verbose.txt"
    ptxas_info: dict = {}
    if cubin_verbose_path.exists():
        ptxas_info = parse_ptxas_verbose(cubin_verbose_path.read_text())

    # Occupancy per kernel (one entry per "Used N registers" line)
    occupancies = []
    smem_list = ptxas_info.get("smem", []) or [0]
    for i, regs in enumerate(ptxas_info.get("registers", [])):
        smem = smem_list[i] if i < len(smem_list) else smem_list[0]
        occupancies.append(analyze_occupancy(regs, threads_per_block, smem, arch_spec))

    # Roofline — pick the first SASS function (typically the fp16 path)
    roofline = None
    if sass_stats and sass_stats.functions:
        roofline = analyze_kernel(sass_stats.functions[0], arch_spec)
    workload = WorkloadRoofline(H=H, I=I, K=K, batch=1)

    # Plots
    plot_paths = {}
    if sass_stats and sass_stats.functions:
        fn = sass_stats.functions[0]
        p = plots_dir / "instruction_mix.png"
        plot_instruction_mix(fn, p, f"SASS instruction mix — {tag}")
        plot_paths["Instruction mix (SASS)"] = "plots/instruction_mix.png"

        p = plots_dir / "memory_widths.png"
        plot_memory_widths(fn, p, f"Memory access widths — {tag}")
        plot_paths["Memory access widths"] = "plots/memory_widths.png"

    if roofline:
        p = plots_dir / "roofline.png"
        plot_roofline(roofline, workload, arch_spec, p,
                      f"Roofline — {tag} on {arch_spec.typical_gpu_name}")
        plot_paths["Roofline"] = "plots/roofline.png"

    md = render_markdown(config, arch_spec, ptx_stats, sass_stats,
                         occupancies, ptxas_info, roofline, workload, plot_paths)
    md_path = out_dir / "analysis.md"
    md_path.write_text(md)
    print(f"[done] {md_path}")
    return md_path


def compare(models: list[str], arch: str, dtype: str,
            threads_per_block: int, out_root: Path,
            skip_compile: bool = False) -> Path:
    """Run analyze_one for each model, then build a side-by-side compare.md."""
    summaries = []
    for name in models:
        cfg = ModelRegistry.by_name(name)
        analyze_one(cfg.name, cfg.hidden_size, cfg.intermediate_size,
                    cfg.num_experts, cfg.top_k, arch, dtype,
                    threads_per_block, out_root, skip_compile=skip_compile)

        # Re-load the parsed data for the table
        out_dir = out_root / cfg.name
        ptx = parse_ptx(out_dir / "kernel.ptx")
        sass_path = out_dir / "kernel.sass"
        sass = parse_sass(sass_path) if sass_path.exists() else None
        cubin_verbose = out_dir / "ptxas_cubin_verbose.txt"
        ptxas_info = parse_ptxas_verbose(cubin_verbose.read_text()) if cubin_verbose.exists() else {}

        regs = (ptxas_info.get("registers") or [0])[0]
        smem = (ptxas_info.get("smem") or [0])[0]
        spill = sum(ptxas_info.get("spill stores", []) + ptxas_info.get("spill loads", []))
        arch_spec = get_arch(arch)
        occ = analyze_occupancy(regs, threads_per_block, smem, arch_spec)
        rl = analyze_kernel(sass.functions[0], arch_spec) if sass else None
        wl = WorkloadRoofline(cfg.hidden_size, cfg.intermediate_size, cfg.top_k, 1)

        summaries.append({
            "name": cfg.name,
            "H": cfg.hidden_size,
            "I": cfg.intermediate_size,
            "E": cfg.num_experts,
            "K": cfg.top_k,
            "regs": regs,
            "smem": smem,
            "spill_bytes": spill,
            "occupancy_pct": occ.theoretical_occupancy * 100,
            "ai_static": rl.arithmetic_intensity if rl else 0.0,
            "ai_workload_b1": wl.arithmetic_intensity_fp16,
            "ldg_128_pct": (rl.bytes_loaded and (
                sass.functions[0].ldg_widths.get(128, 0) * 16
                / max(1, rl.bytes_loaded) * 100)) if (rl and sass) else 0,
            "ffma": sass.functions[0].ffma if sass else 0,
            "hmma": sass.functions[0].hmma if sass else 0,
            "total_insns": sass.functions[0].total_insns if sass else 0,
        })

    # Build markdown table
    md = [f"# FlashMoE kernel comparison — {arch} / {dtype}\n"]
    md.append("| metric | " + " | ".join(s["name"] for s in summaries) + " |")
    md.append("|" + "---|" * (len(summaries) + 1))
    keys = [
        ("H, I, E, K", lambda s: f"{s['H']}, {s['I']}, {s['E']}, {s['K']}"),
        ("regs/thread", lambda s: s['regs']),
        ("smem bytes", lambda s: s['smem']),
        ("spill bytes", lambda s: s['spill_bytes']),
        ("theoretical occupancy", lambda s: f"{s['occupancy_pct']:.1f}%"),
        ("AI (static, SASS)", lambda s: f"{s['ai_static']:.3f}"),
        ("AI (workload, B=1)", lambda s: f"{s['ai_workload_b1']:.3f}"),
        ("LDG.128 share of bytes", lambda s: f"{s['ldg_128_pct']:.1f}%"),
        ("FFMA (fp32 FMA)", lambda s: s['ffma']),
        ("HMMA (tensor)", lambda s: s['hmma']),
        ("total SASS insns", lambda s: s['total_insns']),
    ]
    for label, fn in keys:
        md.append("| " + label + " | "
                  + " | ".join(str(fn(s)) for s in summaries) + " |")

    out_path = out_root / f"compare_{'_'.join(models)}.md"
    out_path.write_text("\n".join(md))
    print(f"[done] {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_a = sub.add_parser("analyze", help="Analyze one (model | dims)")
    ap_a.add_argument("--model", help="registered model name (overrides dims)")
    ap_a.add_argument("--hidden", type=int, default=2048)
    ap_a.add_argument("--inter", type=int, default=768)
    ap_a.add_argument("--experts", type=int, default=128)
    ap_a.add_argument("--topk", type=int, default=8)
    ap_a.add_argument("--arch", default="sm_89",
                      help=f"GPU arch (one of {sorted(ARCH_DB.keys())})")
    ap_a.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap_a.add_argument("--tpb", type=int, default=128,
                      help="Threads per block assumed in occupancy analysis")
    ap_a.add_argument("--out", default="ptx_output")
    ap_a.add_argument("--skip-compile", action="store_true",
                      help="Skip nvcc, reuse existing PTX/SASS")

    ap_c = sub.add_parser("compare", help="Compare multiple registered models")
    ap_c.add_argument("--models", required=True,
                      help="comma-separated registered model names")
    ap_c.add_argument("--arch", default="sm_89")
    ap_c.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap_c.add_argument("--tpb", type=int, default=128)
    ap_c.add_argument("--out", default="ptx_output")
    ap_c.add_argument("--skip-compile", action="store_true")

    args = ap.parse_args()

    out_root = Path(args.out)

    if args.cmd == "analyze":
        tag, H, I, E, K = _resolve_model(args)
        analyze_one(tag, H, I, E, K, args.arch, args.dtype,
                    args.tpb, out_root, skip_compile=args.skip_compile)
    elif args.cmd == "compare":
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        compare(models, args.arch, args.dtype,
                args.tpb, out_root, skip_compile=args.skip_compile)

    return 0


if __name__ == "__main__":
    sys.exit(main())
