"""Build the per-model analysis.md + plots."""

from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .ptx_parser import PtxStats
from .sass_parser import SassStats, SassFunctionStats
from .archs import ArchSpec
from .roofline import RooflineResult, WorkloadRoofline
from .instruction_mix import categorize, CATEGORIES
from .occupancy import OccupancyResult


_CATEGORY_COLORS = {
    "Memory (global)":      "#1f77b4",
    "Memory (shared)":      "#aec7e8",
    "Memory (local/spill)": "#d62728",
    "Memory (atomic)":      "#ff9896",
    "Compute (FP)":         "#2ca02c",
    "Compute (INT)":        "#98df8a",
    "Compute (Tensor)":     "#9467bd",
    "Control flow":         "#ff7f0e",
    "Synchronization":      "#bcbd22",
    "Other":                "#7f7f7f",
}


def plot_instruction_mix(fn: SassFunctionStats, out_path: Path,
                          title: str | None = None) -> None:
    mix = categorize(fn)
    labels = [c for c in CATEGORIES if mix.counts.get(c, 0) > 0]
    sizes = [mix.counts[c] for c in labels]
    colors = [_CATEGORY_COLORS[c] for c in labels]

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, _, autotxt = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, wedgeprops=dict(linewidth=0.5, edgecolor="white"),
    )
    for t in autotxt:
        t.set_fontsize(9); t.set_color("white"); t.set_weight("bold")
    ax.set_title(title or "SASS instruction mix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_roofline(rl: RooflineResult, wl: WorkloadRoofline | None,
                  arch: ArchSpec, out_path: Path, title: str | None = None) -> None:
    import numpy as np
    fig, ax = plt.subplots(figsize=(9, 6))

    ai_range = np.logspace(-2, 3, 400)
    bw_line = arch.dram_bw_gbps * 1e9 * ai_range / 1e12
    cuda_peak = np.full_like(ai_range, arch.fp16_tflops)
    roof_cuda = np.minimum(bw_line, cuda_peak)
    ax.loglog(ai_range, roof_cuda, color="#0f3460", linewidth=2,
              label=f"CUDA-core roof ({arch.fp16_tflops:.0f} TFLOPS, {arch.dram_bw_gbps:.0f} GB/s)")

    if arch.tc_fp16_tflops:
        tc_peak = np.full_like(ai_range, arch.tc_fp16_tflops)
        roof_tc = np.minimum(bw_line, tc_peak)
        ax.loglog(ai_range, roof_tc, color="#9467bd", linewidth=2, linestyle="--",
                  label=f"Tensor-core roof ({arch.tc_fp16_tflops:.0f} TFLOPS)")

    # Our point: from static SASS analysis
    if rl.arithmetic_intensity > 0:
        ax.scatter([rl.arithmetic_intensity], [rl.bandwidth_bound_tflops],
                   color="#e94560", s=160, marker="*", zorder=5,
                   label=f"FlashMoE static AI = {rl.arithmetic_intensity:.3f}")

    # Workload "ideal" point (analytical, derived from H/I/K)
    if wl is not None:
        ideal_ai = wl.arithmetic_intensity_fp16
        ideal_perf = arch.dram_bw_gbps * 1e9 * ideal_ai / 1e12
        ideal_perf = min(ideal_perf, arch.fp16_tflops)
        ax.scatter([ideal_ai], [ideal_perf], color="#2ca02c", s=140, marker="^",
                   zorder=5, label=f"Workload AI = {ideal_ai:.3f} (B=1)")

    ax.axvline(arch.ridge_point_fp16, color="#0f3460", alpha=0.3, linestyle=":")
    ax.text(arch.ridge_point_fp16, arch.fp16_tflops * 1.2,
            f"ridge\n{arch.ridge_point_fp16:.1f}", color="#0f3460",
            ha="center", fontsize=8)

    ax.set_xlabel("Arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("Throughput (TFLOPS)")
    ax.set_title(title or f"Roofline — {arch.typical_gpu_name} ({arch.sm_arch})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_memory_widths(fn: SassFunctionStats, out_path: Path,
                        title: str | None = None) -> None:
    """Stacked bar of memory access widths (global vs shared vs local)."""
    widths = sorted({*fn.ldg_widths, *fn.stg_widths,
                     *fn.lds_widths, *fn.sts_widths,
                     *fn.ldl_widths, *fn.stl_widths})
    if not widths:
        return
    x_positions = list(range(len(widths)))

    fig, ax = plt.subplots(figsize=(8, 5))
    categories = [
        ("LDG (global load)",  fn.ldg_widths, "#1f77b4"),
        ("STG (global store)", fn.stg_widths, "#aec7e8"),
        ("LDS (shared load)",  fn.lds_widths, "#2ca02c"),
        ("STS (shared store)", fn.sts_widths, "#98df8a"),
        ("LDL (local load — spill)",  fn.ldl_widths, "#d62728"),
        ("STL (local store — spill)", fn.stl_widths, "#ff9896"),
    ]
    bottom = [0] * len(widths)
    for label, src, color in categories:
        vals = [src.get(w, 0) for w in widths]
        if sum(vals) == 0:
            continue
        ax.bar(x_positions, vals, label=label, bottom=bottom, color=color,
               edgecolor="black", linewidth=0.5)
        bottom = [b + v for b, v in zip(bottom, vals)]

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{w}-bit" for w in widths])
    ax.set_xlabel("Access width")
    ax.set_ylabel("Instruction count")
    ax.set_title(title or "Memory access widths (SASS)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ── Markdown report ──────────────────────────────────────────────────


def _table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def render_markdown(
    config: dict,
    arch: ArchSpec,
    ptx: PtxStats,
    sass: SassStats | None,
    occupancies: list[OccupancyResult],
    ptxas_info: dict,
    roofline: RooflineResult | None,
    workload: WorkloadRoofline | None,
    plot_paths: dict[str, str],
) -> str:
    lines: list[str] = []
    lines.append(f"# Kernel analysis — {config.get('model', 'unknown')}")
    lines.append("")
    lines.append(f"**GPU target:** {arch.typical_gpu_name} ({arch.sm_arch}, {arch.name})")
    lines.append(f"**Source:** `{config.get('source', '?')}`")
    lines.append("")
    lines.append("## Configuration\n")
    cfg_rows = [[k, v] for k, v in config.items()]
    lines.append(_table(["key", "value"], cfg_rows))
    lines.append("")

    # ── ptxas verbose ─────────────────────────────────────────
    lines.append("## Resource usage (`ptxas -v`)\n")
    if ptxas_info:
        rows = []
        for k, v in ptxas_info.items():
            rows.append([k, v if isinstance(v, (int, str)) else ", ".join(map(str, v))])
        lines.append(_table(["resource", "value"], rows))
    else:
        lines.append("(no ptxas info parsed)")
    lines.append("")

    # ── Occupancy ──────────────────────────────────────────────
    if occupancies:
        lines.append("## Occupancy analysis\n")
        lines.append(f"_Block size assumed: {occupancies[0].threads_per_block} threads_\n")
        rows = []
        for i, occ in enumerate(occupancies):
            rows.append([
                f"kernel {i}",
                occ.regs_per_thread,
                occ.max_threads_per_sm_by_regs,
                occ.max_threads_per_sm_by_smem,
                occ.effective_max_threads_per_sm,
                occ.max_blocks_per_sm,
                f"{occ.theoretical_occupancy * 100:.1f}%",
                "⚠" if occ.spill_risk else "✓",
            ])
        lines.append(_table(
            ["", "regs/thread", "max_thr/SM (regs)", "max_thr/SM (smem)",
             "effective max_thr/SM", "blocks/SM", "occupancy", "spill risk"],
            rows))
        lines.append("")

    # ── PTX top-level ──────────────────────────────────────────
    lines.append("## PTX instruction mix\n")
    k = ptx.aggregate
    lines.append(_table(
        ["metric", "value"],
        [
            ["PTX size (bytes)", f"{ptx.ptx_bytes:,}"],
            ["entries", ", ".join(ptx.entries) if ptx.entries else "?"],
            ["total instructions (PTX)", f"{k.instructions:,}"],
            ["inline-asm blocks", k.inline_asm_blocks],
            ["loads", k.loads],
            ["stores", k.stores],
            ["atomics", k.atomics],
            ["fma total (fp16/fp32/fp64)",
             f"{k.fma_total} ({k.fma_f16} / {k.fma_f32} / {k.fma_f64})"],
            ["branches", k.branches],
            ["bar.sync", k.bar_sync],
            ["membar / fence", k.fences],
            ["shfl", k.shfl],
        ],
    ))
    lines.append("")
    lines.append("### PTX virtual register declarations\n")
    rows = [[kind, n] for kind, n in ptx.register_decls.items()]
    lines.append(_table(["type", "count"], rows))
    lines.append("")

    # ── SASS ──────────────────────────────────────────────────
    if sass and sass.functions:
        for fn in sass.functions:
            short = fn.name.split("(")[0].split("<")[0]
            lines.append(f"## SASS — `{short}`\n")
            lines.append(_table(
                ["metric", "value"],
                [
                    ["total SASS instructions", f"{fn.total_insns:,}"],
                    ["FFMA (fp32 FMA)", fn.ffma],
                    ["HFMA2 (fp16x2 FMA)", fn.hfma2],
                    ["HMMA / IMMA (tensor cores)", fn.hmma + fn.imma],
                    ["IMAD (int multiply-add)", fn.imad],
                    ["IADD3 (int add)", fn.iadd3],
                    ["global loads (LDG)", sum(fn.ldg_widths.values())],
                    ["global stores (STG)", sum(fn.stg_widths.values())],
                    ["shared loads (LDS)", sum(fn.lds_widths.values())],
                    ["shared stores (STS)", sum(fn.sts_widths.values())],
                    ["local loads (LDL, spill)", sum(fn.ldl_widths.values())],
                    ["local stores (STL, spill)", sum(fn.stl_widths.values())],
                    ["atomics", fn.atomics],
                    ["branches", fn.branches],
                    ["BAR.SYNC", fn.barriers],
                    ["MEMBAR", fn.membars],
                    ["SHFL", fn.shfl],
                ],
            ))
            lines.append("")
            lines.append("### Memory access widths\n")
            ldg = dict(fn.ldg_widths); stg = dict(fn.stg_widths)
            lds = dict(fn.lds_widths); sts = dict(fn.sts_widths)
            ldl = dict(fn.ldl_widths); stl = dict(fn.stl_widths)
            widths = sorted({*ldg, *stg, *lds, *sts, *ldl, *stl})
            rows = []
            for w in widths:
                rows.append([f"{w}-bit",
                             ldg.get(w, 0), stg.get(w, 0),
                             lds.get(w, 0), sts.get(w, 0),
                             ldl.get(w, 0), stl.get(w, 0)])
            lines.append(_table(
                ["width", "LDG", "STG", "LDS", "STS", "LDL", "STL"], rows))
            lines.append("")

    # ── Roofline ───────────────────────────────────────────────
    if roofline:
        lines.append("## Arithmetic intensity & roofline\n")
        rows = [
            ["FLOPs (from SASS FFMA+HFMA2+...)", f"{roofline.flops:,}"],
            ["Bytes loaded (LDG total)", f"{roofline.bytes_loaded:,}"],
            ["Bytes stored (STG total)", f"{roofline.bytes_stored:,}"],
            ["Static arithmetic intensity",
             f"{roofline.arithmetic_intensity:.3f} FLOP/byte"],
            ["Ridge point (CUDA-core fp16)",
             f"{roofline.ridge_point_fp16:.1f} FLOP/byte"],
            ["Memory-bound (vs CUDA-core)?",
             "yes" if roofline.is_memory_bound else "no"],
            ["DRAM-bound ceiling at this AI",
             f"{roofline.bandwidth_bound_tflops:.3f} TFLOPS"],
        ]
        if roofline.ridge_point_tc:
            rows.append(["Ridge point (tensor core)",
                         f"{roofline.ridge_point_tc:.1f} FLOP/byte"])
            rows.append(["Memory-bound (vs tensor-core)?",
                         "yes" if roofline.is_memory_bound_vs_tc else "no"])
        lines.append(_table(["metric", "value"], rows))
        lines.append("")
        if workload:
            lines.append("### Workload-derived AI (analytical, from H/I/K)\n")
            lines.append(_table(["batch", "AI (FLOP/byte)"], [
                [b, f"{workload.amortized_intensity_fp16(b):.3f}"]
                for b in (1, 2, 4, 8, 16, 32, 64, 128)
            ]))
            lines.append("")

    # ── Plots ─────────────────────────────────────────────────
    if plot_paths:
        lines.append("## Plots\n")
        for name, p in plot_paths.items():
            lines.append(f"### {name}\n")
            lines.append(f"![{name}]({p})\n")

    return "\n".join(lines)
