"""Generate markdown report + plots from a sweep results directory.

Usage:
    python -m benchmarks.report benchmarks/results/<run_id>/
"""

from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .workloads import BenchmarkResult
from .models import ModelRegistry


def load_results(run_dir: Path) -> tuple[dict, list[BenchmarkResult]]:
    metadata = json.loads((run_dir / "metadata.json").read_text())
    raw = json.loads((run_dir / "results.json").read_text())
    results = [BenchmarkResult.model_validate(r) for r in raw]
    return metadata, results


def group_by(results, *keys):
    out = defaultdict(list)
    for r in results:
        key = tuple(getattr(r, k) for k in keys)
        out[key].append(r)
    return out


def plot_latency_vs_batch(results, model_name, out_path):
    by_runner = group_by([r for r in results if r.model_name == model_name],
                          "runner")
    if not by_runner:
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    for (runner,), rs in sorted(by_runner.items()):
        rs = sorted(rs, key=lambda r: r.batch_size)
        xs = [r.batch_size for r in rs if r.error is None]
        ys = [r.mean_ms for r in rs if r.error is None]
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", label=runner, linewidth=2)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Latency per forward (ms)")
    ax.set_title(f"{model_name} — Latency vs Batch")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def plot_speedup_vs_batch(results, model_name, baseline_runner, out_path):
    """Plot speedup of every runner over baseline_runner for this model."""
    rs = [r for r in results if r.model_name == model_name and r.error is None]
    by_batch = group_by(rs, "batch_size")

    speedups: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for (batch,), batch_rs in by_batch.items():
        base = next((r for r in batch_rs if r.runner == baseline_runner), None)
        if base is None:
            continue
        for r in batch_rs:
            if r.runner == baseline_runner:
                continue
            if r.mean_ms > 0:
                speedups[r.runner].append((batch, base.mean_ms / r.mean_ms))

    if not speedups:
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    for runner, points in speedups.items():
        points.sort()
        xs, ys = zip(*points)
        ax.plot(xs, ys, marker="s", label=f"{runner} vs {baseline_runner}",
                linewidth=2)
    ax.axhline(1.0, color="grey", linestyle="--", alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch size")
    ax.set_ylabel(f"Speedup over {baseline_runner}")
    ax.set_title(f"{model_name} — Speedup vs {baseline_runner}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def markdown_table(results) -> str:
    """One row per (model, batch, runner). Sorted by model then batch."""
    lines = ["| model | batch | dtype | runner | mean ms | p50 | p99 | GB/s | note |",
             "|---|---|---|---|---|---|---|---|---|"]
    rs = sorted(results, key=lambda r: (r.model_name, r.batch_size, r.runner))
    for r in rs:
        if r.error:
            note = r.error
            mean = "—"
            p50 = p99 = gbps = "—"
        else:
            note = r.notes or ""
            mean = f"{r.mean_ms:.4f}"
            p50 = f"{r.p50_ms:.3f}" if r.p50_ms else "—"
            p99 = f"{r.p99_ms:.3f}" if r.p99_ms else "—"
            gbps = f"{r.achieved_gbps:.1f}" if r.achieved_gbps else "—"
        lines.append(
            f"| {r.model_name} | {r.batch_size} | {r.dtype} | {r.runner} | "
            f"{mean} | {p50} | {p99} | {gbps} | {note} |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()

    metadata, results = load_results(args.run_dir)

    plots_dir = args.run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    model_names = sorted(set(r.model_name for r in results))
    plot_files = []
    for name in model_names:
        latency_path = plots_dir / f"latency_{name}.png"
        if plot_latency_vs_batch(results, name, latency_path):
            plot_files.append(latency_path)

        # Only plot speedup if vLLM ran for this model
        speedup_path = plots_dir / f"speedup_{name}_vs_vllm.png"
        if plot_speedup_vs_batch(results, name, "vllm", speedup_path):
            plot_files.append(speedup_path)

    report = []
    report.append(f"# FlashMoE Benchmark Report — {metadata['run_id']}\n")
    report.append(f"**Device:** {metadata['device']}  ")
    report.append(f"**Run timestamp:** {metadata['timestamp']}  ")
    report.append(f"**Models:** {', '.join(metadata['models'])}  ")
    report.append(f"**Batches:** {metadata['batches']}  ")
    report.append(f"**Dtypes:** {metadata['dtypes']}  ")
    report.append(f"**Runners:** {', '.join(metadata['runners'])}\n")
    report.append("## Results\n")
    report.append(markdown_table(results))
    report.append("\n## Plots\n")
    for p in plot_files:
        rel = p.relative_to(args.run_dir)
        report.append(f"![{p.stem}]({rel})\n")

    out = args.run_dir / "report.md"
    out.write_text("\n".join(report))
    print(f"Wrote {out}")
    print(f"Plots: {len(plot_files)} in {plots_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
