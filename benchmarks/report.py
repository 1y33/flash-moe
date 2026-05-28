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


_RUNNER_COLORS = {
    "flashmoe":      "#e94560",   # red — ours
    "vllm":          "#0f3460",   # deep blue
    "pytorch-eager": "#9aa0a6",   # grey
}


def plot_latency_bars(results, model_name, out_path):
    """Grouped bar chart: x=batch size, bars per runner, y=latency (ms)."""
    rs = [r for r in results if r.model_name == model_name and r.error is None]
    if not rs:
        return False

    batches = sorted({r.batch_size for r in rs})
    runners = sorted({r.runner for r in rs})

    lookup = {(r.runner, r.batch_size): r for r in rs}

    n_runners = len(runners)
    bar_width = 0.8 / max(n_runners, 1)
    x_positions = list(range(len(batches)))

    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(batches)), 5))

    for i, runner in enumerate(runners):
        means = [lookup[(runner, b)].mean_ms if (runner, b) in lookup else 0.0
                 for b in batches]
        xs = [p + (i - (n_runners - 1) / 2) * bar_width for p in x_positions]
        color = _RUNNER_COLORS.get(runner, None)
        ax.bar(xs, means, bar_width, label=runner,
               color=color, edgecolor="black", linewidth=0.6)
        for x, y in zip(xs, means):
            if y > 0:
                ax.text(x, y, f"{y:.3f}", ha="center", va="bottom",
                        fontsize=9, fontweight="bold")

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"b={b}" for b in batches])
    ax.set_ylabel("Latency per forward (ms)")
    ax.set_title(f"{model_name} — Latency")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_axisbelow(True)
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
        if plot_latency_bars(results, name, latency_path):
            plot_files.append(latency_path)

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
