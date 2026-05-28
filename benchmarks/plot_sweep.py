"""Plot how FlashMoE scales as we vary one dim at a time.

Reads a sweep results.json and emits one PNG per varied axis (H, I, E, K).
Each plot shows mean_ms vs the swept dim, one line per runner. The baseline
Qwen3 point (H=2048 I=768 E=128 K=8) is included on every plot so each axis
is anchored.

Usage:
    python -m benchmarks.plot_sweep benchmarks/results/<run_id>/
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .models import ModelRegistry
from . import models  # noqa: F401 — register side effects


_RUNNER_COLORS = {
    "flashmoe":      "#e94560",
    "vllm":          "#0f3460",
    "pytorch-eager": "#9aa0a6",
}

# Baseline (Qwen3) anchor used on every per-axis plot
BASELINE = {"H": 2048, "I": 768, "E": 128, "K": 8}


def axis_value(model_name: str, axis: str) -> int | None:
    """Extract the swept value from a name like 'synth-H1024' or 'synth-EK128x16'."""
    if model_name == "qwen3-30b-a3b":
        return BASELINE[axis]
    if not model_name.startswith("synth-"):
        return None
    body = model_name[len("synth-"):]
    # Single-axis names: H1024, I256, E64, K2
    m = re.fullmatch(rf"{axis}(\d+)", body)
    if m:
        return int(m.group(1))
    # Cross-axis names like EK128x16 or IK1024x12: not single-axis, skip.
    return None


def collect_axis(results, axis):
    """{runner: [(dim_value, mean_ms), ...]} sorted by dim_value."""
    points = defaultdict(list)
    for r in results:
        if r.get("error"):
            continue
        v = axis_value(r["model_name"], axis)
        if v is None:
            continue
        points[r["runner"]].append((v, r["mean_ms"], r.get("p50_ms")))
    for runner in points:
        points[runner].sort()
    return points


def plot_axis(results, axis, out_path):
    pts = collect_axis(results, axis)
    if not pts:
        return False
    fig, ax = plt.subplots(figsize=(8, 5))
    for runner, vals in pts.items():
        xs = [v[0] for v in vals]
        ys_mean = [v[1] for v in vals]
        ys_p50 = [v[2] if v[2] else v[1] for v in vals]
        color = _RUNNER_COLORS.get(runner, None)
        ax.plot(xs, ys_mean, marker="o", label=f"{runner} mean",
                color=color, linewidth=2)
        ax.plot(xs, ys_p50, marker="x", linestyle="--", label=f"{runner} p50",
                color=color, alpha=0.6, linewidth=1)
    # Highlight the baseline value
    ax.axvline(BASELINE[axis], color="black", linestyle=":", alpha=0.4,
               label=f"baseline {axis}={BASELINE[axis]}")
    ax.set_xlabel(f"{axis} (with other dims fixed at baseline)")
    ax.set_ylabel("Latency per forward (ms)")
    ax.set_title(f"FlashMoE scaling on axis {axis}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def plot_speedup_vs_pytorch(results, out_path):
    """Bar chart of (pytorch_mean / flashmoe_mean) per model, sorted descending."""
    by_model = defaultdict(dict)
    for r in results:
        if r.get("error"):
            continue
        by_model[r["model_name"]][r["runner"]] = r["mean_ms"]

    rows = []
    for name, runners in by_model.items():
        if "flashmoe" not in runners or "pytorch-eager" not in runners:
            continue
        rows.append((name, runners["pytorch-eager"] / runners["flashmoe"]))
    rows.sort(key=lambda x: x[1], reverse=True)
    if not rows:
        return False

    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(rows)), 5))
    names = [r[0] for r in rows]
    sps = [r[1] for r in rows]
    bars = ax.bar(names, sps, color="#e94560", edgecolor="black", linewidth=0.5)
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5,
               label="parity with PyTorch eager")
    for bar, s in zip(bars, sps):
        ax.text(bar.get_x() + bar.get_width() / 2, s,
                f"{s:.2f}x", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Speedup vs PyTorch eager (× higher = better)")
    ax.set_title("FlashMoE speedup vs PyTorch eager across all configs")
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def plot_vllm_comparison(results, out_path):
    """For every model that has both vllm and flashmoe data, plot the latency pair."""
    by_model = defaultdict(dict)
    for r in results:
        if r.get("error"):
            continue
        by_model[r["model_name"]][r["runner"]] = r

    rows = []
    for name, runners in by_model.items():
        if "flashmoe" not in runners or "vllm" not in runners:
            continue
        f, v = runners["flashmoe"], runners["vllm"]
        rows.append((name, f["mean_ms"], v["mean_ms"]))
    if not rows:
        return False
    rows.sort(key=lambda x: x[2] / x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(rows)), 5))
    x = list(range(len(rows)))
    w = 0.35
    flash = [r[1] for r in rows]
    vllm = [r[2] for r in rows]
    ax.bar([i - w / 2 for i in x], flash, w, label="flashmoe",
           color="#e94560", edgecolor="black", linewidth=0.5)
    ax.bar([i + w / 2 for i in x], vllm, w, label="vllm",
           color="#0f3460", edgecolor="black", linewidth=0.5)
    for i, (_, f, v) in enumerate(rows):
        ratio = v / f
        marker = "WIN" if ratio > 1.0 else "lose"
        color = "green" if ratio > 1.0 else "red"
        ax.text(i, max(f, v), f"{ratio:.2f}x", ha="center", va="bottom",
                fontsize=8, color=color, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], rotation=45, ha="right")
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("FlashMoE vs vLLM — head-to-head across configs (label = vllm/flash)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()

    raw = json.loads((args.run_dir / "results.json").read_text())
    out_dir = args.run_dir / "plots"
    out_dir.mkdir(exist_ok=True)

    made = []
    for axis in ("H", "I", "E", "K"):
        path = out_dir / f"scaling_{axis}.png"
        if plot_axis(raw, axis, path):
            made.append(path)

    sp_path = out_dir / "speedup_vs_pytorch.png"
    if plot_speedup_vs_pytorch(raw, sp_path):
        made.append(sp_path)

    vllm_path = out_dir / "headtohead_vllm.png"
    if plot_vllm_comparison(raw, vllm_path):
        made.append(vllm_path)

    for p in made:
        print(f"wrote {p}")
    print(f"Total: {len(made)} plots")
    return 0


if __name__ == "__main__":
    sys.exit(main())
