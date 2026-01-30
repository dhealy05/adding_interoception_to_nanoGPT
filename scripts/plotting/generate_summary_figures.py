#!/usr/bin/env python3
"""Generate summary comparison figures that combine data across runs.

These are figures that synthesize/compare data rather than just copying:
- regime-reactions.png: All regimes on one plot with distinct colors
- regime-deltas.png: Bar chart comparing regime effects
- memory comparison plots
- controls comparison plots

Usage:
  python scripts/plotting/generate_summary_figures.py \
    --shakespeare out/shakespeare \
    --memory out/memory \
    --controls out/controls \
    --figures figures
"""

import argparse
import csv
import json
import os
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_reaction_log(path):
    """Load JSONL reaction log."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    records.sort(key=lambda r: r.get("step", 0))
    return records


def series_for_metric(records, key):
    """Extract step/value series for a metric."""
    steps = []
    vals = []
    for rec in records:
        if key in rec and rec[key] is not None:
            steps.append(rec["step"])
            vals.append(rec[key])
    return np.array(steps), np.array(vals)


def build_regime_windows(records):
    """Extract regime windows from records."""
    by_step = {r["step"]: r for r in records}
    steps = sorted(by_step)
    regime_ids = sorted({rid for r in records for rid in r.get("regimes", [])})
    windows = {}
    for rid in regime_ids:
        rsteps = [s for s in steps if rid in by_step[s].get("regimes", [])]
        if rsteps:
            windows[rid] = (min(rsteps), max(rsteps) + 1)
    return windows


# =============================================================================
# Regime summary figures
# =============================================================================

REGIME_COLORS = {
    "lr_heatwave": "#e74c3c",      # red
    "sensory_fog": "#3498db",       # blue
    "false_feedback": "#9b59b6",    # purple
    "self_clamp": "#f39c12",        # orange
    "bewilderment": "#2ecc71",      # green
}


def plot_regime_reactions(records, out_path, title="Perturbation Regime Signatures"):
    """Multi-panel plot showing all regimes with distinct colors."""
    windows = build_regime_windows(records)
    if not windows:
        print(f"  No regimes found, skipping {out_path}")
        return

    metrics = [
        ("loss", "Loss"),
        ("self_drift", "State Drift"),
        ("state_effect", "State Effect"),
        ("stats_entropy", "Entropy"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (key, label) in zip(axes, metrics):
        steps, vals = series_for_metric(records, key)
        if len(steps) == 0:
            ax.set_title(label)
            continue

        # Plot baseline
        ax.plot(steps, vals, color="#888888", alpha=0.5, linewidth=0.8)

        # Highlight each regime
        for rid, (start, end) in windows.items():
            color = REGIME_COLORS.get(rid, "#888888")
            ax.axvspan(start, end, color=color, alpha=0.2)
            mask = (steps >= start) & (steps < end)
            if mask.any():
                ax.plot(steps[mask], vals[mask], color=color, linewidth=2, label=rid)

        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)

    # Legend
    handles = []
    labels = []
    for rid, color in REGIME_COLORS.items():
        if rid in windows:
            handles.append(plt.Line2D([0], [0], color=color, linewidth=3))
            labels.append(rid.replace("_", " ").title())

    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels),
                   bbox_to_anchor=(0.5, 0.02), fontsize=10)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_regime_deltas(records, out_path):
    """Bar chart comparing regime effects on metrics."""
    windows = build_regime_windows(records)
    if not windows:
        print(f"  No regimes found, skipping {out_path}")
        return

    by_step = {r["step"]: r for r in records}
    steps = sorted(by_step)

    metrics = ["loss", "self_drift", "state_effect", "stats_entropy"]
    metric_labels = ["Δ Loss", "Δ Drift", "Δ Effect", "Δ Entropy"]

    # Compute deltas
    regime_deltas = {}
    baseline_steps = [s for s in steps if not by_step[s].get("regimes")]

    for rid, (start, end) in windows.items():
        prior = [s for s in baseline_steps if s < start][-100:]
        during = [s for s in steps if start <= s < end]

        deltas = {}
        for m in metrics:
            base_vals = [by_step[s].get(m) for s in prior if by_step[s].get(m) is not None]
            during_vals = [by_step[s].get(m) for s in during if by_step[s].get(m) is not None]
            if base_vals and during_vals:
                deltas[m] = np.mean(during_vals) - np.mean(base_vals)
            else:
                deltas[m] = 0
        regime_deltas[rid] = deltas

    # Bar chart
    regime_names = list(regime_deltas.keys())
    x = np.arange(len(metrics))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#e74c3c", "#3498db", "#9b59b6", "#f39c12", "#2ecc71"]
    for i, rid in enumerate(regime_names):
        offset = (i - len(regime_names) / 2 + 0.5) * width
        values = [regime_deltas[rid].get(m, 0) for m in metrics]
        ax.bar(x + offset, values, width, label=rid.replace("_", " ").title(),
               color=colors[i % len(colors)])

    ax.set_ylabel("Delta from Baseline", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(loc="best", fontsize=9)
    ax.set_title("Regime Effects: Each Perturbation Produces Distinct Signatures",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =============================================================================
# Memory comparison figures
# =============================================================================

def plot_memory_comparison(memory_root, out_dir):
    """Generate memory architecture comparison plots."""
    configs = [
        ("mem-ema-decay-0.1_s10_k1", "EMA Baseline"),
        ("mem-buf-l64_s10", "Buffer L=64"),
        ("mem-buf-l128_s10", "Buffer L=128"),
    ]

    metrics = [
        ("self_update_gate", "Memory Gate"),
        ("self_drift", "State Drift"),
        ("state_effect", "State Effect"),
    ]

    for metric_key, metric_label in metrics:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)

        for ax, (config_id, config_label) in zip(axes, configs):
            log_path = os.path.join(memory_root, config_id, "reaction_log.jsonl")
            if not os.path.exists(log_path):
                ax.set_title(f"{config_label}\n(data not found)")
                continue

            records = load_reaction_log(log_path)
            steps, vals = series_for_metric(records, metric_key)

            if len(steps) > 0:
                ax.plot(steps, vals, linewidth=1, color="#1f77b4")
                ax.axvspan(1000, 1500, color="#F2C94C", alpha=0.25)

            ax.set_title(config_label, fontweight="bold")
            ax.set_xlabel("Step")
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel(metric_label)
        fig.suptitle(f"Memory Architecture Comparison: {metric_label}",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        out_path = os.path.join(out_dir, f"memory-{metric_key}.png")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# =============================================================================
# Controls comparison figures
# =============================================================================

def plot_controls_comparison(controls_root, out_dir):
    """Generate control experiment comparison plots."""
    variants = [
        ("vanilla-regimes", "Vanilla"),
        ("self-regimes", "Self-State"),
        ("fixed-bias-0.5", "Fixed 0.5x"),
        ("fixed-bias-1.0", "Fixed 1.0x"),
        ("self-dim1-regimes", "Dim=1"),
    ]

    # Load regime delta CSVs
    delta_data = {}
    for variant_id, variant_label in variants:
        csv_path = os.path.join(controls_root, variant_id, "regime_deltas.csv")
        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                delta_data[variant_label] = list(reader)

    if not delta_data:
        print("  No control delta data found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Loss delta comparison
    ax1 = axes[0]
    for label, rows in delta_data.items():
        regimes = [r.get('regime', '') for r in rows]
        loss_deltas = [float(r.get('loss_delta', 0)) for r in rows]
        ax1.scatter(range(len(regimes)), loss_deltas, label=label, s=80, alpha=0.7)

    ax1.set_xlabel("Regime Index")
    ax1.set_ylabel("Δ Loss")
    ax1.set_title("Loss Response by Configuration", fontweight="bold")
    ax1.legend(loc="best", fontsize=8)
    ax1.axhline(0, color="black", linewidth=0.5)
    ax1.grid(True, alpha=0.3)

    # Entropy delta comparison
    ax2 = axes[1]
    for label, rows in delta_data.items():
        regimes = [r.get('regime', '') for r in rows]
        ent_deltas = [float(r.get('stats_entropy_delta', 0)) for r in rows]
        ax2.scatter(range(len(regimes)), ent_deltas, label=label, s=80, alpha=0.7)

    ax2.set_xlabel("Regime Index")
    ax2.set_ylabel("Δ Entropy")
    ax2.set_title("Entropy Response by Configuration", fontweight="bold")
    ax2.legend(loc="best", fontsize=8)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "controls-comparison.png")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shakespeare", default="out/shakespeare")
    ap.add_argument("--openwebtext", default="out/openwebtext")
    ap.add_argument("--memory", default="out/memory")
    ap.add_argument("--controls", default="out/controls")
    ap.add_argument("--figures", default="figures")
    args = ap.parse_args()

    print("Generating summary figures...\n")

    # 1. Regime summary figures (shakespeare)
    print("1. Regime summary figures (shakespeare):")
    regimes_log = os.path.join(args.shakespeare, "regimes-all", "reaction_log.jsonl")
    if os.path.exists(regimes_log):
        records = load_reaction_log(regimes_log)
        plot_regime_reactions(
            records,
            os.path.join(args.figures, "perturbation", "regime-reactions.png"),
            title="Perturbation Regimes Produce Distinct Signatures"
        )
        plot_regime_deltas(
            records,
            os.path.join(args.figures, "perturbation", "regime-deltas.png")
        )
    else:
        print(f"  Not found: {regimes_log}")

    # 2. Regime summary figures (openwebtext)
    print("\n2. Regime summary figures (openwebtext):")
    regimes_log_owt = os.path.join(args.openwebtext, "regimes-all", "reaction_log.jsonl")
    if os.path.exists(regimes_log_owt):
        records = load_reaction_log(regimes_log_owt)
        plot_regime_reactions(
            records,
            os.path.join(args.figures, "openwebtext", "regime-reactions.png"),
            title="OpenWebText: Regime Signatures Generalize"
        )
        plot_regime_deltas(
            records,
            os.path.join(args.figures, "openwebtext", "regime-deltas.png")
        )
    else:
        print(f"  Not found: {regimes_log_owt}")

    # 3. Memory comparison
    print("\n3. Memory comparison figures:")
    if os.path.exists(args.memory):
        plot_memory_comparison(args.memory, os.path.join(args.figures, "memory"))
    else:
        print(f"  Not found: {args.memory}")

    # 4. Controls comparison
    print("\n4. Controls comparison figures:")
    if os.path.exists(args.controls):
        plot_controls_comparison(args.controls, os.path.join(args.figures, "controls"))
    else:
        print(f"  Not found: {args.controls}")

    print(f"\nDone! Summary figures saved to: {args.figures}/")


if __name__ == "__main__":
    main()
