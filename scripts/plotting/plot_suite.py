#!/usr/bin/env python3
"""Plot reaction logs for a suite out_root.

Usage:
  python scripts/plotting/plot_suite.py --out_root out/suites/2026-01-24-unified \
    --patch_start 1200 --patch_end 1400
"""

import argparse
import json
import os
from collections import defaultdict

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_METRICS = [
    ("loss", "loss"),
    ("self_norm", "self_norm"),
    ("self_delta", "self_delta"),
    ("self_drift", "self_drift"),
    ("state_effect", "state_effect"),
    ("stats_entropy", "stats_entropy"),
    ("stats_top1_conf", "stats_top1_conf"),
    ("stats_grad_norm", "stats_grad_norm"),
    ("self_update_mlp_norm", "self_update_mlp_norm"),
    ("self_update_mem_norm", "self_update_mem_norm"),
    ("self_update_mem_clamped_norm", "self_update_mem_clamped_norm"),
    ("self_update_cos", "self_update_cos"),
    ("self_update_gate", "self_update_gate"),
]


def load_reaction_log(path):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    records.sort(key=lambda r: r.get('step', 0))
    return records


def series_for_metric(records, key):
    steps = []
    vals = []
    for rec in records:
        if key in rec and rec[key] is not None:
            steps.append(rec['step'])
            vals.append(rec[key])
    return steps, vals


def find_reaction_logs(out_root):
    logs = {}
    for root, _, files in os.walk(out_root):
        if 'reaction_log.jsonl' in files:
            run_id = os.path.basename(root)
            logs[run_id] = os.path.join(root, 'reaction_log.jsonl')
    return logs


def normalize_pair_name(name):
    return name.replace('patch', '').replace('baseline', '').replace('-', '').replace('_', '')


def plot_run(records, run_id, out_dir):
    metrics = [(k, label) for k, label in DEFAULT_METRICS if any(k in r for r in records)]
    if not metrics:
        return
    n = len(metrics)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3 * rows), sharex=False)
    axes = axes.flatten() if n > 1 else [axes]
    for ax, (key, label) in zip(axes, metrics):
        steps, vals = series_for_metric(records, key)
        if steps:
            ax.plot(steps, vals, linewidth=1)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
    for ax in axes[n:]:
        ax.axis('off')
    fig.tight_layout()
    out_path = os.path.join(out_dir, f"{run_id}_reactions.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_comparison(patch_records, base_records, out_dir, prefix, patch_start, patch_end, label_a, label_b):
    for key, label in DEFAULT_METRICS:
        p_steps, p_vals = series_for_metric(patch_records, key)
        b_steps, b_vals = series_for_metric(base_records, key)
        if not p_steps and not b_steps:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        if b_steps:
            ax.plot(b_steps, b_vals, label=label_b, linewidth=1)
        if p_steps:
            ax.plot(p_steps, p_vals, label=label_a, linewidth=1)
        if patch_start is not None and patch_end is not None:
            if patch_start == patch_end:
                ax.axvline(patch_start, color='red', alpha=0.5, linestyle='--')
            else:
                ax.axvspan(patch_start, patch_end, color='red', alpha=0.1)
        ax.set_title(f"{label}: {label_a} vs {label_b}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        out_path = os.path.join(out_dir, f"{prefix}_{key}_{label_a}_vs_{label_b}.png".replace(' ', '_'))
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--patch_start', type=int, default=1200)
    ap.add_argument('--patch_end', type=int, default=1400)
    ap.add_argument('--swap_step', type=int, default=1000)
    args = ap.parse_args(argv)

    out_root = args.out_root
    plot_dir = os.path.join(out_root, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    logs = find_reaction_logs(out_root)
    if not logs:
        print('No reaction_log.jsonl files found')
        return 1

    records_by_run = {}
    for run_id, path in logs.items():
        records = load_reaction_log(path)
        records_by_run[run_id] = records
        plot_run(records, run_id, plot_dir)

    # Pair patch vs baseline
    patch_runs = [r for r in records_by_run if 'patch' in r]
    base_runs = [r for r in records_by_run if 'baseline' in r]
    paired = []
    for p in patch_runs:
        p_key = normalize_pair_name(p)
        for b in base_runs:
            if normalize_pair_name(b) == p_key:
                paired.append((p, b, p_key))
                break
    if len(paired) == 1:
        p, b, key = paired[0]
        plot_comparison(records_by_run[p], records_by_run[b], plot_dir, key or 'patch', args.patch_start, args.patch_end, 'patch', 'baseline')
    else:
        for idx, (p, b, key) in enumerate(paired):
            plot_comparison(records_by_run[p], records_by_run[b], plot_dir, key or f"patch{idx}", args.patch_start, args.patch_end, 'patch', 'baseline')

    # Swap comparisons: swapped vs continued
    swapped_runs = [r for r in records_by_run if r.endswith('swapped')]
    for swapped in swapped_runs:
        base = swapped.replace('swapped', 'continued')
        if base not in records_by_run:
            base = swapped.replace('-swapped', '')
        if base not in records_by_run:
            continue
        prefix = swapped.replace('-swapped', '')
        plot_comparison(
            records_by_run[swapped],
            records_by_run[base],
            plot_dir,
            f"swap_{prefix}",
            args.swap_step,
            args.swap_step,
            'swapped',
            'continued',
        )

    print(f"wrote plots to {plot_dir}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main(os.sys.argv[1:]))
