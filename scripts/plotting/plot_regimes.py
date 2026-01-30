#!/usr/bin/env python3
"""Plot per-regime reaction windows from a reaction_log.jsonl."""

import argparse
import json
import os
from pathlib import Path

import matplotlib

if not os.environ.get('MPLCONFIGDIR'):
    os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_records(path):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    records.sort(key=lambda r: r.get('step', 0))
    return records


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--reaction_log', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--pad', type=int, default=100)
    args = ap.parse_args(argv)

    records = load_records(args.reaction_log)
    if not records:
        raise SystemExit('no records found')

    by_step = {r['step']: r for r in records}
    steps = sorted(by_step)

    regime_ids = sorted({rid for r in records for rid in r.get('regimes', [])})
    windows = {}
    for rid in regime_ids:
        rsteps = [s for s in steps if rid in by_step[s].get('regimes', [])]
        if rsteps:
            windows[rid] = (min(rsteps), max(rsteps) + 1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Precompute series for speed.
    series = {
        'loss': [(s, by_step[s].get('loss')) for s in steps],
        'self_drift': [(s, by_step[s].get('self_drift')) for s in steps],
        'state_effect': [(s, by_step[s].get('state_effect')) for s in steps],
        'stats_entropy': [(s, by_step[s].get('stats_entropy')) for s in steps],
        'stats_top1_conf': [(s, by_step[s].get('stats_top1_conf')) for s in steps],
        'self_update_mlp_norm': [(s, by_step[s].get('self_update_mlp_norm')) for s in steps],
        'self_update_mem_norm': [(s, by_step[s].get('self_update_mem_norm')) for s in steps],
        'self_update_mem_clamped_norm': [(s, by_step[s].get('self_update_mem_clamped_norm')) for s in steps],
        'self_update_cos': [(s, by_step[s].get('self_update_cos')) for s in steps],
        'self_update_gate': [(s, by_step[s].get('self_update_gate')) for s in steps],
    }

    for rid, (start, end) in windows.items():
        lo = max(min(steps), start - args.pad)
        hi = min(max(steps), end + args.pad)

        def windowed(key):
            pts = [(s, v) for s, v in series[key] if lo <= s <= hi and v is not None]
            if not pts:
                return [], []
            xs, ys = zip(*pts)
            return xs, ys

        fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
        axes = axes.flatten()

        for ax in axes:
            ax.axvspan(start, end, color='#F2C94C', alpha=0.25, label='regime')

        # Loss
        xs, ys = windowed('loss')
        axes[0].plot(xs, ys, color='#1f77b4')
        axes[0].set_title('loss')

        # Self drift
        xs, ys = windowed('self_drift')
        axes[1].plot(xs, ys, color='#d62728')
        axes[1].set_title('self_drift')

        # State effect
        xs, ys = windowed('state_effect')
        axes[2].plot(xs, ys, color='#2ca02c')
        axes[2].set_title('state_effect')

        # Entropy + top1
        xs_e, ys_e = windowed('stats_entropy')
        xs_t, ys_t = windowed('stats_top1_conf')
        if xs_e:
            axes[3].plot(xs_e, ys_e, color='#9467bd', label='entropy')
        if xs_t:
            axes[3].plot(xs_t, ys_t, color='#8c564b', label='top1_conf')
        axes[3].set_title('stats')
        axes[3].legend(loc='best', fontsize=8)

        fig.suptitle(f"regime {rid} ({start}-{end})")
        for ax in axes:
            ax.grid(True, alpha=0.2)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = out_dir / f"regime_{rid}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        xs_mlp, ys_mlp = windowed('self_update_mlp_norm')
        xs_mem, ys_mem = windowed('self_update_mem_norm')
        xs_mem_clamped, ys_mem_clamped = windowed('self_update_mem_clamped_norm')
        xs_cos, ys_cos = windowed('self_update_cos')
        xs_gate, ys_gate = windowed('self_update_gate')
        if xs_mlp or xs_mem or xs_mem_clamped or xs_cos or xs_gate:
            fig2, axes2 = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
            for ax in axes2:
                ax.axvspan(start, end, color='#F2C94C', alpha=0.25, label='regime')
            if xs_mlp:
                axes2[0].plot(xs_mlp, ys_mlp, color='#1f77b4', label='mlp_update_norm')
            if xs_mem:
                axes2[0].plot(xs_mem, ys_mem, color='#ff7f0e', label='memory_update_norm')
            if xs_mem_clamped:
                axes2[0].plot(xs_mem_clamped, ys_mem_clamped, color='#9467bd', label='memory_clamped_norm')
            axes2[0].set_title('update norms')
            axes2[0].legend(loc='best', fontsize=8)
            if xs_cos:
                axes2[1].plot(xs_cos, ys_cos, color='#2ca02c')
            axes2[1].set_title('update cosine similarity')
            if xs_gate:
                axes2[2].plot(xs_gate, ys_gate, color='#d62728')
            axes2[2].set_title('memory gate')
            for ax in axes2:
                ax.grid(True, alpha=0.2)
            fig2.suptitle(f"regime {rid} updates ({start}-{end})")
            fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
            out_path2 = out_dir / f"regime_{rid}_update.png"
            fig2.savefig(out_path2, dpi=150)
            plt.close(fig2)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
