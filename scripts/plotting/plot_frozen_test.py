#!/usr/bin/env python3
"""Generate plot for frozen-weight causal test results."""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='out/frozen-weight-test.png')
    args = ap.parse_args()

    # Data from the frozen-weight tests (3 seeds, 2000 iters)
    seeds = ['Seed 1', 'Seed 2', 'Seed 3']

    # Entropy values for each condition
    data = {
        'zero':       [1.8643, 1.9084, 1.9741],
        'checkpoint': [1.7670, 1.7873, 1.7808],
        'confident':  [1.8115, 1.8057, 1.7934],
        'uncertain':  [1.9608, 2.2013, 2.2472],
        'random':     [2.0036, 1.8106, 2.2431],
    }

    # Delta from zero
    deltas = {
        'checkpoint': [-0.0973, -0.1211, -0.1932],
        'confident':  [-0.0528, -0.1027, -0.1806],
        'uncertain':  [+0.0965, +0.2928, +0.2731],
        'random':     [+0.1393, -0.0978, +0.2690],
    }

    # Effect sizes
    effect_sizes = [0.1493, 0.3955, 0.4538]
    ckpt_effects = [0.0973, 0.1211, 0.1932]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Plot 1: Entropy by condition across seeds
    ax1 = axes[0]
    x = np.arange(len(seeds))
    width = 0.15

    colors = {
        'zero': '#888888',
        'checkpoint': '#2ecc71',
        'confident': '#3498db',
        'uncertain': '#e74c3c',
        'random': '#9b59b6'
    }

    for i, (name, values) in enumerate(data.items()):
        offset = (i - 2) * width
        bars = ax1.bar(x + offset, values, width, label=name, color=colors[name])

    ax1.set_xlabel('Training Seed')
    ax1.set_ylabel('Entropy')
    ax1.set_title('Output Entropy by Self-State Condition')
    ax1.set_xticks(x)
    ax1.set_xticklabels(seeds)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.axhline(y=np.mean(data['zero']), color='gray', linestyle='--', alpha=0.5)

    # Plot 2: Delta from zero baseline
    ax2 = axes[1]
    conditions = ['checkpoint', 'confident', 'uncertain', 'random']
    x2 = np.arange(len(conditions))
    width2 = 0.25

    for i, seed in enumerate(seeds):
        values = [deltas[c][i] for c in conditions]
        ax2.bar(x2 + (i - 1) * width2, values, width2, label=seed, alpha=0.8)

    ax2.set_xlabel('Self-State Condition')
    ax2.set_ylabel('Δ Entropy (from zero)')
    ax2.set_title('Entropy Change from Zero Baseline')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(conditions)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_ylim(-0.25, 0.35)

    # Add annotations for direction
    ax2.annotate('← confident\n(reduces entropy)', xy=(0.5, -0.20), fontsize=8, ha='center', color='#3498db')
    ax2.annotate('uncertain →\n(increases entropy)', xy=(2.5, 0.30), fontsize=8, ha='center', color='#e74c3c')

    # Plot 3: Effect sizes and checkpoint effect
    ax3 = axes[2]
    x3 = np.arange(len(seeds))
    width3 = 0.35

    ax3.bar(x3 - width3/2, effect_sizes, width3, label='Effect Size\n|confident - uncertain|', color='#3498db')
    ax3.bar(x3 + width3/2, ckpt_effects, width3, label='Checkpoint Effect\n|checkpoint - zero|', color='#2ecc71')

    ax3.set_xlabel('Training Seed')
    ax3.set_ylabel('Effect Magnitude')
    ax3.set_title('Causal Effect Sizes')
    ax3.set_xticks(x3)
    ax3.set_xticklabels(seeds)
    ax3.legend(loc='upper left', fontsize=8)

    # Add mean lines
    ax3.axhline(y=np.mean(effect_sizes), color='#3498db', linestyle='--', alpha=0.5)
    ax3.axhline(y=np.mean(ckpt_effects), color='#2ecc71', linestyle='--', alpha=0.5)

    plt.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Saved: {args.out}")


if __name__ == '__main__':
    main()
