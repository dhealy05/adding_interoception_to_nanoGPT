#!/usr/bin/env python3
"""Frozen-weight causal test with automatic direction discovery.

Discovers confident/uncertain directions for the specific checkpoint,
then runs the frozen-weight test using those directions.
"""

import argparse
import json
import os
import sys

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.model import GPT, GPTConfig


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = GPTConfig(**ckpt['model_args'])
    model = GPT(cfg)
    state_dict = ckpt['model']
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, ckpt


def get_batch(data_dir, block_size, batch_size, device, seed):
    data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    torch.manual_seed(seed)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    return x.to(device)


def evaluate(model, x, self_state, device):
    self_state = self_state.to(device).float()
    with torch.no_grad():
        logits, _ = model(x, self_state=self_state)
        logits = logits[:, -1, :]
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean().item()
    top1 = probs.max(dim=-1).values.mean().item()
    return entropy, top1, probs


def discover_directions(model, x, dim, device, magnitude=0.5, top_k=5):
    """Discover which dimensions affect entropy in each direction."""
    zero_state = torch.zeros(dim)
    base_ent, _, _ = evaluate(model, x, zero_state, device)

    effects = []
    for d in range(dim):
        pos_state = torch.zeros(dim)
        pos_state[d] = magnitude
        pos_ent, _, _ = evaluate(model, x, pos_state, device)
        effects.append((d, pos_ent - base_ent))

    effects_sorted = sorted(effects, key=lambda x: x[1])

    # Top-k confident (most negative = reduces entropy most)
    confident_dims = [d for d, e in effects_sorted[:top_k]]
    # Top-k uncertain (most positive = increases entropy most)
    uncertain_dims = [d for d, e in effects_sorted[-top_k:][::-1]]

    return confident_dims, uncertain_dims, effects_sorted


def create_directional_state(dims, dim_total, magnitude, sign=1.0):
    """Create a state vector with given magnitude on specified dims."""
    state = torch.zeros(dim_total)
    for d in dims:
        state[d] = sign * magnitude
    return state


def plot_results(summary, effect_size, ckpt_effect, out_path):
    """Generate frozen-weight test figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Extract data
    conditions = ["zero", "checkpoint", "confident", "uncertain", "random"]
    entropies = [summary[c]["entropy"] for c in conditions]
    deltas = [summary[c]["d_entropy"] for c in conditions[1:]]  # skip zero

    colors = {
        "zero": "#888888",
        "checkpoint": "#2ecc71",
        "confident": "#3498db",
        "uncertain": "#e74c3c",
        "random": "#9b59b6",
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Plot 1: Entropy by condition
    ax1 = axes[0]
    x = np.arange(len(conditions))
    bars = ax1.bar(x, entropies, color=[colors[c] for c in conditions])
    ax1.set_xlabel("Self-State Condition")
    ax1.set_ylabel("Entropy")
    ax1.set_title("Output Entropy by Self-State Condition", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(conditions, rotation=15)
    ax1.axhline(y=summary["zero"]["entropy"], color="gray", linestyle="--", alpha=0.5)

    # Plot 2: Delta from zero baseline
    ax2 = axes[1]
    delta_conditions = ["checkpoint", "confident", "uncertain", "random"]
    x2 = np.arange(len(delta_conditions))
    delta_colors = [colors[c] for c in delta_conditions]
    ax2.bar(x2, deltas, color=delta_colors)
    ax2.set_xlabel("Self-State Condition")
    ax2.set_ylabel("Δ Entropy (from zero)")
    ax2.set_title("Entropy Change from Zero Baseline", fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(delta_conditions, rotation=15)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    # Annotations
    ax2.annotate("← reduces entropy", xy=(1, min(deltas) - 0.02),
                 fontsize=8, ha="center", color="#3498db")
    ax2.annotate("increases entropy →", xy=(2, max(deltas) + 0.02),
                 fontsize=8, ha="center", color="#e74c3c")

    # Plot 3: Effect sizes
    ax3 = axes[2]
    effect_labels = ["Effect Size\n|conf - unc|", "Checkpoint Effect\n|ckpt - zero|"]
    effect_values = [effect_size, ckpt_effect]
    effect_colors = ["#3498db", "#2ecc71"]
    ax3.bar(range(2), effect_values, color=effect_colors)
    ax3.set_ylabel("Effect Magnitude")
    ax3.set_title("Causal Effect Sizes", fontweight="bold")
    ax3.set_xticks(range(2))
    ax3.set_xticklabels(effect_labels)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--dataset', default='shakespeare_char')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--magnitude', type=float, default=0.3)
    ap.add_argument('--top_k', type=int, default=5,
                    help='Number of top dims to use for confident/uncertain')
    ap.add_argument('--num_batches', type=int, default=5)
    ap.add_argument('--num_random', type=int, default=50,
                    help='Number of random directions for control statistics')
    ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--out', default='', help='Path to save results JSON')
    ap.add_argument('--plot', default='', help='Path to save results plot (PNG)')
    args = ap.parse_args()

    model, ckpt = load_model(args.ckpt, args.device)
    block_size = ckpt['model_args']['block_size']
    dim = ckpt['model_args'].get('self_state_dim', 32)
    data_dir = os.path.join(REPO_ROOT, 'data', args.dataset)

    # Discovery batch
    x_discover = get_batch(data_dir, block_size, args.batch_size, args.device, args.seed)

    if not args.quiet:
        print(f"Checkpoint: {args.ckpt}")
        print(f"Discovering directions (top_k={args.top_k}, magnitude={args.magnitude})...")

    confident_dims, uncertain_dims, effects = discover_directions(
        model, x_discover, dim, args.device, args.magnitude, args.top_k)

    if not args.quiet:
        print(f"  Confident dims: {confident_dims}")
        print(f"  Uncertain dims: {uncertain_dims}")

    # Create state variants
    checkpoint_state = ckpt.get('self_state', torch.zeros(dim)).float()
    zero_state = torch.zeros(dim)
    confident_state = create_directional_state(confident_dims, dim, args.magnitude, sign=1.0)
    uncertain_state = create_directional_state(uncertain_dims, dim, args.magnitude, sign=1.0)

    # Random placebo with matched norm
    torch.manual_seed(999)
    random_state = torch.randn(dim)
    random_state = random_state / random_state.norm() * confident_state.norm()

    variants = {
        'zero': zero_state,
        'checkpoint': checkpoint_state,
        'confident': confident_state,
        'uncertain': uncertain_state,
        'random': random_state
    }

    # Evaluate across batches (shared across all variants)
    results = {name: {'entropy': [], 'top1': []} for name in variants}
    x_batches = []
    for batch_idx in range(args.num_batches):
        x = get_batch(data_dir, block_size, args.batch_size, args.device,
                      args.seed + 1000 + batch_idx)
        x_batches.append(x)
        for name, state in variants.items():
            ent, top1, _ = evaluate(model, x, state, args.device)
            results[name]['entropy'].append(ent)
            results[name]['top1'].append(top1)

    # Aggregate
    zero_ent_list = results['zero']['entropy']
    zero_top1_list = results['zero']['top1']
    zero_ent = np.mean(zero_ent_list)
    zero_top1 = np.mean(zero_top1_list)

    summary = {}
    for name in variants:
        ent = np.mean(results[name]['entropy'])
        top1 = np.mean(results[name]['top1'])
        summary[name] = {
            'entropy': ent,
            'top1': top1,
            'd_entropy': ent - zero_ent,
            'd_top1': top1 - zero_top1
        }

    # Effect size
    effect_size = abs(summary['confident']['entropy'] - summary['uncertain']['entropy'])
    ckpt_effect = abs(summary['checkpoint']['entropy'] - zero_ent)

    # Sign consistency across batches for learned directions
    conf_deltas = [e - z for e, z in zip(results['confident']['entropy'], zero_ent_list)]
    unc_deltas = [e - z for e, z in zip(results['uncertain']['entropy'], zero_ent_list)]
    conf_sign = float(np.mean([d < 0 for d in conf_deltas])) if conf_deltas else 0.0
    unc_sign = float(np.mean([d > 0 for d in unc_deltas])) if unc_deltas else 0.0

    # Random-direction control stats (many random draws, norm-matched)
    random_stats = None
    if args.num_random > 0:
        rand_deltas = []
        for i in range(args.num_random):
            torch.manual_seed(1000 + i)
            rand_state = torch.randn(dim)
            rand_state = rand_state / rand_state.norm() * confident_state.norm()
            ent_vals = []
            for x in x_batches:
                ent, _, _ = evaluate(model, x, rand_state, args.device)
                ent_vals.append(ent)
            delta = float(np.mean(ent_vals) - zero_ent)
            rand_deltas.append(delta)
        rand_deltas = np.array(rand_deltas, dtype=np.float64)
        rand_sign = float(np.mean(rand_deltas < 0.0))
        random_stats = {
            'num_random': int(args.num_random),
            'mean_d_entropy': float(np.mean(rand_deltas)),
            'std_d_entropy': float(np.std(rand_deltas)),
            'sign_consistency': rand_sign,
        }

    if not args.quiet:
        print(f"\n{'State':<12} {'Entropy':>10} {'Top-1':>10} {'Δ Ent':>10} {'Δ Top1':>10}")
        print("-" * 55)
        for name in ['zero', 'checkpoint', 'confident', 'uncertain', 'random']:
            s = summary[name]
            print(f"{name:<12} {s['entropy']:>10.4f} {s['top1']:>10.4f} "
                  f"{s['d_entropy']:>+10.4f} {s['d_top1']:>+10.4f}")

        print(f"\nEffect size (|confident - uncertain|): {effect_size:.4f}")
        print(f"Checkpoint effect (vs zero): {ckpt_effect:.4f}")
        print(f"Checkpoint norm: {checkpoint_state.norm().item():.4f}")
        print(f"Sign consistency (confident Δentropy<0): {conf_sign:.2f}")
        print(f"Sign consistency (uncertain Δentropy>0): {unc_sign:.2f}")
        if random_stats is not None:
            print(
                f"Random control (n={random_stats['num_random']}): "
                f"Δentropy mean={random_stats['mean_d_entropy']:.4f} "
                f"std={random_stats['std_d_entropy']:.4f} "
                f"sign_consistency={random_stats['sign_consistency']:.2f}"
            )

        # Checks
        conf_ok = summary['confident']['d_entropy'] < -0.01
        unc_ok = summary['uncertain']['d_entropy'] > 0.01
        print(f"\nChecks:")
        print(f"  confident < zero: {'✓' if conf_ok else '✗'}")
        print(f"  uncertain > zero: {'✓' if unc_ok else '✗'}")

    # Build result dict for programmatic use
    result = {
        'ckpt': args.ckpt,
        'effect_size': effect_size,
        'ckpt_effect': ckpt_effect,
        'ckpt_norm': checkpoint_state.norm().item(),
        'confident_dims': confident_dims,
        'uncertain_dims': uncertain_dims,
        'summary': summary,
        'sign_consistency': {'confident': conf_sign, 'uncertain': unc_sign},
        'random_control': random_stats,
    }

    # Save to JSON if --out specified
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        if not args.quiet:
            print(f"\nResults saved to: {args.out}")

    # Generate plot if --plot specified
    if args.plot:
        plot_results(summary, effect_size, ckpt_effect, args.plot)

    return result


if __name__ == '__main__':
    main()
