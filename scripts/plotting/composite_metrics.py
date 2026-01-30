#!/usr/bin/env python3
"""Compute desmotic valence/intensity metrics from self_state dumps + reaction log."""

import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np
import torch

import matplotlib

if not os.environ.get('MPLCONFIGDIR'):
    os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'

matplotlib.use('Agg')
import matplotlib.pyplot as plt

SELF_STATE_RE = re.compile(r"self_state_(\d+)\.pt$")


def load_self_states(run_dir):
    run_path = Path(run_dir)
    items = []
    for path in run_path.glob('self_state_*.pt'):
        match = SELF_STATE_RE.search(path.name)
        if not match:
            continue
        step = int(match.group(1))
        try:
            tensor = torch.load(path, map_location='cpu')
        except Exception:
            continue
        if hasattr(tensor, 'detach'):
            tensor = tensor.detach().cpu()
        arr = np.array(tensor, dtype=np.float64).reshape(-1)
        items.append((step, arr))
    items.sort(key=lambda x: x[0])
    if not items:
        return [], np.zeros((0, 0), dtype=np.float64)
    steps = [s for s, _ in items]
    states = np.stack([x for _, x in items], axis=0)
    return steps, states


def load_reaction_log(path):
    if not os.path.isfile(path):
        return {}
    records = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            step = int(rec.get('step', -1))
            if step >= 0:
                records[step] = rec
    return records


def compute_pca_basis(x, rank):
    if x.shape[0] == 0:
        return np.zeros(x.shape[1], dtype=np.float64), np.zeros((x.shape[1], 0), dtype=np.float64)
    mu = x.mean(axis=0)
    centered = x - mu
    if rank <= 0:
        return mu, np.zeros((x.shape[1], 0), dtype=np.float64)
    # SVD for PCA basis
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    rank = min(rank, vt.shape[0])
    basis = vt[:rank].T
    return mu, basis


def distance_to_manifold(x, mu, basis):
    residual = x - mu
    if basis.shape[1] == 0:
        return np.linalg.norm(residual, axis=1)
    proj = residual @ basis @ basis.T
    off = residual - proj
    return np.linalg.norm(off, axis=1)


def standardize(values, mask, eps):
    base = values[mask]
    if base.size == 0:
        mean = float(np.mean(values))
        std = float(np.std(values))
    else:
        mean = float(np.mean(base))
        std = float(np.std(base))
    return (values - mean) / (std + eps), mean, std


def zscore(values, base_values, eps):
    if base_values.size == 0:
        mean = float(np.mean(values))
        std = float(np.std(values))
    else:
        mean = float(np.mean(base_values))
        std = float(np.std(base_values))
    return (values - mean) / (std + eps), mean, std


def ema(values, alpha):
    if alpha <= 0:
        return values
    out = np.zeros_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def intensity_transform(x, fn):
    if fn == 'tanh':
        return 0.5 * (np.tanh(x) + 1.0)
    if fn == 'softsign':
        return 0.5 * (x / (1.0 + np.abs(x)) + 1.0)
    return sigmoid(x)


def build_regime_windows(records, steps):
    windows = {}
    for step in steps:
        rec = records.get(int(step))
        if not rec:
            continue
        for rid in rec.get('regimes', []) or []:
            if rid not in windows:
                windows[rid] = [step, step]
            else:
                windows[rid][0] = min(windows[rid][0], step)
                windows[rid][1] = max(windows[rid][1], step)
    return {rid: (lo, hi + 1) for rid, (lo, hi) in windows.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--out_dir', default='')
    ap.add_argument('--reaction_log', default='')
    ap.add_argument('--baseline_start', type=int, default=None)
    ap.add_argument('--baseline_end', type=int, default=None)
    ap.add_argument('--baseline_len', type=int, default=0)
    ap.add_argument('--local_baseline_len', type=int, default=0)
    ap.add_argument('--patch_start', type=int, default=None)
    ap.add_argument('--patch_end', type=int, default=None)
    ap.add_argument('--attractor_window', type=int, default=200)
    ap.add_argument('--pca_rank', type=int, default=4)
    ap.add_argument('--epsilon', type=float, default=1e-6)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--beta', type=float, default=1.0)
    ap.add_argument('--gamma', type=float, default=1.0)
    ap.add_argument('--delta', type=float, default=1.0)
    ap.add_argument('--eta1', type=float, default=1.0)
    ap.add_argument('--eta2', type=float, default=1.0)
    ap.add_argument('--intensity_scale', type=float, default=0.5)
    ap.add_argument('--intensity_fn', choices=['sigmoid', 'tanh', 'softsign'], default='sigmoid')
    ap.add_argument('--smooth_ema', type=float, default=0.0)
    args = ap.parse_args(argv)

    run_dir = args.run_dir
    out_dir = args.out_dir or os.path.join(run_dir, 'desmotic')
    reaction_log = args.reaction_log or os.path.join(run_dir, 'reaction_log.jsonl')

    steps, states = load_self_states(run_dir)
    if len(steps) == 0:
        raise SystemExit(f'no self_state_*.pt files found in {run_dir}')

    reactions = load_reaction_log(reaction_log)

    aligned_steps = []
    xs = []
    entropy = []
    influence = []
    for step, x in zip(steps, states):
        rec = reactions.get(step)
        if not rec:
            continue
        if rec.get('stats_entropy') is None:
            continue
        if rec.get('state_effect') is None:
            continue
        aligned_steps.append(step)
        xs.append(x)
        entropy.append(float(rec.get('stats_entropy')))
        influence.append(float(rec.get('state_effect')))

    if len(aligned_steps) < 3:
        raise SystemExit('not enough aligned steps with entropy/state_effect')

    steps = np.array(aligned_steps, dtype=np.int64)
    xs = np.stack(xs, axis=0)
    entropy = np.array(entropy, dtype=np.float64)
    influence = np.array(influence, dtype=np.float64)

    # Derivatives (per-step, adjusted for step spacing)
    dot = np.zeros_like(xs)
    if len(xs) > 1:
        dt = (steps[1:] - steps[:-1]).astype(np.float64)
        dt[dt == 0] = 1.0
        dot[1:] = (xs[1:] - xs[:-1]) / dt[:, None]

    ddot = np.zeros_like(xs)
    if len(xs) > 2:
        dt2 = (steps[2:] - steps[1:-1]).astype(np.float64)
        dt2[dt2 == 0] = 1.0
        ddot[2:] = (dot[2:] - dot[1:-1]) / dt2[:, None]

    s_t = np.linalg.norm(dot, axis=1)
    kappa = np.linalg.norm(ddot, axis=1) / np.power(1.0 + np.square(s_t), 1.5)

    # Attractor manifold (PCA on last K dumps)
    k = min(args.attractor_window, xs.shape[0])
    attractor_slice = xs[-k:]
    mu, basis = compute_pca_basis(attractor_slice, args.pca_rank)
    d_t = distance_to_manifold(xs, mu, basis)

    # Baseline mask
    base_start = args.baseline_start if args.baseline_start is not None else steps.min()
    if args.baseline_end is not None:
        base_end = args.baseline_end
    elif args.patch_start is not None:
        base_end = args.patch_start
    else:
        base_end = steps.min() + int(0.2 * (steps.max() - steps.min()))
    baseline_mask = (steps >= base_start) & (steps < base_end)
    if args.baseline_len > 0:
        idx = np.where(baseline_mask)[0]
        if idx.size > args.baseline_len:
            keep = idx[-args.baseline_len:]
            baseline_mask = np.zeros_like(baseline_mask)
            baseline_mask[keep] = True

    # Control loss term
    base_influence = influence[baseline_mask]
    if base_influence.size == 0:
        i_star = float(np.median(influence))
    else:
        i_star = float(np.median(base_influence))
    i_star = max(i_star, args.epsilon)
    c_t = np.log(i_star / (influence + args.epsilon))

    d_t_z, d_mu, d_std = standardize(d_t, baseline_mask, args.epsilon)
    s_t_z, s_mu, s_std = standardize(s_t, baseline_mask, args.epsilon)
    u_t_z, u_mu, u_std = standardize(entropy, baseline_mask, args.epsilon)
    c_t_z, c_mu, c_std = standardize(c_t, baseline_mask, args.epsilon)

    windows = {}
    if args.local_baseline_len > 0:
        windows = build_regime_windows(reactions, steps)

    d_t_z_local = d_t_z.copy()
    s_t_z_local = s_t_z.copy()
    u_t_z_local = u_t_z.copy()
    c_t_z_local = c_t_z.copy()

    if windows:
        for _, (start, end) in windows.items():
            win_idx = np.where((steps >= start) & (steps < end))[0]
            if win_idx.size == 0:
                continue
            base_idx = np.where(steps < start)[0]
            if base_idx.size == 0:
                continue
            base_idx = base_idx[-args.local_baseline_len:]
            if base_idx.size < 2:
                continue
            d_t_z_local[win_idx], _, _ = zscore(d_t[win_idx], d_t[base_idx], args.epsilon)
            s_t_z_local[win_idx], _, _ = zscore(s_t[win_idx], s_t[base_idx], args.epsilon)
            u_t_z_local[win_idx], _, _ = zscore(entropy[win_idx], entropy[base_idx], args.epsilon)
            c_t_z_local[win_idx], _, _ = zscore(c_t[win_idx], c_t[base_idx], args.epsilon)

    valence = -(
        args.alpha * d_t_z
        + args.beta * s_t_z
        + args.gamma * u_t_z
        + args.delta * c_t_z
    )
    valence_local = -(
        args.alpha * d_t_z_local
        + args.beta * s_t_z_local
        + args.gamma * u_t_z_local
        + args.delta * c_t_z_local
    )

    if args.smooth_ema > 0:
        valence = ema(valence, args.smooth_ema)
        valence_local = ema(valence_local, args.smooth_ema)

    r_t = np.zeros_like(valence)
    r_t[1:] = np.abs(valence[1:] - valence[:-1])
    r_t_local = np.zeros_like(valence_local)
    r_t_local[1:] = np.abs(valence_local[1:] - valence_local[:-1])

    r_t_z, _, _ = standardize(r_t, baseline_mask, args.epsilon)
    r_t_local_z, _, _ = standardize(r_t_local, baseline_mask, args.epsilon)
    kappa_z, _, _ = standardize(kappa, baseline_mask, args.epsilon)

    phi_t = intensity_transform(
        args.intensity_scale * (args.eta1 * r_t_z + args.eta2 * kappa_z),
        args.intensity_fn,
    )
    phi_t_local = intensity_transform(
        args.intensity_scale * (args.eta1 * r_t_local_z + args.eta2 * kappa_z),
        args.intensity_fn,
    )

    contrib_d = args.alpha * d_t_z
    contrib_s = args.beta * s_t_z
    contrib_u = args.gamma * u_t_z
    contrib_c = args.delta * c_t_z

    contrib_d_local = args.alpha * d_t_z_local
    contrib_s_local = args.beta * s_t_z_local
    contrib_u_local = args.gamma * u_t_z_local
    contrib_c_local = args.delta * c_t_z_local

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_path = out_path / 'desmotic_metrics.csv'
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'step', 'd_t', 's_t', 'entropy', 'influence', 'control_loss',
            'valence', 'valence_local', 'intensity', 'intensity_local', 'r_t', 'r_t_local', 'kappa',
            'd_t_z', 's_t_z', 'u_t_z', 'c_t_z',
            'd_t_z_local', 's_t_z_local', 'u_t_z_local', 'c_t_z_local',
            'contrib_d', 'contrib_s', 'contrib_u', 'contrib_c',
            'contrib_d_local', 'contrib_s_local', 'contrib_u_local', 'contrib_c_local',
        ])
        for i, step in enumerate(steps):
            writer.writerow([
                int(step), float(d_t[i]), float(s_t[i]), float(entropy[i]),
                float(influence[i]), float(c_t[i]), float(valence[i]),
                float(valence_local[i]), float(phi_t[i]), float(phi_t_local[i]),
                float(r_t[i]), float(r_t_local[i]), float(kappa[i]),
                float(d_t_z[i]), float(s_t_z[i]), float(u_t_z[i]), float(c_t_z[i]),
                float(d_t_z_local[i]), float(s_t_z_local[i]), float(u_t_z_local[i]), float(c_t_z_local[i]),
                float(contrib_d[i]), float(contrib_s[i]), float(contrib_u[i]), float(contrib_c[i]),
                float(contrib_d_local[i]), float(contrib_s_local[i]),
                float(contrib_u_local[i]), float(contrib_c_local[i]),
            ])

    def shade(ax):
        if args.patch_start is not None and args.patch_end is not None:
            ax.axvspan(args.patch_start, args.patch_end, color='#F2C94C', alpha=0.25)

    use_local = bool(windows)
    valence_plot = valence_local if use_local else valence
    phi_plot = phi_t_local if use_local else phi_t
    d_plot = d_t_z_local if use_local else d_t_z
    s_plot = s_t_z_local if use_local else s_t_z
    u_plot = u_t_z_local if use_local else u_t_z
    c_plot = c_t_z_local if use_local else c_t_z
    contrib_d_plot = contrib_d_local if use_local else contrib_d
    contrib_s_plot = contrib_s_local if use_local else contrib_s
    contrib_u_plot = contrib_u_local if use_local else contrib_u
    contrib_c_plot = contrib_c_local if use_local else contrib_c

    # Plot 1: valence + intensity
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(steps, valence_plot, color='#1f77b4', label='valence')
    shade(axes[0])
    axes[0].set_ylabel('V_t')
    axes[0].grid(True, alpha=0.2)
    axes[1].plot(steps, phi_plot, color='#d62728', label='intensity')
    shade(axes[1])
    axes[1].set_ylabel('Phi_t')
    axes[1].set_xlabel('step')
    axes[1].grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path / 'desmotic_valence_intensity.png', dpi=150)
    plt.close(fig)

    # Plot 2: standardized components
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(steps, d_plot, label='d_t_z', color='#2ca02c')
    ax.plot(steps, s_plot, label='s_t_z', color='#9467bd')
    ax.plot(steps, u_plot, label='u_t_z', color='#8c564b')
    ax.plot(steps, c_plot, label='c_t_z', color='#ff7f0e')
    shade(ax)
    ax.set_ylabel('standardized component')
    ax.set_xlabel('step')
    ax.grid(True, alpha=0.2)
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path / 'desmotic_components.png', dpi=150)
    plt.close(fig)

    # Plot 3: contributions to distress
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    stack = [
        np.maximum(contrib_d_plot, 0.0),
        np.maximum(contrib_s_plot, 0.0),
        np.maximum(contrib_u_plot, 0.0),
        np.maximum(contrib_c_plot, 0.0),
    ]
    ax.stackplot(
        steps,
        *stack,
        labels=['distance', 'instability', 'uncertainty', 'control'],
        colors=['#2ca02c', '#9467bd', '#8c564b', '#ff7f0e'],
        alpha=0.7,
    )
    ax.plot(steps, -valence_plot, color='#000000', linewidth=1.0, label='-valence')
    shade(ax)
    ax.set_ylabel('contribution to distress')
    ax.set_xlabel('step')
    ax.grid(True, alpha=0.2)
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path / 'desmotic_contributions.png', dpi=150)
    plt.close(fig)

    # Plot 4: distance + control loss
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(steps, d_t, color='#2ca02c', label='distance')
    shade(axes[0])
    axes[0].set_ylabel('d_t')
    axes[0].grid(True, alpha=0.2)
    axes[1].plot(steps, c_t, color='#ff7f0e', label='control_loss')
    shade(axes[1])
    axes[1].set_ylabel('C_t')
    axes[1].set_xlabel('step')
    axes[1].grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path / 'desmotic_distance_control.png', dpi=150)
    plt.close(fig)

    print(f"wrote {csv_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
