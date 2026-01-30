#!/usr/bin/env python3
"""Analyze persistent self-state dumps and logs.

Usage:
  python scripts/analysis/self_state_analysis.py --out_dir out-dir \
    --log /path/to/log \
    --ckpt /path/to/ckpt.pt

Expects:
  - self_state_<iter>.pt files in out_dir (saved by train.py)
  - optional self_stats_<iter>.pt files in out_dir (loss, grad_norm, step_frac, entropy, top1_conf)

Outputs:
  - self_state_dim_corr.csv (per-dim correlations)
  - self_state_dim_effect.csv (per-dim causal effects if --ckpt provided)
  - self_state_clusters.csv (k-means labels if sklearn available)
"""

import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch

try:
    from sklearn.cluster import KMeans
except Exception:
    KMeans = None

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from src.model import GPT, GPTConfig

ITER_RE = re.compile(r"self_state_(\d+)\.pt")
STAT_RE = re.compile(r"self_stats_(\d+)\.pt")
LOG_ITER_RE = re.compile(r"^iter (\d+):")
LOG_LOSS_RE = re.compile(r"^iter (\d+): loss ([0-9.]+)")
LOG_SELF_RE = re.compile(r"^self_norm=([0-9.]+), self_delta=([0-9.]+), self_drift=([0-9.]+)")

STAT_FIELDS = ['loss', 'grad_norm', 'step_frac', 'entropy', 'top1_conf']


def load_states(out_dir):
    paths = sorted(glob.glob(os.path.join(out_dir, "self_state_*.pt")))
    if not paths:
        raise SystemExit("No self_state_*.pt files found in out_dir")
    steps = []
    states = []
    for p in paths:
        m = ITER_RE.search(os.path.basename(p))
        if not m:
            continue
        step = int(m.group(1))
        state = torch.load(p, map_location='cpu').float().numpy()
        steps.append(step)
        states.append(state)
    steps = np.array(steps)
    states = np.stack(states, axis=0)
    return steps, states


def load_stats(stats_dir):
    paths = sorted(glob.glob(os.path.join(stats_dir, "self_stats_*.pt")))
    if not paths:
        return None, None
    steps = []
    stats = []
    for p in paths:
        m = STAT_RE.search(os.path.basename(p))
        if not m:
            continue
        step = int(m.group(1))
        vec = torch.load(p, map_location='cpu').float().numpy()
        steps.append(step)
        stats.append(vec)
    steps = np.array(steps)
    stats = np.stack(stats, axis=0)
    return steps, stats


def parse_log(log_path):
    if not log_path:
        return {}
    cur = None
    stats = defaultdict(dict)
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            m = LOG_ITER_RE.match(line)
            if m:
                cur = int(m.group(1))
                continue
            if cur is None:
                continue
            m = LOG_LOSS_RE.match(line)
            if m:
                stats[cur]['loss'] = float(m.group(2))
                continue
            m = LOG_SELF_RE.match(line)
            if m:
                stats[cur]['self_norm'] = float(m.group(1))
                stats[cur]['self_delta'] = float(m.group(2))
                stats[cur]['self_drift'] = float(m.group(3))
    return stats


def write_csv(path, header, rows):
    with open(path, 'w') as f:
        f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join(str(x) for x in row) + '\n')


def corrcoef(x, y):
    if len(x) < 2:
        return 0.0
    x = np.asarray(x)
    y = np.asarray(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def align_steps(steps_a, arr_a, steps_b, arr_b):
    idx_b = {int(s): i for i, s in enumerate(steps_b)}
    keep = [i for i, s in enumerate(steps_a) if int(s) in idx_b]
    if not keep:
        return None, None, None
    aligned_a = arr_a[keep]
    aligned_b = np.stack([arr_b[idx_b[int(steps_a[i])]] for i in keep], axis=0)
    aligned_steps = steps_a[keep]
    return aligned_steps, aligned_a, aligned_b


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model_args = ckpt['model_args']
    cfg = GPTConfig(**model_args)
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


def effect_metric(base, other, metric):
    if metric == 'abs':
        return float((other - base).abs().mean().item())
    if metric == 'kl':
        base_logp = torch.log_softmax(base, dim=-1)
        other_logp = torch.log_softmax(other, dim=-1)
        base_p = base_logp.exp()
        kl = (base_p * (base_logp - other_logp)).sum(dim=-1).mean()
        return float(kl.item())
    raise ValueError(f"unknown metric: {metric}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--log', default='')
    ap.add_argument('--stats_dir', default='')
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--ckpt', default='')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--effect_batch_size', type=int, default=8)
    ap.add_argument('--effect_seed', type=int, default=123)
    ap.add_argument('--effect_metric', choices=['abs', 'kl'], default='abs')
    args = ap.parse_args()

    if not args.stats_dir:
        args.stats_dir = args.out_dir

    steps, states = load_states(args.out_dir)
    stats_steps, stats_mat = load_stats(args.stats_dir)
    log_stats = parse_log(args.log)

    # correlations per dimension
    corr_rows = []
    if stats_mat is not None:
        aligned_steps, aligned_states, aligned_stats = align_steps(steps, states, stats_steps, stats_mat)
    else:
        aligned_steps = None
        aligned_states = None
        aligned_stats = None

    for d in range(states.shape[1]):
        if aligned_stats is not None:
            dim_vals = aligned_states[:, d]
            corr_vals = [corrcoef(dim_vals, aligned_stats[:, i]) for i in range(aligned_stats.shape[1])]
        else:
            dim_vals = states[:, d]
            loss_vals = [log_stats.get(int(s), {}).get('loss', np.nan) for s in steps]
            loss_vals = np.array(loss_vals)
            mask = ~np.isnan(loss_vals)
            corr_loss = corrcoef(dim_vals[mask], loss_vals[mask]) if mask.any() else 0.0
            corr_vals = [corr_loss, 0.0, 0.0, 0.0, 0.0]
        corr_rows.append((d, *corr_vals))

    out_corr = os.path.join(args.out_dir, 'self_state_dim_corr.csv')
    write_csv(out_corr, ['dim'] + [f"corr_{f}" for f in STAT_FIELDS], corr_rows)

    # k-means clustering (optional)
    out_labels = None
    if KMeans is not None:
        k = max(2, args.k)
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = km.fit_predict(states)
        out_labels = os.path.join(args.out_dir, 'self_state_clusters.csv')
        write_csv(out_labels, ['step', 'cluster'], zip(steps, labels))

    # causal effect analysis (optional)
    if args.ckpt:
        model, ckpt = load_model(args.ckpt, args.device)
        model_args = ckpt['model_args']
        block_size = model_args['block_size']
        dataset = ckpt.get('config', {}).get('dataset', 'shakespeare_char')
        data_dir = os.path.join(REPO_ROOT, 'data', dataset)
        x = get_batch(data_dir, block_size, args.effect_batch_size, args.device, args.effect_seed)

        if 'self_state' in ckpt:
            base_state = ckpt['self_state'].to(args.device).float()
        else:
            base_state = torch.tensor(states.mean(axis=0), device=args.device)

        dim_min = states.min(axis=0)
        dim_max = states.max(axis=0)

        with torch.no_grad():
            logits_base, _ = model(x, self_state=base_state)
            logits_base = logits_base[:, -1, :]

        effect_rows = []
        for d in range(states.shape[1]):
            state_min = base_state.clone()
            state_min[d] = float(dim_min[d])
            state_max = base_state.clone()
            state_max[d] = float(dim_max[d])
            with torch.no_grad():
                logits_min, _ = model(x, self_state=state_min)
                logits_max, _ = model(x, self_state=state_max)
                logits_min = logits_min[:, -1, :]
                logits_max = logits_max[:, -1, :]
            eff_min = effect_metric(logits_base, logits_min, args.effect_metric)
            eff_max = effect_metric(logits_base, logits_max, args.effect_metric)
            eff_peak = max(eff_min, eff_max)
            effect_rows.append((d, float(dim_min[d]), float(dim_max[d]), eff_min, eff_max, eff_peak))

        out_effect = os.path.join(args.out_dir, 'self_state_dim_effect.csv')
        write_csv(out_effect, ['dim', 'min', 'max', 'effect_min', 'effect_max', 'effect_peak'], effect_rows)

    print(f"wrote {out_corr}")
    if out_labels:
        print(f"wrote {out_labels}")
    if args.ckpt:
        print(f"wrote {out_effect}")
    if KMeans is None:
        print("sklearn not available; skipping clustering")


if __name__ == '__main__':
    main()
