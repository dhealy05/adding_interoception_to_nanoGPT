#!/usr/bin/env python3
"""Find which self_state dimensions affect entropy/confidence.

Scans each dimension independently to find:
- Dims where positive values decrease entropy (confident direction)
- Dims where positive values increase entropy (uncertain direction)
"""

import argparse
import os
import sys

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
    return entropy, top1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--dataset', default='shakespeare_char')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--magnitude', type=float, default=0.5)
    args = ap.parse_args()

    model, ckpt = load_model(args.ckpt, args.device)
    block_size = ckpt['model_args']['block_size']
    dim = ckpt['model_args'].get('self_state_dim', 32)
    data_dir = os.path.join(REPO_ROOT, 'data', args.dataset)

    x = get_batch(data_dir, block_size, args.batch_size, args.device, args.seed)

    # Baseline with zero state
    zero_state = torch.zeros(dim)
    base_ent, base_top1 = evaluate(model, x, zero_state, args.device)
    print(f"Baseline (zero state): entropy={base_ent:.4f}, top1={base_top1:.4f}")
    print(f"\nScanning dimensions with magnitude={args.magnitude}...")
    print(f"{'Dim':<5} {'Δ Entropy (pos)':>15} {'Δ Entropy (neg)':>15} {'Effect':>10}")
    print("-" * 50)

    effects = []
    for d in range(dim):
        # Positive direction
        pos_state = torch.zeros(dim)
        pos_state[d] = args.magnitude
        pos_ent, pos_top1 = evaluate(model, x, pos_state, args.device)

        # Negative direction
        neg_state = torch.zeros(dim)
        neg_state[d] = -args.magnitude
        neg_ent, neg_top1 = evaluate(model, x, neg_state, args.device)

        d_pos = pos_ent - base_ent
        d_neg = neg_ent - base_ent

        # Effect: negative d_pos means positive values reduce entropy (confident)
        # positive d_pos means positive values increase entropy (uncertain)
        effects.append((d, d_pos, d_neg))
        print(f"{d:<5} {d_pos:>+15.4f} {d_neg:>+15.4f}", end="")
        if d_pos < -0.01:
            print("  ← confident(+)")
        elif d_pos > 0.01:
            print("  ← uncertain(+)")
        else:
            print()

    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)

    confident_dims = [d for d, d_pos, d_neg in effects if d_pos < -0.01]
    uncertain_dims = [d for d, d_pos, d_neg in effects if d_pos > 0.01]

    print(f"\nConfident dims (positive reduces entropy): {confident_dims}")
    print(f"Uncertain dims (positive increases entropy): {uncertain_dims}")

    # Best single dims
    effects_sorted = sorted(effects, key=lambda x: x[1])
    print(f"\nTop 5 confident (most negative Δ entropy):")
    for d, d_pos, d_neg in effects_sorted[:5]:
        print(f"  dim {d}: Δ={d_pos:+.4f}")

    print(f"\nTop 5 uncertain (most positive Δ entropy):")
    for d, d_pos, d_neg in effects_sorted[-5:][::-1]:
        print(f"  dim {d}: Δ={d_pos:+.4f}")


if __name__ == '__main__':
    main()
