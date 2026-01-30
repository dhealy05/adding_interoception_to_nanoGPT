#!/usr/bin/env python3
"""Evaluate synthetic self_state impact on a fixed batch.

Usage:
  python scripts/analysis/eval_self_state.py \
    --ckpt out/self-analysis/ckpt.pt \
    --state out/self-analysis/self_state_confident.pt \
    --dataset shakespeare_char
"""

import argparse
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--state', required=True)
    ap.add_argument('--dataset', default='shakespeare_char')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--seed', type=int, default=123)
    args = ap.parse_args()

    model, ckpt = load_model(args.ckpt, args.device)
    block_size = ckpt['model_args']['block_size']
    data_dir = os.path.join(REPO_ROOT, 'data', args.dataset)

    x = get_batch(data_dir, block_size, args.batch_size, args.device, args.seed)
    state = torch.load(args.state, map_location=args.device).float()
    zero_state = torch.zeros_like(state)

    with torch.no_grad():
        logits_state, _ = model(x, self_state=state)
        logits_zero, _ = model(x, self_state=zero_state)
        logits_state = logits_state[:, -1, :]
        logits_zero = logits_zero[:, -1, :]

    # effect metrics
    abs_diff = (logits_state - logits_zero).abs().mean().item()
    p0 = torch.softmax(logits_zero, dim=-1)
    p1 = torch.softmax(logits_state, dim=-1)
    kl = (p0 * (torch.log(p0 + 1e-9) - torch.log(p1 + 1e-9))).sum(dim=-1).mean().item()
    ent0 = -(p0 * torch.log(p0 + 1e-9)).sum(dim=-1).mean().item()
    ent1 = -(p1 * torch.log(p1 + 1e-9)).sum(dim=-1).mean().item()
    top0 = p0.max(dim=-1).values.mean().item()
    top1 = p1.max(dim=-1).values.mean().item()

    print(f"abs_logit_diff={abs_diff:.6f}")
    print(f"kl(p0||p1)={kl:.6f}")
    print(f"entropy_zero={ent0:.6f} entropy_state={ent1:.6f}")
    print(f"top1_zero={top0:.6f} top1_state={top1:.6f}")


if __name__ == '__main__':
    main()
