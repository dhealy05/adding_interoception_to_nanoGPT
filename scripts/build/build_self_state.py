#!/usr/bin/env python3
"""Build a synthetic self_state from presets.

Usage:
  python scripts/build/build_self_state.py \
    --ckpt out/self-analysis/ckpt.pt \
    --dump_dir out/self-analysis \
    --config configs/self_state_presets.json \
    --preset confident \
    --out out/self-analysis/self_state_confident.pt
"""

import argparse
import json
import os
import glob
import re

import numpy as np
import torch

ITER_RE = re.compile(r"self_state_(\d+)\.pt")


def load_dump_minmax(dump_dir):
    paths = sorted(glob.glob(os.path.join(dump_dir, "self_state_*.pt")))
    if not paths:
        raise SystemExit("No self_state_*.pt files found in dump_dir")
    states = []
    for p in paths:
        state = torch.load(p, map_location='cpu').float().numpy()
        states.append(state)
    states = np.stack(states, axis=0)
    return states.min(axis=0), states.max(axis=0)


def load_base_state(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'self_state' not in ckpt:
        raise SystemExit("Checkpoint missing self_state")
    return ckpt['self_state'].float()


def apply_ops(state, ops, dim_min, dim_max):
    for op in ops:
        op_name = op['op']
        if op_name == 'set_all':
            val = op['value']
            state[:] = float(val)
            continue
        dims = op.get('dims', [])
        if not dims:
            continue
        raw_val = op.get('value', 0.0)
        if raw_val == 'min':
            vals = dim_min
        elif raw_val == 'max':
            vals = dim_max
        else:
            vals = float(raw_val)
        if op_name == 'set':
            for d in dims:
                state[d] = float(vals[d]) if isinstance(vals, np.ndarray) else float(vals)
        elif op_name == 'add':
            for d in dims:
                state[d] += float(vals[d]) if isinstance(vals, np.ndarray) else float(vals)
        elif op_name == 'scale':
            for d in dims:
                state[d] *= float(vals[d]) if isinstance(vals, np.ndarray) else float(vals)
        else:
            raise ValueError(f"unknown op: {op_name}")
    return state


def clip_state(state, dim_min, dim_max, mode):
    if mode == 'minmax':
        return torch.max(torch.min(state, torch.tensor(dim_max)), torch.tensor(dim_min))
    if mode in (None, '', 'none'):
        return state
    raise ValueError(f"unknown clip mode: {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--dump_dir', required=True)
    ap.add_argument('--config', required=True)
    ap.add_argument('--preset', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        cfg = json.load(f)

    presets = cfg.get('presets', {})
    if args.preset not in presets:
        raise SystemExit(f"Preset not found: {args.preset}")

    defaults = cfg.get('defaults', {})
    base_mode = presets[args.preset].get('base', defaults.get('base', 'checkpoint'))
    clip_mode = presets[args.preset].get('clip', defaults.get('clip', 'minmax'))

    dim_min, dim_max = load_dump_minmax(args.dump_dir)

    if base_mode == 'checkpoint':
        state = load_base_state(args.ckpt)
    elif base_mode == 'zero':
        state = torch.zeros_like(torch.tensor(dim_min))
    elif base_mode == 'mean':
        # mean of dumps
        paths = sorted(glob.glob(os.path.join(args.dump_dir, "self_state_*.pt")))
        states = [torch.load(p, map_location='cpu').float() for p in paths]
        state = torch.stack(states, dim=0).mean(dim=0)
    else:
        raise ValueError(f"unknown base: {base_mode}")

    state = apply_ops(state.clone(), presets[args.preset].get('ops', []), dim_min, dim_max)
    state = clip_state(state, dim_min, dim_max, clip_mode)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(state.cpu(), args.out)
    print(f"wrote {args.out}")


if __name__ == '__main__':
    main()
