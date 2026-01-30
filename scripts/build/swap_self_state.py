#!/usr/bin/env python3
"""Swap self_state and self_stats_prev between two checkpoints.

Usage:
  python scripts/build/swap_self_state.py --ckpt_a path/to/a/ckpt.pt --ckpt_b path/to/b/ckpt.pt \
    --out_a out/swap-a-swapped --out_b out/swap-b-swapped
"""

import argparse
import os

import torch


def swap_fields(ckpt_a, ckpt_b, fields):
    for field in fields:
        if field not in ckpt_a or field not in ckpt_b:
            raise KeyError(f"missing {field} in checkpoints")
        ckpt_a[field], ckpt_b[field] = ckpt_b[field], ckpt_a[field]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt_a', required=True)
    ap.add_argument('--ckpt_b', required=True)
    ap.add_argument('--out_a', required=True)
    ap.add_argument('--out_b', required=True)
    args = ap.parse_args(argv)

    ckpt_a = torch.load(args.ckpt_a, map_location='cpu')
    ckpt_b = torch.load(args.ckpt_b, map_location='cpu')

    swap_fields(ckpt_a, ckpt_b, ['self_state', 'self_stats_prev'])

    os.makedirs(args.out_a, exist_ok=True)
    os.makedirs(args.out_b, exist_ok=True)

    torch.save(ckpt_a, os.path.join(args.out_a, 'ckpt.pt'))
    torch.save(ckpt_b, os.path.join(args.out_b, 'ckpt.pt'))

    print(f"wrote {os.path.join(args.out_a, 'ckpt.pt')}")
    print(f"wrote {os.path.join(args.out_b, 'ckpt.pt')}")
    return 0


if __name__ == '__main__':
    import sys
    raise SystemExit(main(sys.argv[1:]))
