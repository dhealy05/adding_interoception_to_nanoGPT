#!/usr/bin/env python3
"""Compare gate behavior in an ambiguous window between two runs.

Reports mean/std and AUC using only gate values in the specified window.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_gate_values(path, start, end, key):
    vals = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            step = rec.get("step")
            if step is None or step < start or step > end:
                continue
            v = rec.get(key)
            if v is None:
                continue
            if isinstance(v, float) and math.isnan(v):
                continue
            vals.append(float(v))
    return vals


def auc_from_scores(pos, neg):
    # Rank-based AUC (equivalent to Mann-Whitney U).
    scores = np.array(pos + neg, dtype=np.float64)
    labels = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int64)
    order = np.argsort(scores, kind="mergesort")
    scores_sorted = scores[order]
    labels_sorted = labels[order]
    ranks = np.zeros_like(scores_sorted, dtype=np.float64)
    i = 0
    rank = 1
    n = len(scores_sorted)
    while i < n:
        j = i
        while j + 1 < n and scores_sorted[j + 1] == scores_sorted[i]:
            j += 1
        avg_rank = (rank + rank + (j - i)) / 2.0
        ranks[i : j + 1] = avg_rank
        rank += (j - i + 1)
        i = j + 1
    pos_rank_sum = ranks[labels_sorted == 1].sum()
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def summary(vals):
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    arr = np.array(vals, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": len(vals)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_a", required=True, help="reaction_log.jsonl for run A")
    ap.add_argument("--run_b", required=True, help="reaction_log.jsonl for run B")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--key", default="self_update_gate")
    ap.add_argument("--out_json", default="")
    args = ap.parse_args(argv)

    a_vals = load_gate_values(args.run_a, args.start, args.end, args.key)
    b_vals = load_gate_values(args.run_b, args.start, args.end, args.key)

    auc = auc_from_scores(a_vals, b_vals)
    a_sum = summary(a_vals)
    b_sum = summary(b_vals)

    out = {
        "window": {"start": args.start, "end": args.end},
        "key": args.key,
        "run_a": {"path": args.run_a, **a_sum},
        "run_b": {"path": args.run_b, **b_sum},
        "auc": auc,
    }

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
