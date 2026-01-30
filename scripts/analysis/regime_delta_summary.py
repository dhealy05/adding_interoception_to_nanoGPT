#!/usr/bin/env python3
"""Summarize per-regime metric deltas from a reaction_log.jsonl.

Usage:
  python scripts/analysis/regime_delta_summary.py \
    --reaction_log out/run/reaction_log.jsonl \
    --out_csv out/run/regime_deltas.csv \
    --baseline_len 100 \
    --plot out/run/regime_delta_scatter.png
"""

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_METRICS = [
    "loss",
    "stats_entropy",
    "stats_top1_conf",
    "self_drift",
    "state_effect",
]


def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    records.sort(key=lambda r: r.get("step", 0))
    return records


def build_windows(records):
    by_step = {r["step"]: r for r in records}
    steps = sorted(by_step)
    regime_ids = sorted({rid for r in records for rid in r.get("regimes", [])})
    windows = {}
    for rid in regime_ids:
        rsteps = [s for s in steps if rid in by_step[s].get("regimes", [])]
        if rsteps:
            windows[rid] = (min(rsteps), max(rsteps) + 1)
    return windows, by_step, steps


def mean_over(by_step, steps, key):
    vals = [by_step[s].get(key) for s in steps if by_step[s].get(key) is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reaction_log", required=True)
    ap.add_argument("--out_csv", default="")
    ap.add_argument("--baseline_len", type=int, default=100)
    ap.add_argument("--metrics", default="")
    ap.add_argument("--plot", default="")
    ap.add_argument("--title", default="")
    args = ap.parse_args(argv)

    records = load_records(args.reaction_log)
    if not records:
        raise SystemExit("no records found")

    windows, by_step, steps = build_windows(records)
    if not windows:
        raise SystemExit("no regimes found in log")

    baseline_steps = [s for s in steps if not by_step[s].get("regimes")]

    metrics = DEFAULT_METRICS
    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    out_csv = args.out_csv
    if not out_csv:
        out_csv = os.path.splitext(args.reaction_log)[0] + "_regime_deltas.csv"
    Path(os.path.dirname(out_csv) or ".").mkdir(parents=True, exist_ok=True)

    rows = []
    for rid, (start, end) in windows.items():
        prior = [s for s in baseline_steps if s < start]
        if args.baseline_len > 0:
            prior = prior[-args.baseline_len:]
        during = [s for s in steps if start <= s < end and rid in by_step[s].get("regimes", [])]
        row = {
            "regime": rid,
            "start": start,
            "end": end,
            "n_baseline": len(prior),
            "n_during": len(during),
        }
        for m in metrics:
            base_mean = mean_over(by_step, prior, m)
            during_mean = mean_over(by_step, during, m)
            delta = None
            if base_mean is not None and during_mean is not None:
                delta = during_mean - base_mean
            row[f"{m}_baseline"] = base_mean
            row[f"{m}_during"] = during_mean
            row[f"{m}_delta"] = delta
        rows.append(row)

    # write csv
    fieldnames = ["regime", "start", "end", "n_baseline", "n_during"]
    for m in metrics:
        fieldnames += [f"{m}_baseline", f"{m}_during", f"{m}_delta"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # optional scatter plot (Δentropy vs Δloss)
    if args.plot:
        xs = []
        ys = []
        labels = []
        for row in rows:
            dx = row.get("loss_delta")
            dy = row.get("stats_entropy_delta")
            if dx is None or dy is None:
                continue
            xs.append(dx)
            ys.append(dy)
            labels.append(row["regime"])
        if xs and ys:
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(xs, ys, s=40)
            for x, y, lab in zip(xs, ys, labels):
                ax.text(x, y, lab, fontsize=8, ha="left", va="bottom")
            ax.axhline(0, color="gray", linewidth=0.8, alpha=0.5)
            ax.axvline(0, color="gray", linewidth=0.8, alpha=0.5)
            ax.set_xlabel("Δ loss")
            ax.set_ylabel("Δ entropy")
            if args.title:
                ax.set_title(args.title)
            fig.tight_layout()
            Path(os.path.dirname(args.plot) or ".").mkdir(parents=True, exist_ok=True)
            fig.savefig(args.plot, dpi=150)
            plt.close(fig)

    print(f"wrote {out_csv}")
    if args.plot:
        print(f"wrote {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
