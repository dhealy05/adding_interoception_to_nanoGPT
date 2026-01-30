#!/usr/bin/env python3
"""Summarize memory decay sweep runs under an out_root.

Looks for subdirs named mem-decay-*_s*_k* and mem-off with regime_deltas.csv.
Outputs summary CSV and optional plot.
"""

import argparse
import csv
import os
import re
from pathlib import Path

import matplotlib

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MEM_RE = re.compile(r"mem-decay-([0-9.]+)(?:_s([0-9.]+))?(?:_k([0-9.]+))?")


def parse_decay(name):
    m = MEM_RE.search(name)
    if not m:
        return None
    try:
        decay = float(m.group(1))
    except Exception:
        return None
    scale = None
    clamp = None
    if m.group(2) is not None:
        try:
            scale = float(m.group(2))
        except Exception:
            scale = None
    if m.group(3) is not None:
        try:
            clamp = float(m.group(3))
        except Exception:
            clamp = None
    return decay, scale, clamp


def load_regime_deltas(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def mean_abs(rows, key):
    vals = []
    for row in rows:
        v = row.get(key)
        if v in (None, ""):
            continue
        try:
            vals.append(abs(float(v)))
        except Exception:
            continue
    return sum(vals) / len(vals) if vals else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--out_csv", default="")
    ap.add_argument("--plot", default="")
    args = ap.parse_args(argv)

    out_root = Path(args.out_root)
    runs = []
    for child in out_root.iterdir():
        if not child.is_dir():
            continue
        if child.name == "mem-off":
            decay = None
            scale = None
            clamp = None
        else:
            parsed = parse_decay(child.name)
            if parsed is None:
                continue
            decay, scale, clamp = parsed
        if decay is None and child.name != "mem-off":
            continue
        csv_path = child / "regime_deltas.csv"
        if not csv_path.exists():
            continue
        rows = load_regime_deltas(csv_path)
        runs.append((decay, scale, clamp, child.name, rows))

    if not runs:
        raise SystemExit("no memory runs found with regime_deltas.csv")

    runs.sort(key=lambda x: (
        x[0] is None,
        x[0] if x[0] is not None else -1,
        x[1] if x[1] is not None else -1,
        x[2] if x[2] is not None else -1,
    ))
    out_csv = args.out_csv or str(out_root / "memory_decay_summary.csv")
    Path(os.path.dirname(out_csv) or ".").mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for decay, scale, clamp, name, rows in runs:
        summary_rows.append({
            "run": name,
            "memory_decay": decay if decay is not None else "off",
            "memory_scale": scale if scale is not None else "",
            "memory_clamp_k": clamp if clamp is not None else "",
            "mean_abs_self_drift_delta": mean_abs(rows, "self_drift_delta"),
            "mean_abs_state_effect_delta": mean_abs(rows, "state_effect_delta"),
        })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "memory_decay",
                "memory_scale",
                "memory_clamp_k",
                "mean_abs_self_drift_delta",
                "mean_abs_state_effect_delta",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    if args.plot:
        combos = {}
        for row in summary_rows:
            if row["memory_decay"] == "off":
                continue
            try:
                d = float(row["memory_decay"])
            except Exception:
                continue
            scale = row.get("memory_scale")
            clamp = row.get("memory_clamp_k")
            combo_key = (scale, clamp)
            combos.setdefault(combo_key, []).append((d, row))

        fig, axes = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
        for combo_key, items in combos.items():
            items.sort(key=lambda x: x[0])
            decays = [d for d, _ in items]
            drift = [it["mean_abs_self_drift_delta"] for _, it in items]
            effect = [it["mean_abs_state_effect_delta"] for _, it in items]
            label = f"s={combo_key[0]}, k={combo_key[1]}"
            axes[0].plot(decays, drift, marker="o", label=label)
            axes[1].plot(decays, effect, marker="o", label=label)
        axes[0].set_ylabel("mean |Δself_drift|")
        axes[1].set_ylabel("mean |Δstate_effect|")
        axes[1].set_xlabel("self_memory_decay (1/τ)")
        for ax in axes:
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=8)
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
