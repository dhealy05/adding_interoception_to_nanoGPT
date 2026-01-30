#!/usr/bin/env python3
"""Summarize memory buffer grid runs under an out_root.

Looks for subdirs matching:
  - mem-ema-decay-{decay}_s{scale}_k{clamp}
  - mem-buf-l{length}_s{stride}

Outputs summary CSV and optional plot comparing EMA vs buffer architectures.
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


EMA_RE = re.compile(r"mem-ema-decay-([0-9.]+)_s([0-9.]+)_k([0-9.]+)")
BUF_RE = re.compile(r"mem-buf-l(\d+)_s(\d+)")


def parse_run(name):
    """Return (mode, params_dict) or None if not recognized."""
    m = EMA_RE.search(name)
    if m:
        return "ema", {
            "decay": float(m.group(1)),
            "scale": float(m.group(2)),
            "clamp_k": float(m.group(3)),
        }
    m = BUF_RE.search(name)
    if m:
        return "buffer", {
            "buffer_len": int(m.group(1)),
            "buffer_stride": int(m.group(2)),
        }
    return None


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
        parsed = parse_run(child.name)
        if parsed is None:
            continue
        mode, params = parsed
        csv_path = child / "regime_deltas.csv"
        if not csv_path.exists():
            continue
        rows = load_regime_deltas(csv_path)
        runs.append((mode, params, child.name, rows))

    if not runs:
        raise SystemExit("no memory runs found with regime_deltas.csv")

    # Sort: EMA first, then buffers by length
    runs.sort(key=lambda x: (
        0 if x[0] == "ema" else 1,
        x[1].get("buffer_len", 0),
        x[1].get("buffer_stride", 0),
    ))

    out_csv = args.out_csv or str(out_root / "memory_buffer_summary.csv")
    Path(os.path.dirname(out_csv) or ".").mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for mode, params, name, rows in runs:
        row = {
            "run": name,
            "mode": mode,
            "mean_abs_self_drift_delta": mean_abs(rows, "self_drift_delta"),
            "mean_abs_state_effect_delta": mean_abs(rows, "state_effect_delta"),
            "mean_abs_update_gate_delta": mean_abs(rows, "update_gate_delta"),
        }
        row.update(params)
        summary_rows.append(row)

    fieldnames = [
        "run", "mode", "buffer_len", "buffer_stride", "decay", "scale", "clamp_k",
        "mean_abs_self_drift_delta", "mean_abs_state_effect_delta", "mean_abs_update_gate_delta",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    if args.plot:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # Group by mode
        ema_runs = [r for r in summary_rows if r["mode"] == "ema"]
        buf_runs = [r for r in summary_rows if r["mode"] == "buffer"]

        metrics = [
            ("mean_abs_self_drift_delta", "Self Drift Δ"),
            ("mean_abs_state_effect_delta", "State Effect Δ"),
            ("mean_abs_update_gate_delta", "Update Gate Δ"),
        ]

        for ax, (key, label) in zip(axes, metrics):
            # Plot EMA as horizontal line
            if ema_runs:
                ema_val = ema_runs[0].get(key)
                if ema_val is not None:
                    ax.axhline(ema_val, color="red", linestyle="--", label="EMA baseline")

            # Plot buffer runs by length (group by stride=10)
            buf_s10 = [r for r in buf_runs if r.get("buffer_stride") == 10]
            if buf_s10:
                buf_s10.sort(key=lambda r: r.get("buffer_len", 0))
                lens = [r.get("buffer_len", 0) for r in buf_s10]
                vals = [r.get(key) for r in buf_s10]
                ax.plot(lens, vals, "o-", color="blue", label="Buffer (stride=10)")

            ax.set_xlabel("Buffer Length")
            ax.set_ylabel(label)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)

        fig.suptitle("Memory Architecture Comparison")
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
