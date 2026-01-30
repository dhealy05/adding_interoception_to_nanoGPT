#!/usr/bin/env python3
"""Compare stability/control metrics between two runs.

Metrics:
  - self_drift mean/std within windows
  - drift spike frequency (threshold from baseline window)
  - curvature (2nd-diff norm) from self_state dumps
  - state_effect mean/min and recovery slopes within windows
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


def load_reaction_log(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    records.sort(key=lambda r: r.get("step", 0))
    return records


def series_in_window(records, key, start, end):
    vals = []
    steps = []
    for rec in records:
        step = rec.get("step")
        if step is None or step < start or step > end:
            continue
        v = rec.get(key)
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        steps.append(step)
        vals.append(float(v))
    return np.array(steps, dtype=np.int64), np.array(vals, dtype=np.float64)


def load_states(state_dir):
    state_dir = Path(state_dir)
    states = {}
    for path in state_dir.glob("self_state_*.pt"):
        try:
            step = int(path.stem.split("_")[-1])
        except Exception:
            continue
        states[step] = path
    return dict(sorted(states.items()))


def curvature_from_states(states, start, end):
    steps = [s for s in states.keys() if start <= s <= end]
    steps.sort()
    if len(steps) < 3:
        return None
    curvatures = []
    prev2 = None
    prev1 = None
    for step in steps:
        vec = torch.load(states[step], map_location="cpu").float().view(-1)
        if prev2 is not None:
            curv = (vec - 2 * prev1 + prev2).norm().item()
            curvatures.append(curv)
        prev2 = prev1
        prev1 = vec
    if not curvatures:
        return None
    arr = np.array(curvatures, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p95": float(np.percentile(arr, 95)),
        "n": int(arr.size),
    }


def recovery_slope(steps, vals):
    if len(vals) < 2:
        return float("nan")
    min_idx = int(np.argmin(vals))
    if min_idx == len(vals) - 1:
        return float("nan")
    return float((vals[-1] - vals[min_idx]) / (steps[-1] - steps[min_idx]))


def linear_slope(steps, vals):
    if len(vals) < 2:
        return float("nan")
    x = steps.astype(np.float64)
    y = vals.astype(np.float64)
    coef = np.polyfit(x, y, deg=1)
    return float(coef[0])


def summarize_run(records, states, windows, baseline, spike_k):
    out = {}
    # baseline for drift spikes
    base_steps, base_drift = series_in_window(records, "self_drift", baseline[0], baseline[1])
    base_mean = float(base_drift.mean()) if base_drift.size else float("nan")
    base_std = float(base_drift.std()) if base_drift.size else float("nan")
    spike_thresh = base_mean + spike_k * base_std if base_drift.size else float("nan")
    out["baseline_drift_mean"] = base_mean
    out["baseline_drift_std"] = base_std
    out["drift_spike_threshold"] = spike_thresh

    for label, (start, end) in windows.items():
        w = {}
        steps, drift = series_in_window(records, "self_drift", start, end)
        if drift.size:
            w["self_drift_mean"] = float(drift.mean())
            w["self_drift_std"] = float(drift.std())
            if not math.isnan(spike_thresh):
                spikes = int(np.sum(drift > spike_thresh))
                w["drift_spikes"] = spikes
                w["drift_spike_freq"] = float(spikes / drift.size)
            else:
                w["drift_spikes"] = None
                w["drift_spike_freq"] = None
        else:
            w["self_drift_mean"] = None
            w["self_drift_std"] = None
            w["drift_spikes"] = None
            w["drift_spike_freq"] = None

        # curvature
        if states is not None:
            curv = curvature_from_states(states, start, end)
            w["curvature"] = curv
        else:
            w["curvature"] = None

        # state_effect dynamics
        s_steps, s_effect = series_in_window(records, "state_effect", start, end)
        if s_effect.size:
            w["state_effect_mean"] = float(s_effect.mean())
            w["state_effect_min"] = float(s_effect.min())
            w["state_effect_recovery_slope"] = recovery_slope(s_steps, s_effect)
            w["state_effect_linear_slope"] = linear_slope(s_steps, s_effect)
        else:
            w["state_effect_mean"] = None
            w["state_effect_min"] = None
            w["state_effect_recovery_slope"] = None
            w["state_effect_linear_slope"] = None

        out[label] = w
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_a_log", required=True)
    ap.add_argument("--run_b_log", required=True)
    ap.add_argument("--run_a_state_dir", default="")
    ap.add_argument("--run_b_state_dir", default="")
    ap.add_argument("--baseline_start", type=int, default=800)
    ap.add_argument("--baseline_end", type=int, default=999)
    ap.add_argument("--amb_start", type=int, default=1000)
    ap.add_argument("--amb_end", type=int, default=1150)
    ap.add_argument("--future_start", type=int, default=1150)
    ap.add_argument("--future_end", type=int, default=1500)
    ap.add_argument("--spike_k", type=float, default=2.0)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args(argv)

    rec_a = load_reaction_log(args.run_a_log)
    rec_b = load_reaction_log(args.run_b_log)
    states_a = load_states(args.run_a_state_dir or Path(args.run_a_log).parent)
    states_b = load_states(args.run_b_state_dir or Path(args.run_b_log).parent)

    windows = {
        "ambiguous": (args.amb_start, args.amb_end),
        "future": (args.future_start, args.future_end),
    }
    baseline = (args.baseline_start, args.baseline_end)

    out = {
        "run_a": {
            "log": args.run_a_log,
            "state_dir": args.run_a_state_dir or str(Path(args.run_a_log).parent),
            "metrics": summarize_run(rec_a, states_a, windows, baseline, args.spike_k),
        },
        "run_b": {
            "log": args.run_b_log,
            "state_dir": args.run_b_state_dir or str(Path(args.run_b_log).parent),
            "metrics": summarize_run(rec_b, states_b, windows, baseline, args.spike_k),
        },
    }

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
