#!/usr/bin/env python3
"""Analyze state_effect/self_drift as early warning signals.

Outputs JSON summary and optional lead-lag plot.

Usage:
  python scripts/analysis/state_effect_monitoring.py \
    --reaction_log out/run/reaction_log.jsonl \
    --out_json out/run/state_effect_monitoring.json \
    --plot out/run/state_effect_leadlag.png
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

import matplotlib

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def pearson(x, y):
    if len(x) < 2:
        return 0.0
    x = np.asarray(x)
    y = np.asarray(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 2:
        return 0.0
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return pearson(rx, ry)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reaction_log", required=True)
    ap.add_argument("--out_json", default="")
    ap.add_argument("--plot", default="")
    ap.add_argument("--max_lag", type=int, default=50)
    ap.add_argument("--loss_k", type=float, default=2.0)
    ap.add_argument("--entropy_k", type=float, default=2.0)
    ap.add_argument("--effect_k", type=float, default=1.0)
    ap.add_argument("--lead_window", type=int, default=50)
    ap.add_argument("--recovery_hold", type=int, default=5)
    args = ap.parse_args(argv)

    records = load_records(args.reaction_log)
    if not records:
        raise SystemExit("no records found")

    windows, by_step, steps = build_windows(records)
    baseline_steps = [s for s in steps if not by_step[s].get("regimes")]
    if not baseline_steps:
        baseline_steps = steps[: max(1, len(steps) // 5)]

    # Build arrays
    loss = {s: by_step[s].get("loss") for s in steps}
    entropy = {s: by_step[s].get("stats_entropy") for s in steps}
    state_effect = {s: by_step[s].get("state_effect") for s in steps}
    self_drift = {s: by_step[s].get("self_drift") for s in steps}

    def baseline_stats(series):
        vals = [series[s] for s in baseline_steps if series[s] is not None]
        if not vals:
            return None, None
        return float(np.mean(vals)), float(np.std(vals) + 1e-8)

    loss_mu, loss_std = baseline_stats(loss)
    ent_mu, ent_std = baseline_stats(entropy)
    eff_mu, eff_std = baseline_stats(state_effect)
    drift_mu, drift_std = baseline_stats(self_drift)

    # Lead-lag correlations: corr(state_effect_t, loss_{t+lag})
    lags = range(-args.max_lag, args.max_lag + 1)
    corr_loss = []
    corr_entropy = []
    for lag in lags:
        xs_loss = []
        ys_loss = []
        xs_ent = []
        ys_ent = []
        for s in steps:
            s2 = s + lag
            if s2 not in by_step:
                continue
            if state_effect[s] is None:
                continue
            if loss[s2] is not None:
                xs_loss.append(state_effect[s])
                ys_loss.append(loss[s2])
            if entropy[s2] is not None:
                xs_ent.append(state_effect[s])
                ys_ent.append(entropy[s2])
        corr_loss.append(pearson(xs_loss, ys_loss) if xs_loss and ys_loss else 0.0)
        corr_entropy.append(pearson(xs_ent, ys_ent) if xs_ent and ys_ent else 0.0)

    # Event prediction: state_effect drops predict future loss/entropy spikes
    def event_metrics(series, mu, std, threshold, lead_window):
        if mu is None or std is None:
            return None
        events = [s for s in steps if series[s] is not None and series[s] > threshold]
        if not events:
            return {"n_events": 0}
        return events

    loss_thresh = None if loss_mu is None else loss_mu + args.loss_k * loss_std
    ent_thresh = None if ent_mu is None else ent_mu + args.entropy_k * ent_std
    eff_drop = None if eff_mu is None else eff_mu - args.effect_k * eff_std

    loss_spikes = set()
    if loss_thresh is not None:
        loss_spikes = {s for s in steps if loss[s] is not None and loss[s] > loss_thresh}
    ent_spikes = set()
    if ent_thresh is not None:
        ent_spikes = {s for s in steps if entropy[s] is not None and entropy[s] > ent_thresh}

    effect_drops = set()
    if eff_drop is not None:
        effect_drops = {s for s in steps if state_effect[s] is not None and state_effect[s] < eff_drop}

    def precision_recall(pred_steps, event_steps, lead_window):
        if not pred_steps:
            return {"precision": 0.0, "recall": 0.0, "n_pred": 0, "n_event": len(event_steps)}
        # precision: fraction of preds that are followed by event within window
        hits = 0
        for s in pred_steps:
            for t in range(s, s + lead_window + 1):
                if t in event_steps:
                    hits += 1
                    break
        precision = hits / len(pred_steps)
        # recall: fraction of events preceded by pred within window
        hits_r = 0
        for e in event_steps:
            for t in range(e - lead_window, e + 1):
                if t in pred_steps:
                    hits_r += 1
                    break
        recall = hits_r / len(event_steps) if event_steps else 0.0
        return {
            "precision": float(precision),
            "recall": float(recall),
            "n_pred": len(pred_steps),
            "n_event": len(event_steps),
        }

    pred_metrics = {
        "loss_spike": precision_recall(effect_drops, loss_spikes, args.lead_window),
        "entropy_spike": precision_recall(effect_drops, ent_spikes, args.lead_window),
    }

    # Recovery time analysis
    recovery = []
    if windows and loss_mu is not None and loss_std is not None:
        for rid, (start, end) in windows.items():
            # average drift during regime
            drift_vals = [self_drift[s] for s in steps if start <= s < end and self_drift[s] is not None]
            eff_vals = [state_effect[s] for s in steps if start <= s < end and state_effect[s] is not None]
            avg_drift = float(np.mean(drift_vals)) if drift_vals else None
            avg_eff = float(np.mean(eff_vals)) if eff_vals else None
            # recovery time: first step after end where loss within baseline band for hold window
            rec_time = None
            for i, s in enumerate([t for t in steps if t >= end]):
                window_steps = [t for t in steps if s <= t < s + args.recovery_hold]
                if not window_steps:
                    break
                ok = True
                for t in window_steps:
                    if loss[t] is None:
                        ok = False
                        break
                    if not (loss_mu - loss_std <= loss[t] <= loss_mu + loss_std):
                        ok = False
                        break
                if ok:
                    rec_time = s - end
                    break
            recovery.append({
                "regime": rid,
                "start": start,
                "end": end,
                "avg_drift": avg_drift,
                "avg_state_effect": avg_eff,
                "recovery_time": rec_time,
            })

    # Correlation between drift/effect and recovery time
    drift_vals = [r["avg_drift"] for r in recovery if r["avg_drift"] is not None and r["recovery_time"] is not None]
    eff_vals = [r["avg_state_effect"] for r in recovery if r["avg_state_effect"] is not None and r["recovery_time"] is not None]
    rec_vals = [r["recovery_time"] for r in recovery if r["recovery_time"] is not None]

    drift_corr = pearson(drift_vals, rec_vals) if drift_vals and rec_vals else 0.0
    drift_spearman = spearman(drift_vals, rec_vals) if drift_vals and rec_vals else 0.0
    eff_corr = pearson(eff_vals, rec_vals) if eff_vals and rec_vals else 0.0
    eff_spearman = spearman(eff_vals, rec_vals) if eff_vals and rec_vals else 0.0

    result = {
        "reaction_log": args.reaction_log,
        "baseline": {
            "loss_mean": loss_mu,
            "loss_std": loss_std,
            "entropy_mean": ent_mu,
            "entropy_std": ent_std,
            "state_effect_mean": eff_mu,
            "state_effect_std": eff_std,
            "self_drift_mean": drift_mu,
            "self_drift_std": drift_std,
        },
        "lead_lag": {
            "lags": list(lags),
            "corr_state_effect_loss": corr_loss,
            "corr_state_effect_entropy": corr_entropy,
        },
        "event_prediction": {
            "loss_threshold": loss_thresh,
            "entropy_threshold": ent_thresh,
            "state_effect_drop_threshold": eff_drop,
            "lead_window": args.lead_window,
            "metrics": pred_metrics,
        },
        "recovery": {
            "per_regime": recovery,
            "corr_drift_recovery_time": drift_corr,
            "spearman_drift_recovery_time": drift_spearman,
            "corr_effect_recovery_time": eff_corr,
            "spearman_effect_recovery_time": eff_spearman,
        },
    }

    out_json = args.out_json or os.path.splitext(args.reaction_log)[0] + "_state_effect_monitoring.json"
    Path(os.path.dirname(out_json) or ".").mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    if args.plot:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(list(lags), corr_loss, label="corr(state_effect_t, loss_{t+lag})")
        ax.plot(list(lags), corr_entropy, label="corr(state_effect_t, entropy_{t+lag})")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("lag (steps)")
        ax.set_ylabel("correlation")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=150)
        plt.close(fig)

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
