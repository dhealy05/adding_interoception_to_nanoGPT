#!/usr/bin/env python3
"""Analyze the self_state update MLP inputs and hidden units.

Outputs:
  - input_sensitivity.csv + plot: per-feature sensitivity curves
  - input_ablation.csv: effect of neutralizing each feature
  - input_grad.csv: finite-diff gradient norms per feature
  - update_norm_corr.csv: correlation of update_norm with each input
  - probe outputs (JSON): regime vs baseline and regime ID classification

Usage:
  python scripts/analysis/self_update_mlp_analysis.py \\
    --ckpt out/run/self-regimes/ckpt.pt \\
    --reaction_log out/run/self-regimes/reaction_log.jsonl \\
    --out_dir out/run/self-regimes/mlp_analysis
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
matplotlib.use("Agg")
import matplotlib.pyplot as plt


STAT_FIELDS = ["stats_loss", "stats_grad_norm", "stats_step_frac", "stats_entropy", "stats_top1_conf"]


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


def extract_stats(records):
    rows = []
    for rec in records:
        vec = []
        ok = True
        for key in STAT_FIELDS:
            if key not in rec or rec[key] is None:
                ok = False
                break
            vec.append(float(rec[key]))
        if not ok:
            continue
        rows.append((rec.get("step", 0), rec.get("regimes", []) or [], np.array(vec, dtype=np.float64)))
    return rows


def baseline_mask(rows):
    has_regimes = any(len(r[1]) > 0 for r in rows)
    if not has_regimes:
        return np.array([True] * len(rows), dtype=bool)
    return np.array([len(r[1]) == 0 for r in rows], dtype=bool)


def load_model(ckpt_path, device):
    from src.model import GPT, GPTConfig
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = GPTConfig(**ckpt["model_args"])
    model = GPT(cfg)
    state_dict = ckpt["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, ckpt


def compute_update(controller, stats_vec, state_vec):
    stats = torch.tensor(stats_vec, dtype=torch.float32).view(1, -1)
    state = torch.tensor(state_vec, dtype=torch.float32).view(-1)
    mem_delta = torch.zeros_like(state)
    raw, _, _, _ = controller.compute_updates(state, stats, memory_delta=mem_delta, memory_scale=0.0)
    return raw.view(-1)


def compute_hidden(controller, stats_vec):
    if controller.update_type != "mlp":
        return None
    with torch.no_grad():
        x = torch.tensor(stats_vec, dtype=torch.float32).view(1, -1)
        if controller.memory_enabled:
            state = torch.zeros(controller.state_proj.in_features, dtype=torch.float32)
            mem_delta = torch.zeros_like(state)
            x = controller._build_features(state, x, mem_delta)
        h = controller.update_trunk(x)
    return h.view(-1).cpu().numpy()


def corrcoef(x, y):
    if len(x) < 2:
        return 0.0
    x = np.asarray(x)
    y = np.asarray(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def probe_classifier(X, y, kfold=5, seed=123):
    # Try sklearn; fallback to nearest centroid
    use_sklearn = False
    try:
        import inspect
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score
        use_sklearn = True
    except Exception:
        use_sklearn = False

    # stratified folds
    rng = np.random.default_rng(seed)
    labels = np.unique(y)
    indices = {lab: np.where(y == lab)[0].tolist() for lab in labels}
    for lab in labels:
        rng.shuffle(indices[lab])
    folds = [[] for _ in range(kfold)]
    for lab in labels:
        for i, idx in enumerate(indices[lab]):
            folds[i % kfold].append(idx)

    accs = []
    bal_accs = []
    aucs = []
    for i in range(kfold):
        test_idx = np.array(folds[i], dtype=int)
        train_idx = np.array([j for f, fold in enumerate(folds) if f != i for j in fold], dtype=int)
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        if use_sklearn:
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            kwargs = {"max_iter": 1000}
            sig = inspect.signature(LogisticRegression)
            if "multi_class" in sig.parameters:
                kwargs["multi_class"] = "multinomial"
            if "solver" in sig.parameters:
                kwargs["solver"] = "lbfgs"
            clf = LogisticRegression(**kwargs)
            clf.fit(X_train_s, y_train)
            preds = clf.predict(X_test_s)
            scores = None
            if hasattr(clf, "decision_function"):
                scores = clf.decision_function(X_test_s)
            elif hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X_test_s)
                # use positive class prob if binary
                if probs.shape[1] == 2:
                    scores = probs[:, 1]
        else:
            # nearest centroid
            centroids = {lab: X_train[y_train == lab].mean(axis=0) for lab in labels}
            centers = np.stack([centroids[lab] for lab in labels], axis=0)
            preds = []
            for x in X_test:
                dists = np.sum((centers - x) ** 2, axis=1)
                preds.append(labels[int(np.argmin(dists))])
            preds = np.array(preds)
            scores = None

        # accuracy
        accs.append(float(np.mean(preds == y_test)))

        # balanced accuracy
        per_class = []
        for lab in labels:
            mask = y_test == lab
            if not np.any(mask):
                continue
            per_class.append(float(np.mean(preds[mask] == y_test[mask])))
        bal_accs.append(float(np.mean(per_class)) if per_class else 0.0)

        # AUC for binary only, if scores available
        if use_sklearn and scores is not None and len(labels) == 2:
            try:
                aucs.append(float(roc_auc_score(y_test, scores)))
            except Exception:
                pass

    result = {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "balanced_accuracy_mean": float(np.mean(bal_accs)),
        "balanced_accuracy_std": float(np.std(bal_accs)),
    }
    if aucs:
        result["auc_mean"] = float(np.mean(aucs))
        result["auc_std"] = float(np.std(aucs))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--reaction_log", required=True)
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num_points", type=int, default=25)
    ap.add_argument("--q_lo", type=float, default=0.05)
    ap.add_argument("--q_hi", type=float, default=0.95)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--probe_task", choices=["regime_vs_baseline", "regime_id"], default="regime_vs_baseline")
    ap.add_argument("--probe_features", choices=["hidden", "update", "stats"], default="hidden")
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args(argv)

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.reaction_log), "mlp_analysis")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    records = load_records(args.reaction_log)
    rows = extract_stats(records)
    if not rows:
        raise SystemExit("no stats found in reaction log")

    steps = np.array([r[0] for r in rows], dtype=np.int64)
    stats = np.stack([r[2] for r in rows], axis=0)
    regimes = [r[1] for r in rows]

    base_mask = baseline_mask(rows)
    base_stats = stats[base_mask]
    base_mean = base_stats.mean(axis=0)
    base_std = base_stats.std(axis=0) + 1e-8

    model, ckpt = load_model(args.ckpt, args.device)
    controller = model.self_state_controller
    if controller is None:
        raise SystemExit("checkpoint has no self_state_controller")

    if "self_state" in ckpt:
        base_state = ckpt["self_state"].float().cpu().numpy()
    else:
        base_state = np.zeros(controller.state_proj.in_features, dtype=np.float32)

    # Sensitivity curves
    q_lo = np.quantile(base_stats, args.q_lo, axis=0)
    q_hi = np.quantile(base_stats, args.q_hi, axis=0)
    sensitivity_rows = []
    for i, name in enumerate(STAT_FIELDS):
        vals = np.linspace(q_lo[i], q_hi[i], args.num_points)
        base_out = compute_update(controller, base_mean, base_state).detach().cpu().numpy()
        for v in vals:
            vec = base_mean.copy()
            vec[i] = v
            out = compute_update(controller, vec, base_state).detach().cpu().numpy()
            update_norm = float(np.linalg.norm(out))
            delta_norm = float(np.linalg.norm(out - base_out))
            sensitivity_rows.append((name, float(v), update_norm, delta_norm))

    sens_path = os.path.join(out_dir, "input_sensitivity.csv")
    with open(sens_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "value", "update_norm", "delta_norm"])
        writer.writerows(sensitivity_rows)

    # Sensitivity plot (one subplot per feature)
    fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharey=False)
    axes = axes.flatten()
    for i, name in enumerate(STAT_FIELDS):
        ax = axes[i]
        xs = [r[1] for r in sensitivity_rows if r[0] == name]
        ys = [r[3] for r in sensitivity_rows if r[0] == name]
        ax.plot(xs, ys, color="#1f77b4")
        ax.set_title(name)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("value")
        ax.set_ylabel("Δ update norm")
    for ax in axes[len(STAT_FIELDS):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "input_sensitivity.png"), dpi=150)
    plt.close(fig)

    # Feature ablation: set each feature to baseline mean for each sample
    ablation_rows = []
    for i, name in enumerate(STAT_FIELDS):
        diffs = []
        for vec in stats:
            base_out = compute_update(controller, vec, base_state).detach().cpu().numpy()
            ablate = vec.copy()
            ablate[i] = base_mean[i]
            out = compute_update(controller, ablate, base_state).detach().cpu().numpy()
            diffs.append(np.linalg.norm(base_out - out))
        ablation_rows.append((name, float(np.mean(diffs)), float(np.std(diffs))))
    ablate_path = os.path.join(out_dir, "input_ablation.csv")
    with open(ablate_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "mean_delta_norm", "std_delta_norm"])
        writer.writerows(ablation_rows)

    # Finite-diff gradient norms at baseline mean
    grad_rows = []
    base_out = compute_update(controller, base_mean, base_state).detach().cpu().numpy()
    for i, name in enumerate(STAT_FIELDS):
        eps = 0.1 * base_std[i]
        vec = base_mean.copy()
        vec[i] += eps
        out = compute_update(controller, vec, base_state).detach().cpu().numpy()
        grad_norm = np.linalg.norm(out - base_out) / (eps + 1e-8)
        grad_rows.append((name, float(base_mean[i]), float(base_std[i]), float(eps), float(grad_norm)))
    grad_path = os.path.join(out_dir, "input_grad.csv")
    with open(grad_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "baseline_mean", "baseline_std", "eps", "grad_norm"])
        writer.writerows(grad_rows)

    # Update norm correlations with inputs
    update_norms = []
    for vec in stats:
        out = compute_update(controller, vec, base_state).detach().cpu().numpy()
        update_norms.append(np.linalg.norm(out))
    corr_rows = []
    for i, name in enumerate(STAT_FIELDS):
        corr_rows.append((name, corrcoef(stats[:, i], update_norms)))
    corr_path = os.path.join(out_dir, "update_norm_corr.csv")
    with open(corr_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "corr_update_norm"])
        writer.writerows(corr_rows)

    # Optional probe
    if args.probe:
        # features
        feats = []
        labels = []
        for vec, regs in zip(stats, regimes):
            label = None
            if args.probe_task == "regime_vs_baseline":
                label = "regime" if len(regs) > 0 else "baseline"
            else:
                if len(regs) != 1:
                    continue
                label = regs[0]
            if args.probe_features == "stats":
                feat_vec = vec
            elif args.probe_features == "update":
                feat_vec = compute_update(controller, vec, base_state).detach().cpu().numpy()
            else:
                hidden = compute_hidden(controller, vec)
                if hidden is None:
                    continue
                feat_vec = hidden
            feats.append(feat_vec)
            labels.append(label)

        if feats:
            X = np.stack(feats, axis=0)
            y = np.array(labels)
            metrics = probe_classifier(X, y, kfold=args.kfold, seed=args.seed)
            # shuffled baseline
            rng = np.random.default_rng(args.seed)
            y_shuf = y.copy()
            rng.shuffle(y_shuf)
            metrics_shuf = probe_classifier(X, y_shuf, kfold=args.kfold, seed=args.seed + 1)
            probe_out = {
                "task": args.probe_task,
                "features": args.probe_features,
                "kfold": args.kfold,
                "n_samples": int(X.shape[0]),
                "n_classes": int(len(np.unique(y))),
                "accuracy_mean": metrics.get("accuracy_mean"),
                "accuracy_std": metrics.get("accuracy_std"),
                "balanced_accuracy_mean": metrics.get("balanced_accuracy_mean"),
                "balanced_accuracy_std": metrics.get("balanced_accuracy_std"),
                "auc_mean": metrics.get("auc_mean"),
                "auc_std": metrics.get("auc_std"),
                "shuffle_accuracy_mean": metrics_shuf.get("accuracy_mean"),
                "shuffle_accuracy_std": metrics_shuf.get("accuracy_std"),
                "shuffle_balanced_accuracy_mean": metrics_shuf.get("balanced_accuracy_mean"),
                "shuffle_balanced_accuracy_std": metrics_shuf.get("balanced_accuracy_std"),
                "shuffle_auc_mean": metrics_shuf.get("auc_mean"),
                "shuffle_auc_std": metrics_shuf.get("auc_std"),
            }
            out_path = os.path.join(out_dir, f"probe_{args.probe_task}_{args.probe_features}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(probe_out, f, indent=2)

    print(f"wrote {sens_path}")
    print(f"wrote {ablate_path}")
    print(f"wrote {grad_path}")
    print(f"wrote {corr_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
