#!/usr/bin/env python3
"""Estimate regime separability with a simple classifier + shuffled-label baseline.

Usage:
  python scripts/analysis/regime_separability.py \
    --reaction_log out/run/reaction_log.jsonl \
    --out_json out/run/regime_separability.json \
    --features loss,stats_entropy,stats_top1_conf \
    --window 10 \
    --kfold 5
"""

import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np


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


def build_samples(records, features, window, max_per_class):
    # group by regime id using contiguous windows
    by_regime = defaultdict(list)
    for rec in records:
        regimes = rec.get("regimes", []) or []
        if len(regimes) != 1:
            # skip baseline or overlapping regimes
            continue
        rid = regimes[0]
        by_regime[rid].append(rec)

    X = []
    y = []
    for rid, items in by_regime.items():
        if window <= 1:
            chunks = [[rec] for rec in items]
        else:
            chunks = [items[i:i + window] for i in range(0, len(items), window)]
        feats = []
        for chunk in chunks:
            vec = []
            ok = True
            for key in features:
                vals = [c.get(key) for c in chunk if c.get(key) is not None]
                if not vals:
                    ok = False
                    break
                vec.append(float(np.mean(vals)))
            if not ok:
                continue
            feats.append(vec)
        if max_per_class > 0 and len(feats) > max_per_class:
            feats = feats[:max_per_class]
        for vec in feats:
            X.append(vec)
            y.append(rid)
    return np.array(X, dtype=np.float64), np.array(y)


def stratified_kfold_indices(y, k, seed=0):
    rng = random.Random(seed)
    classes = defaultdict(list)
    for i, label in enumerate(y):
        classes[label].append(i)
    for idxs in classes.values():
        rng.shuffle(idxs)
    folds = [[] for _ in range(k)]
    for idxs in classes.values():
        for i, idx in enumerate(idxs):
            folds[i % k].append(idx)
    return folds


def nearest_centroid_fit_predict(X_train, y_train, X_test):
    centroids = {}
    for label in np.unique(y_train):
        centroids[label] = X_train[y_train == label].mean(axis=0)
    preds = []
    labels = list(centroids.keys())
    centers = np.stack([centroids[l] for l in labels], axis=0)
    for x in X_test:
        dists = np.sum((centers - x) ** 2, axis=1)
        preds.append(labels[int(np.argmin(dists))])
    return np.array(preds)


def eval_classifier(X, y, kfold, seed):
    # Try sklearn if available; otherwise fall back to nearest centroid.
    use_sklearn = False
    try:
        import inspect
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        use_sklearn = True
    except Exception:
        use_sklearn = False

    folds = stratified_kfold_indices(y, kfold, seed=seed)
    accs = []
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
        else:
            preds = nearest_centroid_fit_predict(X_train, y_train, X_test)
        acc = float(np.mean(preds == y_test))
        accs.append(acc)
    return accs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--reaction_log", required=True)
    ap.add_argument("--out_json", default="")
    ap.add_argument("--features", default="loss,stats_entropy,stats_top1_conf")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--max_per_class", type=int, default=0)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args(argv)

    features = [f.strip() for f in args.features.split(",") if f.strip()]
    records = load_records(args.reaction_log)
    X, y = build_samples(records, features, args.window, args.max_per_class)
    if X.size == 0:
        raise SystemExit("no samples built (check features/window)")

    accs = eval_classifier(X, y, args.kfold, seed=args.seed)

    # shuffled-label baseline
    rng = np.random.default_rng(args.seed)
    y_shuf = y.copy()
    rng.shuffle(y_shuf)
    accs_shuf = eval_classifier(X, y_shuf, args.kfold, seed=args.seed + 1)

    result = {
        "reaction_log": args.reaction_log,
        "features": features,
        "window": args.window,
        "kfold": args.kfold,
        "max_per_class": args.max_per_class,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(np.unique(y))),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "shuffle_accuracy_mean": float(np.mean(accs_shuf)),
        "shuffle_accuracy_std": float(np.std(accs_shuf)),
    }

    out_json = args.out_json
    if not out_json:
        out_json = os.path.splitext(args.reaction_log)[0] + "_separability.json"
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
