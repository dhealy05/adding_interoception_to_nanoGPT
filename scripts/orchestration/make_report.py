#!/usr/bin/env python3
"""Generate a lightweight markdown report from a suite out_root."""

import argparse
import csv
import json
import os
import re
import sys

import torch

ABS_RE = re.compile(r"abs_logit_diff=([0-9.]+)")
KL_RE = re.compile(r"kl\(p0\|\|p1\)=([0-9.]+)")
ENT_RE = re.compile(r"entropy_zero=([0-9.]+) entropy_state=([0-9.]+)")
TOP1_RE = re.compile(r"top1_zero=([0-9.]+) top1_state=([0-9.]+)")
ITER_RE = re.compile(r"^iter (\d+):")
SELF_RE = re.compile(r"^self_norm=([0-9.]+), self_delta=([0-9.]+), self_drift=([0-9.]+)")
STATE_EFFECT_RE = re.compile(r"^state_effect=([0-9.]+)")


def read_best_val(path):
    if not os.path.isfile(path):
        return None
    ckpt = torch.load(path, map_location='cpu')
    val = ckpt.get('best_val_loss')
    try:
        return float(val)
    except Exception:
        return None


def parse_eval_log(path):
    if not os.path.isfile(path):
        return None
    metrics = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            m = ABS_RE.search(line)
            if m:
                metrics['abs_logit_diff'] = float(m.group(1))
            m = KL_RE.search(line)
            if m:
                metrics['kl'] = float(m.group(1))
            m = ENT_RE.search(line)
            if m:
                metrics['entropy_zero'] = float(m.group(1))
                metrics['entropy_state'] = float(m.group(2))
            m = TOP1_RE.search(line)
            if m:
                metrics['top1_zero'] = float(m.group(1))
                metrics['top1_state'] = float(m.group(2))
    required = {'abs_logit_diff', 'kl', 'entropy_zero', 'entropy_state', 'top1_zero', 'top1_state'}
    if required.issubset(metrics.keys()):
        return metrics
    return None


def parse_self_log(path):
    if not os.path.isfile(path):
        return None
    rows = {}
    cur_iter = None
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            m = ITER_RE.match(line)
            if m:
                cur_iter = int(m.group(1))
                rows.setdefault(cur_iter, {})
                continue
            if cur_iter is None:
                continue
            m = SELF_RE.match(line)
            if m:
                rows[cur_iter]['self_norm'] = float(m.group(1))
                rows[cur_iter]['self_delta'] = float(m.group(2))
                rows[cur_iter]['self_drift'] = float(m.group(3))
                continue
            m = STATE_EFFECT_RE.match(line)
            if m:
                rows[cur_iter]['state_effect'] = float(m.group(1))
    return rows


def mean_in_window(rows, key, start, end):
    vals = [v.get(key) for it, v in rows.items() if start <= it < end and v.get(key) is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def read_top_dims(path, col, n=5):
    if not os.path.isfile(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            try:
                row['dim'] = int(row['dim'])
                row[col] = float(row[col])
            except Exception:
                continue
            rows.append(row)
    rows.sort(key=lambda r: abs(r[col]), reverse=True)
    return rows[:n]


def read_top_effects(path, n=5):
    if not os.path.isfile(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            try:
                row['dim'] = int(row['dim'])
                row['effect_peak'] = float(row['effect_peak'])
            except Exception:
                continue
            rows.append(row)
    rows.sort(key=lambda r: r['effect_peak'], reverse=True)
    return rows[:n]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--patch_start', type=int, default=1200)
    ap.add_argument('--patch_end', type=int, default=1400)
    ap.add_argument('--swap_step', type=int, default=1000)
    ap.add_argument('--swap_window', type=int, default=100)
    ap.add_argument('--regimes_out_dir', default='', help='Optional regimes run dir for summary')
    args = ap.parse_args(argv)

    out_root = args.out_root
    log_dir = os.path.join(out_root, 'logs')

    report_path = os.path.join(out_root, 'report.md')
    lines = []
    lines.append(f"# Suite Report ({os.path.basename(out_root)})")
    lines.append("")

    self_ckpt = os.path.join(out_root, 'self-analysis', 'ckpt.pt')
    noself_ckpt = os.path.join(out_root, 'self-analysis-noself', 'ckpt.pt')
    self_val = read_best_val(self_ckpt)
    noself_val = read_best_val(noself_ckpt)
    if self_val is not None or noself_val is not None:
        lines.append("## Val Loss")
        if self_val is not None:
            lines.append(f"- self_state best_val_loss: {self_val:.4f}")
        if noself_val is not None:
            lines.append(f"- no_self best_val_loss: {noself_val:.4f}")
        if self_val is not None and noself_val is not None:
            lines.append(f"- delta (no_self - self): {(noself_val - self_val):.4f}")
        lines.append("")

    corr_path = os.path.join(out_root, 'self-analysis', 'self_state_dim_corr.csv')
    effect_path = os.path.join(out_root, 'self-analysis', 'self_state_dim_effect.csv')
    top_step = read_top_dims(corr_path, 'corr_step_frac', n=5)
    top_loss = read_top_dims(corr_path, 'corr_loss', n=5)
    top_effect = read_top_effects(effect_path, n=5)
    if top_step or top_loss or top_effect:
        lines.append("## Top Dims")
        if top_step:
            lines.append("- corr_step_frac:")
            for row in top_step:
                lines.append(f"  - dim {row['dim']}: {row['corr_step_frac']:.4f}")
        if top_loss:
            lines.append("- corr_loss:")
            for row in top_loss:
                lines.append(f"  - dim {row['dim']}: {row['corr_loss']:.4f}")
        if top_effect:
            lines.append("- effect_peak:")
            for row in top_effect:
                lines.append(f"  - dim {row['dim']}: {row['effect_peak']:.5f}")
        lines.append("")

    patch_log = os.path.join(log_dir, 'patch.log')
    base_log = os.path.join(log_dir, 'baseline.log')
    patch_rows = parse_self_log(patch_log)
    base_rows = parse_self_log(base_log)
    if patch_rows and base_rows:
        lines.append("## Patch vs Baseline (log-derived)")
        patch_drift = mean_in_window(patch_rows, 'self_drift', args.patch_start, args.patch_end)
        base_drift = mean_in_window(base_rows, 'self_drift', args.patch_start, args.patch_end)
        if patch_drift is not None and base_drift is not None:
            lines.append(f"- mean self_drift {args.patch_start}-{args.patch_end}: patch={patch_drift:.4f}, baseline={base_drift:.4f}")
        patch_effect = mean_in_window(patch_rows, 'state_effect', args.patch_start, args.patch_end)
        base_effect = mean_in_window(base_rows, 'state_effect', args.patch_start, args.patch_end)
        if patch_effect is not None and base_effect is not None:
            lines.append(f"- mean state_effect {args.patch_start}-{args.patch_end}: patch={patch_effect:.4f}, baseline={base_effect:.4f}")
        lines.append("")

    swap_pairs = [
        ("swap_a_swapped.log", "swap_a_continued.log", "A"),
        ("swap_b_swapped.log", "swap_b_continued.log", "B"),
    ]
    swap_summaries = []
    for swapped_log, cont_log, label in swap_pairs:
        swapped_rows = parse_self_log(os.path.join(log_dir, swapped_log))
        cont_rows = parse_self_log(os.path.join(log_dir, cont_log))
        if not swapped_rows or not cont_rows:
            continue
        start = args.swap_step
        end = args.swap_step + args.swap_window
        s_drift = mean_in_window(swapped_rows, 'self_drift', start, end)
        c_drift = mean_in_window(cont_rows, 'self_drift', start, end)
        s_eff = mean_in_window(swapped_rows, 'state_effect', start, end)
        c_eff = mean_in_window(cont_rows, 'state_effect', start, end)
        swap_summaries.append((label, s_drift, c_drift, s_eff, c_eff, start, end))

    if swap_summaries:
        lines.append("## Swap vs Continued (log-derived)")
        for label, s_drift, c_drift, s_eff, c_eff, start, end in swap_summaries:
            if s_drift is not None and c_drift is not None:
                lines.append(f"- mean self_drift {start}-{end} (swap {label}): swapped={s_drift:.4f}, continued={c_drift:.4f}")
            if s_eff is not None and c_eff is not None:
                lines.append(f"- mean state_effect {start}-{end} (swap {label}): swapped={s_eff:.4f}, continued={c_eff:.4f}")
        lines.append("")

    # Optional regimes summary (precomputed run dir with reaction_log.jsonl)
    regimes_dir = args.regimes_out_dir
    if regimes_dir:
        from pathlib import Path

        log_path = Path(regimes_dir) / 'reaction_log.jsonl'
        if log_path.exists():
            records = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
            records.sort(key=lambda r: r.get('step', 0))
            by_step = {r['step']: r for r in records}
            steps = sorted(by_step)
            baseline_steps = [s for s in steps if not by_step[s].get('regimes')]

            def mean_over(step_list, key):
                vals = [by_step[s].get(key) for s in step_list if by_step[s].get(key) is not None]
                return sum(vals) / len(vals) if vals else None

            regime_ids = sorted({rid for r in records for rid in r.get('regimes', [])})
            windows = {}
            for rid in regime_ids:
                rsteps = [s for s in steps if rid in by_step[s].get('regimes', [])]
                if rsteps:
                    windows[rid] = (min(rsteps), max(rsteps) + 1)

            def baseline_before(start, n=100):
                prior = [s for s in baseline_steps if s < start]
                return prior[-n:] if len(prior) >= n else prior

            def baseline_after(end, n=100):
                after = [s for s in baseline_steps if s >= end]
                return after[:n] if len(after) >= n else after

            metrics = ['loss', 'self_drift', 'state_effect', 'stats_entropy', 'stats_top1_conf']
            lines.append("## Regime Expansion (log-derived)")
            for rid, (start, end) in windows.items():
                pre_steps = baseline_before(start)
                during_steps = [s for s in steps if start <= s < end]
                post_steps = baseline_after(end)
                pre = {m: mean_over(pre_steps, m) for m in metrics}
                during = {m: mean_over(during_steps, m) for m in metrics}
                post = {m: mean_over(post_steps, m) for m in metrics}
                lines.append(f"- {rid} ({start}-{end})")
                for m in metrics:
                    if pre[m] is None or during[m] is None or post[m] is None:
                        continue
                    delta = during[m] - pre[m]
                    lines.append(
                        f"  - {m}: pre={pre[m]:.4f} during={during[m]:.4f} post={post[m]:.4f} delta={delta:.4f}"
                    )
            lines.append("")
    conf_eval = parse_eval_log(os.path.join(log_dir, 'eval_confident.log'))
    unc_eval = parse_eval_log(os.path.join(log_dir, 'eval_uncertain.log'))
    if conf_eval or unc_eval:
        lines.append("## Preset Eval (log-derived)")
        if conf_eval:
            lines.append(f"- confident: abs_logit_diff={conf_eval['abs_logit_diff']:.6f}, kl={conf_eval['kl']:.6f}, entropy {conf_eval['entropy_zero']:.4f}->{conf_eval['entropy_state']:.4f}, top1 {conf_eval['top1_zero']:.4f}->{conf_eval['top1_state']:.4f}")
        if unc_eval:
            lines.append(f"- uncertain: abs_logit_diff={unc_eval['abs_logit_diff']:.6f}, kl={unc_eval['kl']:.6f}, entropy {unc_eval['entropy_zero']:.4f}->{unc_eval['entropy_state']:.4f}, top1 {unc_eval['top1_zero']:.4f}->{unc_eval['top1_state']:.4f}")
        lines.append("")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines).strip() + '\n')

    print(f"wrote {report_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
