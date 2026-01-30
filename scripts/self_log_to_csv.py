#!/usr/bin/env python3
"""Parse nanoGPT persistent-self logs into CSV.

Usage:
  python scripts/self_log_to_csv.py /path/to/log [output.csv]
  python scripts/self_log_to_csv.py /path/to/log1 /path/to/log2 ...

If no output is given, writes <log>.csv next to each input.
"""

import csv
import os
import re
import sys

ITER_RE = re.compile(r"^iter (\d+):")
LOSS_RE = re.compile(r"^iter (\d+): loss ([0-9.]+)")
SELF_RE = re.compile(r"^self_norm=([0-9.]+), self_delta=([0-9.]+), self_drift=([0-9.]+)")
EFFECT_RE = re.compile(r"^state_effect=([0-9.]+)")


def parse_log(path):
    rows = {}
    cur_iter = None
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            m = ITER_RE.match(line)
            if m:
                cur_iter = int(m.group(1))
                rows.setdefault(cur_iter, {})
                m2 = LOSS_RE.match(line)
                if m2:
                    rows[cur_iter]['loss'] = float(m2.group(2))
                continue
            if cur_iter is None:
                continue
            m = SELF_RE.match(line)
            if m:
                rows[cur_iter]['self_norm'] = float(m.group(1))
                rows[cur_iter]['self_delta'] = float(m.group(2))
                rows[cur_iter]['self_drift'] = float(m.group(3))
                continue
            m = EFFECT_RE.match(line)
            if m:
                rows[cur_iter]['state_effect'] = float(m.group(1))
    return rows


def write_csv(rows, out_path):
    fieldnames = ['iter', 'loss', 'self_norm', 'self_delta', 'self_drift', 'state_effect']
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for it in sorted(rows.keys()):
            row = {'iter': it}
            row.update(rows[it])
            writer.writerow(row)


def main(argv):
    if len(argv) < 2:
        print("usage: python scripts/self_log_to_csv.py /path/to/log [output.csv] [more logs...]", file=sys.stderr)
        return 2

    if len(argv) == 3 and argv[2].endswith('.csv') and os.path.isfile(argv[1]):
        logs = [argv[1]]
        outputs = [argv[2]]
    else:
        logs = [p for p in argv[1:] if os.path.isfile(p)]
        outputs = [p + '.csv' for p in logs]

    if not logs:
        print("No valid log files found.", file=sys.stderr)
        return 2

    for log_path, out_path in zip(logs, outputs):
        rows = parse_log(log_path)
        write_csv(rows, out_path)
        print(f"wrote {out_path}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
