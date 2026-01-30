#!/usr/bin/env python3
"""Convert reaction JSONL logs to CSV.

Usage:
  python scripts/reaction_log_to_csv.py /path/to/reaction_log.jsonl [output.csv]
  python scripts/reaction_log_to_csv.py /path/to/log1 /path/to/log2 ...
"""

import csv
import json
import os
import sys


def parse_log(path):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_csv(records, out_path):
    keys = set()
    for rec in records:
        keys.update(rec.keys())
    keys.discard('step')
    keys.discard('regimes')
    fieldnames = ['step', 'regimes'] + sorted(keys)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = dict(rec)
            regimes = row.get('regimes')
            if isinstance(regimes, list):
                row['regimes'] = '|'.join(str(r) for r in regimes)
            writer.writerow(row)


def main(argv):
    if len(argv) < 2:
        print("usage: python scripts/reaction_log_to_csv.py /path/to/log [output.csv] [more logs...]", file=sys.stderr)
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
        records = parse_log(log_path)
        write_csv(records, out_path)
        print(f"wrote {out_path}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
