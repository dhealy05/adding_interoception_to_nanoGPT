#!/usr/bin/env python3
"""Run a suite of training/analysis steps from a YAML config.

Usage:
  python scripts/orchestration/run_suite.py --suite configs/suites/shakespeare_core.yaml \
    --out_root out/suites/2026-01-24-unified
"""

import argparse
import datetime as dt
import json
import os
import subprocess
import sys

import yaml


def git_short_hash():
    try:
        out = subprocess.check_output([
            'git', 'rev-parse', '--short', 'HEAD'
        ], stderr=subprocess.DEVNULL, text=True).strip()
        return out or 'unknown'
    except Exception:
        return 'unknown'


def format_obj(obj, context):
    if isinstance(obj, str):
        try:
            return obj.format_map(context)
        except Exception:
            return obj
    if isinstance(obj, dict):
        return {k: format_obj(v, context) for k, v in obj.items()}
    if isinstance(obj, list):
        return [format_obj(v, context) for v in obj]
    return obj


def load_suite(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if isinstance(data, dict) and 'suite' in data:
        return data['suite']
    return data


def build_command(step):
    kind = step.get('kind')
    if kind == 'train':
        args = step.get('args', {})
        config = args.get('config')
        if not config:
            raise ValueError(f"train step {step.get('id')} missing args.config")
        cmd = [sys.executable, 'src/train.py', config]
        overrides = args.get('overrides', {})
        for key, value in overrides.items():
            if isinstance(value, bool):
                value = 'True' if value else 'False'
            cmd.append(f"--{key}={value}")
        return cmd
    if kind == 'script':
        cmd_path = step.get('cmd')
        if not cmd_path:
            raise ValueError(f"script step {step.get('id')} missing cmd")
        args = step.get('args', [])
        if cmd_path.endswith('.py'):
            return [sys.executable, cmd_path] + list(args)
        return [cmd_path] + list(args)
    raise ValueError(f"unknown step kind: {kind}")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--suite', required=True, help='Path to suite YAML')
    ap.add_argument('--out_root', default='', help='Override suite out_root')
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args(argv)

    suite = load_suite(args.suite)
    if not isinstance(suite, dict):
        raise SystemExit('suite config must be a mapping')

    now = dt.datetime.now()
    context = {
        'date': now.strftime('%Y-%m-%d'),
        'datetime': now.strftime('%Y-%m-%d_%H%M%S'),
        'git': git_short_hash(),
    }

    out_root_override = bool(args.out_root)
    if out_root_override:
        out_root = args.out_root
    else:
        template = suite.get('out_root')
        if template:
            out_root = format_obj(template, context)
        else:
            out_root = f"out/suites/{context['date']}-{context['git']}"
    context['out_root'] = out_root

    suite = format_obj(suite, context)
    if out_root_override:
        suite['out_root'] = out_root

    steps = suite.get('steps', [])
    if not steps:
        raise SystemExit('suite has no steps')

    os.makedirs(out_root, exist_ok=True)
    log_dir = os.path.join(out_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    manifest_path = os.path.join(out_root, 'manifest.json')

    manifest = {
        'suite': suite.get('name', os.path.basename(args.suite)),
        'suite_path': args.suite,
        'out_root': out_root,
        'git': context['git'],
        'date': context['date'],
        'datetime': context['datetime'],
        'steps': {},
    }

    if args.resume and os.path.isfile(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)

    for step in steps:
        step_id = step.get('id') or step.get('name')
        if not step_id:
            raise SystemExit('step missing id')
        if args.resume:
            prev = manifest.get('steps', {}).get(step_id)
            if prev and prev.get('status') == 'success':
                print(f"[skip] {step_id}")
                continue

        cmd = build_command(step)
        log_path = os.path.join(log_dir, f"{step_id}.log")
        print(f"[run] {step_id}: {' '.join(cmd)}")
        if args.dry_run:
            continue

        start = dt.datetime.now().isoformat(timespec='seconds')
        with open(log_path, 'w', encoding='utf-8') as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
            ret = proc.wait()
        end = dt.datetime.now().isoformat(timespec='seconds')

        status = 'success' if ret == 0 else 'failed'
        manifest.setdefault('steps', {})[step_id] = {
            'id': step_id,
            'kind': step.get('kind'),
            'cmd': cmd,
            'status': status,
            'returncode': ret,
            'log_path': log_path,
            'start': start,
            'end': end,
        }
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)

        if ret != 0:
            print(f"[fail] {step_id} (returncode={ret})")
            return ret

    print(f"suite complete: {out_root}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
