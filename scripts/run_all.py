#!/usr/bin/env python3
"""Master orchestration script for nanodesmo experiments.

Runs the complete experimental pipeline from scratch with clear dependencies:

  Phase 1: Cleanup           - Remove old outputs
  Phase 2: Shakespeare       - Core experiments (regimes, swaps, baselines)
  Phase 3: Frozen test (shk) - Causal leverage test on shakespeare checkpoint
  Phase 4: OpenWebText       - Generalization experiments
  Phase 5: Frozen test (owt) - Causal leverage test on openwebtext checkpoint
  Phase 6: Memory            - Memory architecture comparison
  Phase 7: Controls          - Ablation experiments
  Phase 8: Collect figures   - Copy per-run figures to figures/
  Phase 9: Summary figures   - Generate cross-run comparison figures

Usage:
  pipenv run python scripts/run_all.py              # Run everything
  pipenv run python scripts/run_all.py --dry_run    # Show what would run
  pipenv run python scripts/run_all.py --only memory  # Run only memory suite

See docs/PIPELINE.md for full data flow documentation.
"""

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(msg, level="info"):
    """Print timestamped log message."""
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = {"info": "", "skip": "[skip] ", "fail": "[FAIL] ", "ok": "[ok] "}
    print(f"[{ts}] {prefix.get(level, '')}{msg}")


def run_cmd(cmd, dry_run=False):
    """Run a command, return exit code."""
    if dry_run:
        print(f"    $ {' '.join(cmd)}")
        return 0
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode


def phase_cleanup(dry_run=False):
    """Phase 1: Remove old output directories."""
    log("Phase 1: Cleanup")
    for d in ["out", "figures"]:
        path = os.path.join(REPO_ROOT, d)
        if os.path.exists(path):
            if dry_run:
                print(f"    Would remove: {path}")
            else:
                shutil.rmtree(path)
                print(f"    Removed: {path}")
        else:
            print(f"    Not found: {path}")


def phase_suite(name, yaml_path, out_root, dry_run=False, resume=False):
    """Run a suite via run_suite.py."""
    log(name)
    cmd = [
        sys.executable,
        "scripts/orchestration/run_suite.py",
        "--suite", yaml_path,
        "--out_root", out_root,
    ]
    if resume:
        cmd.append("--resume")
    if dry_run:
        print(f"    $ {' '.join(cmd)}")
        return 0
    return run_cmd(cmd, dry_run)


def phase_frozen_test(name, ckpt_path, dataset, out_json, out_plot, dry_run=False):
    """Run frozen-weight causal test."""
    log(name)

    if not dry_run and not os.path.exists(os.path.join(REPO_ROOT, ckpt_path)):
        log(f"Checkpoint not found: {ckpt_path}", "skip")
        return 1

    cmd = [
        sys.executable,
        "scripts/analysis/frozen_test_auto.py",
        "--ckpt", ckpt_path,
        "--dataset", dataset,
        "--device", "cpu",
        "--batch_size", "32",
        "--num_batches", "5",
        "--magnitude", "0.3",
        "--top_k", "5",
        "--num_random", "50",
        "--out", out_json,
        "--plot", out_plot,
    ]
    if dry_run:
        print(f"    $ {' '.join(cmd)}")
        return 0
    return run_cmd(cmd, dry_run)


def phase_collect_figures(shakespeare, openwebtext, memory, controls, figures_dir, dry_run=False):
    """Phase 8: Collect figures from suite outputs."""
    log("Phase 8: Collect figures")
    cmd = [
        sys.executable,
        "scripts/plotting/collect_figures.py",
        "--shakespeare", shakespeare,
        "--openwebtext", openwebtext,
        "--memory", memory,
        "--controls", controls,
        "--figures", figures_dir,
    ]
    if dry_run:
        print(f"    $ {' '.join(cmd)}")
        return 0
    return run_cmd(cmd, dry_run)


def phase_summary_figures(shakespeare, openwebtext, memory, controls, figures_dir, dry_run=False):
    """Phase 9: Generate cross-run summary figures."""
    log("Phase 9: Summary figures")
    cmd = [
        sys.executable,
        "scripts/plotting/generate_summary_figures.py",
        "--shakespeare", shakespeare,
        "--openwebtext", openwebtext,
        "--memory", memory,
        "--controls", controls,
        "--figures", figures_dir,
    ]
    if dry_run:
        print(f"    $ {' '.join(cmd)}")
        return 0
    return run_cmd(cmd, dry_run)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry_run", action="store_true",
                    help="Show commands without executing")
    ap.add_argument("--skip_cleanup", action="store_true",
                    help="Skip cleanup phase")
    ap.add_argument("--resume", action="store_true",
                    help="Resume suites from where they left off")
    ap.add_argument("--only",
                    choices=["shakespeare", "openwebtext", "memory", "controls",
                             "frozen", "collect", "summary"],
                    help="Run only a specific phase")
    args = ap.parse_args()

    # Output paths
    shakespeare = "out/shakespeare"
    openwebtext = "out/openwebtext"
    memory = "out/memory"
    controls = "out/controls"
    figures_dir = "figures"

    log("=" * 60)
    log("nanodesmo experimental pipeline")
    log("=" * 60)

    # Phase 1: Cleanup
    if not args.skip_cleanup and not args.resume and args.only is None:
        phase_cleanup(args.dry_run)

    # Phase 2: Shakespeare core
    if args.only in (None, "shakespeare"):
        ret = phase_suite(
            "Phase 2: Shakespeare core",
            "configs/suites/shakespeare_core.yaml",
            shakespeare,
            args.dry_run,
            args.resume
        )
        if ret != 0 and not args.dry_run:
            log("Shakespeare suite failed", "fail")
            return ret

    # Phase 3: Frozen-weight test (shakespeare)
    if args.only in (None, "frozen"):
        ret = phase_frozen_test(
            "Phase 3: Frozen test (shakespeare)",
            f"{shakespeare}/self-analysis/ckpt.pt",
            "shakespeare_char",
            f"{shakespeare}/frozen_test_results.json",
            f"{shakespeare}/frozen_test.png",
            args.dry_run
        )
        if ret != 0 and not args.dry_run:
            log("Frozen test (shakespeare) failed (continuing)", "fail")

    # Phase 4: OpenWebText
    if args.only in (None, "openwebtext"):
        ret = phase_suite(
            "Phase 4: OpenWebText",
            "configs/suites/openwebtext_progress.yaml",
            openwebtext,
            args.dry_run,
            args.resume
        )
        if ret != 0 and not args.dry_run:
            log("OpenWebText suite failed", "fail")
            return ret

    # Phase 5: Frozen-weight test (openwebtext)
    if args.only in (None, "frozen"):
        ret = phase_frozen_test(
            "Phase 5: Frozen test (openwebtext)",
            f"{openwebtext}/self-analysis/ckpt.pt",
            "openwebtext_small",
            f"{openwebtext}/frozen_test_results.json",
            f"{openwebtext}/frozen_test.png",
            args.dry_run
        )
        if ret != 0 and not args.dry_run:
            log("Frozen test (openwebtext) failed (continuing)", "fail")

    # Phase 6: Memory
    if args.only in (None, "memory"):
        ret = phase_suite(
            "Phase 6: Memory",
            "configs/suites/memory_buffer_grid.yaml",
            memory,
            args.dry_run,
            args.resume
        )
        if ret != 0 and not args.dry_run:
            log("Memory suite failed", "fail")
            return ret

    # Phase 7: Controls
    if args.only in (None, "controls"):
        ret = phase_suite(
            "Phase 7: Controls",
            "configs/suites/reviewer_controls.yaml",
            controls,
            args.dry_run,
            args.resume
        )
        if ret != 0 and not args.dry_run:
            log("Controls suite failed", "fail")
            return ret

    # Phase 8: Collect figures
    if args.only in (None, "collect"):
        phase_collect_figures(shakespeare, openwebtext, memory, controls, figures_dir, args.dry_run)

    # Phase 9: Summary figures
    if args.only in (None, "summary"):
        phase_summary_figures(shakespeare, openwebtext, memory, controls, figures_dir, args.dry_run)

    log("=" * 60)
    log("Pipeline complete")
    log("=" * 60)

    if not args.dry_run:
        print("\nOutput directories:")
        for name, path in [("shakespeare", shakespeare), ("openwebtext", openwebtext),
                           ("memory", memory), ("controls", controls), ("figures", figures_dir)]:
            exists = "OK" if os.path.exists(os.path.join(REPO_ROOT, path)) else "MISSING"
            print(f"  {name:12} {path:20} [{exists}]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
