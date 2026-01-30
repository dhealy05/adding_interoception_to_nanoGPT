#!/usr/bin/env python3
"""Collect figures from suite outputs into a single figures/ directory.

This script copies key plots from the various suite output directories
into a consolidated figures/ directory for the README and paper.

Usage:
  python scripts/plotting/collect_figures.py \
    --shakespeare out/shakespeare \
    --memory out/memory \
    --controls out/controls \
    --figures figures
"""

import argparse
import os
import shutil
from pathlib import Path


def copy_file(src, dst, quiet=False):
    """Copy a file, creating parent directories as needed."""
    if not os.path.exists(src):
        if not quiet:
            print(f"  [skip] Not found: {src}")
        return False
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  {src} -> {dst}")
    return True


def copy_glob(src_dir, pattern, dst_dir, quiet=False):
    """Copy all files matching a glob pattern."""
    src_path = Path(src_dir)
    if not src_path.exists():
        if not quiet:
            print(f"  [skip] Directory not found: {src_dir}")
        return 0
    count = 0
    for src_file in src_path.glob(pattern):
        dst_file = Path(dst_dir) / src_file.name
        copy_file(str(src_file), str(dst_file), quiet)
        count += 1
    return count


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shakespeare", default="out/shakespeare",
                    help="Shakespeare core suite output directory")
    ap.add_argument("--openwebtext", default="out/openwebtext",
                    help="OpenWebText suite output directory")
    ap.add_argument("--memory", default="out/memory",
                    help="Memory suite output directory")
    ap.add_argument("--controls", default="out/controls",
                    help="Controls suite output directory")
    ap.add_argument("--figures", default="figures",
                    help="Output figures directory")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress 'not found' messages")
    args = ap.parse_args()

    figures = args.figures
    print(f"Collecting figures to: {figures}/\n")

    # =========================================================================
    # 1. Frozen-weight test (produced by frozen_test_auto.py)
    # =========================================================================
    print("1. Frozen-weight test:")
    copy_file(
        f"{args.shakespeare}/frozen_test.png",
        f"{figures}/frozen-weight-test.png",
        args.quiet
    )

    # =========================================================================
    # 2. Perturbation/regime plots from shakespeare suite
    # =========================================================================
    print("\n2. Perturbation regime plots:")
    regimes_plots = f"{args.shakespeare}/regimes-all/plots"
    copy_glob(regimes_plots, "regime_*.png", f"{figures}/perturbation", args.quiet)

    # =========================================================================
    # 3. Memory comparison plots
    # =========================================================================
    print("\n3. Memory comparison plots:")
    memory_configs = [
        ("mem-ema-decay-0.1_s10_k1", "ema-baseline"),
        ("mem-buf-l64_s10", "buffer-l64"),
        ("mem-buf-l128_s10", "buffer-l128"),
    ]
    for config_id, label in memory_configs:
        plots_dir = f"{args.memory}/{config_id}/plots"
        if os.path.exists(plots_dir):
            # Copy regime plots with prefixed names
            for src_file in Path(plots_dir).glob("regime_*.png"):
                dst_file = f"{figures}/memory/{label}_{src_file.name}"
                copy_file(str(src_file), dst_file, args.quiet)

    # Also copy the summary plot if it exists
    copy_file(
        f"{args.memory}/memory_decay_summary.png",
        f"{figures}/memory/memory_decay_summary.png",
        args.quiet
    )

    # =========================================================================
    # 4. Control experiment plots
    # =========================================================================
    print("\n4. Control experiment plots:")
    control_variants = [
        ("vanilla-regimes", "vanilla"),
        ("self-regimes", "self-state"),
        ("fixed-bias-0.5", "fixed-bias-0.5"),
        ("fixed-bias-1.0", "fixed-bias-1.0"),
        ("self-dim1-regimes", "dim1"),
    ]
    for variant_id, label in control_variants:
        # Delta scatter plots
        copy_file(
            f"{args.controls}/{variant_id}/regime_delta_scatter.png",
            f"{figures}/controls/{label}_delta_scatter.png",
            args.quiet
        )

    # MLP analysis plots
    mlp_dir = f"{args.controls}/self-regimes/mlp_analysis"
    copy_glob(mlp_dir, "*.png", f"{figures}/controls/mlp", args.quiet)

    # State effect monitoring
    copy_file(
        f"{args.controls}/self-regimes/state_effect_leadlag.png",
        f"{figures}/controls/state_effect_leadlag.png",
        args.quiet
    )

    # =========================================================================
    # 5. Composite/desmotic figures
    # =========================================================================
    print("\n5. Composite (desmotic) figures:")
    desmotic_sources = [
        (f"{args.shakespeare}/shakespeare-char-patch/desmotic", "shakespeare-patch"),
        (f"{args.shakespeare}/regimes-all/desmotic", "shakespeare-regimes"),
        (f"{args.openwebtext}/shakespeare-char-patch/desmotic", "openwebtext-patch"),
        (f"{args.openwebtext}/regimes-all/desmotic", "openwebtext-regimes"),
    ]
    for src_dir, prefix in desmotic_sources:
        if os.path.exists(src_dir):
            for src_file in Path(src_dir).glob("*.png"):
                dst_file = f"{figures}/composite/{prefix}_{src_file.name}"
                copy_file(str(src_file), dst_file, args.quiet)

    # =========================================================================
    # 6. Suite reports
    # =========================================================================
    print("\n6. Suite reports:")
    for name, root in [("shakespeare", args.shakespeare), ("openwebtext", args.openwebtext)]:
        copy_file(f"{root}/report.md", f"{figures}/reports/{name}_report.md", args.quiet)

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\nDone! Figures collected in: {figures}/")

    # List what was created
    if os.path.exists(figures):
        total = sum(1 for _ in Path(figures).rglob("*.png"))
        print(f"Total PNG files: {total}")


if __name__ == "__main__":
    main()
