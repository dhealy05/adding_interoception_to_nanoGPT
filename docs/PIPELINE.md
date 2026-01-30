# Nanodesmo Experimental Pipeline

This document describes the complete data flow from raw experiments to final figures.

## Quick Start

```bash
# Run everything from scratch
pipenv run python scripts/run_all.py

# Dry run (see what would execute)
pipenv run python scripts/run_all.py --dry_run

# Resume after interruption
pipenv run python scripts/run_all.py --resume

# Run only a specific phase
pipenv run python scripts/run_all.py --only memory
```

---

## Pipeline Phases

```
Phase 1: Cleanup           Remove out/ and figures/
    │
    ▼
Phase 2: Shakespeare       configs/suites/shakespeare_core.yaml
    │                      → out/shakespeare/
    │                         ├── self-analysis/ckpt.pt
    │                         ├── regimes-all/reaction_log.jsonl
    │                         ├── regimes-all/plots/
    │                         └── */desmotic/
    ▼
Phase 3: Frozen test (shk) scripts/analysis/frozen_test_auto.py
    │                      ← out/shakespeare/self-analysis/ckpt.pt
    │                      → out/shakespeare/frozen_test_results.json
    │                      → out/shakespeare/frozen_test.png
    ▼
Phase 4: OpenWebText       configs/suites/openwebtext_progress.yaml
    │                      → out/openwebtext/
    │                         ├── self-analysis/ckpt.pt
    │                         ├── regimes-all/reaction_log.jsonl
    │                         └── */desmotic/
    ▼
Phase 5: Frozen test (owt) scripts/analysis/frozen_test_auto.py
    │                      ← out/openwebtext/self-analysis/ckpt.pt
    │                      → out/openwebtext/frozen_test_results.json
    │                      → out/openwebtext/frozen_test.png
    ▼
Phase 6: Memory            configs/suites/memory_buffer_grid.yaml
    │                      → out/memory/
    │                         ├── mem-ema-*/reaction_log.jsonl
    │                         ├── mem-buf-l64_*/reaction_log.jsonl
    │                         └── mem-buf-l128_*/reaction_log.jsonl
    ▼
Phase 7: Controls          configs/suites/reviewer_controls.yaml
    │                      → out/controls/
    │                         ├── vanilla-regimes/regime_deltas.csv
    │                         ├── self-regimes/regime_deltas.csv
    │                         ├── fixed-bias-*/regime_deltas.csv
    │                         └── self-regimes/mlp_analysis/
    ▼
Phase 8: Collect figures   scripts/plotting/collect_figures.py
    │                      Copies per-run plots from out/ to figures/
    ▼
Phase 9: Summary figures   scripts/plotting/generate_summary_figures.py
                           Generates cross-run comparison plots
```

---

## Data → Figure Traceability

| Finding | Phase | Data Source | Figure |
|---------|-------|-------------|--------|
| **Causal leverage** | 3, 5 | `frozen_test_auto.py` | `frozen-weight-test.png` |
| **Regime signatures** | 2 | `regimes-all/reaction_log.jsonl` | `perturbation/regime-reactions.png` |
| **Regime deltas** | 2 | `regimes-all/reaction_log.jsonl` | `perturbation/regime-deltas.png` |
| **Memory tradeoff** | 6 | `mem-*/reaction_log.jsonl` | `memory/memory-*.png` |
| **Control ablations** | 7 | `*/regime_deltas.csv` | `controls/controls-comparison.png` |
| **Generalization** | 4 | `openwebtext/regimes-all/` | `openwebtext/regime-*.png` |
| **Composite metrics** | 2, 4 | `*/desmotic/*.png` | `composite/*.png` |

---

## Output Directory Structure

```
out/
├── shakespeare/                    # Phase 2: shakespeare_core suite
│   ├── self-analysis/
│   │   └── ckpt.pt                 # Checkpoint for frozen test
│   ├── self-analysis-noself/
│   ├── regimes-all/
│   │   ├── reaction_log.jsonl      # Raw metrics per step
│   │   ├── plots/                  # Per-regime individual plots
│   │   └── desmotic/               # Composite metric figures
│   ├── shakespeare-char-patch/
│   │   └── desmotic/
│   ├── shakespeare-char-baseline/
│   ├── swap-a/, swap-b/, swap-a-swapped/, swap-b-swapped/
│   ├── frozen_test_results.json    # Phase 3 output
│   ├── frozen_test.png             # Phase 3 output
│   └── manifest.json
│
├── openwebtext/                    # Phase 4: openwebtext_progress suite
│   ├── self-analysis/
│   │   └── ckpt.pt
│   ├── regimes-all/
│   │   ├── reaction_log.jsonl
│   │   └── desmotic/
│   ├── frozen_test_results.json    # Phase 5 output
│   ├── frozen_test.png             # Phase 5 output
│   └── manifest.json
│
├── memory/                         # Phase 6: memory_buffer_grid suite
│   ├── mem-ema-decay-0.1_s10_k1/
│   │   ├── reaction_log.jsonl
│   │   ├── plots/
│   │   └── regime_deltas.csv
│   ├── mem-buf-l64_s10/
│   ├── mem-buf-l128_s10/
│   ├── memory_decay_summary.csv
│   └── memory_decay_summary.png
│
└── controls/                       # Phase 7: reviewer_controls suite
    ├── vanilla-regimes/
    │   ├── reaction_log.jsonl
    │   ├── regime_deltas.csv
    │   └── regime_delta_scatter.png
    ├── self-regimes/
    │   ├── reaction_log.jsonl
    │   ├── regime_deltas.csv
    │   ├── mlp_analysis/
    │   └── state_effect_leadlag.png
    ├── fixed-bias-0.5/
    ├── fixed-bias-1.0/
    └── self-dim1-regimes/

figures/                            # Phases 8-9: Final collected output
├── frozen-weight-test.png          # From out/shakespeare/
├── perturbation/
│   ├── regime-reactions.png        # Summary: all regimes on one plot
│   ├── regime-deltas.png           # Summary: bar chart comparison
│   └── regime_*.png                # Per-regime detail plots
├── memory/
│   ├── memory-self_drift.png       # Cross-config comparison
│   ├── memory-state_effect.png
│   └── *.png                       # Per-config plots
├── controls/
│   ├── controls-comparison.png     # Summary comparison
│   ├── *_delta_scatter.png         # Per-variant scatter plots
│   └── mlp/                        # MLP analysis plots
├── openwebtext/
│   ├── regime-reactions.png
│   └── regime-deltas.png
└── composite/
    └── *_desmotic_*.png            # Copied from desmotic/ dirs
```

---

## Suite Configuration Files

| Suite | Config | Purpose |
|-------|--------|---------|
| Shakespeare core | `configs/suites/shakespeare_core.yaml` | Main experiments: regimes, swaps, baselines |
| OpenWebText | `configs/suites/openwebtext_progress.yaml` | Generalization to different dataset |
| Memory | `configs/suites/memory_buffer_grid.yaml` | EMA vs buffer memory comparison |
| Controls | `configs/suites/reviewer_controls.yaml` | Ablations: vanilla, fixed-bias, dim=1 |

---

## Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_all.py` | Master orchestrator - runs full pipeline |
| `scripts/orchestration/run_suite.py` | Runs a single suite from YAML config |
| `scripts/analysis/frozen_test_auto.py` | Frozen-weight causal test |
| `scripts/plotting/collect_figures.py` | Copies per-run plots to figures/ |
| `scripts/plotting/generate_summary_figures.py` | Generates cross-run comparison plots |
| `scripts/analysis/regime_delta_summary.py` | Computes regime effect deltas |
| `scripts/plotting/plot_regimes.py` | Plots per-regime windows |
| `scripts/plotting/composite_metrics.py` | Generates desmotic figures |

---

## Running Individual Phases

```bash
# Just shakespeare suite
pipenv run python scripts/orchestration/run_suite.py \
  --suite configs/suites/shakespeare_core.yaml \
  --out_root out/shakespeare

# Just frozen test (requires checkpoint)
pipenv run python scripts/analysis/frozen_test_auto.py \
  --ckpt out/shakespeare/self-analysis/ckpt.pt \
  --out out/shakespeare/frozen_test_results.json \
  --plot out/shakespeare/frozen_test.png

# Just collect figures
pipenv run python scripts/plotting/collect_figures.py \
  --shakespeare out/shakespeare \
  --memory out/memory \
  --controls out/controls \
  --figures figures

# Just summary figures
pipenv run python scripts/plotting/generate_summary_figures.py \
  --shakespeare out/shakespeare \
  --memory out/memory \
  --controls out/controls \
  --figures figures
```

---

## Validation Checklist

After running the full pipeline:

- [ ] `figures/frozen-weight-test.png` exists and shows effect size > 0.3
- [ ] `figures/perturbation/regime-reactions.png` shows distinct regime signatures
- [ ] `figures/memory/` contains comparison plots
- [ ] `figures/controls/controls-comparison.png` shows ablation results
- [ ] All README figure references resolve
