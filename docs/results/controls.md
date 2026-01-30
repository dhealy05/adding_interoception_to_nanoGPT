# Control Experiments Results

**Pipeline phase:** 7 (controls)
**Suite:** `configs/suites/reviewer_controls.yaml`
**Output:** `out/controls/`

## What We Ran

Ablation experiments to verify the self-state mechanism is doing something non-trivial:

| Condition | Description |
|-----------|-------------|
| **vanilla** | No self-state at all |
| **self** | Full self-state (baseline) |
| **fixed-bias** | Static embedding bias (no dynamics) |
| **dim1** | 1-dimensional self-state |

Each condition ran with and without perturbation regimes.

## What We Got

**Vanilla vs Self-state (with regimes):**

| Metric | Vanilla | Self-state |
|--------|---------|------------|
| self_drift | — | 0.030 (bewilderment) |
| state_effect | — | -0.18 (bewilderment) |
| Regime classification | 77-80% | 100% |

Vanilla has no self_drift or state_effect columns (expected — no self-state mechanism).

**Fixed-bias ablation:**

Static injection doesn't reproduce the dynamic patterns. The learned dynamics matter.

**1D ablation:**

1-dimensional state shows weaker, sometimes absent responses to perturbations. The multi-dimensional structure carries meaningful information.

## Key Finding

The self-state effects are:
1. **Not just a constant** — fixed-bias doesn't replicate them
2. **Not reducible to 1D** — multi-dimensional structure matters
3. **Unique to having a self-state** — vanilla baseline lacks these signals entirely

## Data Files

- `out/controls/vanilla-regimes/regime_deltas.csv` — vanilla baseline
- `out/controls/self-regimes/regime_deltas.csv` — full self-state
- `out/controls/fixed-bias-*/regime_deltas.csv` — fixed-bias ablations
- `out/controls/self-dim1-regimes/regime_deltas.csv` — 1D ablation
