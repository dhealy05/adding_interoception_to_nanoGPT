# Adding Interoception to NanoGPT

Interoception is sometimes called the "eighth sense", but really it's a collection of senses, the ones telling your body when you start to get thirsty, or hungry, or tired. It's feedback, basically, in the most fundamental form.

What does "feedback" mean to a fixed-weight LLM? Probably not much: no matter what happens, the weights will not change. What about during training? Weights are not yet decided: some feedback is possible.

Interoception is not possible in a vanilla LLM implementation without architectural changes. To add it, we augment the model with a 32-dimensional "self-state" vector:

```
                              ┌─────────────────────────────────┐
                              │                                 │
                              ▼                                 │
┌─────────┐    ┌──────────────────────┐    ┌───────────┐    ┌───┴───────┐
│  Input  │───▶│  Embeddings + Bias   │───▶│Transformer│───▶│  Logits   │
└─────────┘    └──────────────────────┘    └───────────┘    └─────┬─────┘
                        ▲                                         │
                        │                                         ▼
               ┌────────┴────────┐                      ┌─────────────────┐
               │   self_state    │◀─────────────────────│   Statistics    │
               │    (32-dim)     │      MLP update      │   from batch    │
               └─────────────────┘                      └─────────────────┘
```

The self-state persists across training steps. Each step, it receives statistics from the current batch and updates itself through a learned MLP. Then it injects a bias into the model's embeddings, where the model can learn to read it.

---

## What the Self-State Sees

The self-state receives five signals from each training batch:

- **Loss** — how wrong the predictions were
- **Entropy** — how spread out the probability distribution is (uncertain = high entropy)
- **Top-1 confidence** — probability assigned to the most likely next token
- **Gradient norm** — how aggressively the optimizer wants to change weights
- **Training progress** — what fraction of training is complete

From these, the MLP produces an update. The state changes slowly (80% old, 20% new), so it acts as a smoothed summary of recent training conditions.

---

## What We Measure

To see whether the self-state is doing anything interesting, we track:

- **state_drift** — how fast the state is moving (L2 distance between successive states)
- **state_effect** — how much the state actually changes the model's outputs
- **state_norm** — the overall magnitude of the state vector

---

## What We Find

The self-state learns directions that matter. When we freeze the weights and manually steer the state toward its learned "confident" direction, entropy drops. Steer it toward "uncertain," entropy rises. Random directions of the same magnitude? Nothing consistent.

![Frozen-weight causal test](figures/frozen-weight-test.png)
*Frozen-weight test. Learned directions (confident/uncertain) produce consistent effects; random directions do not.*

| Condition | Δ entropy vs zero | Sign consistency |
|-----------|-------------------|------------------|
| Confident (learned) | **-0.09** | 100% (always reduces) |
| Uncertain (learned) | **+0.26** | 100% (always increases) |
| Random (n=50, norm-matched) | +0.04 ± 0.10 | 44% (mixed) |

The state has found structure that the weights have learned to use.

**Effect size** = |entropy(confident) - entropy(uncertain)| = **0.35**

This measures the spread between steering toward high vs low entropy. Values >0.3 indicate meaningful causal leverage—the learned directions produce reliably different outcomes.

### What Each Test Shows Us

| Test | Question | Answer |
|------|----------|--------|
| Frozen-weight steering | Does the state do anything? | Yes — learned directions reliably shift model outputs; random ones don't |
| Fixed-bias ablation | Is it just adding a constant? | No — static injection doesn't reproduce the dynamic patterns |
| 1D ablation | Does it need to be multi-dimensional? | Yes — 1D state shows weaker, sometimes absent responses |
| Vanilla comparison | Are the effects unique to having a self-state? | Partially — loss/entropy shifts happen either way, but state-specific signals (drift, effect) add new information |
| Regime classification | Can you tell what's wrong from the state? | Yes — with state channels, you get perfect classification of perturbation type |
| Memory experiments | Does remembering help? | Tradeoff — more memory means more stability but slower adaptation |

### Perturbation Regimes

We stress-test the system by injecting various perturbations during training windows:

| Regime | What it does |
|--------|--------------|
| Target corruption | Corrupt labels with probability 0.5 |
| Input masking | Mask input tokens |
| Statistics bias | Bias the statistics fed to self-state |
| State freeze | Prevent self-state updates |
| LR perturbation | Multiply learning rate |

Each regime produces a distinct signature in the state-specific channels:

![Perturbation regime reactions](figures/perturbation/regime-reactions.png)
*Different perturbation regimes produce distinct signatures in state-specific channels.*

| Regime | loss Δ | entropy Δ | drift Δ | effect Δ |
|--------|--------|-----------|---------|----------|
| bewilderment | +1.65 | +1.48 | +0.030 | -0.18 |
| sensory_fog | +0.26 | +0.27 | -0.002 | +0.07 |
| lr_heatwave | -0.01 | -0.05 | +0.27 | -0.01 |
| self_clamp | -0.03 | -0.02 | +0.21 | -0.19 |
| false_feedback | -0.05 | +0.01 | -0.002 | +0.01 |

Each regime has a unique fingerprint. With just loss, entropy, and confidence, you can classify regimes with 80% accuracy. Add state_drift and state_effect? **Perfect classification.**

### Memory: Stability vs Control

We also tested giving the self-state explicit memory of past states (a ring buffer with attention) instead of just exponential moving average smoothing.

| Memory type | Memory share | Restoring force | Control leverage |
|-------------|--------------|-----------------|------------------|
| EMA (decay 0.1) | 0.10 | weak | higher |
| Buffer (L=64) | 0.34 | strong | lower |
| Buffer (L=128) | 0.43 | very strong | lowest |

| EMA baseline | Buffer (L=64) | Buffer (L=128) |
|:------------:|:-------------:|:--------------:|
| ![EMA](figures/memory/ema-baseline_regime_bewilderment.png) | ![L64](figures/memory/buffer-l64_regime_bewilderment.png) | ![L128](figures/memory/buffer-l128_regime_bewilderment.png) |

*Update dynamics under target corruption. Buffer memory shows stronger restoring force (memory fraction increases with buffer size).*

**The tradeoff:** More memory → more stability → less control leverage. The state gets anchored to its history, which helps it resist perturbation but weakens its ability to rapidly influence outputs.

### What It Adds Up To

The model learns to *use* its internal state. It actively develops structure that the weights learn to read. The state becomes a low-dimensional summary of "how things are going" that has causal influence on outputs.

*Can we make a model's internal condition observable and manipulable?* For the toy model, the answer is yes. The self-state gives you something to point at, measure, and intervene on—which is more than you get from loss curves alone.

---

## What Is a Self-State?

In this case the self state acts like a training log of sorts: it's a compressed, smoothed-out trace of training conditions. High loss? Unstable gradients? The state carries an echo of that even after the moment passes.

The swap experiments show this most clearly: when you take a state trained on one trajectory and drop it into a model with a different history, things break. The state is *entangled* with the weights it grew up alongside; they learned to read each other.

What happens if you scale it up?

It's not obvious. But plausibly it would be interesting to observe what a persistent state trained alongside the weights would learn.

---

## Related Ideas, and How This Differs

Persistent state in neural networks isn't new. But most of it lives in a different place than what we're doing here. The self-state sits in an unusual spot:

|  | Persists across training | Model reads it (learned) | Derived from model's own signals |
|--|--------------------------|--------------------------|----------------------------------|
| RNN hidden state | ✗ | ✓ | ✗ |
| FiLM conditioning | ✗ | ✓ | ✗ |
| Hypernetworks | ✗ | ✓ | ✗ |
| BatchNorm stats | ✓ | ✗ (fixed formula) | ✓ |
| Meta-learning | ✓ | ✗ (modifies optimizer) | ✓ |
| **Self-state** | ✓ | ✓ | ✓ |

It's the combination that's unusual: a state that (1) accumulates over training, (2) is updated by a learned function of the model's own statistics, and (3) is read by the model through learned weights. The model is, in a limited sense, learning to condition itself on its own history.

---

## Documentation

- [docs/PIPELINE.md](docs/PIPELINE.md) — How to run the full experiment pipeline
- [docs/results/shakespeare.md](docs/results/shakespeare.md) — Core training, regime signatures
- [docs/results/frozen-test.md](docs/results/frozen-test.md) — Causal test results
- [docs/results/memory.md](docs/results/memory.md) — EMA vs buffer comparison
- [docs/results/controls.md](docs/results/controls.md) — Ablation experiments
- [docs/results/openwebtext.md](docs/results/openwebtext.md) — Generalization to OpenWebText

---

## Quick Start

```bash
# Install dependencies
pipenv install

# Prepare data
pipenv run python data/shakespeare_char/prepare.py

# Run the full pipeline (all experiments + figures)
pipenv run python scripts/run_all.py

# Or run individual phases:
pipenv run python scripts/run_all.py --only shakespeare
pipenv run python scripts/run_all.py --only frozen_test
pipenv run python scripts/run_all.py --only figures
```

---

## Limitations

1. **Scale**: 0.80M parameters, CPU training. Whether these dynamics appear at larger scales is unknown.
2. **Dataset**: Primarily Shakespeare characters. OpenWebText generalization in [docs/generalization.md](docs/generalization.md).
3. **Inference**: The self-state only updates during training. We haven't explored inference-time dynamics.
