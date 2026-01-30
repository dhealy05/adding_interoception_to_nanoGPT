# Suite Report (shakespeare)

## Val Loss
- self_state best_val_loss: 2.0670
- no_self best_val_loss: 2.0773
- delta (no_self - self): 0.0103

## Top Dims
- corr_step_frac:
  - dim 12: 0.9568
  - dim 2: -0.9567
  - dim 9: -0.9486
  - dim 22: -0.9472
  - dim 27: 0.9423
- corr_loss:
  - dim 8: -0.7778
  - dim 12: -0.7640
  - dim 9: 0.7421
  - dim 2: 0.7324
  - dim 10: 0.7324
- effect_peak:
  - dim 25: 0.04173
  - dim 31: 0.03585
  - dim 23: 0.03546
  - dim 2: 0.03270
  - dim 5: 0.02497

## Patch vs Baseline (log-derived)
- mean self_drift 1200-1400: patch=0.0050, baseline=0.0005
- mean state_effect 1200-1400: patch=0.4919, baseline=0.9876

## Swap vs Continued (log-derived)
- mean self_drift 1000-1100 (swap A): swapped=0.3224, continued=0.0018
- mean state_effect 1000-1100 (swap A): swapped=0.7280, continued=0.7604
- mean self_drift 1000-1100 (swap B): swapped=0.3722, continued=0.0015
- mean state_effect 1000-1100 (swap B): swapped=0.6308, continued=0.6213

## Regime Expansion (log-derived)
- bewilderment (1200-1400)
  - loss: pre=2.2844 during=3.7683 post=2.3203 delta=1.4839
  - self_drift: pre=0.0029 during=0.0221 post=0.0110 delta=0.0191
  - state_effect: pre=1.1513 during=0.5076 post=0.4985 delta=-0.6437
  - stats_entropy: pre=2.2948 during=3.6060 post=2.5135 delta=1.3112
  - stats_top1_conf: pre=0.3387 during=0.1529 post=0.3115 delta=-0.1858
- false_feedback (900-1000)
  - loss: pre=2.3682 during=2.3205 post=2.2844 delta=-0.0477
  - self_drift: pre=0.0229 during=0.0095 post=0.0029 delta=-0.0134
  - state_effect: pre=0.6733 during=0.6049 post=1.1513 delta=-0.0684
  - stats_entropy: pre=2.3892 during=2.3578 post=2.2948 delta=-0.0314
  - stats_top1_conf: pre=0.3153 during=0.3243 post=0.3387 delta=0.0090
- lr_heatwave (300-350)
  - loss: pre=2.5134 during=2.5181 post=2.4640 delta=0.0046
  - self_drift: pre=0.0364 during=0.3136 post=0.2164 delta=0.2771
  - state_effect: pre=0.6278 during=0.3844 post=0.4543 delta=-0.2435
  - stats_entropy: pre=2.5547 during=2.5335 post=2.5107 delta=-0.0212
  - stats_top1_conf: pre=0.2891 during=0.2863 post=0.2841 delta=-0.0028
- self_clamp (1100-1200)
  - loss: pre=2.2844 during=2.2798 post=2.3203 delta=-0.0047
  - self_drift: pre=0.0029 during=0.1202 post=0.0110 delta=0.1173
  - state_effect: pre=1.1513 during=0.7766 post=0.4985 delta=-0.3747
  - stats_entropy: pre=2.2948 during=2.2855 post=2.5135 delta=-0.0092
  - stats_top1_conf: pre=0.3387 during=0.3373 post=0.3115 delta=-0.0014
- sensory_fog (600-800)
  - loss: pre=2.4256 during=2.6530 post=2.3682 delta=0.2274
  - self_drift: pre=0.0471 during=0.0690 post=0.0229 delta=0.0219
  - state_effect: pre=0.7133 during=0.9932 post=0.6733 delta=0.2799
  - stats_entropy: pre=2.4261 during=2.6429 post=2.3892 delta=0.2168
  - stats_top1_conf: pre=0.3036 during=0.2647 post=0.3153 delta=-0.0390

## Preset Eval (log-derived)
- confident: abs_logit_diff=0.347150, kl=0.020830, entropy 2.1222->2.0466, top1 0.4075->0.4217
- uncertain: abs_logit_diff=0.313089, kl=0.016717, entropy 2.1222->2.0518, top1 0.4075->0.4174
