# Suite Report (openwebtext)

## Val Loss
- self_state best_val_loss: 2.1956
- no_self best_val_loss: 2.2304
- delta (no_self - self): 0.0348

## Top Dims
- corr_step_frac:
  - dim 18: -0.9667
  - dim 7: 0.9663
  - dim 24: 0.9645
  - dim 14: -0.9621
  - dim 9: 0.9500
- corr_loss:
  - dim 3: -0.9027
  - dim 11: 0.8521
  - dim 26: 0.7861
  - dim 2: -0.7639
  - dim 21: 0.7609
- effect_peak:
  - dim 26: 0.05602
  - dim 15: 0.04358
  - dim 8: 0.03953
  - dim 11: 0.03842
  - dim 7: 0.03308

## Patch vs Baseline (log-derived)
- mean self_drift 1200-1400: patch=0.0760, baseline=0.0031
- mean state_effect 1200-1400: patch=0.3669, baseline=0.5594

## Swap vs Continued (log-derived)
- mean self_drift 1000-1100 (swap A): swapped=0.4020, continued=0.0016
- mean state_effect 1000-1100 (swap A): swapped=0.4076, continued=0.3915
- mean self_drift 1000-1100 (swap B): swapped=0.4404, continued=0.0023
- mean state_effect 1000-1100 (swap B): swapped=0.5668, continued=0.5387

## Regime Expansion (log-derived)
- bewilderment (1200-1400)
  - loss: pre=2.4395 during=4.7703 post=2.4431 delta=2.3309
  - self_drift: pre=0.0082 during=0.0685 post=0.0090 delta=0.0603
  - state_effect: pre=0.4743 during=0.5416 post=0.4688 delta=0.0673
  - stats_entropy: pre=2.4260 during=4.4618 post=2.6060 delta=2.0357
  - stats_top1_conf: pre=0.3101 during=0.1652 post=0.3089 delta=-0.1449
- false_feedback (900-1000)
  - loss: pre=2.4883 during=2.4704 post=2.4395 delta=-0.0179
  - self_drift: pre=0.0310 during=0.0222 post=0.0082 delta=-0.0087
  - state_effect: pre=0.5870 during=0.6402 post=0.4743 delta=0.0533
  - stats_entropy: pre=2.5176 during=2.4988 post=2.4260 delta=-0.0188
  - stats_top1_conf: pre=0.2888 during=0.2917 post=0.3101 delta=0.0028
- lr_heatwave (300-350)
  - loss: pre=2.6184 during=2.6269 post=2.5721 delta=0.0085
  - self_drift: pre=0.0119 during=0.0216 post=0.0132 delta=0.0097
  - state_effect: pre=0.8296 during=0.4450 post=0.6281 delta=-0.3846
  - stats_entropy: pre=2.7154 during=2.6630 post=2.6249 delta=-0.0524
  - stats_top1_conf: pre=0.2537 during=0.2593 post=0.2573 delta=0.0056
- self_clamp (1100-1200)
  - loss: pre=2.4395 during=2.4224 post=2.4431 delta=-0.0171
  - self_drift: pre=0.0082 during=0.1671 post=0.0090 delta=0.1589
  - state_effect: pre=0.4743 during=1.5935 post=0.4688 delta=1.1192
  - stats_entropy: pre=2.4260 during=2.4501 post=2.6060 delta=0.0240
  - stats_top1_conf: pre=0.3101 during=0.3077 post=0.3089 delta=-0.0023
- sensory_fog (600-800)
  - loss: pre=2.5565 during=2.7052 post=2.4883 delta=0.1487
  - self_drift: pre=0.0132 during=0.0574 post=0.0310 delta=0.0441
  - state_effect: pre=0.4265 during=1.1460 post=0.5870 delta=0.7195
  - stats_entropy: pre=2.5885 during=2.7072 post=2.5176 delta=0.1187
  - stats_top1_conf: pre=0.2731 during=0.2451 post=0.2888 delta=-0.0279

## Preset Eval (log-derived)
- confident: abs_logit_diff=0.392844, kl=0.037358, entropy 2.3627->2.3265, top1 0.3127->0.3370
- uncertain: abs_logit_diff=0.421392, kl=0.043198, entropy 2.3627->2.3276, top1 0.3127->0.3366
