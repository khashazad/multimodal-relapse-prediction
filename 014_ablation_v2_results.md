# 014 — Ablation v2: Union vs All Feature Sets

## Summary

Re-ran all 7 ablation experiments with both `union` (24 features) and `all` (69 features) feature sets. 810 SLURM jobs total. Goal: determine whether the notebook's best model (union features) responds differently to regularization techniques than the full feature set.

**Key finding**: Focal loss + label smoothing is the best technique for both feature sets, achieving **0.8454 AUROC** (all, g=1.0,a=0.7,ls=0.2) and **0.8410 AUROC** (union, g=1.0,a=0.5,ls=0.2). The gap between feature sets is small (~0.004), suggesting the 24-feature union set captures most of the signal.

## Best Config per Technique

| Feature Set | Technique | Best AUROC | AUPRC | F1 | Best Config |
|---|---|---|---|---|---|
| all | baseline | 0.8079±0.098 | 0.6885 | 0.2518 | — |
| all | focal | 0.8143±0.114 | 0.7277 | 0.3589 | g=2.0, a=0.5 |
| all | **focal+smooth** | **0.8454±0.087** | 0.7450 | 0.4384 | g=1.0, a=0.7, ls=0.2 |
| all | label_smooth | 0.8385±0.093 | 0.7550 | 0.1463 | ls=0.2 |
| all | weight_decay | 0.8275±0.112 | 0.7077 | 0.2410 | wd=1e-4 |
| all | stoch_depth | 0.8133±0.106 | 0.7079 | 0.1749 | sd=0.2 |
| all | rope | 0.7538±0.084 | 0.6118 | 0.1512 | — |
| union | baseline | 0.8050±0.105 | 0.7068 | 0.4049 | — |
| union | focal | 0.8341±0.089 | 0.7162 | 0.2399 | g=3.0, a=0.5 |
| union | **focal+smooth** | **0.8410±0.096** | 0.7370 | 0.2865 | g=1.0, a=0.5, ls=0.2 |
| union | label_smooth | 0.8133±0.098 | 0.7501 | 0.2648 | ls=0.1 |
| union | weight_decay | 0.8153±0.091 | 0.7357 | 0.5382 | wd=1e-3 |
| union | stoch_depth | 0.8161±0.115 | 0.7321 | 0.2856 | sd=0.1 |
| union | rope | 0.7849±0.103 | 0.7039 | 0.2248 | — |

## Union vs All Comparison (best config per technique)

| Technique | Union AUROC | All AUROC | Delta | Winner |
|---|---|---|---|---|
| baseline | 0.8050 | 0.8079 | -0.003 | all |
| focal | 0.8341 | 0.8143 | +0.020 | union |
| focal+smooth | 0.8410 | 0.8454 | -0.004 | all |
| label_smooth | 0.8133 | 0.8385 | -0.025 | all |
| weight_decay | 0.8153 | 0.8275 | -0.012 | all |
| stoch_depth | 0.8161 | 0.8133 | +0.003 | union |
| rope | 0.7849 | 0.7538 | +0.031 | union |

## Overall Top 10 Configs

| Rank | Config | AUROC | AUPRC | F1 |
|---|---|---|---|---|
| 1 | all + focal(g=1.0,a=0.7) + ls=0.2 | 0.8454±0.087 | 0.7450 | 0.4384 |
| 2 | union + focal(g=1.0,a=0.5) + ls=0.2 | 0.8410±0.096 | 0.7370 | 0.2865 |
| 3 | all + ls=0.2 | 0.8385±0.093 | 0.7550 | 0.1463 |
| 4 | union + focal(g=3.0,a=0.5) | 0.8341±0.089 | 0.7162 | 0.2399 |
| 5 | union + focal(g=1.0,a=0.7) + ls=0.15 | 0.8326±0.109 | 0.7701 | 0.5966 |
| 6 | all + focal(g=0.5,a=0.5) + ls=0.15 | 0.8307±0.110 | 0.7127 | 0.3605 |
| 7 | all + wd=1e-4 | 0.8275±0.112 | 0.7077 | 0.2410 |
| 8 | union + focal(g=1.0,a=0.3) + ls=0.1 | 0.8265±0.107 | 0.7469 | 0.0000 |
| 9 | all + focal(g=1.0,a=0.5) + ls=0.1 | 0.8248±0.122 | 0.7450 | 0.2914 |
| 10 | all + focal(g=0.5,a=0.7) + ls=0.2 | 0.8247±0.094 | 0.7332 | 0.5752 |

## Key Observations

1. **Focal+smooth dominates**: top 2 configs both use focal loss + label smoothing. The combination provides ~+0.04 AUROC over baselines for both feature sets.

2. **Feature set gap is small**: baselines differ by only 0.003 AUROC (all=0.808, union=0.805). With optimal regularization, both reach ~0.84. The 45 extra features in `all` add minimal signal.

3. **Union benefits more from focal loss alone**: union+focal reaches 0.834 vs all+focal at 0.814 (+0.020). The sparser feature set may produce cleaner gradients for focal loss reweighting.

4. **All benefits more from label smoothing alone**: all+ls reaches 0.839 vs union+ls at 0.813 (+0.025). Higher-dimensional inputs may benefit more from soft targets.

5. **RoPE hurts both**: worst technique for both feature sets (0.754 all, 0.785 union), well below baselines. With only 7 timesteps, positional encoding adds noise.

6. **Weight decay helps all more**: all+wd=1e-4 gives 0.828 (+0.020 over baseline) while union+wd=1e-3 gives 0.815 (+0.010). More features = more parameters to regularize.

7. **F1 is noisy**: F1 varies wildly (0.00–0.60) even for configs with similar AUROC, reflecting threshold sensitivity in this imbalanced setting.

8. **Fold 8 (P9) NaN**: all+rope fold 8 produced NaN AUROC, excluded from that mean. Likely a degenerate prediction (all same class).

## Recommended Next Steps

- Best recipe to carry forward: **focal(g=1.0, a=0.7) + label_smoothing=0.2** on all features (AUROC 0.8454)
- For union specifically: **focal(g=1.0, a=0.5) + label_smoothing=0.2** (AUROC 0.8410)
- Consider combining with weight_decay=1e-4 (not yet tested jointly)
- Consider rank #5 (union + focal g=1.0,a=0.7,ls=0.15) which has the best F1 (0.5966) and AUPRC (0.7701)

## Config Files

14 configs in `configs/ablv2_*.json`, outputs in `outputs/ablations_v2/`.
