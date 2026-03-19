# 015b — Transformer-Only Ensemble Results

## Summary

| Method | Mean AUROC | Std | Mean AUPRC | Std | N Folds |
|--------|-----------|-----|-----------|-----|---------|
| Single best (per-fold)         | 0.8569 | 0.0814 | 0.7572 | 0.1855 | 9 |
| Simple average                 | 0.8262 | 0.0944 | 0.7061 | 0.2170 | 9 |
| Rank average                   | 0.9119 | 0.0558 | 0.8526 | 0.1226 | 9 |
| Weighted average               | 0.8489 | 0.0823 | 0.7422 | 0.1789 | 9 |
| Stacking (LR)                  | 0.8242 | 0.1056 | 0.7099 | 0.2096 | 9 |
| FiLM transformer               | 0.8433 | 0.0951 | 0.7326 | 0.2055 | 9 |
| TTA_entropy_ln_lr0.0001        | 0.8327 | 0.0955 | 0.7311 | 0.1790 | 9 |
| TTA_entropy_ln_lr0.001         | 0.8329 | 0.0957 | 0.7327 | 0.1793 | 9 |
| TTA_feature_renorm_lr0.0001    | 0.8195 | 0.1019 | 0.7203 | 0.1738 | 9 |
| TTA_feature_renorm_lr0.001     | 0.8194 | 0.1020 | 0.7200 | 0.1739 | 9 |


## Per-Fold AUROC

| Fold | Patient | Single best (pe |  Simple average |    Rank average | Weighted averag |   Stacking (LR) | FiLM transforme | TTA_entropy_ln_ | TTA_entropy_ln_ | TTA_feature_ren | TTA_feature_ren |
|------|---------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| 0    | P1      |          0.7229 |          0.7242 |          0.8938 |          0.7258 |          0.7633 |          0.7217 |          0.7192 |          0.7192 |          0.6883 |          0.6883 |
| 1    | P2      |          0.7916 |          0.7368 |          0.8046 |          0.7851 |          0.7426 |          0.8623 |          0.7916 |          0.7902 |          0.7394 |          0.7394 |
| 2    | P3      |          0.8638 |          0.7596 |          0.8555 |          0.7612 |          0.7382 |          0.8900 |          0.8050 |          0.8061 |          0.7607 |          0.7607 |
| 3    | P4      |          0.8708 |          0.8992 |          0.9414 |          0.9091 |          0.9474 |          0.8521 |          0.8708 |          0.8719 |          0.8521 |          0.8521 |
| 4    | P5      |          0.7936 |          0.7989 |          0.8931 |          0.8004 |          0.7614 |          0.7289 |          0.6730 |          0.6720 |          0.7045 |          0.7036 |
| 5    | P6      |          0.8211 |          0.8684 |          0.9102 |          0.8694 |          0.8609 |          0.7850 |          0.8146 |          0.8160 |          0.8112 |          0.8112 |
| 6    | P7      |          0.8857 |          0.7220 |          0.9399 |          0.8619 |          0.6750 |          0.7720 |          0.8857 |          0.8857 |          0.8643 |          0.8643 |
| 7    | P8      |          0.9893 |          0.9872 |          1.0000 |          0.9872 |          0.9807 |          1.0000 |          0.9818 |          0.9786 |          0.9850 |          0.9850 |
| 8    | P9      |          0.9731 |          0.9394 |          0.9684 |          0.9402 |          0.9478 |          0.9773 |          0.9529 |          0.9562 |          0.9697 |          0.9697 |


## Statistical Tests

  Wilcoxon (Simple average > Single best (per-fold)): stat=14.0, p=0.8496 (not significant)
  Wilcoxon (Rank average > Single best (per-fold)): stat=42.0, p=0.0098 (significant)
  Wilcoxon (Weighted average > Single best (per-fold)): stat=21.0, p=0.5898 (not significant)
  Wilcoxon (Stacking (LR) > Single best (per-fold)): stat=16.0, p=0.7871 (not significant)
  Wilcoxon (FiLM transformer > Single best (per-fold)): stat=18.0, p=0.7148 (not significant)
  Wilcoxon (TTA_entropy_ln_lr0.0001 > Single best (per-fold)): stat=0.0, p=1.0000 (not significant)
  Wilcoxon (TTA_entropy_ln_lr0.001 > Single best (per-fold)): stat=1.0, p=0.9961 (not significant)
  Wilcoxon (TTA_feature_renorm_lr0.0001 > Single best (per-fold)): stat=0.0, p=1.0000 (not significant)
  Wilcoxon (TTA_feature_renorm_lr0.001 > Single best (per-fold)): stat=0.0, p=1.0000 (not significant)

## Transformer Subset Analysis

| Method | Mean AUROC | Std | Mean AUPRC | Std | N Folds |
|--------|-----------|-----|-----------|-----|---------|
| Rank: All 6 transformers (6)   | 0.9119 | 0.0558 | 0.8526 | 0.1226 | 9 |
| Avg: All 6 transformers (6)    | 0.8262 | 0.0944 | 0.7061 | 0.2170 | 9 |
| Rank: Top-4 (d>=512) (4)       | 0.9021 | 0.0597 | 0.8433 | 0.1197 | 9 |
| Avg: Top-4 (d>=512) (4)        | 0.8483 | 0.0769 | 0.7489 | 0.1878 | 9 |
| Rank: Top-2 (d=1024) (2)       | 0.8683 | 0.0737 | 0.7942 | 0.1499 | 9 |
| Avg: Top-2 (d=1024) (2)        | 0.8191 | 0.0838 | 0.7256 | 0.1676 | 9 |
| Rank: Trans+FiLM (8)           | 0.9383 | 0.0453 | 0.9167 | 0.0740 | 9 |
| Avg: Trans+FiLM (8)            | 0.8316 | 0.0889 | 0.7247 | 0.2002 | 9 |


### Subset Per-Fold AUROC

| Fold | Patient | Rank: All 6 tra | Rank: Top-4 (d> | Rank: Top-2 (d= | Rank: Trans+FiL |
|------|---------|----------------|----------------|----------------|----------------|
| 0    | P1      |          0.8938 |          0.8517 |          0.7875 |          0.9129 |
| 1    | P2      |          0.8046 |          0.8082 |          0.8165 |          0.8571 |
| 2    | P3      |          0.8555 |          0.8937 |          0.8590 |          0.8910 |
| 3    | P4      |          0.9414 |          0.9463 |          0.9321 |          0.9901 |
| 4    | P5      |          0.8931 |          0.8454 |          0.7639 |          0.9169 |
| 5    | P6      |          0.9102 |          0.8694 |          0.8167 |          0.9442 |
| 6    | P7      |          0.9399 |          0.9494 |          0.9461 |          0.9521 |
| 7    | P8      |          1.0000 |          0.9989 |          0.9893 |          1.0000 |
| 8    | P9      |          0.9684 |          0.9554 |          0.9032 |          0.9806 |


### Subset vs Baseline (Wilcoxon)

  Wilcoxon (Rank: All 6 transformers (6) > Single best (per-fold)): stat=42.0, p=0.0098 (significant)
  Wilcoxon (Rank: Top-4 (d>=512) (4) > Single best (per-fold)): stat=42.0, p=0.0098 (significant)
  Wilcoxon (Rank: Top-2 (d=1024) (2) > Single best (per-fold)): stat=21.0, p=0.3711 (not significant)
  Wilcoxon (Rank: Trans+FiLM (8) > Single best (per-fold)): stat=45.0, p=0.0020 (significant)

## Diagnosis Group Breakdown


### single_best — by diagnosis group
  bipolar: AUROC=0.8853 (±0.0727, n=6)
  schiz-spectrum: AUROC=0.8387 (±0.0470, n=2)
  brief-psychotic: AUROC=0.7229 (±0.0000, n=1)

### simple_avg — by diagnosis group
  bipolar: AUROC=0.8754 (±0.0780, n=6)
  schiz-spectrum: AUROC=0.7294 (±0.0074, n=2)
  brief-psychotic: AUROC=0.7242 (±0.0000, n=1)

### rank_avg — by diagnosis group
  bipolar: AUROC=0.9281 (±0.0479, n=6)
  schiz-spectrum: AUROC=0.8722 (±0.0676, n=2)
  brief-psychotic: AUROC=0.8938 (±0.0000, n=1)

### weighted_avg — by diagnosis group
  bipolar: AUROC=0.8779 (±0.0780, n=6)
  schiz-spectrum: AUROC=0.8235 (±0.0384, n=2)
  brief-psychotic: AUROC=0.7258 (±0.0000, n=1)

### stacking — by diagnosis group
  bipolar: AUROC=0.8728 (±0.0944, n=6)
  schiz-spectrum: AUROC=0.7088 (±0.0338, n=2)
  brief-psychotic: AUROC=0.7633 (±0.0000, n=1)

### film — by diagnosis group
  bipolar: AUROC=0.8722 (±0.0968, n=6)
  schiz-spectrum: AUROC=0.8172 (±0.0451, n=2)
  brief-psychotic: AUROC=0.7217 (±0.0000, n=1)

### TTA_entropy_ln_lr0.0001 — by diagnosis group
  bipolar: AUROC=0.8497 (±0.1025, n=6)
  schiz-spectrum: AUROC=0.8387 (±0.0470, n=2)
  brief-psychotic: AUROC=0.7192 (±0.0000, n=1)

### TTA_entropy_ln_lr0.001 — by diagnosis group
  bipolar: AUROC=0.8501 (±0.1025, n=6)
  schiz-spectrum: AUROC=0.8380 (±0.0478, n=2)
  brief-psychotic: AUROC=0.7192 (±0.0000, n=1)

### TTA_feature_renorm_lr0.0001 — by diagnosis group
  bipolar: AUROC=0.8472 (±0.1026, n=6)
  schiz-spectrum: AUROC=0.8018 (±0.0625, n=2)
  brief-psychotic: AUROC=0.6883 (±0.0000, n=1)

### TTA_feature_renorm_lr0.001 — by diagnosis group
  bipolar: AUROC=0.8471 (±0.1028, n=6)
  schiz-spectrum: AUROC=0.8018 (±0.0625, n=2)
  brief-psychotic: AUROC=0.6883 (±0.0000, n=1)

### subset_rank_All 6 transformers — by diagnosis group
  bipolar: AUROC=0.9281 (±0.0479, n=6)
  schiz-spectrum: AUROC=0.8722 (±0.0676, n=2)
  brief-psychotic: AUROC=0.8938 (±0.0000, n=1)

### subset_avg_All 6 transformers — by diagnosis group
  bipolar: AUROC=0.8754 (±0.0780, n=6)
  schiz-spectrum: AUROC=0.7294 (±0.0074, n=2)
  brief-psychotic: AUROC=0.7242 (±0.0000, n=1)

### subset_rank_Top-4 (d>=512) — by diagnosis group
  bipolar: AUROC=0.9182 (±0.0532, n=6)
  schiz-spectrum: AUROC=0.8788 (±0.0706, n=2)
  brief-psychotic: AUROC=0.8517 (±0.0000, n=1)

### subset_avg_Top-4 (d>=512) — by diagnosis group
  bipolar: AUROC=0.8776 (±0.0762, n=6)
  schiz-spectrum: AUROC=0.7887 (±0.0381, n=2)
  brief-psychotic: AUROC=0.7917 (±0.0000, n=1)

### subset_rank_Top-2 (d=1024) — by diagnosis group
  bipolar: AUROC=0.8774 (±0.0743, n=6)
  schiz-spectrum: AUROC=0.8813 (±0.0648, n=2)
  brief-psychotic: AUROC=0.7875 (±0.0000, n=1)

### subset_avg_Top-2 (d=1024) — by diagnosis group
  bipolar: AUROC=0.8283 (±0.0883, n=6)
  schiz-spectrum: AUROC=0.8414 (±0.0497, n=2)
  brief-psychotic: AUROC=0.7192 (±0.0000, n=1)

### subset_rank_trans_film — by diagnosis group
  bipolar: AUROC=0.9538 (±0.0399, n=6)
  schiz-spectrum: AUROC=0.9046 (±0.0475, n=2)
  brief-psychotic: AUROC=0.9129 (±0.0000, n=1)

### subset_avg_trans_film — by diagnosis group
  bipolar: AUROC=0.8754 (±0.0772, n=6)
  schiz-spectrum: AUROC=0.7559 (±0.0090, n=2)
  brief-psychotic: AUROC=0.7208 (±0.0000, n=1)