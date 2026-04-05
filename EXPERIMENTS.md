# Experiment Catalog

Binary per-day relapse prediction from wearable sensor data (9 patients, LOSO cross-validation).

## Results Summary

| Phase | Method | Features | AUROC | Folds | AUPRC | Source |
|-------|--------|----------|-------|-------|-------|--------|
| Notebook | XGBoost baseline | sleep+steps | ~0.530 | 9/9 | — | `notebooks/main.ipynb` |
| Notebook | Logistic Regression | sleep+steps | ~0.580 | 9/9 | — | `notebooks/main.ipynb` |
| Notebook | MLP | 24 feat | ~0.550 | 9/9 | — | `notebooks/main.ipynb` |
| Notebook | Transformer (bipolar-6) | 24 feat | 0.849 | 6/6 | 0.794 | `notebooks/main.ipynb` |
| Notebook | Transformer (all-9) | 24 feat | 0.793 | 9/9 | 0.705 | `notebooks/main.ipynb` |
| Framework | Focal loss (d=32, exploration) | 69 feat | 0.664 | 9/9 | 0.545 | `docs/results/exp_focal.md` |
| Framework | Baseline BCE (d=1024) | 69 feat | 0.824 | 8/9 | 0.689 | `docs/results/ablation_baseline.md` |
| Framework | Focal loss (γ=1, α=0.5) | 69 feat | 0.822 | 9/9 | 0.739 | `docs/results/ablation_focal.md` |
| Framework | Focal+smooth (v1) | 69 feat | 0.857 | 5/9 ⚠️ | 0.690 | `docs/results/ablation_focal_smooth.md` |
| Framework | Label smoothing | 69 feat | 0.839 | 9/9 | 0.704 | `docs/results/ablation_label_smooth.md` |
| Framework | **Training recipe (focal+gc)** | **69 feat** | **0.842** | **9/9** | **0.749** | `docs/results/ablation_recipe.md` |
| Framework | Focal+smooth (v2, all) | 69 feat | 0.841 | 6/9 ⚠️ | 0.725 | `docs/results/ablv2_all_focal_smooth.md` |
| Framework | Focal+smooth (v2, union) | 24 feat | 0.827 | 8/9 ⚠️ | 0.686 | `docs/results/ablv2_union_focal_smooth.md` |
| Framework | Focal+smooth (v2 stab, all) | 69 feat | *pending* | 9/9 | *pending* | `docs/results/ablv2_all_focal_smooth_v2.md` |
| Framework | Focal+smooth (v2 stab, union) | 24 feat | *pending* | 9/9 | *pending* | `docs/results/ablv2_union_focal_smooth_v2.md` |

⚠️ = mean computed over fewer than 9 folds (NaN folds excluded). Scientifically invalid for LOSO reporting.

**Headline result: single transformer with focal loss + grad_clip → 0.842 AUROC** (d=1024, 3L, 4H, LOSO 9/9 patients). Pending stabilized focal+smooth rerun.

## Experiment Phases

### Phase 1: Feature Engineering (notebooks/main.ipynb)

Progressive feature extraction from 5 wearable modalities:
- **Sleep**: main episode duration, nap count/duration, z-score deviations from personal baseline
- **Steps**: daily totals, inverted z-score (fewer steps = higher risk)
- **HRV**: nighttime RMSSD/SDNN from accelerometer-filtered heart rate
- **Circadian**: hourly activity profiles from gyroscope + accelerometer fusion (relative amplitude, intradaily variability, cosinor, L5/M10)
- **Demographics**: age, sex, diagnosis

24-feature "union" set captures most signal (confirmed in ablation v2).

### Phase 2: Traditional ML (notebooks/main.ipynb)

XGBoost (0.53) → Logistic Regression (0.58) → MLP (0.55). All underperformed relative to the sequential transformer.

### Phase 3: Transformer Scaling (notebooks/main.ipynb → framework)

- d=32 transformer: ~0.51 (underfit)
- d=1024 transformer: 0.849 (bipolar-6), 0.793 (all-9)
- Moved to experiment framework for systematic grid search

### Phase 4: Architecture Variants (framework, exploration configs)

Four transformer variants tested: bottleneck (v2), gated fusion (v3), DANN domain-adversarial (v4). None beat the standard transformer v1.

### Phase 5: Ablation Study V1 (framework, 621 SLURM jobs)

Systematic comparison of loss functions and regularization with d=1024, 3L, 4H. Baseline=0.824 (8/9 folds).

| Technique | AUROC | Folds | Δ Baseline |
|-----------|-------|-------|------------|
| Focal+smooth | 0.857 | 5/9 ⚠️ | — |
| Label smoothing (ε=0.05) | 0.839 | 9/9 | +0.015 |
| Stochastic depth (p=0.3) | 0.824 | 8/9 | +0.000 |
| Focal loss (γ=1, α=0.5) | 0.822 | 9/9 | −0.002 |
| Weight decay (λ=1e-3) | 0.811 | 8/9 | −0.013 |
| RoPE | 0.758 | 8/9 | −0.066 |

### Phase 6: Training Recipe Sweep (framework, 216 SLURM jobs)

Tested optimizer (Adam/AdamW), LR scheduling (none/cosine), gradient clipping, warmup. Best: adam + fixed LR + grad_clip=1.0 → 0.842 (9/9 folds). Cosine schedule actively hurt performance.

### Phase 7: Ablation V2 (framework, 810 SLURM jobs)

Re-ran all ablations with both union (24) and all (69) feature sets. "All" feature set configs are fully valid (9/9 folds) except focal+smooth (6/9). Union configs have sporadic NaN folds.

### Phase 8: Stabilized Focal+Smooth Rerun (pending)

Added training stabilization (NaN recovery, checkpoint guards, auto grad_clip for focal+smooth). New configs: `ablv2_all_focal_smooth_v2`, `ablv2_union_focal_smooth_v2` (grad_clip=1.0, AdamW). Same sweep grid as original. To be submitted to SLURM.

## Key Findings

1. **Loss function > regularization**: label smoothing (+0.015) helps; focal loss alone is neutral vs baseline but combined with grad clipping gives best result (0.842)
2. **Mild focal focus is optimal**: γ=1.0 >> γ=2.0 or 3.0 — with only 9 patients and ~20 relapse days per fold, aggressive down-weighting loses gradient signal
3. **RoPE hurts short sequences**: 7-day windows are too short for relative position bias to help
4. **Bipolar subset is stronger**: bipolar-only (6 patients) AUROC 0.849 vs all-9 at 0.793 — non-bipolar patients (P1 brief psychotic, P2 schizophreniform, P7 schizophrenia) are harder
5. **24 features ≈ 69 features**: the union feature set captures most signal; extra features add noise without improving AUROC
6. **Per-patient variance dominates**: oracle AUROC across all configs is 0.907, but no single config achieves this — the 0.062 gap requires per-patient adaptation

## Negative Results

- **Recipe sweep produced best valid result**: adam + grad_clip=1.0 → 0.842 (9/9), the only fully-valid result above 0.84
- **Cosine schedule hurts**: counter-intuitively, fixed LR oscillation aids best-epoch checkpoint selection
- **NaN folds are widespread**: V1 baseline, stoch_depth, weight_decay, rope all have 1 NaN fold; focal+smooth has 4-5 NaN folds. Training instability without gradient clipping
- **Ensemble methods** (0.912–0.938): not credible with 9 patients; removed from repository
- **Unsupervised pretraining** (BumbleBee autoencoder): 0.48 AUROC — worse than random when used alone, and fusion with supervised features (0.72) hurt vs supervised-only

## Reproduction

### Headline result (0.842 AUROC, pending stabilized rerun)

```bash
# Preprocess data
python scripts/preprocess_data.py

# Best fully-valid result: training recipe (focal + grad_clip)
bash scripts/submit_slurm.sh -n ablation_recipe

# Stabilized focal+smooth rerun (pending)
bash scripts/submit_slurm.sh -n ablv2_all_focal_smooth_v2
bash scripts/submit_slurm.sh -n ablv2_union_focal_smooth_v2
```

Config: `configs/ablation_recipe.json` — focal loss (γ=1.0, α=0.5) + grad_clip=1.0, 9-fold LOSO.
Stabilized: `configs/ablv2_all_focal_smooth_v2.json` — focal + label smoothing + grad_clip=1.0 + AdamW.

### Notebook results

```bash
jupyter notebook notebooks/main.ipynb
```

Run all cells sequentially. Requires patient data in `data/original/track1/`.
