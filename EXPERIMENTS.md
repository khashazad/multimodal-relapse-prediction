# Experiment Catalog

Binary per-day relapse prediction from wearable sensor data (9 patients, LOSO cross-validation).

## Results Summary

| Phase | Method | Features | AUROC | AUPRC | Source |
|-------|--------|----------|-------|-------|--------|
| Notebook | XGBoost baseline | sleep+steps | ~0.530 | — | `notebooks/main.ipynb` |
| Notebook | Logistic Regression | sleep+steps | ~0.580 | — | `notebooks/main.ipynb` |
| Notebook | MLP | 24 feat | ~0.550 | — | `notebooks/main.ipynb` |
| Notebook | Transformer (bipolar-6) | 24 feat | 0.849 | 0.794 | `notebooks/main.ipynb` |
| Notebook | Transformer (all-9) | 24 feat | 0.793 | 0.705 | `notebooks/main.ipynb` |
| Framework | Focal loss (d=32, exploration) | 69 feat | 0.664 | 0.545 | `docs/experiment-log/exp_focal.md` |
| Framework | Baseline BCE (d=1024) | 69 feat | 0.808 | 0.689 | `docs/experiment-log/011_*` |
| Framework | **Focal loss (γ=1, α=0.5)** | **69 feat** | **0.845** | **0.762** | `docs/experiment-log/011_*` |
| Framework | Focal+smooth (v1) | 69 feat | 0.838 | 0.775 | `docs/experiment-log/011_*` |
| Framework | Label smoothing | 69 feat | 0.833 | 0.752 | `docs/experiment-log/011_*` |
| Framework | Training recipe sweep | 69 feat | 0.842 | — | `docs/experiment-log/012_*` |
| Framework | Focal+smooth (v2, all) | 69 feat | 0.845 | 0.745 | `docs/experiment-log/014_*` |
| Framework | Focal+smooth (v2, union) | 24 feat | 0.841 | 0.737 | `docs/experiment-log/014_*` |

**Headline result: single transformer with focal loss → 0.845 AUROC** (d=1024, 3 layers, 4 heads, LOSO across 9 patients).

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

Systematic comparison of loss functions and regularization with d=1024, 3L, 4H:

| Technique | AUROC | Δ Baseline |
|-----------|-------|------------|
| **Focal loss (γ=1, α=0.5)** | **0.845** | **+0.037** |
| Focal+smooth | 0.838 | +0.030 |
| Label smoothing (ε=0.2) | 0.833 | +0.025 |
| Stochastic depth (p=0.3) | 0.820 | +0.012 |
| Weight decay (λ=1e-3) | 0.812 | +0.004 |
| RoPE | 0.756 | −0.052 |

### Phase 6: Training Recipe Sweep (framework, 216 SLURM jobs)

Tested optimizer (Adam/AdamW), LR scheduling (none/cosine), gradient clipping, warmup. **No recipe beat the baseline 0.845.** Cosine schedule actively hurt performance.

### Phase 7: Ablation V2 (framework, 810 SLURM jobs)

Re-ran all ablations with both union (24) and all (69) feature sets. Confirmed feature set gap is small (~0.004 AUROC). Best config: focal+smooth at 0.845 AUROC.

## Key Findings

1. **Loss function > regularization**: focal loss (+0.037) and label smoothing (+0.025) dominate; architectural regularization (stochastic depth, weight decay) give marginal gains
2. **Mild focal focus is optimal**: γ=1.0 >> γ=2.0 or 3.0 — with only 9 patients and ~20 relapse days per fold, aggressive down-weighting loses gradient signal
3. **RoPE hurts short sequences**: 7-day windows are too short for relative position bias to help
4. **Bipolar subset is stronger**: bipolar-only (6 patients) AUROC 0.849 vs all-9 at 0.793 — non-bipolar patients (P1 brief psychotic, P2 schizophreniform, P7 schizophrenia) are harder
5. **24 features ≈ 69 features**: the union feature set captures most signal; extra features add noise without improving AUROC
6. **Per-patient variance dominates**: oracle AUROC across all configs is 0.907, but no single config achieves this — the 0.062 gap requires per-patient adaptation

## Negative Results

- **Training recipe sweep found nothing**: cosine schedule, AdamW, gradient clipping, warmup — none improved over baseline Adam + fixed LR
- **Cosine schedule hurts**: counter-intuitively, fixed LR oscillation aids best-epoch checkpoint selection
- **AdamW is unstable**: 11% NaN rate without gradient clipping
- **Ensemble methods** (0.912–0.938): not credible with 9 patients; removed from repository
- **Unsupervised pretraining** (BumbleBee autoencoder): 0.48 AUROC — worse than random when used alone, and fusion with supervised features (0.72) hurt vs supervised-only

## Reproduction

### Headline result (0.845 AUROC)

```bash
# Preprocess data
python scripts/preprocess_data.py

# Run ablation with focal loss
bash scripts/run.sh -n ablation_focal

# Or submit to SLURM
bash scripts/submit_slurm.sh -n ablation_focal
```

Config: `configs/ablation_focal.json` — sweeps γ ∈ {1.0, 2.0, 3.0} × α ∈ {0.3, 0.5, 0.7} with 9-fold LOSO.

### Notebook results

```bash
jupyter notebook notebooks/main.ipynb
```

Run all cells sequentially. Requires patient data in `data/original/track1/`.
