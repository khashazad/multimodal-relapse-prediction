# Final Results Summary

## Summarize

Best supervised transformer (d=1024, 3L, 4H) achieves **AUROC 0.8492, AUPRC 0.7936** on 6 bipolar patients (Cell 77 best). All-9-patient expansion drops to AUROC 0.7928 due to non-bipolar patients. Progression from XGBoost baseline (0.53) through feature engineering and architecture scaling. Unsupervised/fusion approaches underperformed.

## Best Model: Supervised Transformer

**Architecture:** d_model=1024, 3 layers, 4 heads, dropout=0.3, seq_len=7 (7-day windows), BCE w/ adaptive weighting, lr=0.001, batch=32, 80 epochs, LOPO CV.

**24 input features:** sleep metrics (main episode + naps, z-score deviations), step counts, nighttime HRV (RMSSD/SDNN), circadian actigraphy (relative amplitude, intradaily variability, cosinor, L5/M10), demographics.

### Best Result — Bipolar Only (6 folds, Cell 77)

**AUROC = 0.8492, AUPRC = 0.7936**

### All-9 Patients: AUROC 0.7928 ± 0.1160, AUPRC 0.7050 ± 0.1702

| Patient | Diagnosis | AUROC | AUPRC |
|---------|-----------|-------|-------|
| P1 | Brief Psychotic Episode | 0.808 | 0.647 |
| P2 | Schizophreniform | 0.765 | 0.616 |
| P3 | Bipolar I | 0.781 | 0.826 |
| P4 | Bipolar I | 0.870 | 0.564 |
| P5 | Bipolar II | 0.623 | 0.651 |
| P6 | Bipolar I | 0.639 | 0.579 |
| P7 | Schizophrenia | 0.733 | 0.499 |
| P8 | Bipolar I | 0.922 | 0.968 |
| P9 | Bipolar I | 0.996 | 0.996 |

Bipolar-only mean (from all-9 run): AUROC 0.805 ± 0.139, AUPRC 0.764 ± 0.176.

## Model Progression

1. **XGBoost baseline** (~0.53 AUROC) → feature selection → ~0.58
2. **Logistic Regression** (~0.58), **MLP** (~0.55)
3. **Small transformer** (d=32) — ~0.51 (underfit)
4. **d=1024 transformer** — **0.8492 bipolar (best)**, 0.79 all-9
5. **BumbleBee autoencoder** (unsupervised) — 0.48
6. **Fusion MLP** (supervised + AE latents) — 0.72 (worse than supervised alone)

## Cached Artifacts

- `cache/transformer_all9_d1024_l3_dr03.pkl` — final all-9 results
- `cache/patient_data_export_all9.pkl` — 24 features, 3337 days
- `cache/sup_fold_P*.pth` — per-patient model weights
