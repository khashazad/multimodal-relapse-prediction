# Experiment: Gated Fusion (`exp_gated`)

## Overview

This experiment replaces the baseline's cross-modal attention fusion with a **gated fusion** mechanism. Each modality receives a learned sigmoid gate that controls its contribution to the fused representation. Missing modalities are zeroed out before normalization, making the fusion naturally robust to sensor dropout.

**Key change from baseline:** `model_name: "transformer_v3"` (GatedFusion replaces FusionTransformer)

## Configuration

| Parameter | Value |
|---|---|
| Model | `transformer_v3` (Gated Fusion) |
| Loss function | Weighted BCE |
| d_model | 64 |
| Attention heads | 4 |
| Encoder layers | 1 |
| Fusion layers | 1 |
| Dropout | 0.1 |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Batch size | 32 |
| Epochs (max) | 100 |
| Early stopping patience | 15 |
| Default threshold | 0.5 |
| Calibrate threshold | Yes |
| Model parameters | 266,054 |
| Seed | 42 |

## Per-Fold Results (Default Threshold = 0.5)

| Fold | Patient | Best Epoch | Val AUROC | Val AUPRC | Val F1 | Test AUROC | Test AUPRC | Test F1 |
|------|---------|-----------|-----------|-----------|--------|------------|------------|---------|
| 0 | P1 | 1 | 0.923 | 0.876 | 0.000 | 0.386 | 0.276 | 0.000 |
| 1 | P2 | 2 | 0.888 | 0.820 | 0.531 | 0.517 | 0.341 | 0.000 |
| 2 | P3 | 6 | 0.668 | 0.574 | 0.255 | 0.842 | 0.856 | 0.237 |
| 3 | P4 | 0 | NaN* | NaN* | 0.000 | 0.254 | 0.087 | 0.000 |
| 4 | P5 | 4 | 0.656 | 0.576 | 0.000 | 0.635 | 0.690 | 0.479 |
| 5 | P6 | 1 | 0.667 | 0.680 | 0.650 | 0.561 | 0.566 | 0.555 |
| 6 | P7 | 32 | 0.568 | 0.241 | 0.000 | 0.307 | 0.186 | 0.000 |
| 7 | P8 | 41 | 0.567 | 0.171 | 0.000 | 0.464 | 0.776 | 0.277 |
| 8 | P9 | 1 | 0.692 | 0.376 | 0.390 | 1.000 | 1.000 | 1.000 |
| **Mean** | | | **0.703** | **0.539** | **0.203** | **0.552** | **0.531** | **0.283** |

*\*Fold 3 (P4): Validation set has 0 positive samples (n_pos=0), making AUROC/AUPRC undefined. Mean val AUROC/AUPRC excludes this fold.*

## Per-Fold Results (Calibrated Threshold)

| Fold | Threshold | Cal Val F1 | Cal Val Prec | Cal Val Rec | Cal Test F1 | Cal Test Prec | Cal Test Rec |
|------|-----------|------------|--------------|-------------|-------------|---------------|--------------|
| 0 | 0.40 | 0.918 | 0.848 | 1.000 | 0.388 | 0.295 | 0.565 |
| 1 | 0.15 | 0.832 | 0.737 | 0.955 | 0.000 | 0.000 | 0.000 |
| 2 | 0.20 | 0.320 | 0.800 | 0.200 | 0.448 | 0.938 | 0.294 |
| 3 | 0.05 | 0.000 | 0.000 | 0.000 | 0.059 | 0.043 | 0.091 |
| 4 | 0.05 | 0.598 | 0.542 | 0.667 | 0.629 | 0.557 | 0.721 |
| 5 | 0.50 | 0.650 | 0.487 | 0.974 | 0.555 | 0.471 | 0.673 |
| 6 | 0.05 | 0.000 | 0.000 | 0.000 | 0.108 | 0.143 | 0.087 |
| 7 | 0.10 | 0.125 | 0.143 | 0.111 | 0.294 | 0.714 | 0.185 |
| 8 | 0.45 | 0.508 | 0.372 | 0.800 | 0.941 | 0.889 | 1.000 |
| **Mean** | | **0.439** | | | **0.380** | | |

## Comparison with Baseline

| Metric | Baseline | Gated | Delta |
|--------|----------|-------|-------|
| Val AUROC (mean, excl fold 3) | 0.700 | 0.703 | +0.003 |
| Test AUROC (mean) | 0.560 | 0.552 | -0.008 |
| Test AUPRC (mean) | 0.537 | 0.531 | -0.006 |
| Test F1 (mean, default) | 0.286 | 0.283 | -0.003 |
| Cal Test F1 (mean) | N/A | 0.380 | -- |

## Aggregate Summary

| Metric | Value |
|--------|-------|
| Mean test AUROC | 0.552 |
| Mean test AUPRC | 0.531 |
| Mean test F1 (default) | 0.283 |
| Mean test F1 (calibrated) | 0.380 |
| Mean calibrated threshold | 0.22 |
| Model parameters | 266,054 |
| Total runtime | ~909s (~15.1 min) |

## Key Observations

### 1. Gated fusion performs on par with the baseline
Mean test AUROC (0.552) and test F1 (0.283) are nearly identical to the baseline (0.560, 0.286). The gated fusion mechanism neither helps nor hurts compared to the cross-modal attention fusion in v1.

### 2. Smaller model, similar performance
At 266K parameters (vs 316K baseline), the gated model is 16% smaller because it replaces the multi-layer fusion transformer with simple per-modality linear gates. This suggests the fusion transformer's additional capacity is not being utilized effectively.

### 3. Some folds allow extended training
Fold 6 (P7) trains to epoch 32 and fold 7 (P8) to epoch 41 — much longer than the baseline where most folds stop by epoch 5. The gated fusion's simpler architecture may have a smoother loss landscape that delays early stopping, though this doesn't translate to better test metrics.

### 4. Fold 0 (P1) shows extreme val-test discordance
Val AUROC is 0.923 (highest across all experiments for this fold) but test AUROC drops to 0.386. The gated fusion may be memorizing validation-specific patterns. Calibrated test F1 (0.388) partially recovers via threshold adjustment.

### 5. Fold 1 (P2) calibration fails completely
Despite strong val AUROC (0.888), calibrated test F1 is 0.000 — the model predicts zero relapse cases on the test set at the calibrated threshold of 0.15. This fold exhibits a severe distribution shift between val and test sequences.

### 6. Fold 8 (P9) achieves perfect scores
Test AUROC, AUPRC, and F1 are all 1.000 (even at default threshold), confirming P9's relapse pattern is trivially separable. This is the only experiment where P9 achieves a perfect test F1 without calibration.
