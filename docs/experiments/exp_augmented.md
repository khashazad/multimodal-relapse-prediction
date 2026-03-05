# Experiment: Data Augmentation (`exp_augmented`)

## Overview

This experiment applies **all augmentation strategies** (jittering, magnitude warping, and cross-patient mixup) to the baseline `transformer_v1` architecture. Augmentation aims to increase the effective training set size and reduce overfitting on the small ~2,900-window training set.

**Key change from baseline:** `augmentation: "all"` (jittering + magnitude warping + mixup)

## Configuration

| Parameter | Value |
|---|---|
| Model | `transformer_v1` |
| Loss function | Weighted BCE |
| Augmentation | all (jitter + warp + mixup) |
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
| Model parameters | 316,033 |
| Seed | 42 |

## Per-Fold Results (Default Threshold = 0.5)

| Fold | Patient | Best Epoch | Val AUROC | Val AUPRC | Val F1 | Test AUROC | Test AUPRC | Test F1 |
|------|---------|-----------|-----------|-----------|--------|------------|------------|---------|
| 0 | P1 | 1 | 0.825 | 0.719 | 0.000 | 0.361 | 0.260 | 0.000 |
| 1 | P2 | 4 | 0.874 | 0.782 | 0.000 | 0.580 | 0.420 | 0.000 |
| 2 | P3 | 7 | 0.790 | 0.719 | 0.140 | 0.524 | 0.633 | 0.222 |
| 3 | P4 | 0 | NaN* | NaN* | 0.000 | 0.284 | 0.086 | 0.049 |
| 4 | P5 | 6 | 0.788 | 0.694 | 0.000 | 0.641 | 0.660 | 0.463 |
| 5 | P6 | 1 | 0.609 | 0.623 | 0.537 | 0.524 | 0.523 | 0.439 |
| 6 | P7 | 20 | 0.559 | 0.222 | 0.000 | 0.397 | 0.218 | 0.176 |
| 7 | P8 | 24 | 0.540 | 0.138 | 0.000 | 0.626 | 0.821 | 0.036 |
| 8 | P9 | 1 | 0.757 | 0.486 | 0.390 | 1.000 | 1.000 | 0.984 |
| **Mean** | | | **0.718** | **0.548** | **0.119** | **0.549** | **0.513** | **0.263** |

*\*Fold 3 (P4): Validation set has 0 positive samples (n_pos=0), making AUROC/AUPRC undefined. Mean val AUROC/AUPRC excludes this fold.*

## Per-Fold Results (Calibrated Threshold)

| Fold | Threshold | Cal Val F1 | Cal Val Prec | Cal Val Rec | Cal Test F1 | Cal Test Prec | Cal Test Rec |
|------|-----------|------------|--------------|-------------|-------------|---------------|--------------|
| 0 | 0.15 | 0.758 | 0.658 | 0.893 | 0.432 | 0.314 | 0.696 |
| 1 | 0.05 | 0.240 | 1.000 | 0.136 | 0.000 | 0.000 | 0.000 |
| 2 | 0.05 | 0.255 | 0.857 | 0.150 | 0.250 | 0.615 | 0.157 |
| 3 | 0.05 | 0.000 | 0.000 | 0.000 | 0.047 | 0.031 | 0.091 |
| 4 | 0.05 | 0.098 | 1.000 | 0.051 | 0.566 | 0.667 | 0.492 |
| 5 | 0.25 | 0.643 | 0.493 | 0.923 | 0.567 | 0.462 | 0.735 |
| 6 | 0.05 | 0.000 | 0.000 | 0.000 | 0.158 | 0.200 | 0.130 |
| 7 | 0.05 | 0.000 | 0.000 | 0.000 | 0.070 | 0.667 | 0.037 |
| 8 | 0.30 | 0.535 | 0.373 | 0.950 | 0.865 | 0.762 | 1.000 |
| **Mean** | | **0.281** | | | **0.328** | | |

## Comparison with Baseline

| Metric | Baseline | Augmented | Delta |
|--------|----------|-----------|-------|
| Val AUROC (mean, excl fold 3) | 0.700 | 0.718 | +0.018 |
| Test AUROC (mean) | 0.560 | 0.549 | -0.011 |
| Test AUPRC (mean) | 0.537 | 0.513 | -0.024 |
| Test F1 (mean, default) | 0.286 | 0.263 | -0.023 |
| Cal Test F1 (mean) | N/A | 0.328 | -- |

## Aggregate Summary

| Metric | Value |
|--------|-------|
| Mean test AUROC | 0.549 |
| Mean test AUPRC | 0.513 |
| Mean test F1 (default) | 0.263 |
| Mean test F1 (calibrated) | 0.328 |
| Mean calibrated threshold | 0.11 |
| Total runtime | ~928s (~15.5 min) |

## Key Observations

### 1. Augmentation provides no improvement over baseline
Mean test AUROC (0.549) and test F1 (0.263) are both slightly *below* the baseline (0.560, 0.286). Augmentation slightly improves validation AUROC (0.718 vs 0.700) but this does not transfer to test performance.

### 2. Augmentation allows slightly longer training
Best epochs extend further (fold 6: epoch 20, fold 7: epoch 24) compared to the baseline where most folds peak at epoch 0-5. The augmented data slows overfitting but does not ultimately produce better test metrics.

### 3. Calibrated thresholds are extremely low
Most calibrated thresholds are 0.05, the minimum in the sweep. Many folds show calibrated val F1 of 0.0, indicating the model cannot distinguish relapse from stable even with threshold tuning on some folds. The mean calibrated test F1 (0.328) is the lowest among all experiments with calibration.

### 4. Fold 1 (P2) shows a calibration failure
Calibrated threshold of 0.05 achieves val F1=0.240 but test F1=0.000 — the model predicts no relapse cases at all on the test set. This highlights the fragility of threshold calibration when val and test distributions differ.

### 5. Augmentation may be too aggressive
Using "all" augmentation strategies simultaneously (jittering + warping + mixup) may introduce excessive noise that masks the subtle relapse signals. Future experiments could try individual augmentation strategies or more conservative noise levels.
