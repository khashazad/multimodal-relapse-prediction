# Experiment: Bottleneck Fusion (`exp_bottleneck`)

## Overview

This experiment replaces the baseline's `transformer_v1` with **`transformer_v2`**, which introduces learnable **bottleneck tokens** into the fusion layer. Instead of concatenating all modality embeddings directly, the model routes cross-modal information through a small set of bottleneck tokens, encouraging a compressed shared representation and reducing quadratic attention cost.

**Key change from baseline:** `model_name: "transformer_v2"` with `num_bottleneck_tokens=4`, plus `calibrate_threshold: true`

## Configuration

| Parameter | Value |
|---|---|
| Model | `transformer_v2` |
| Bottleneck tokens | 4 |
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
| Model parameters | 365,953 |
| Seed | 42 |

## Per-Fold Results (Default Threshold = 0.5)

| Fold | Patient | Best Epoch | Val AUROC | Val AUPRC | Val F1 | Test AUROC | Test AUPRC | Test F1 |
|------|---------|-----------|-----------|-----------|--------|------------|------------|---------|
| 0 | P1 | 1 | 0.606 | 0.528 | 0.000 | 0.506 | 0.325 | 0.000 |
| 1 | P2 | 1 | 0.892 | 0.854 | 0.087 | 0.547 | 0.393 | 0.000 |
| 2 | P3 | 3 | 0.827 | 0.752 | 0.292 | 0.620 | 0.703 | 0.328 |
| 3 | P4 | 0 | NaN* | NaN* | 0.000 | 0.222 | 0.081 | 0.000 |
| 4 | P5 | 3 | 0.827 | 0.823 | 0.000 | 0.696 | 0.663 | 0.515 |
| 5 | P6 | 1 | 0.661 | 0.635 | 0.528 | 0.323 | 0.407 | 0.170 |
| 6 | P7 | 1 | 0.279 | 0.147 | 0.164 | 0.787 | 0.477 | 0.642 |
| 7 | P8 | 23 | 0.622 | 0.204 | 0.000 | 0.278 | 0.639 | 0.064 |
| 8 | P9 | 1 | 0.493 | 0.427 | 0.340 | 0.980 | 0.979 | 0.871 |
| **Mean** | | | **0.651** | **0.546** | **0.157** | **0.551** | **0.518** | **0.288** |

*\*Fold 3 (P4): Validation set has 0 positive samples (n_pos=0), making AUROC/AUPRC undefined. Mean val AUROC/AUPRC excludes this fold.*

## Per-Fold Results (Calibrated Threshold)

Threshold calibration searches for the threshold on the validation set that maximizes F1, then applies it to the test set.

| Fold | Threshold | Cal Val F1 | Cal Val Prec | Cal Val Rec | Cal Test F1 | Cal Test Prec | Cal Test Rec |
|------|-----------|------------|--------------|-------------|-------------|---------------|--------------|
| 0 | 0.15 | 0.675 | 0.519 | 0.964 | 0.495 | 0.329 | 1.000 |
| 1 | 0.20 | 0.820 | 0.732 | 0.932 | 0.521 | 0.528 | 0.514 |
| 2 | 0.05 | 0.641 | 0.658 | 0.625 | 0.561 | 0.742 | 0.451 |
| 3 | 0.05 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 4 | 0.05 | 0.597 | 0.944 | 0.436 | 0.639 | 0.639 | 0.639 |
| 5 | 0.20 | 0.678 | 0.513 | 1.000 | 0.473 | 0.378 | 0.633 |
| 6 | 0.05 | 0.345 | 0.209 | 1.000 | 0.400 | 0.250 | 1.000 |
| 7 | 0.05 | 0.000 | 0.000 | 0.000 | 0.064 | 0.222 | 0.037 |
| 8 | 0.05 | 0.396 | 0.247 | 1.000 | 0.647 | 0.478 | 1.000 |
| **Mean** | | **0.461** | | | **0.422** | | |

## Comparison with Baseline (Weighted BCE)

### Default Threshold (0.5)

| Metric | Baseline | Bottleneck | Delta |
|--------|----------|------------|-------|
| Val AUROC (mean, excl fold 3) | 0.700 | 0.651 | -0.049 |
| Test AUROC (mean) | 0.560 | 0.551 | -0.009 |
| Test AUPRC (mean) | 0.537 | 0.518 | -0.019 |
| Test F1 (mean) | 0.286 | 0.288 | +0.002 |

### Calibrated Threshold

| Metric | Baseline | Bottleneck | Delta |
|--------|----------|------------|-------|
| Cal Test F1 (mean) | N/A | 0.422 | -- |

*Note: Baseline results do not include calibrated metrics.*

## Comparison with Focal Loss

### Default Threshold (0.5)

| Metric | Focal | Bottleneck | Delta |
|--------|-------|------------|-------|
| Test AUROC (mean) | 0.664 | 0.551 | -0.113 |
| Test AUPRC (mean) | 0.545 | 0.518 | -0.027 |
| Test F1 (mean) | 0.275 | 0.288 | +0.013 |

### Calibrated Threshold

| Metric | Focal | Bottleneck | Delta |
|--------|-------|------------|-------|
| Cal Test F1 (mean) | 0.612 | 0.422 | **-0.190** |

## Aggregate Summary

| Metric | Value |
|--------|-------|
| Mean test AUROC | 0.551 |
| Mean test AUPRC | 0.518 |
| Mean test F1 (default) | 0.288 |
| Mean test F1 (calibrated) | 0.422 |
| Mean calibrated threshold | 0.09 |
| Total runtime | ~802s (~13.4 min) |

## Key Observations

### 1. Bottleneck fusion does not improve over baseline
Mean test AUROC (0.551) is essentially the same as baseline (0.560), with a marginal delta of -0.009. The additional 50K parameters from the bottleneck tokens (+16% over baseline) do not translate to better ranking performance.

### 2. Calibrated F1 lags behind focal loss
With calibration, bottleneck achieves a mean test F1 of 0.422, substantially lower than focal loss's 0.612. The bottleneck architecture paired with weighted BCE does not produce predictions that benefit as much from threshold tuning as focal loss does.

### 3. Fold 3 (P4) fails completely under calibration
Unlike focal loss (which achieves cal test F1 = 0.239 for fold 3 despite the broken validation set), bottleneck produces cal test F1 = 0.000 for fold 3 -- the calibration finds no useful threshold when val has zero positives.

### 4. Fold 7 (P8) is the longest-training fold
Fold 7 reaches best epoch 23, far beyond all other folds (which peak at epoch 0-3). P8's validation dynamics are unique, with the model continuing to improve for many more epochs. Despite this, test performance remains poor (test AUROC = 0.278).

### 5. High variance persists
Test AUROC ranges from 0.222 (fold 3) to 0.980 (fold 8). The bottleneck mechanism does not reduce cross-patient variability compared to baseline or focal loss.

### 6. Bottleneck tokens add parameters without regularization benefit
At 365,953 parameters, this is the largest model tested. The bottleneck tokens were intended to create an information bottleneck for better generalization, but with only ~2,900 training windows, the additional capacity may contribute to overfitting rather than improved fusion.
