# Experiment: Focal Loss (`exp_focal`)

## Overview

This experiment replaces the baseline's weighted binary cross-entropy loss with **focal loss** to address the severe class imbalance between relapse and stable days. Focal loss down-weights easy (well-classified) examples and focuses training on hard negatives/positives.

**Key change from baseline:** `loss_fn: "focal"` with `focal_alpha=0.75`, `focal_gamma=2.0`

## Configuration

| Parameter | Value |
|---|---|
| Model | `transformer_v1` |
| Loss function | Focal (alpha=0.75, gamma=2.0) |
| d_model | 32 |
| Attention heads | 2 |
| Encoder layers | 1 |
| Fusion layers | 1 |
| Dropout | 0.3 |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Batch size | 32 |
| Epochs (max) | 100 |
| Early stopping patience | 15 |
| Default threshold | 0.5 |
| Calibrate threshold | Yes |
| Model parameters | 83,265 |
| Seed | 42 |

## Per-Fold Results (Default Threshold = 0.5)

| Fold | Patient | Best Epoch | Val AUROC | Val AUPRC | Val F1 | Test AUROC | Test AUPRC | Test F1 |
|------|---------|-----------|-----------|-----------|--------|------------|------------|---------|
| 0 | P1 | 2 | 0.628 | 0.631 | 0.000 | 0.436 | 0.286 | 0.000 |
| 1 | P2 | 2 | 0.974 | 0.945 | 0.400 | 0.492 | 0.331 | 0.000 |
| 2 | P3 | 15 | 0.753 | 0.681 | 0.091 | 0.796 | 0.803 | 0.313 |
| 3 | P4 | 0 | NaN* | NaN* | 0.000 | 0.649 | 0.169 | 0.069 |
| 4 | P5 | 6 | 0.911 | 0.862 | 0.000 | 0.522 | 0.522 | 0.203 |
| 5 | P6 | 3 | 0.558 | 0.489 | 0.584 | 0.599 | 0.519 | 0.395 |
| 6 | P7 | 4 | 0.248 | 0.142 | 0.044 | 0.640 | 0.324 | 0.514 |
| 7 | P8 | 1 | 0.338 | 0.093 | 0.000 | 0.841 | 0.952 | 0.000 |
| 8 | P9 | 7 | 0.717 | 0.462 | 0.439 | 1.000 | 1.000 | 0.984 |
| **Mean** | | | **0.641** | **0.538** | **0.173** | **0.664** | **0.545** | **0.275** |

*\*Fold 3 (P4): Validation set has 0 positive samples (n_pos=0), making AUROC/AUPRC undefined. Mean val AUROC/AUPRC excludes this fold.*

## Per-Fold Results (Calibrated Threshold)

Threshold calibration searches for the threshold on the validation set that maximizes F1, then applies it to the test set.

| Fold | Threshold | Cal Val F1 | Cal Val Prec | Cal Val Rec | Cal Test F1 | Cal Test Prec | Cal Test Rec |
|------|-----------|------------|--------------|-------------|-------------|---------------|--------------|
| 0 | 0.40 | 0.659 | 0.491 | 1.000 | 0.488 | 0.339 | 0.870 |
| 1 | 0.45 | 0.846 | 0.733 | 1.000 | 0.468 | 0.386 | 0.595 |
| 2 | 0.05 | 0.673 | 0.569 | 0.825 | 0.813 | 0.694 | 0.980 |
| 3 | 0.05 | 0.000 | 0.000 | 0.000 | 0.239 | 0.136 | 1.000 |
| 4 | 0.25 | 0.805 | 0.767 | 0.846 | 0.643 | 0.570 | 0.738 |
| 5 | 0.05 | 0.634 | 0.464 | 1.000 | 0.628 | 0.458 | 1.000 |
| 6 | 0.05 | 0.345 | 0.209 | 1.000 | 0.400 | 0.250 | 1.000 |
| 7 | 0.05 | 0.205 | 0.114 | 1.000 | 0.871 | 0.771 | 1.000 |
| 8 | 0.30 | 0.494 | 0.328 | 1.000 | 0.955 | 0.914 | 1.000 |
| **Mean** | | **0.518** | | | **0.612** | | |

## Comparison with Baseline (Weighted BCE)

### Default Threshold (0.5)

| Metric | Baseline | Focal | Delta |
|--------|----------|-------|-------|
| Val AUROC (mean, excl fold 3) | 0.700 | 0.641 | -0.059 |
| Test AUROC (mean) | 0.560 | 0.664 | **+0.104** |
| Test AUPRC (mean) | 0.537 | 0.545 | +0.008 |
| Test F1 (mean) | 0.286 | 0.275 | -0.011 |

### Calibrated Threshold

| Metric | Baseline | Focal | Delta |
|--------|----------|-------|-------|
| Cal Test F1 (mean) | N/A | 0.612 | -- |

*Note: Baseline results do not include calibrated metrics.*

## Aggregate Summary

| Metric | Value |
|--------|-------|
| Mean test AUROC | 0.664 |
| Mean test AUPRC | 0.545 |
| Mean test F1 (default) | 0.275 |
| Mean test F1 (calibrated) | 0.612 |
| Mean calibrated threshold | 0.18 |
| Total runtime | ~611s (~10 min) |

## Key Observations

### 1. Focal loss improves test AUROC but not validation AUROC
Mean test AUROC improved from 0.560 (baseline) to 0.664, a notable +0.104 gain. However, validation AUROC dropped from 0.700 to 0.641. This suggests focal loss produces better-ranking predictions on unseen patients despite appearing weaker on the held-out validation sequences of the same training distribution.

### 2. Threshold calibration is critical
At the default 0.5 threshold, focal loss achieves a mean test F1 of only 0.275 (comparable to baseline's 0.286). After calibration, mean test F1 jumps to **0.612** -- a 2.2x improvement. Most calibrated thresholds are very low (0.05), indicating the model's predicted probabilities are generally well below 0.5 even for relapse days.

### 3. Early stopping occurs very early
Best epochs range from 0-15, with most folds peaking within the first few epochs (median best epoch = 3). The model quickly overfits, consistent with the small training set (~2,850-2,950 windows) and the 83K parameter model.

### 4. Fold 3 (P4) remains problematic
P4's validation set contains zero relapse days, making model selection impossible. The model defaults to epoch 0 with undefined AUROC. This is a structural issue with the LOSO setup, not specific to focal loss.

### 5. High variance across folds persists
Test AUROC ranges from 0.436 (fold 0) to 1.000 (fold 8), and calibrated test F1 ranges from 0.239 (fold 3) to 0.955 (fold 8). The per-patient variability remains a fundamental challenge.

### 6. Focal loss shifts predictions toward low confidence
The majority of calibrated thresholds are 0.05, meaning the model's output probabilities cluster near zero. Focal loss's gamma=2.0 aggressively down-weights easy negatives, which may cause the model to output very low probabilities overall. This makes threshold calibration essential.
