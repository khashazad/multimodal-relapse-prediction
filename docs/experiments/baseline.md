# Experiment: Baseline (`train`)

## Overview

The baseline experiment establishes a reference point for multimodal relapse prediction using a transformer fusion architecture with **weighted binary cross-entropy** loss. It uses the default `transformer_v1` model with no threshold calibration.

**Key characteristics:** `loss_fn: "weighted_bce"`, `d_model=64`, no calibrated threshold

## Configuration

| Parameter | Value |
|---|---|
| Model | `transformer_v1` |
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
| Calibrate threshold | No |
| Model parameters | 316,033 |
| Seed | 42 |

## Per-Fold Results (Default Threshold = 0.5)

| Fold | Patient | Best Epoch | Val AUROC | Val AUPRC | Val F1 | Test AUROC | Test AUPRC | Test F1 |
|------|---------|-----------|-----------|-----------|--------|------------|------------|---------|
| 0 | P1 | 1 | 0.843 | 0.721 | 0.000 | 0.389 | 0.272 | 0.000 |
| 1 | P2 | 1 | 0.864 | 0.755 | 0.452 | 0.724 | 0.555 | 0.050 |
| 2 | P3 | 5 | 0.839 | 0.774 | 0.182 | 0.578 | 0.692 | 0.254 |
| 3 | P4 | 0 | NaN* | NaN* | 0.000 | 0.149 | 0.075 | 0.000 |
| 4 | P5 | 4 | 0.742 | 0.597 | 0.000 | 0.653 | 0.653 | 0.345 |
| 5 | P6 | 1 | 0.608 | 0.634 | 0.467 | 0.545 | 0.543 | 0.395 |
| 6 | P7 | 1 | 0.331 | 0.158 | 0.122 | 0.773 | 0.416 | 0.542 |
| 7 | P8 | 21 | 0.629 | 0.182 | 0.182 | 0.232 | 0.626 | 0.000 |
| 8 | P9 | 1 | 0.745 | 0.485 | 0.429 | 1.000 | 1.000 | 0.984 |
| **Mean** | | | **0.700** | **0.538** | **0.204** | **0.560** | **0.537** | **0.286** |

*\*Fold 3 (P4): Validation set has 0 positive samples (n_pos=0), making AUROC/AUPRC undefined. Mean val AUROC/AUPRC excludes this fold.*

## Aggregate Summary

| Metric | Value |
|--------|-------|
| Mean test AUROC | 0.560 |
| Mean test AUPRC | 0.537 |
| Mean test F1 (default) | 0.286 |
| Mean val AUROC (excl fold 3) | 0.700 |
| Total runtime | ~436s (~7.3 min) |

## Key Observations

### 1. The model overfits almost immediately
Six of nine folds have a best epoch of 0 or 1, indicating the model peaks on validation within the first two epochs and then degrades. Only fold 7 (P8) trains to epoch 21. This is consistent with ~316K parameters trained on only ~2,900 windows.

### 2. Default threshold produces many zero-F1 folds
At the fixed 0.5 threshold, the model predicts all-negative in folds 0, 3, and 7, yielding F1 = 0. The model's predicted probabilities likely cluster well below 0.5 for relapse days, making threshold calibration essential for downstream experiments.

### 3. Large val-test discordance
Several folds show strong validation AUROC but poor test AUROC (fold 0: 0.843 val vs 0.389 test; fold 1: 0.864 val vs 0.724 test). This suggests distribution shift between a patient's validation and test sequences, a fundamental challenge in the LOSO setup.

### 4. Fold 3 (P4) is structurally broken
P4's validation set has zero relapse days, making AUROC/AUPRC undefined and model selection meaningless. The model defaults to epoch 0, producing the worst test AUROC (0.149). This is a dataset-level issue, not a model issue.

### 5. High variance across folds
Test AUROC ranges from 0.149 (fold 3) to 1.000 (fold 8). Test F1 ranges from 0.000 to 0.984. The per-patient variability reflects the heterogeneity of relapse patterns across the 9 patients.

### 6. Fold 8 (P9) is anomalously perfect
P9 achieves test AUROC = 1.000 and test F1 = 0.984 across baseline and all subsequent experiments, suggesting this patient's relapse pattern is easily separable from stable days.
