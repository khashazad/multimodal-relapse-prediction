# Experiment: Domain-Adversarial Training (`exp_dann`)

## Overview

This experiment adds **Domain-Adversarial Neural Network (DANN)** training to the baseline transformer architecture, encouraging the fused representation to be patient-invariant. A gradient reversal layer negates gradients from a patient discriminator head, pushing the encoder toward features that generalize across patients.

**Key change from baseline:** `model_name: "transformer_v4"` with `dann_lambda=0.1`, `dann_warmup_epochs=30`

## Configuration

| Parameter | Value |
|---|---|
| Model | `transformer_v4` (DANN) |
| Loss function | Weighted BCE |
| DANN lambda | 0.1 |
| DANN warmup epochs | 30 |
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
| Model parameters | 318,410 |
| Seed | 42 |

## Per-Fold Results (Default Threshold = 0.5)

| Fold | Patient | Best Epoch | Val AUROC | Val AUPRC | Val F1 | Test AUROC | Test AUPRC | Test F1 |
|------|---------|-----------|-----------|-----------|--------|------------|------------|---------|
| 0 | P1 | 1 | 0.847 | 0.715 | 0.000 | 0.325 | 0.249 | 0.000 |
| 1 | P2 | 1 | 0.882 | 0.790 | 0.563 | 0.711 | 0.545 | 0.093 |
| 2 | P3 | 7 | 0.816 | 0.761 | 0.095 | 0.566 | 0.670 | 0.203 |
| 3 | P4 | 0 | NaN* | NaN* | 0.000 | 0.241 | 0.082 | 0.044 |
| 4 | P5 | 5 | 0.804 | 0.699 | 0.098 | 0.652 | 0.649 | 0.535 |
| 5 | P6 | 1 | 0.651 | 0.675 | 0.481 | 0.579 | 0.598 | 0.462 |
| 6 | P7 | 1 | 0.363 | 0.169 | 0.231 | 0.792 | 0.436 | 0.571 |
| 7 | P8 | 12 | 0.554 | 0.144 | 0.000 | 0.493 | 0.731 | 0.069 |
| 8 | P9 | 1 | 0.731 | 0.463 | 0.372 | 1.000 | 1.000 | 0.984 |
| **Mean** | | | **0.706** | **0.552** | **0.205** | **0.595** | **0.551** | **0.329** |

*\*Fold 3 (P4): Validation set has 0 positive samples (n_pos=0), making AUROC/AUPRC undefined. Mean val AUROC/AUPRC excludes this fold.*

## Per-Fold Results (Calibrated Threshold)

| Fold | Threshold | Cal Val F1 | Cal Val Prec | Cal Val Rec | Cal Test F1 | Cal Test Prec | Cal Test Rec |
|------|-----------|------------|--------------|-------------|-------------|---------------|--------------|
| 0 | 0.20 | 0.812 | 0.683 | 1.000 | 0.444 | 0.310 | 0.783 |
| 1 | 0.30 | 0.832 | 0.737 | 0.955 | 0.571 | 0.550 | 0.595 |
| 2 | 0.05 | 0.292 | 0.875 | 0.175 | 0.250 | 0.615 | 0.157 |
| 3 | 0.05 | 0.000 | 0.000 | 0.000 | 0.085 | 0.056 | 0.182 |
| 4 | 0.05 | 0.182 | 0.800 | 0.103 | 0.606 | 0.688 | 0.541 |
| 5 | 0.25 | 0.636 | 0.500 | 0.872 | 0.531 | 0.469 | 0.612 |
| 6 | 0.20 | 0.349 | 0.211 | 1.000 | 0.400 | 0.250 | 1.000 |
| 7 | 0.05 | 0.000 | 0.000 | 0.000 | 0.067 | 0.333 | 0.037 |
| 8 | 0.30 | 0.531 | 0.386 | 0.850 | 0.901 | 0.821 | 1.000 |
| **Mean** | | **0.404** | | | **0.428** | | |

## Comparison with Baseline

| Metric | Baseline | DANN | Delta |
|--------|----------|------|-------|
| Val AUROC (mean, excl fold 3) | 0.700 | 0.706 | +0.006 |
| Test AUROC (mean) | 0.560 | 0.595 | **+0.035** |
| Test AUPRC (mean) | 0.537 | 0.551 | +0.014 |
| Test F1 (mean, default) | 0.286 | 0.329 | **+0.043** |
| Cal Test F1 (mean) | N/A | 0.428 | -- |

## Aggregate Summary

| Metric | Value |
|--------|-------|
| Mean test AUROC | 0.595 |
| Mean test AUPRC | 0.551 |
| Mean test F1 (default) | 0.329 |
| Mean test F1 (calibrated) | 0.428 |
| Mean calibrated threshold | 0.16 |
| Total runtime | ~744s (~12.4 min) |

## Key Observations

### 1. DANN provides modest improvements over baseline
Mean test AUROC improves from 0.560 to 0.595 (+0.035), and default-threshold test F1 improves from 0.286 to 0.329 (+0.043). The adversarial training does appear to help with cross-patient generalization, though the effect is smaller than expected.

### 2. The discriminator loss plateaus at ~2.05 (near-random)
The DANN loss at the final epoch is consistently around 2.02-2.05 across all folds, close to log(9) = 2.20 (random chance for 9-class classification). This indicates the gradient reversal is successfully preventing the encoder from encoding patient-specific information, but the small lambda (0.1) and long warmup (30 epochs) mean the adversarial signal is relatively weak.

### 3. Early stopping occurs very early despite DANN
Best epochs range from 0-12, with most folds peaking at epoch 1. The DANN warmup of 30 epochs means most folds stop before the adversarial loss reaches full strength. The adversarial regularization does not prevent the rapid overfitting seen in the baseline.

### 4. Calibrated F1 underperforms focal loss
Mean calibrated test F1 is 0.428, compared to 0.612 for exp_focal. This suggests that the model architecture matters less than the loss function and model size for this task — the smaller focal model (83K params, d_model=32) generalizes better than the DANN model (318K params, d_model=64).

### 5. Fold 3 (P4) and fold 8 (P9) follow the same pattern
P4 remains structurally broken (0 val positives), and P9 remains near-perfect (test AUROC=1.000), consistent across all experiments.
