# Experiment Results: Multimodal Transformer Fusion (Baseline)

## Experiment Setup

| Parameter | Value |
|-----------|-------|
| Model | MultimodalRelapseTransformer |
| Parameters | 316,033 |
| d_model | 64 |
| Attention heads | 4 |
| Encoder layers | 1 (per modality) |
| Fusion layers | 1 |
| Dropout | 0.1 |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Optimizer | AdamW + CosineAnnealingLR |
| Loss function | Weighted BCE (auto pos_weight) |
| Batch size | 32 |
| Max epochs | 100 |
| Early stopping | Patience 15 on val AUROC |
| Window size | 7 days |
| Threshold | 0.5 |
| Seed | 42 |
| Hardware | RTX 4090 (1 GPU per fold) |

## Per-Fold Results

### Validation Metrics

| Fold | Test Patient | AUROC | AUPRC | F1 | Precision | Recall | Accuracy | Samples (pos/neg) | Best Epoch |
|------|-------------|-------|-------|-----|-----------|--------|----------|-------------------|------------|
| 0 | P1 | 0.8426 | 0.7211 | 0.0000 | 0.0000 | 0.0000 | 0.5333 | 60 (28/32) | 1 |
| 1 | P2 | 0.8643 | 0.7547 | 0.4516 | 0.7778 | 0.3182 | 0.6852 | 108 (44/64) | 1 |
| 2 | P3 | 0.8385 | 0.7743 | 0.1818 | 1.0000 | 0.1000 | 0.6087 | 92 (40/52) | 5 |
| 3 | P4 | NaN | NaN | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 71 (0/71) | 0 |
| 4 | P5 | 0.7416 | 0.5973 | 0.0000 | 0.0000 | 0.0000 | 0.5714 | 91 (39/52) | 4 |
| 5 | P6 | 0.6080 | 0.6336 | 0.4667 | 0.6667 | 0.3590 | 0.6190 | 84 (39/45) | 1 |
| 6 | P7 | 0.3311 | 0.1578 | 0.1224 | 0.1000 | 0.1579 | 0.5275 | 91 (19/72) | 1 |
| 7 | P8 | 0.6286 | 0.1822 | 0.1818 | 0.1538 | 0.2222 | 0.7722 | 79 (9/70) | 21 |
| 8 | P9 | 0.7451 | 0.4850 | 0.4286 | 0.4091 | 0.4500 | 0.7037 | 81 (20/61) | 1 |

### Test Metrics

| Fold | Test Patient | AUROC | AUPRC | F1 | Precision | Recall | Accuracy | Samples (pos/neg) | Time |
|------|-------------|-------|-------|-----|-----------|--------|----------|-------------------|------|
| 0 | P1 | 0.3885 | 0.2717 | 0.0000 | 0.0000 | 0.0000 | 0.5857 | 70 (23/47) | 45s |
| 1 | P2 | 0.7237 | 0.5554 | 0.0500 | 0.3333 | 0.0270 | 0.6514 | 109 (37/72) | 43s |
| 2 | P3 | 0.5782 | 0.6917 | 0.2540 | 0.6667 | 0.1569 | 0.4535 | 86 (51/35) | 53s |
| 3 | P4 | 0.1493 | 0.0746 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 92 (11/81) | 37s |
| 4 | P5 | 0.6527 | 0.6530 | 0.3448 | 0.5769 | 0.2459 | 0.5043 | 115 (61/54) | 39s |
| 5 | P6 | 0.5450 | 0.5434 | 0.3947 | 0.5556 | 0.3061 | 0.5701 | 107 (49/58) | 41s |
| 6 | P7 | 0.7732 | 0.4161 | 0.5417 | 0.5200 | 0.5652 | 0.7609 | 92 (23/69) | 42s |
| 7 | P8 | 0.2315 | 0.6257 | 0.0000 | 0.0000 | 0.0000 | 0.2143 | 70 (54/16) | 93s |
| 8 | P9 | 1.0000 | 1.0000 | 0.9841 | 1.0000 | 0.9688 | 0.9851 | 67 (32/35) | 43s |

## Aggregate Statistics (Mean +/- Std across LOSO folds)

| Metric | Validation | Test |
|--------|-----------|------|
| AUROC | 0.7000 +/- 0.1658 | 0.5602 +/- 0.2541 |
| AUPRC | 0.5383 +/- 0.2299 | 0.5369 +/- 0.2492 |
| F1 | 0.2037 +/- 0.1866 | 0.2855 +/- 0.3109 |
| Precision | 0.3453 +/- 0.3617 | 0.4058 +/- 0.3308 |
| Recall | 0.1786 +/- 0.1598 | 0.2522 +/- 0.3100 |
| Accuracy | 0.6690 +/- 0.1397 | 0.5806 +/- 0.2011 |

Note: Validation AUROC/AUPRC averages computed over 8 folds (fold 3 excluded due to single-class validation set). All other metrics averaged over all 9 folds.

## Observations

### High-Variance Results

The high standard deviation across folds (e.g., test AUROC std = 0.25) is characteristic of LOSO cross-validation with only 9 subjects. Each patient has distinct sensor wearing patterns and relapse profiles, making generalization inherently difficult.

### Per-Fold Analysis

- **Fold 3 (P4)**: Validation set contains zero relapse days (71 stable, 0 relapse), making AUROC undefined. The model cannot learn meaningful decision boundaries without positive validation examples. Test AUROC of 0.15 (below chance) confirms this — the model's predictions are anti-correlated with the true labels for this patient.

- **Fold 8 (P9)**: Perfect test scores (AUROC = 1.0, F1 = 0.98) suggest P9's relapse pattern is highly distinct from their stable period, making it trivially separable. This inflates aggregate metrics and should be interpreted cautiously.

- **Folds 0, 4**: Val AUROC is reasonable (0.84, 0.74) but F1 = 0 — the model ranks relapse days higher in probability but the default threshold (0.5) fails to classify any as positive. Threshold tuning per-fold could improve F1.

- **Fold 7 (P8)**: Longest training (21 epochs before early stopping) but poor test performance (AUROC = 0.23). The validation set has extreme class imbalance (9 pos / 70 neg), making val AUROC unreliable as a model selection criterion for this fold.

- **Fold 6 (P7)**: Inverted val/test pattern — low val AUROC (0.33) but high test AUROC (0.77). This suggests the validation data distribution differs substantially from the test data for P7.

### Early Stopping Behavior

Most folds stop at epoch 1 (6 out of 9 folds), indicating the model overfits rapidly after the first epoch. This is expected with ~300K parameters trained on only ~2,900 windows from 8 patients. The model captures the bulk of its useful signal in the first gradient update.

### Class Imbalance

Relapse prevalence varies substantially across patients:
- Highest: P3 test (59% relapse), P8 test (77% relapse)
- Lowest: P4 val (0% relapse), P4 test (12% relapse)

The weighted BCE loss compensates at the population level, but per-patient imbalance remains a challenge for LOSO evaluation.

## Total Training Time

| Metric | Value |
|--------|-------|
| Total wall time (all folds) | ~7 minutes |
| Average per fold | ~48 seconds |
| Longest fold | Fold 7 (93s, 21 epochs) |
| Shortest fold | Fold 3 (37s, 0 useful epochs) |
