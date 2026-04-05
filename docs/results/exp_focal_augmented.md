# Experiment: `exp_focal_augmented`

Focal loss + small model + data augmentation (jitter+scale). Tests if augmentation helps with the focal+small recipe

## Configuration

| Parameter | Value |
|---|---|
| model_name | transformer_v1 |
| d_model | 32 |
| nhead | 2 |
| num_encoder_layers | 1 |
| num_fusion_layers | 1 |
| dropout | 0.3 |
| lr | 0.0001 |
| weight_decay | 0.01 |
| batch_size | 32 |
| epochs | 100 |
| patience | 15 |
| loss_fn | focal |
| focal_alpha | 0.75 |
| focal_gamma | 2.0 |
| augmentation | all |
| seed | 42 |
| threshold | 0.5 |
| calibrate_threshold | True |
| data_dir | data/processed/track1 |


## Per-Fold Results

| Fold | Patient | Best Epoch | Val AUROC | Val AUPRC | Val F1 | Test AUROC | Test AUPRC | Test F1 |
|------|---------|-----------|-----------|-----------|--------|------------|------------|---------|
| 0 | P1 | 2 | 0.683 | 0.615 | 0.000 | 0.443 | 0.287 | 0.000 |
| 1 | P2 | 2 | 0.974 | 0.943 | 0.240 | 0.495 | 0.320 | 0.000 |
| 2 | P3 | 5 | 0.729 | 0.678 | 0.095 | 0.861 | 0.891 | 0.349 |
| 3 | P4 | 0 | — | — | 0.000 | 0.715 | 0.190 | 0.308 |
| 4 | P5 | 13 | 0.884 | 0.828 | 0.000 | 0.580 | 0.540 | 0.282 |
| 5 | P6 | 4 | 0.575 | 0.554 | 0.600 | 0.561 | 0.498 | 0.577 |
| 6 | P7 | 4 | 0.311 | 0.154 | 0.098 | 0.631 | 0.314 | 0.500 |
| 7 | P8 | 1 | 0.383 | 0.106 | 0.000 | 0.905 | 0.971 | 0.000 |
| 8 | P9 | 5 | 0.602 | 0.356 | 0.378 | 1.000 | 1.000 | 0.857 |
| **Mean** | | | **0.643** | | | **0.688** | **0.557** | **0.319** |


## Calibrated Threshold Results

| Fold | Threshold | Cal Val F1 | Cal Val Prec | Cal Val Rec | Cal Test F1 | Cal Test Prec | Cal Test Rec |
|------|-----------|------------|--------------|-------------|-------------|---------------|--------------|
| 0 | 0.40 | 0.683 | 0.519 | 1.000 | 0.488 | 0.339 | 0.870 |
| 1 | 0.45 | 0.898 | 0.815 | 1.000 | 0.333 | 0.317 | 0.351 |
| 2 | 0.15 | 0.635 | 0.465 | 1.000 | 0.750 | 0.600 | 1.000 |
| 3 | 0.05 | 0.000 | 0.000 | 0.000 | 0.227 | 0.128 | 1.000 |
| 4 | 0.20 | 0.780 | 0.744 | 0.821 | 0.652 | 0.595 | 0.721 |
| 5 | 0.40 | 0.655 | 0.494 | 0.974 | 0.571 | 0.452 | 0.776 |
| 6 | 0.20 | 0.349 | 0.211 | 1.000 | 0.386 | 0.242 | 0.957 |
| 7 | 0.45 | 0.212 | 0.118 | 1.000 | 0.898 | 0.828 | 0.981 |
| 8 | 0.35 | 0.476 | 0.312 | 1.000 | 0.985 | 0.970 | 1.000 |
| **Mean** | | | | | **0.588** | | |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Val AUROC | 0.643 | 0.213 |
| Test AUROC | 0.688 | 0.184 |
| Test AUPRC | 0.557 | 0.299 |
| Test F1 | 0.319 | 0.278 |
| Cal Test F1 | 0.588 | 0.243 |
| Total runtime | 461s | |
