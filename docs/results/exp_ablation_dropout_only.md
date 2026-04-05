# Experiment: `exp_ablation_dropout_only`

Ablation: large model with weighted BCE but dropout=0.3 to isolate dropout effect

## Configuration

| Parameter | Value |
|---|---|
| model_name | transformer_v1 |
| d_model | 64 |
| nhead | 4 |
| num_encoder_layers | 1 |
| num_fusion_layers | 1 |
| dropout | 0.3 |
| lr | 0.0001 |
| weight_decay | 0.01 |
| batch_size | 32 |
| epochs | 100 |
| patience | 15 |
| loss_fn | weighted_bce |
| seed | 42 |
| threshold | 0.5 |
| calibrate_threshold | True |
| data_dir | data/processed/track1 |


## Per-Fold Results

| Fold | Patient | Best Epoch | Val AUROC | Val AUPRC | Val F1 | Test AUROC | Test AUPRC | Test F1 |
|------|---------|-----------|-----------|-----------|--------|------------|------------|---------|
| 0 | P1 | 1 | 0.914 | 0.815 | 0.000 | 0.368 | 0.263 | 0.000 |
| 1 | P2 | 3 | 0.865 | 0.749 | 0.163 | 0.729 | 0.540 | 0.000 |
| 2 | P3 | 10 | 0.768 | 0.672 | 0.133 | 0.633 | 0.697 | 0.136 |
| 3 | P4 | 0 | — | — | 0.000 | 0.299 | 0.090 | 0.143 |
| 4 | P5 | 7 | 0.648 | 0.545 | 0.000 | 0.651 | 0.658 | 0.356 |
| 5 | P6 | 1 | 0.621 | 0.660 | 0.492 | 0.619 | 0.616 | 0.487 |
| 6 | P7 | 1 | 0.330 | 0.158 | 0.200 | 0.730 | 0.363 | 0.491 |
| 7 | P8 | 0 | 0.500 | 0.114 | 0.205 | 0.500 | 0.771 | 0.871 |
| 8 | P9 | 1 | 0.743 | 0.488 | 0.421 | 0.999 | 0.999 | 0.984 |
| **Mean** | | | **0.674** | | | **0.614** | **0.555** | **0.385** |


## Calibrated Threshold Results

| Fold | Threshold | Cal Val F1 | Cal Val Prec | Cal Val Rec | Cal Test F1 | Cal Test Prec | Cal Test Rec |
|------|-----------|------------|--------------|-------------|-------------|---------------|--------------|
| 0 | 0.35 | 0.862 | 0.757 | 1.000 | 0.457 | 0.340 | 0.696 |
| 1 | 0.05 | 0.716 | 0.784 | 0.659 | 0.308 | 0.533 | 0.216 |
| 2 | 0.05 | 0.213 | 0.714 | 0.125 | 0.308 | 0.714 | 0.196 |
| 3 | 0.05 | 0.000 | 0.000 | 0.000 | 0.133 | 0.088 | 0.273 |
| 4 | 0.05 | 0.136 | 0.600 | 0.077 | 0.569 | 0.646 | 0.508 |
| 5 | 0.45 | 0.646 | 0.533 | 0.821 | 0.569 | 0.517 | 0.633 |
| 6 | 0.05 | 0.345 | 0.209 | 1.000 | 0.400 | 0.250 | 1.000 |
| 7 | 0.05 | 0.205 | 0.114 | 1.000 | 0.871 | 0.771 | 1.000 |
| 8 | 0.40 | 0.519 | 0.351 | 1.000 | 0.810 | 0.681 | 1.000 |
| **Mean** | | | | | **0.492** | | |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Val AUROC | 0.674 | 0.180 |
| Test AUROC | 0.614 | 0.197 |
| Test AUPRC | 0.555 | 0.262 |
| Test F1 | 0.385 | 0.339 |
| Cal Test F1 | 0.492 | 0.227 |
| Total runtime | 358s | |
