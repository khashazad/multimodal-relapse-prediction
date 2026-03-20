# Experiment: `exp_ablation_loss_only`

Ablation: focal loss with LARGE model (d=64) to isolate loss function effect

## Configuration

| Parameter | Value |
|---|---|
| model_name | transformer_v1 |
| d_model | 64 |
| nhead | 4 |
| num_encoder_layers | 1 |
| num_fusion_layers | 1 |
| dropout | 0.1 |
| lr | 0.0001 |
| weight_decay | 0.01 |
| batch_size | 32 |
| epochs | 100 |
| patience | 15 |
| loss_fn | focal |
| focal_alpha | 0.75 |
| focal_gamma | 2.0 |
| seed | 42 |
| threshold | 0.5 |
| calibrate_threshold | True |
| data_dir | data/processed/track1 |


## Per-Fold Results

| Fold | Patient | Best Epoch | Val AUROC | Val AUPRC | Val F1 | Test AUROC | Test AUPRC | Test F1 |
|------|---------|-----------|-----------|-----------|--------|------------|------------|---------|
| 0 | P1 | 1 | 0.709 | 0.589 | 0.000 | 0.366 | 0.264 | 0.000 |
| 1 | P2 | 18 | 0.934 | 0.895 | 0.000 | 0.511 | 0.349 | 0.000 |
| 2 | P3 | 3 | 0.866 | 0.774 | 0.140 | 0.593 | 0.674 | 0.140 |
| 3 | P4 | 0 | — | — | 0.000 | 0.130 | 0.073 | 0.000 |
| 4 | P5 | 9 | 0.784 | 0.717 | 0.050 | 0.614 | 0.623 | 0.544 |
| 5 | P6 | 1 | 0.553 | 0.583 | 0.292 | 0.578 | 0.539 | 0.230 |
| 6 | P7 | 30 | 0.499 | 0.221 | 0.000 | 0.483 | 0.243 | 0.111 |
| 7 | P8 | 49 | 0.597 | 0.171 | 0.200 | 0.227 | 0.629 | 0.000 |
| 8 | P9 | 1 | 0.727 | 0.482 | 0.368 | 1.000 | 1.000 | 0.968 |
| **Mean** | | | **0.708** | | | **0.500** | **0.488** | **0.221** |


## Calibrated Threshold Results

| Fold | Threshold | Cal Val F1 | Cal Val Prec | Cal Val Rec | Cal Test F1 | Cal Test Prec | Cal Test Rec |
|------|-----------|------------|--------------|-------------|-------------|---------------|--------------|
| 0 | 0.25 | 0.709 | 0.549 | 1.000 | 0.500 | 0.333 | 1.000 |
| 1 | 0.05 | 0.128 | 1.000 | 0.068 | 0.000 | 0.000 | 0.000 |
| 2 | 0.10 | 0.795 | 0.767 | 0.825 | 0.581 | 0.643 | 0.529 |
| 3 | 0.05 | 0.000 | 0.000 | 0.000 | 0.074 | 0.043 | 0.273 |
| 4 | 0.05 | 0.370 | 0.667 | 0.256 | 0.639 | 0.655 | 0.623 |
| 5 | 0.05 | 0.634 | 0.464 | 1.000 | 0.628 | 0.458 | 1.000 |
| 6 | 0.05 | 0.000 | 0.000 | 0.000 | 0.347 | 0.250 | 0.565 |
| 7 | 0.10 | 0.345 | 0.250 | 0.556 | 0.121 | 0.333 | 0.074 |
| 8 | 0.40 | 0.531 | 0.386 | 0.850 | 0.955 | 0.914 | 1.000 |
| **Mean** | | | | | **0.427** | | |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Val AUROC | 0.708 | 0.142 |
| Test AUROC | 0.500 | 0.237 |
| Test AUPRC | 0.488 | 0.266 |
| Test F1 | 0.221 | 0.312 |
| Cal Test F1 | 0.427 | 0.298 |
| Total runtime | 469s | |
