# Experiment: `exp_asymmetric_focal`

Asymmetric focal loss: gamma_pos=1.0 (strong positive gradients) gamma_neg=3.0 (suppress easy negatives)

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
| focal_gamma_pos | 1.0 |
| focal_gamma_neg | 3.0 |
| seed | 42 |
| threshold | 0.5 |
| calibrate_threshold | True |
| data_dir | data/processed/track1 |


## Per-Fold Results

| Fold | Patient | Best Epoch | Val AUROC | Val AUPRC | Val F1 | Test AUROC | Test AUPRC | Test F1 |
|------|---------|-----------|-----------|-----------|--------|------------|------------|---------|
| 0 | P1 | 2 | 0.749 | 0.640 | 0.651 | 0.426 | 0.280 | 0.488 |
| 1 | P2 | 2 | 0.957 | 0.918 | 0.779 | 0.542 | 0.335 | 0.507 |
| 2 | P3 | 6 | 0.765 | 0.680 | 0.133 | 0.864 | 0.874 | 0.514 |
| 3 | P4 | 0 | — | — | 0.000 | 0.678 | 0.174 | 0.304 |
| 4 | P5 | 18 | 0.865 | 0.829 | 0.267 | 0.592 | 0.549 | 0.527 |
| 5 | P6 | 4 | 0.513 | 0.513 | 0.650 | 0.565 | 0.519 | 0.571 |
| 6 | P7 | 5 | 0.254 | 0.143 | 0.137 | 0.580 | 0.278 | 0.429 |
| 7 | P8 | 1 | 0.502 | 0.119 | 0.205 | 0.978 | 0.994 | 0.871 |
| 8 | P9 | 7 | 0.698 | 0.480 | 0.373 | 1.000 | 1.000 | 0.984 |
| **Mean** | | | **0.663** | | | **0.692** | **0.556** | **0.577** |


## Calibrated Threshold Results

| Fold | Threshold | Cal Val F1 | Cal Val Prec | Cal Val Rec | Cal Test F1 | Cal Test Prec | Cal Test Rec |
|------|-----------|------------|--------------|-------------|-------------|---------------|--------------|
| 0 | 0.55 | 0.654 | 0.708 | 0.607 | 0.264 | 0.233 | 0.304 |
| 1 | 0.55 | 0.863 | 0.759 | 1.000 | 0.212 | 0.241 | 0.189 |
| 2 | 0.25 | 0.680 | 0.567 | 0.850 | 0.850 | 0.739 | 1.000 |
| 3 | 0.05 | 0.000 | 0.000 | 0.000 | 0.214 | 0.120 | 1.000 |
| 4 | 0.25 | 0.750 | 0.632 | 0.923 | 0.676 | 0.593 | 0.787 |
| 5 | 0.55 | 0.667 | 0.522 | 0.923 | 0.614 | 0.538 | 0.714 |
| 6 | 0.25 | 0.352 | 0.213 | 1.000 | 0.372 | 0.233 | 0.913 |
| 7 | 0.05 | 0.205 | 0.114 | 1.000 | 0.871 | 0.771 | 1.000 |
| 8 | 0.40 | 0.500 | 0.333 | 1.000 | 0.970 | 0.941 | 1.000 |
| **Mean** | | | | | **0.560** | | |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Val AUROC | 0.663 | 0.213 |
| Test AUROC | 0.692 | 0.194 |
| Test AUPRC | 0.556 | 0.305 |
| Test F1 | 0.577 | 0.202 |
| Cal Test F1 | 0.560 | 0.285 |
| Total runtime | 357s | |
