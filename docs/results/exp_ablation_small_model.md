# Experiment: `exp_ablation_small_model`

Ablation: small model (d=32, dropout=0.3) with weighted BCE to isolate model capacity effect

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
| loss_fn | weighted_bce |
| seed | 42 |
| threshold | 0.5 |
| calibrate_threshold | True |
| data_dir | data/processed/track1 |


## Per-Fold Results

| Fold | Patient | Best Epoch | Val AUROC | Val AUPRC | Val F1 | Test AUROC | Test AUPRC | Test F1 |
|------|---------|-----------|-----------|-----------|--------|------------|------------|---------|
| 0 | P1 | 2 | 0.739 | 0.688 | 0.133 | 0.437 | 0.285 | 0.000 |
| 1 | P2 | 2 | 0.976 | 0.949 | 0.944 | 0.500 | 0.323 | 0.159 |
| 2 | P3 | 18 | 0.741 | 0.702 | 0.140 | 0.815 | 0.821 | 0.457 |
| 3 | P4 | 0 | — | — | 0.000 | 0.695 | 0.181 | 0.308 |
| 4 | P5 | 7 | 0.865 | 0.756 | 0.000 | 0.548 | 0.536 | 0.264 |
| 5 | P6 | 4 | 0.571 | 0.548 | 0.608 | 0.514 | 0.465 | 0.577 |
| 6 | P7 | 4 | 0.308 | 0.154 | 0.128 | 0.696 | 0.400 | 0.500 |
| 7 | P8 | 1 | 0.533 | 0.135 | 0.105 | 0.943 | 0.984 | 0.230 |
| 8 | P9 | 5 | 0.684 | 0.402 | 0.378 | 0.999 | 0.999 | 0.951 |
| **Mean** | | | **0.677** | | | **0.683** | **0.555** | **0.383** |


## Calibrated Threshold Results

| Fold | Threshold | Cal Val F1 | Cal Val Prec | Cal Val Rec | Cal Test F1 | Cal Test Prec | Cal Test Rec |
|------|-----------|------------|--------------|-------------|-------------|---------------|--------------|
| 0 | 0.40 | 0.700 | 0.538 | 1.000 | 0.447 | 0.321 | 0.739 |
| 1 | 0.50 | 0.944 | 0.933 | 0.955 | 0.159 | 0.192 | 0.135 |
| 2 | 0.05 | 0.286 | 0.778 | 0.175 | 0.610 | 0.806 | 0.490 |
| 3 | 0.05 | 0.000 | 0.000 | 0.000 | 0.340 | 0.222 | 0.727 |
| 4 | 0.10 | 0.844 | 0.745 | 0.974 | 0.707 | 0.596 | 0.869 |
| 5 | 0.45 | 0.648 | 0.515 | 0.872 | 0.595 | 0.500 | 0.735 |
| 6 | 0.10 | 0.349 | 0.211 | 1.000 | 0.400 | 0.250 | 1.000 |
| 7 | 0.05 | 0.205 | 0.114 | 1.000 | 0.871 | 0.771 | 1.000 |
| 8 | 0.30 | 0.519 | 0.351 | 1.000 | 0.985 | 0.970 | 1.000 |
| **Mean** | | | | | **0.568** | | |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Val AUROC | 0.677 | 0.194 |
| Test AUROC | 0.683 | 0.190 |
| Test AUPRC | 0.555 | 0.289 |
| Test F1 | 0.383 | 0.262 |
| Cal Test F1 | 0.568 | 0.247 |
| Total runtime | 442s | |
