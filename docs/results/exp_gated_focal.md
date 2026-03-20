# Experiment: `exp_gated_focal`

Gated fusion + focal loss + small model. Lightweight per-modality gates may complement focal loss

## Configuration

| Parameter | Value |
|---|---|
| model_name | transformer_v3 |
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
| seed | 42 |
| threshold | 0.5 |
| calibrate_threshold | True |
| data_dir | data/processed/track1 |


## Per-Fold Results

| Fold | Patient | Best Epoch | Val AUROC | Val AUPRC | Val F1 | Test AUROC | Test AUPRC | Test F1 |
|------|---------|-----------|-----------|-----------|--------|------------|------------|---------|
| 0 | P1 | 2 | 0.924 | 0.866 | 0.000 | 0.212 | 0.223 | 0.000 |
| 1 | P2 | 0 | 0.500 | 0.407 | 0.579 | 0.500 | 0.339 | 0.507 |
| 2 | P3 | 14 | 0.713 | 0.633 | 0.000 | 0.798 | 0.776 | 0.073 |
| 3 | P4 | 0 | — | — | 0.000 | 0.593 | 0.150 | 0.080 |
| 4 | P5 | 12 | 0.698 | 0.601 | 0.000 | 0.458 | 0.473 | 0.273 |
| 5 | P6 | 1 | 0.672 | 0.588 | 0.000 | 0.386 | 0.445 | 0.000 |
| 6 | P7 | 8 | 0.496 | 0.263 | 0.389 | 0.373 | 0.220 | 0.240 |
| 7 | P8 | 30 | 0.559 | 0.173 | 0.000 | 0.597 | 0.798 | 0.036 |
| 8 | P9 | 3 | 0.453 | 0.242 | 0.000 | 0.948 | 0.961 | 0.815 |
| **Mean** | | | **0.627** | | | **0.541** | **0.487** | **0.225** |


## Calibrated Threshold Results

| Fold | Threshold | Cal Val F1 | Cal Val Prec | Cal Val Rec | Cal Test F1 | Cal Test Prec | Cal Test Rec |
|------|-----------|------------|--------------|-------------|-------------|---------------|--------------|
| 0 | 0.45 | 0.681 | 0.842 | 0.571 | 0.000 | 0.000 | 0.000 |
| 1 | 0.05 | 0.579 | 0.407 | 1.000 | 0.507 | 0.339 | 1.000 |
| 2 | 0.05 | 0.561 | 0.548 | 0.575 | 0.815 | 0.772 | 0.863 |
| 3 | 0.05 | 0.000 | 0.000 | 0.000 | 0.204 | 0.132 | 0.455 |
| 4 | 0.05 | 0.596 | 0.509 | 0.718 | 0.701 | 0.540 | 1.000 |
| 5 | 0.45 | 0.648 | 0.515 | 0.872 | 0.343 | 0.340 | 0.347 |
| 6 | 0.45 | 0.419 | 0.375 | 0.474 | 0.207 | 0.171 | 0.261 |
| 7 | 0.05 | 0.256 | 0.167 | 0.556 | 0.788 | 0.820 | 0.759 |
| 8 | 0.05 | 0.396 | 0.247 | 1.000 | 0.646 | 0.478 | 1.000 |
| **Mean** | | | | | **0.468** | | |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Val AUROC | 0.627 | 0.146 |
| Test AUROC | 0.541 | 0.213 |
| Test AUPRC | 0.487 | 0.275 |
| Test F1 | 0.225 | 0.261 |
| Cal Test F1 | 0.468 | 0.275 |
| Total runtime | 356s | |
