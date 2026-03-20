# Experiment: `exp_focal_alpha_sweep`

Sweep focal alpha across 5 values (9 folds x 5 = 45 jobs)

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
| focal_alpha | [0.5, 0.65, 0.75, 0.85, 0.9] (sweep) |
| focal_gamma | 2.0 |
| seed | 42 |
| threshold | 0.5 |
| calibrate_threshold | True |
| data_dir | data/processed/track1 |


## Sweep Results (by `focal_alpha`)

| `focal_alpha` | Mean Val AUROC | Mean Test AUROC | Mean Test AUPRC | Mean Test F1 |
|---|---|---|---|---|
| 0.75 | 0.636 | 0.675 | 0.553 | 0.311 |
| 0.65 | 0.644 | 0.668 | 0.584 | 0.184 |
| 0.9 | 0.616 | 0.654 | 0.536 | 0.587 |
| 0.5 | 0.668 | 0.637 | 0.557 | 0.064 |
| 0.85 | 0.663 | 0.630 | 0.519 | 0.534 |


## Per-Fold Results (Best: focal_alpha=0.75)

| Fold | Patient | Best Epoch | Val AUROC | Val AUPRC | Val F1 | Test AUROC | Test AUPRC | Test F1 |
|------|---------|-----------|-----------|-----------|--------|------------|------------|---------|
| 0 | P1 | 2 | 0.671 | 0.604 | 0.000 | 0.441 | 0.286 | 0.000 |
| 1 | P2 | 2 | 0.975 | 0.946 | 0.240 | 0.491 | 0.319 | 0.000 |
| 2 | P3 | 5 | 0.732 | 0.679 | 0.095 | 0.861 | 0.890 | 0.349 |
| 3 | P4 | 0 | — | — | 0.000 | 0.714 | 0.190 | 0.308 |
| 4 | P5 | 7 | 0.882 | 0.796 | 0.000 | 0.544 | 0.531 | 0.256 |
| 5 | P6 | 4 | 0.566 | 0.547 | 0.584 | 0.563 | 0.502 | 0.577 |
| 6 | P7 | 4 | 0.305 | 0.152 | 0.100 | 0.618 | 0.305 | 0.493 |
| 7 | P8 | 1 | 0.338 | 0.093 | 0.000 | 0.841 | 0.952 | 0.000 |
| 8 | P9 | 4 | 0.618 | 0.395 | 0.389 | 0.999 | 0.999 | 0.815 |
| **Mean** | | | **0.636** | | | **0.675** | **0.553** | **0.311** |


## Calibrated Threshold Results (focal_alpha=0.75)

| Fold | Threshold | Cal Val F1 | Cal Val Prec | Cal Val Rec | Cal Test F1 | Cal Test Prec | Cal Test Rec |
|------|-----------|------------|--------------|-------------|-------------|---------------|--------------|
| 0 | 0.40 | 0.683 | 0.519 | 1.000 | 0.488 | 0.339 | 0.870 |
| 1 | 0.45 | 0.898 | 0.815 | 1.000 | 0.333 | 0.317 | 0.351 |
| 2 | 0.15 | 0.640 | 0.471 | 1.000 | 0.750 | 0.600 | 1.000 |
| 3 | 0.05 | 0.000 | 0.000 | 0.000 | 0.234 | 0.133 | 1.000 |
| 4 | 0.25 | 0.800 | 0.679 | 0.974 | 0.699 | 0.600 | 0.836 |
| 5 | 0.40 | 0.650 | 0.487 | 0.974 | 0.585 | 0.469 | 0.776 |
| 6 | 0.20 | 0.349 | 0.211 | 1.000 | 0.386 | 0.242 | 0.957 |
| 7 | 0.05 | 0.205 | 0.114 | 1.000 | 0.871 | 0.771 | 1.000 |
| 8 | 0.35 | 0.400 | 0.250 | 1.000 | 0.955 | 0.914 | 1.000 |
| **Mean** | | | | | **0.589** | | |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Val AUROC | 0.636 | 0.221 |
| Test AUROC | 0.675 | 0.180 |
| Test AUPRC | 0.553 | 0.297 |
| Test F1 | 0.311 | 0.269 |
| Cal Test F1 | 0.589 | 0.234 |
| Total runtime | 365s | |
