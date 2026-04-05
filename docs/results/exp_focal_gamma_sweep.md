# Experiment: `exp_focal_gamma_sweep`

Sweep focal gamma across 5 values (9 folds x 5 = 45 jobs)

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
| focal_gamma | [0.5, 1.0, 1.5, 2.0, 3.0] (sweep) |
| seed | 42 |
| threshold | 0.5 |
| calibrate_threshold | True |
| data_dir | data/processed/track1 |


## Sweep Results (by `focal_gamma`)

| `focal_gamma` | Mean Val AUROC | Mean Test AUROC | Mean Test AUPRC | Mean Test F1 |
|---|---|---|---|---|
| 3.0 | 0.646 | 0.691 | 0.558 | 0.293 |
| 1.5 | 0.645 | 0.683 | 0.578 | 0.257 |
| 2.0 | 0.641 | 0.682 | 0.555 | 0.311 |
| 0.5 | 0.668 | 0.653 | 0.542 | 0.251 |
| 1.0 | 0.614 | 0.648 | 0.557 | 0.244 |


## Per-Fold Results (Best: focal_gamma=3.0)

| Fold | Patient | Best Epoch | Val AUROC | Val AUPRC | Val F1 | Test AUROC | Test AUPRC | Test F1 |
|------|---------|-----------|-----------|-----------|--------|------------|------------|---------|
| 0 | P1 | 2 | 0.661 | 0.591 | 0.000 | 0.434 | 0.283 | 0.000 |
| 1 | P2 | 2 | 0.975 | 0.945 | 0.400 | 0.492 | 0.317 | 0.000 |
| 2 | P3 | 5 | 0.725 | 0.678 | 0.095 | 0.870 | 0.893 | 0.323 |
| 3 | P4 | 0 | — | — | 0.000 | 0.699 | 0.184 | 0.308 |
| 4 | P5 | 12 | 0.871 | 0.808 | 0.000 | 0.558 | 0.527 | 0.262 |
| 5 | P6 | 3 | 0.536 | 0.469 | 0.571 | 0.598 | 0.526 | 0.416 |
| 6 | P7 | 4 | 0.291 | 0.149 | 0.091 | 0.623 | 0.308 | 0.493 |
| 7 | P8 | 1 | 0.475 | 0.115 | 0.000 | 0.948 | 0.985 | 0.000 |
| 8 | P9 | 5 | 0.634 | 0.373 | 0.389 | 0.999 | 0.999 | 0.836 |
| **Mean** | | | **0.646** | | | **0.691** | **0.558** | **0.293** |


## Calibrated Threshold Results (focal_gamma=3.0)

| Fold | Threshold | Cal Val F1 | Cal Val Prec | Cal Val Rec | Cal Test F1 | Cal Test Prec | Cal Test Rec |
|------|-----------|------------|--------------|-------------|-------------|---------------|--------------|
| 0 | 0.05 | 0.636 | 0.467 | 1.000 | 0.495 | 0.329 | 1.000 |
| 1 | 0.45 | 0.815 | 0.688 | 1.000 | 0.472 | 0.362 | 0.676 |
| 2 | 0.25 | 0.619 | 0.526 | 0.750 | 0.857 | 0.750 | 1.000 |
| 3 | 0.05 | 0.000 | 0.000 | 0.000 | 0.214 | 0.120 | 1.000 |
| 4 | 0.25 | 0.747 | 0.654 | 0.872 | 0.676 | 0.603 | 0.770 |
| 5 | 0.05 | 0.634 | 0.464 | 1.000 | 0.628 | 0.458 | 1.000 |
| 6 | 0.05 | 0.345 | 0.209 | 1.000 | 0.400 | 0.250 | 1.000 |
| 7 | 0.05 | 0.205 | 0.114 | 1.000 | 0.871 | 0.771 | 1.000 |
| 8 | 0.40 | 0.427 | 0.291 | 0.800 | 0.985 | 0.970 | 1.000 |
| **Mean** | | | | | **0.622** | | |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Val AUROC | 0.646 | 0.204 |
| Test AUROC | 0.691 | 0.191 |
| Test AUPRC | 0.558 | 0.303 |
| Test F1 | 0.293 | 0.260 |
| Cal Test F1 | 0.622 | 0.237 |
| Total runtime | 359s | |
