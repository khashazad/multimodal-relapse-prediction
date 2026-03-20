# Experiment: `exp_focal_lr_warmup`

Lower LR (5e-5) with 5-epoch linear warmup then constant to address rapid overfitting

## Configuration

| Parameter | Value |
|---|---|
| model_name | transformer_v1 |
| d_model | 32 |
| nhead | 2 |
| num_encoder_layers | 1 |
| num_fusion_layers | 1 |
| dropout | 0.3 |
| lr | 5e-05 |
| weight_decay | 0.01 |
| batch_size | 32 |
| epochs | 100 |
| patience | 15 |
| loss_fn | focal |
| focal_alpha | 0.75 |
| focal_gamma | 2.0 |
| scheduler | warmup_constant |
| warmup_epochs | 5 |
| seed | 42 |
| threshold | 0.5 |
| calibrate_threshold | True |
| data_dir | data/processed/track1 |


## Per-Fold Results

| Fold | Patient | Best Epoch | Val AUROC | Val AUPRC | Val F1 | Test AUROC | Test AUPRC | Test F1 |
|------|---------|-----------|-----------|-----------|--------|------------|------------|---------|
| 0 | P1 | 7 | 0.609 | 0.609 | 0.000 | 0.418 | 0.279 | 0.000 |
| 1 | P2 | 8 | 0.972 | 0.932 | 0.414 | 0.524 | 0.349 | 0.000 |
| 2 | P3 | 22 | 0.771 | 0.664 | 0.091 | 0.838 | 0.853 | 0.424 |
| 3 | P4 | 0 | — | — | 0.000 | 0.713 | 0.192 | 0.286 |
| 4 | P5 | 17 | 0.893 | 0.850 | 0.000 | 0.535 | 0.528 | 0.273 |
| 5 | P6 | 10 | 0.581 | 0.551 | 0.581 | 0.551 | 0.485 | 0.571 |
| 6 | P7 | 1 | 0.545 | 0.303 | 0.000 | 0.624 | 0.311 | 0.000 |
| 7 | P8 | 1 | 0.424 | 0.122 | 0.000 | 0.675 | 0.881 | 0.000 |
| 8 | P9 | 1 | 0.670 | 0.368 | 0.000 | 0.490 | 0.441 | 0.000 |
| **Mean** | | | **0.683** | | | **0.596** | **0.480** | **0.173** |


## Calibrated Threshold Results

| Fold | Threshold | Cal Val F1 | Cal Val Prec | Cal Val Rec | Cal Test F1 | Cal Test Prec | Cal Test Rec |
|------|-----------|------------|--------------|-------------|-------------|---------------|--------------|
| 0 | 0.40 | 0.675 | 0.509 | 1.000 | 0.469 | 0.328 | 0.826 |
| 1 | 0.45 | 0.915 | 0.860 | 0.977 | 0.189 | 0.312 | 0.135 |
| 2 | 0.10 | 0.698 | 0.652 | 0.750 | 0.835 | 0.750 | 0.941 |
| 3 | 0.05 | 0.000 | 0.000 | 0.000 | 0.214 | 0.120 | 1.000 |
| 4 | 0.30 | 0.805 | 0.729 | 0.897 | 0.606 | 0.563 | 0.656 |
| 5 | 0.45 | 0.636 | 0.493 | 0.897 | 0.602 | 0.500 | 0.755 |
| 6 | 0.05 | 0.345 | 0.209 | 1.000 | 0.400 | 0.250 | 1.000 |
| 7 | 0.05 | 0.205 | 0.114 | 1.000 | 0.871 | 0.771 | 1.000 |
| 8 | 0.05 | 0.396 | 0.247 | 1.000 | 0.646 | 0.478 | 1.000 |
| **Mean** | | | | | **0.537** | | |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Val AUROC | 0.683 | 0.172 |
| Test AUROC | 0.596 | 0.122 |
| Test AUPRC | 0.480 | 0.229 |
| Test F1 | 0.173 | 0.209 |
| Cal Test F1 | 0.537 | 0.229 |
| Total runtime | 525s | |
