# Experiment: `ablation_focal`

Ablation Stage 1: Focal loss. 9 folds x 3 gamma x 3 alpha = 81 jobs.

## Configuration

| Parameter | Value |
|---|---|
| method | feature_exp |
| feature_set | all |
| d_model | 1024 |
| n_layers | 3 |
| dropout | 0.3 |
| nhead | 4 |
| seq_len | 7 |
| batch_size | 32 |
| lr | 0.001 |
| n_epochs | 80 |
| seed | 42 |
| data_path | data/processed/patient_data_export_all9.pkl |
| loss_type | focal |
| focal_gamma | [1.0, 2.0, 3.0] (sweep) |
| focal_alpha | [0.25, 0.5, 0.75] (sweep) |


## Sweep Results (by `focal_gamma`, `focal_alpha`)

| `focal_gamma` | `focal_alpha` | Mean Test AUROC | Mean Test AUPRC | Mean Test F1 |
|---|---|---|---|---|
| 1.0 | 0.5 | 0.822 | 0.739 | 0.285 |
| 2.0 | 0.5 | 0.812 | 0.716 | 0.445 |
| 1.0 | 0.25 | 0.807 | 0.726 | 0.008 |
| 3.0 | 0.25 | 0.797 | 0.722 | 0.082 |
| 3.0 | 0.5 | 0.796 | 0.659 | 0.461 |
| 1.0 | 0.75 | 0.782 | 0.661 | 0.583 |
| 2.0 | 0.75 | 0.774 | 0.642 | 0.521 |
| 2.0 | 0.25 | 0.758 | 0.609 | 0.000 |
| 3.0 | 0.75 | 0.748 | 0.616 | 0.578 |


## Per-Fold Results (Best: focal_gamma=1.0, focal_alpha=0.5)

| Fold | Patient | Test AUROC | Test AUPRC | Test F1 |
|------|---------|------------|------------|---------|
| 0 | P1 | 0.668 | 0.508 | 0.000 |
| 1 | P2 | 0.725 | 0.564 | 0.510 |
| 2 | P3 | 0.934 | 0.960 | 0.795 |
| 3 | P4 | 0.872 | 0.814 | 0.230 |
| 4 | P5 | 0.737 | 0.770 | 0.000 |
| 5 | P6 | 0.812 | 0.728 | 0.000 |
| 6 | P7 | 0.777 | 0.449 | 0.000 |
| 7 | P8 | 0.995 | 0.998 | 0.866 |
| 8 | P9 | 0.875 | 0.860 | 0.167 |
| **Mean** | | **0.822** | **0.739** | **0.285** |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Test AUROC | 0.822 | 0.100 |
| Test AUPRC | 0.739 | 0.184 |
| Test F1 | 0.285 | 0.332 |
| Total runtime | 338s | |
