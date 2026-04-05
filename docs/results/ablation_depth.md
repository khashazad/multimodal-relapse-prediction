# Experiment: `ablation_depth`

Ablation: layer depth sweep with focal loss (best from ablation). 9 folds x 6 depths = 54 jobs.

## Configuration

| Parameter | Value |
|---|---|
| method | feature_exp |
| feature_set | all |
| d_model | 1024 |
| n_layers | [1, 2, 3, 4, 5, 6] (sweep) |
| dropout | 0.3 |
| nhead | 4 |
| seq_len | 7 |
| batch_size | 32 |
| lr | 0.001 |
| n_epochs | 80 |
| seed | 42 |
| data_path | data/processed/patient_data_export_all9.pkl |
| loss_type | focal |
| focal_gamma | 1.0 |
| focal_alpha | 0.5 |


## Sweep Results (by `n_layers`)

| `n_layers` | Mean Test AUROC | Mean Test AUPRC | Mean Test F1 |
|---|---|---|---|
| 3 | 0.824 | 0.732 | 0.336 |
| 2 | 0.801 | 0.694 | 0.517 |
| 4 | 0.784 | 0.698 | 0.133 |
| 5 | 0.741 | 0.605 | 0.145 |
| 1 | 0.713 | 0.583 | 0.467 |
| 6 | 0.702 | 0.574 | 0.185 |


## Per-Fold Results (Best: n_layers=3)

| Fold | Patient | Test AUROC | Test AUPRC | Test F1 |
|------|---------|------------|------------|---------|
| 0 | P1 | 0.668 | 0.508 | 0.000 |
| 1 | P2 | 0.725 | 0.564 | 0.510 |
| 2 | P3 | 0.934 | 0.960 | 0.795 |
| 3 | P4 | 0.872 | 0.814 | 0.230 |
| 4 | P5 | 0.737 | 0.770 | 0.000 |
| 5 | P6 | 0.739 | 0.623 | 0.620 |
| 6 | P7 | 0.777 | 0.449 | 0.000 |
| 7 | P8 | 0.995 | 0.998 | 0.866 |
| 8 | P9 | 0.965 | 0.899 | 0.000 |
| **Mean** | | **0.824** | **0.732** | **0.336** |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Test AUROC | 0.824 | 0.113 |
| Test AUPRC | 0.732 | 0.191 |
| Test F1 | 0.336 | 0.344 |
| Total runtime | 314s | |
