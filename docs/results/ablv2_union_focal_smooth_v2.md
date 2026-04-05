# Experiment: `ablv2_union_focal_smooth_v2`

Ablation v2 — Union focal + label smoothing (stabilized: grad_clip + adamw). 9 folds x 2 gamma x 3 alpha x 4 eps = 216 jobs.

## Configuration

| Parameter | Value |
|---|---|
| method | feature_exp |
| feature_set | union |
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
| focal_gamma | [0.5, 1.0] (sweep) |
| focal_alpha | [0.3, 0.5, 0.7] (sweep) |
| label_smoothing | [0.1, 0.15, 0.2, 0.25] (sweep) |
| optimizer | adamw |
| grad_clip | 1.0 |


## Sweep Results (by `focal_gamma`, `focal_alpha`, `label_smoothing`)

| `focal_gamma` | `focal_alpha` | `label_smoothing` | Mean Test AUROC | Mean Test AUPRC | Mean Test F1 |
|---|---|---|---|---|---|
| 0.5 | 0.5 | 0.1 | 0.840 | 0.742 | 0.318 |
| 1.0 | 0.7 | 0.15 | 0.837 | 0.728 | 0.479 |
| 0.5 | 0.7 | 0.1 | 0.828 | 0.713 | 0.490 |
| 0.5 | 0.3 | 0.2 | 0.825 | 0.754 | 0.096 |
| 1.0 | 0.5 | 0.15 | 0.824 | 0.719 | 0.391 |
| 1.0 | 0.5 | 0.25 | 0.822 | 0.716 | 0.381 |
| 1.0 | 0.3 | 0.2 | 0.821 | 0.699 | 0.000 |
| 1.0 | 0.3 | 0.25 | 0.821 | 0.754 | 0.054 |
| 0.5 | 0.3 | 0.15 | 0.808 | 0.695 | 0.000 |
| 0.5 | 0.5 | 0.15 | 0.808 | 0.724 | 0.497 |
| 1.0 | 0.3 | 0.1 | 0.806 | 0.763 | 0.000 |
| 0.5 | 0.5 | 0.2 | 0.805 | 0.690 | 0.245 |
| 1.0 | 0.5 | 0.2 | 0.805 | 0.722 | 0.322 |
| 0.5 | 0.3 | 0.1 | 0.804 | 0.725 | 0.069 |
| 1.0 | 0.3 | 0.15 | 0.800 | 0.668 | 0.000 |
| 1.0 | 0.7 | 0.1 | 0.798 | 0.690 | 0.508 |
| 0.5 | 0.7 | 0.25 | 0.795 | 0.702 | 0.537 |
| 1.0 | 0.5 | 0.1 | 0.785 | 0.660 | 0.356 |
| 0.5 | 0.7 | 0.15 | 0.784 | 0.660 | 0.430 |
| 0.5 | 0.5 | 0.25 | 0.782 | 0.643 | 0.337 |
| 0.5 | 0.3 | 0.25 | 0.782 | 0.701 | 0.000 |
| 1.0 | 0.7 | 0.25 | 0.777 | 0.651 | 0.387 |
| 1.0 | 0.7 | 0.2 | 0.772 | 0.653 | 0.451 |
| 0.5 | 0.7 | 0.2 | 0.768 | 0.608 | 0.377 |


## Per-Fold Results (Best: focal_gamma=0.5, focal_alpha=0.5, label_smoothing=0.1)

| Fold | Patient | Test AUROC | Test AUPRC | Test F1 |
|------|---------|------------|------------|---------|
| 0 | P1 | 0.824 | 0.739 | 0.000 |
| 1 | P2 | 0.728 | 0.657 | 0.510 |
| 2 | P3 | 0.834 | 0.855 | 0.435 |
| 3 | P4 | 0.834 | 0.336 | 0.000 |
| 4 | P5 | — | — | 0.000 |
| 5 | P6 | 0.763 | 0.730 | 0.000 |
| 6 | P7 | 0.795 | 0.664 | 0.407 |
| 7 | P8 | 0.982 | 0.995 | 0.866 |
| 8 | P9 | 0.962 | 0.960 | 0.647 |
| **Mean** | | **0.840** | **0.742** | **0.318** |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Test AUROC | 0.840 | 0.084 |
| Test AUPRC | 0.742 | 0.194 |
| Test F1 | 0.318 | 0.311 |
| Total runtime | 285s | |
