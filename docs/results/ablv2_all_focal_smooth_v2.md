# Experiment: `ablv2_all_focal_smooth_v2`

Ablation v2 — All focal + label smoothing (stabilized: grad_clip + adamw). 9 folds x 2 gamma x 3 alpha x 4 eps = 216 jobs.

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
| focal_gamma | [0.5, 1.0] (sweep) |
| focal_alpha | [0.3, 0.5, 0.7] (sweep) |
| label_smoothing | [0.1, 0.15, 0.2, 0.25] (sweep) |
| optimizer | adamw |
| grad_clip | 1.0 |


## Sweep Results (by `focal_gamma`, `focal_alpha`, `label_smoothing`)

| `focal_gamma` | `focal_alpha` | `label_smoothing` | Mean Test AUROC | Mean Test AUPRC | Mean Test F1 |
|---|---|---|---|---|---|
| 0.5 | 0.7 | 0.25 | 0.904 | 0.825 | 0.248 |
| 1.0 | 0.3 | 0.15 | 0.891 | 0.902 | 0.077 |
| 0.5 | 0.3 | 0.15 | 0.886 | 0.823 | 0.000 |
| 1.0 | 0.7 | 0.25 | 0.885 | 0.812 | 0.281 |
| 1.0 | 0.7 | 0.2 | 0.846 | 0.725 | 0.522 |
| 1.0 | 0.7 | 0.15 | 0.825 | 0.784 | 0.578 |
| 0.5 | 0.7 | 0.2 | 0.819 | 0.697 | 0.578 |
| 0.5 | 0.5 | 0.25 | 0.816 | 0.718 | 0.348 |
| 0.5 | 0.3 | 0.25 | 0.815 | 0.751 | 0.000 |
| 1.0 | 0.5 | 0.15 | 0.809 | 0.713 | 0.189 |
| 1.0 | 0.3 | 0.2 | 0.807 | 0.723 | 0.000 |
| 0.5 | 0.5 | 0.2 | 0.807 | 0.705 | 0.283 |
| 0.5 | 0.7 | 0.15 | 0.803 | 0.681 | 0.534 |
| 0.5 | 0.7 | 0.1 | 0.802 | 0.670 | 0.587 |
| 1.0 | 0.7 | 0.1 | 0.800 | 0.676 | 0.523 |
| 0.5 | 0.3 | 0.2 | 0.799 | 0.647 | 0.000 |
| 1.0 | 0.5 | 0.1 | 0.788 | 0.694 | 0.182 |
| 0.5 | 0.5 | 0.1 | 0.788 | 0.694 | 0.302 |
| 1.0 | 0.3 | 0.25 | 0.786 | 0.668 | 0.086 |
| 0.5 | 0.3 | 0.1 | 0.786 | 0.725 | 0.113 |
| 1.0 | 0.3 | 0.1 | 0.786 | 0.689 | 0.000 |
| 1.0 | 0.5 | 0.25 | 0.779 | 0.688 | 0.360 |
| 1.0 | 0.5 | 0.2 | 0.766 | 0.635 | 0.303 |
| 0.5 | 0.5 | 0.15 | 0.761 | 0.647 | 0.258 |


## Per-Fold Results (Best: focal_gamma=0.5, focal_alpha=0.7, label_smoothing=0.25)

| Fold | Patient | Test AUROC | Test AUPRC | Test F1 |
|------|---------|------------|------------|---------|
| 0 | P1 | — | — | 0.000 |
| 1 | P2 | — | — | 0.000 |
| 2 | P3 | — | — | 0.000 |
| 3 | P4 | — | — | 0.000 |
| 4 | P5 | — | — | 0.000 |
| 5 | P6 | — | — | 0.000 |
| 6 | P7 | 0.776 | 0.510 | 0.407 |
| 7 | P8 | 0.949 | 0.983 | 0.866 |
| 8 | P9 | 0.986 | 0.983 | 0.957 |
| **Mean** | | **0.904** | **0.825** | **0.248** |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Test AUROC | 0.904 | 0.091 |
| Test AUPRC | 0.825 | 0.223 |
| Test F1 | 0.248 | 0.377 |
| Total runtime | 132s | |
