# Experiment: `ablv2_union_focal`

Ablation v2 — Union focal loss. 9 folds x 3 gamma x 3 alpha = 81 jobs.

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
| focal_gamma | [1.0, 2.0, 3.0] (sweep) |
| focal_alpha | [0.25, 0.5, 0.75] (sweep) |


## Sweep Results (by `focal_gamma`, `focal_alpha`)

| `focal_gamma` | `focal_alpha` | Mean Test AUROC | Mean Test AUPRC | Mean Test F1 |
|---|---|---|---|---|
| 2.0 | 0.75 | 0.824 | 0.730 | 0.522 |
| 1.0 | 0.25 | 0.802 | 0.672 | 0.000 |
| 1.0 | 0.5 | 0.800 | 0.681 | 0.272 |
| 2.0 | 0.25 | 0.796 | 0.696 | 0.104 |
| 1.0 | 0.75 | 0.794 | 0.689 | 0.578 |
| 3.0 | 0.5 | 0.792 | 0.667 | 0.299 |
| 3.0 | 0.25 | 0.778 | 0.648 | 0.007 |
| 3.0 | 0.75 | 0.773 | 0.662 | 0.517 |
| 2.0 | 0.5 | 0.762 | 0.671 | 0.364 |


## Per-Fold Results (Best: focal_gamma=2.0, focal_alpha=0.75)

| Fold | Patient | Test AUROC | Test AUPRC | Test F1 |
|------|---------|------------|------------|---------|
| 0 | P1 | 0.898 | 0.751 | 0.510 |
| 1 | P2 | — | — | 0.000 |
| 2 | P3 | 0.728 | 0.766 | 0.748 |
| 3 | P4 | 0.862 | 0.493 | 0.210 |
| 4 | P5 | 0.661 | 0.657 | 0.685 |
| 5 | P6 | 0.726 | 0.761 | 0.620 |
| 6 | P7 | 0.804 | 0.474 | 0.407 |
| 7 | P8 | 0.995 | 0.998 | 0.866 |
| 8 | P9 | 0.921 | 0.943 | 0.647 |
| **Mean** | | **0.824** | **0.730** | **0.522** |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Test AUROC | 0.824 | 0.107 |
| Test AUPRC | 0.730 | 0.176 |
| Test F1 | 0.522 | 0.259 |
| Total runtime | 312s | |
