# Experiment: `ablv2_all_focal`

Ablation v2 — All focal loss. 9 folds x 3 gamma x 3 alpha = 81 jobs.

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
| 2.0 | 0.5 | 0.802 | 0.713 | 0.462 |
| 3.0 | 0.5 | 0.800 | 0.658 | 0.424 |
| 1.0 | 0.5 | 0.796 | 0.705 | 0.395 |
| 3.0 | 0.25 | 0.793 | 0.683 | 0.000 |
| 2.0 | 0.25 | 0.787 | 0.669 | 0.000 |
| 2.0 | 0.75 | 0.781 | 0.665 | 0.578 |
| 1.0 | 0.75 | 0.780 | 0.650 | 0.583 |
| 1.0 | 0.25 | 0.780 | 0.676 | 0.000 |
| 3.0 | 0.75 | 0.735 | 0.600 | 0.578 |


## Per-Fold Results (Best: focal_gamma=2.0, focal_alpha=0.5)

| Fold | Patient | Test AUROC | Test AUPRC | Test F1 |
|------|---------|------------|------------|---------|
| 0 | P1 | 0.679 | 0.597 | 0.510 |
| 1 | P2 | 0.737 | 0.581 | 0.510 |
| 2 | P3 | 0.842 | 0.890 | 0.743 |
| 3 | P4 | 0.882 | 0.488 | 0.210 |
| 4 | P5 | 0.632 | 0.632 | 0.000 |
| 5 | P6 | 0.778 | 0.740 | 0.000 |
| 6 | P7 | 0.686 | 0.506 | 0.407 |
| 7 | P8 | 0.998 | 0.999 | 0.866 |
| 8 | P9 | 0.981 | 0.981 | 0.909 |
| **Mean** | | **0.802** | **0.713** | **0.462** |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Test AUROC | 0.802 | 0.125 |
| Test AUPRC | 0.713 | 0.188 |
| Test F1 | 0.462 | 0.323 |
| Total runtime | 335s | |
