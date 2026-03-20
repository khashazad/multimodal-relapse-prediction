# Experiment: `ablation_label_smooth`

Ablation Stage 2: Label smoothing. 9 folds x 4 eps = 36 jobs.

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
| label_smoothing | [0.05, 0.1, 0.15, 0.2] (sweep) |


## Sweep Results (by `label_smoothing`)

| `label_smoothing` | Mean Test AUROC | Mean Test AUPRC | Mean Test F1 |
|---|---|---|---|
| 0.2 | 0.839 | 0.736 | 0.235 |
| 0.15 | 0.819 | 0.728 | 0.209 |
| 0.05 | 0.813 | 0.675 | 0.161 |
| 0.1 | 0.784 | 0.681 | 0.536 |


## Per-Fold Results (Best: label_smoothing=0.2)

| Fold | Patient | Test AUROC | Test AUPRC | Test F1 |
|------|---------|------------|------------|---------|
| 0 | P1 | 0.740 | 0.577 | 0.000 |
| 1 | P2 | 0.810 | 0.566 | 0.000 |
| 2 | P3 | 0.913 | 0.955 | 0.743 |
| 3 | P4 | 0.862 | 0.456 | 0.415 |
| 4 | P5 | 0.701 | 0.735 | 0.000 |
| 5 | P6 | 0.866 | 0.866 | 0.000 |
| 6 | P7 | 0.699 | 0.485 | 0.308 |
| 7 | P8 | 0.973 | 0.993 | 0.000 |
| 8 | P9 | 0.990 | 0.990 | 0.647 |
| **Mean** | | **0.839** | **0.736** | **0.235** |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Test AUROC | 0.839 | 0.104 |
| Test AUPRC | 0.736 | 0.208 |
| Test F1 | 0.235 | 0.287 |
| Total runtime | 305s | |
