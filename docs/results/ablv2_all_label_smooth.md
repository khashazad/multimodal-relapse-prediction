# Experiment: `ablv2_all_label_smooth`

Ablation v2 — All label smoothing. 9 folds x 4 eps = 36 jobs.

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
| 0.05 | 0.810 | 0.686 | 0.251 |
| 0.1 | 0.807 | 0.739 | 0.413 |
| 0.15 | 0.793 | 0.675 | 0.121 |
| 0.2 | 0.775 | 0.700 | 0.245 |


## Per-Fold Results (Best: label_smoothing=0.05)

| Fold | Patient | Test AUROC | Test AUPRC | Test F1 |
|------|---------|------------|------------|---------|
| 0 | P1 | 0.840 | 0.662 | 0.077 |
| 1 | P2 | 0.658 | 0.543 | 0.510 |
| 2 | P3 | 0.769 | 0.768 | 0.796 |
| 3 | P4 | 0.780 | 0.333 | 0.000 |
| 4 | P5 | 0.794 | 0.755 | 0.000 |
| 5 | P6 | 0.762 | 0.736 | 0.000 |
| 6 | P7 | 0.765 | 0.441 | 0.000 |
| 7 | P8 | 0.983 | 0.995 | 0.873 |
| 8 | P9 | 0.937 | 0.946 | 0.000 |
| **Mean** | | **0.810** | **0.686** | **0.251** |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Test AUROC | 0.810 | 0.093 |
| Test AUPRC | 0.686 | 0.206 |
| Test F1 | 0.251 | 0.349 |
| Total runtime | 300s | |
