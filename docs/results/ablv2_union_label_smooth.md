# Experiment: `ablv2_union_label_smooth`

Ablation v2 — Union label smoothing. 9 folds x 4 eps = 36 jobs.

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
| label_smoothing | [0.05, 0.1, 0.15, 0.2] (sweep) |


## Sweep Results (by `label_smoothing`)

| `label_smoothing` | Mean Test AUROC | Mean Test AUPRC | Mean Test F1 |
|---|---|---|---|
| 0.05 | 0.802 | 0.714 | 0.282 |
| 0.15 | 0.794 | 0.651 | 0.266 |
| 0.2 | 0.776 | 0.698 | 0.389 |
| 0.1 | 0.764 | 0.719 | 0.274 |


## Per-Fold Results (Best: label_smoothing=0.05)

| Fold | Patient | Test AUROC | Test AUPRC | Test F1 |
|------|---------|------------|------------|---------|
| 0 | P1 | 0.746 | 0.598 | 0.000 |
| 1 | P2 | 0.821 | 0.735 | 0.510 |
| 2 | P3 | 0.893 | 0.929 | 0.000 |
| 3 | P4 | 0.699 | 0.374 | 0.196 |
| 4 | P5 | 0.686 | 0.680 | 0.505 |
| 5 | P6 | 0.796 | 0.782 | 0.677 |
| 6 | P7 | 0.649 | 0.386 | 0.000 |
| 7 | P8 | 0.990 | 0.997 | 0.000 |
| 8 | P9 | 0.941 | 0.948 | 0.647 |
| **Mean** | | **0.802** | **0.714** | **0.282** |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Test AUROC | 0.802 | 0.113 |
| Test AUPRC | 0.714 | 0.217 |
| Test F1 | 0.282 | 0.282 |
| Total runtime | 336s | |
