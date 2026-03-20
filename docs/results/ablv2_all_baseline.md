# Experiment: `ablv2_all_baseline`

Ablation v2 — All baseline. 9 jobs.

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


## Per-Fold Results

| Fold | Patient | Test AUROC | Test AUPRC | Test F1 |
|------|---------|------------|------------|---------|
| 0 | P1 | 0.706 | 0.697 | 0.541 |
| 1 | P2 | 0.743 | 0.549 | 0.000 |
| 2 | P3 | 0.796 | 0.870 | 0.000 |
| 3 | P4 | 0.878 | 0.502 | 0.355 |
| 4 | P5 | 0.707 | 0.733 | 0.685 |
| 5 | P6 | 0.816 | 0.837 | 0.000 |
| 6 | P7 | 0.759 | 0.455 | 0.414 |
| 7 | P8 | 0.956 | 0.988 | 0.866 |
| 8 | P9 | 0.955 | 0.919 | 0.000 |
| **Mean** | | **0.813** | **0.728** | **0.318** |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Test AUROC | 0.813 | 0.092 |
| Test AUPRC | 0.728 | 0.181 |
| Test F1 | 0.318 | 0.316 |
| Total runtime | 310s | |
