# Experiment: `ablation_baseline`

Ablation Stage 0: Baseline control — confirm ablation.py reproduces AUROC ~0.834 with all defaults. 9 jobs.

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
| 0 | P1 | 0.708 | 0.558 | 0.148 |
| 1 | P2 | 0.743 | 0.549 | 0.000 |
| 2 | P3 | 0.796 | 0.870 | 0.000 |
| 3 | P4 | 0.878 | 0.502 | 0.355 |
| 4 | P5 | 0.707 | 0.733 | 0.685 |
| 5 | P6 | 0.816 | 0.837 | 0.000 |
| 6 | P7 | — | — | 0.000 |
| 7 | P8 | 0.982 | 0.992 | 0.000 |
| 8 | P9 | 0.964 | 0.964 | 0.000 |
| **Mean** | | **0.824** | **0.751** | **0.132** |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Test AUROC | 0.824 | 0.101 |
| Test AUPRC | 0.751 | 0.182 |
| Test F1 | 0.132 | 0.226 |
| Total runtime | 324s | |
