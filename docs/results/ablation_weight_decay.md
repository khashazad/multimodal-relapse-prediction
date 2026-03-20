# Experiment: `ablation_weight_decay`

Ablation Stage 5: Weight decay. 9 folds x 3 wd = 27 jobs.

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
| weight_decay | [0.0001, 0.001, 0.01] (sweep) |


## Sweep Results (by `weight_decay`)

| `weight_decay` | Mean Test AUROC | Mean Test AUPRC | Mean Test F1 |
|---|---|---|---|
| 0.001 | 0.811 | 0.704 | 0.453 |
| 0.0001 | 0.804 | 0.682 | 0.241 |
| 0.01 | 0.722 | 0.632 | 0.390 |


## Per-Fold Results (Best: weight_decay=0.001)

| Fold | Patient | Test AUROC | Test AUPRC | Test F1 |
|------|---------|------------|------------|---------|
| 0 | P1 | 0.741 | 0.674 | 0.607 |
| 1 | P2 | 0.671 | 0.461 | 0.594 |
| 2 | P3 | 0.911 | 0.941 | 0.800 |
| 3 | P4 | 0.848 | 0.449 | 0.385 |
| 4 | P5 | 0.696 | 0.665 | 0.624 |
| 5 | P6 | — | — | 0.000 |
| 6 | P7 | 0.777 | 0.528 | 0.000 |
| 7 | P8 | 0.911 | 0.975 | 0.226 |
| 8 | P9 | 0.937 | 0.943 | 0.839 |
| **Mean** | | **0.811** | **0.704** | **0.453** |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Test AUROC | 0.811 | 0.098 |
| Test AUPRC | 0.704 | 0.207 |
| Test F1 | 0.453 | 0.300 |
| Total runtime | 327s | |
