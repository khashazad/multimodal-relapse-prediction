# Experiment: `ablv2_all_weight_decay`

Ablation v2 — All weight decay. 9 folds x 3 wd = 27 jobs.

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
| 0.001 | 0.826 | 0.722 | 0.513 |
| 0.0001 | 0.805 | 0.660 | 0.213 |
| 0.01 | 0.733 | 0.651 | 0.480 |


## Per-Fold Results (Best: weight_decay=0.001)

| Fold | Patient | Test AUROC | Test AUPRC | Test F1 |
|------|---------|------------|------------|---------|
| 0 | P1 | 0.677 | 0.629 | 0.148 |
| 1 | P2 | 0.714 | 0.607 | 0.537 |
| 2 | P3 | 0.911 | 0.941 | 0.800 |
| 3 | P4 | 0.846 | 0.327 | 0.440 |
| 4 | P5 | 0.737 | 0.772 | 0.688 |
| 5 | P6 | 0.783 | 0.800 | 0.676 |
| 6 | P7 | 0.821 | 0.468 | 0.000 |
| 7 | P8 | 1.000 | 1.000 | 0.866 |
| 8 | P9 | 0.947 | 0.952 | 0.465 |
| **Mean** | | **0.826** | **0.722** | **0.513** |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Test AUROC | 0.826 | 0.104 |
| Test AUPRC | 0.722 | 0.218 |
| Test F1 | 0.513 | 0.273 |
| Total runtime | 358s | |
