# Experiment: `ablv2_union_weight_decay`

Ablation v2 — Union weight decay. 9 folds x 3 wd = 27 jobs.

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
| weight_decay | [0.0001, 0.001, 0.01] (sweep) |


## Sweep Results (by `weight_decay`)

| `weight_decay` | Mean Test AUROC | Mean Test AUPRC | Mean Test F1 |
|---|---|---|---|
| 0.0001 | 0.814 | 0.732 | 0.507 |
| 0.001 | 0.800 | 0.693 | 0.457 |
| 0.01 | 0.706 | 0.595 | 0.268 |


## Per-Fold Results (Best: weight_decay=0.0001)

| Fold | Patient | Test AUROC | Test AUPRC | Test F1 |
|------|---------|------------|------------|---------|
| 0 | P1 | 0.838 | 0.709 | 0.622 |
| 1 | P2 | 0.750 | 0.677 | 0.565 |
| 2 | P3 | 0.850 | 0.899 | 0.507 |
| 3 | P4 | 0.909 | 0.666 | 0.154 |
| 4 | P5 | 0.643 | 0.645 | 0.532 |
| 5 | P6 | 0.670 | 0.659 | 0.000 |
| 6 | P7 | 0.696 | 0.359 | 0.407 |
| 7 | P8 | 0.990 | 0.997 | 0.902 |
| 8 | P9 | 0.979 | 0.977 | 0.877 |
| **Mean** | | **0.814** | **0.732** | **0.507** |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Test AUROC | 0.814 | 0.124 |
| Test AUPRC | 0.732 | 0.188 |
| Test F1 | 0.507 | 0.279 |
| Total runtime | 306s | |
