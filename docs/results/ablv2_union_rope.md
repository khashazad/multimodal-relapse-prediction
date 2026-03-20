# Experiment: `ablv2_union_rope`

Ablation v2 — Union RoPE. 9 folds x 1 = 9 jobs.

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
| use_rope | True |


## Per-Fold Results

| Fold | Patient | Test AUROC | Test AUPRC | Test F1 |
|------|---------|------------|------------|---------|
| 0 | P1 | 0.780 | 0.716 | 0.000 |
| 1 | P2 | 0.710 | 0.563 | 0.510 |
| 2 | P3 | 0.765 | 0.788 | 0.000 |
| 3 | P4 | 0.802 | 0.457 | 0.000 |
| 4 | P5 | 0.744 | 0.702 | 0.000 |
| 5 | P6 | 0.698 | 0.673 | 0.000 |
| 6 | P7 | 0.588 | 0.367 | 0.000 |
| 7 | P8 | 0.924 | 0.979 | 0.866 |
| 8 | P9 | 0.888 | 0.899 | 0.706 |
| **Mean** | | **0.767** | **0.682** | **0.231** |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Test AUROC | 0.767 | 0.095 |
| Test AUPRC | 0.682 | 0.186 |
| Test F1 | 0.231 | 0.338 |
| Total runtime | 298s | |
