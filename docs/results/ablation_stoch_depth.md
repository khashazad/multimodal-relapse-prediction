# Experiment: `ablation_stoch_depth`

Ablation Stage 3: Stochastic depth. 9 folds x 3 drop_rate = 27 jobs.

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
| stochastic_depth | [0.1, 0.2, 0.3] (sweep) |


## Sweep Results (by `stochastic_depth`)

| `stochastic_depth` | Mean Test AUROC | Mean Test AUPRC | Mean Test F1 |
|---|---|---|---|
| 0.2 | 0.824 | 0.717 | 0.318 |
| 0.3 | 0.803 | 0.724 | 0.297 |
| 0.1 | 0.787 | 0.672 | 0.163 |


## Per-Fold Results (Best: stochastic_depth=0.2)

| Fold | Patient | Test AUROC | Test AUPRC | Test F1 |
|------|---------|------------|------------|---------|
| 0 | P1 | 0.880 | 0.695 | 0.000 |
| 1 | P2 | 0.757 | 0.660 | 0.000 |
| 2 | P3 | 0.821 | 0.849 | 0.743 |
| 3 | P4 | 0.779 | 0.264 | 0.377 |
| 4 | P5 | 0.739 | 0.780 | 0.685 |
| 5 | P6 | 0.799 | 0.783 | 0.646 |
| 6 | P7 | 0.839 | 0.729 | 0.414 |
| 7 | P8 | — | — | 0.000 |
| 8 | P9 | 0.979 | 0.978 | 0.000 |
| **Mean** | | **0.824** | **0.717** | **0.318** |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Test AUROC | 0.824 | 0.072 |
| Test AUPRC | 0.717 | 0.195 |
| Test F1 | 0.318 | 0.306 |
| Total runtime | 295s | |
