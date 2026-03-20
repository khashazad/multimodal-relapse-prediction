# Experiment: `ablv2_all_stoch_depth`

Ablation v2 — All stochastic depth. 9 folds x 3 drop_rate = 27 jobs.

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
| 0.2 | 0.824 | 0.735 | 0.363 |
| 0.3 | 0.807 | 0.699 | 0.324 |
| 0.1 | 0.790 | 0.704 | 0.186 |


## Per-Fold Results (Best: stochastic_depth=0.2)

| Fold | Patient | Test AUROC | Test AUPRC | Test F1 |
|------|---------|------------|------------|---------|
| 0 | P1 | 0.880 | 0.695 | 0.000 |
| 1 | P2 | 0.763 | 0.639 | 0.510 |
| 2 | P3 | 0.821 | 0.849 | 0.743 |
| 3 | P4 | 0.871 | 0.678 | 0.210 |
| 4 | P5 | 0.663 | 0.629 | 0.685 |
| 5 | P6 | 0.783 | 0.692 | 0.696 |
| 6 | P7 | 0.687 | 0.465 | 0.421 |
| 7 | P8 | 0.971 | 0.988 | 0.000 |
| 8 | P9 | 0.979 | 0.978 | 0.000 |
| **Mean** | | **0.824** | **0.735** | **0.363** |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Test AUROC | 0.824 | 0.106 |
| Test AUPRC | 0.735 | 0.162 |
| Test F1 | 0.363 | 0.299 |
| Total runtime | 318s | |
