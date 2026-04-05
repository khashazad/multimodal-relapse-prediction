# Experiment: `ablv2_union_stoch_depth`

Ablation v2 — Union stochastic depth. 9 folds x 3 drop_rate = 27 jobs.

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
| stochastic_depth | [0.1, 0.2, 0.3] (sweep) |


## Sweep Results (by `stochastic_depth`)

| `stochastic_depth` | Mean Test AUROC | Mean Test AUPRC | Mean Test F1 |
|---|---|---|---|
| 0.1 | 0.797 | 0.674 | 0.356 |
| 0.2 | 0.792 | 0.691 | 0.502 |
| 0.3 | 0.769 | 0.711 | 0.593 |


## Per-Fold Results (Best: stochastic_depth=0.1)

| Fold | Patient | Test AUROC | Test AUPRC | Test F1 |
|------|---------|------------|------------|---------|
| 0 | P1 | 0.773 | 0.600 | 0.000 |
| 1 | P2 | 0.748 | 0.629 | 0.000 |
| 2 | P3 | 0.814 | 0.886 | 0.739 |
| 3 | P4 | 0.796 | 0.256 | 0.242 |
| 4 | P5 | 0.635 | 0.589 | 0.685 |
| 5 | P6 | 0.659 | 0.581 | 0.620 |
| 6 | P7 | 0.807 | 0.553 | 0.000 |
| 7 | P8 | 0.947 | 0.984 | 0.000 |
| 8 | P9 | 0.990 | 0.990 | 0.917 |
| **Mean** | | **0.797** | **0.674** | **0.356** |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Test AUROC | 0.797 | 0.110 |
| Test AUPRC | 0.674 | 0.225 |
| Test F1 | 0.356 | 0.359 |
| Total runtime | 287s | |
