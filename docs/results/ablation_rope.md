# Experiment: `ablation_rope`

Ablation Stage 4: RoPE (Rotary Position Embedding). 9 folds x 1 = 9 jobs.

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
| use_rope | True |


## Per-Fold Results

| Fold | Patient | Test AUROC | Test AUPRC | Test F1 |
|------|---------|------------|------------|---------|
| 0 | P1 | 0.677 | 0.504 | 0.000 |
| 1 | P2 | 0.711 | 0.442 | 0.576 |
| 2 | P3 | 0.828 | 0.888 | 0.000 |
| 3 | P4 | 0.729 | 0.372 | 0.210 |
| 4 | P5 | — | — | 0.000 |
| 5 | P6 | 0.717 | 0.709 | 0.620 |
| 6 | P7 | 0.780 | 0.559 | 0.000 |
| 7 | P8 | 0.942 | 0.982 | 0.721 |
| 8 | P9 | 0.681 | 0.701 | 0.647 |
| **Mean** | | **0.758** | **0.645** | **0.308** |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Test AUROC | 0.758 | 0.084 |
| Test AUPRC | 0.645 | 0.201 |
| Test F1 | 0.308 | 0.306 |
| Total runtime | 354s | |
