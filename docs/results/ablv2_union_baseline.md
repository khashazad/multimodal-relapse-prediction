# Experiment: `ablv2_union_baseline`

Ablation v2 — Union baseline. 9 jobs.

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


## Per-Fold Results

| Fold | Patient | Test AUROC | Test AUPRC | Test F1 |
|------|---------|------------|------------|---------|
| 0 | P1 | 0.752 | 0.669 | 0.000 |
| 1 | P2 | 0.718 | 0.527 | 0.000 |
| 2 | P3 | — | — | 0.000 |
| 3 | P4 | 0.796 | 0.362 | 0.000 |
| 4 | P5 | 0.680 | 0.743 | 0.636 |
| 5 | P6 | 0.673 | 0.631 | 0.620 |
| 6 | P7 | 0.816 | 0.558 | 0.000 |
| 7 | P8 | 0.950 | 0.985 | 0.866 |
| 8 | P9 | 0.965 | 0.978 | 0.000 |
| **Mean** | | **0.794** | **0.682** | **0.236** |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Test AUROC | 0.794 | 0.106 |
| Test AUPRC | 0.682 | 0.202 |
| Test F1 | 0.236 | 0.340 |
| Total runtime | 321s | |
