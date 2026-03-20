# Experiment: `ablv2_all_rope`

Ablation v2 — All RoPE. 9 folds x 1 = 9 jobs.

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
| 0 | P1 | 0.679 | 0.437 | 0.000 |
| 1 | P2 | 0.714 | 0.545 | 0.000 |
| 2 | P3 | 0.788 | 0.865 | 0.641 |
| 3 | P4 | 0.729 | 0.372 | 0.210 |
| 4 | P5 | 0.726 | 0.741 | 0.693 |
| 5 | P6 | 0.717 | 0.709 | 0.620 |
| 6 | P7 | 0.780 | 0.559 | 0.000 |
| 7 | P8 | 0.942 | 0.982 | 0.721 |
| 8 | P9 | 0.864 | 0.843 | 0.000 |
| **Mean** | | **0.771** | **0.673** | **0.321** |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Test AUROC | 0.771 | 0.079 |
| Test AUPRC | 0.673 | 0.195 |
| Test F1 | 0.321 | 0.319 |
| Total runtime | 307s | |
