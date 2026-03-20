# Experiment: `ablation_focal_smooth`

Ablation: Focal loss + label smoothing combined. 9 folds x 2 gamma x 3 alpha x 4 eps = 216 jobs.

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
| loss_type | focal |
| focal_gamma | [0.5, 1.0] (sweep) |
| focal_alpha | [0.3, 0.5, 0.7] (sweep) |
| label_smoothing | [0.1, 0.15, 0.2, 0.25] (sweep) |


## Sweep Results (by `focal_gamma`, `focal_alpha`, `label_smoothing`)

| `focal_gamma` | `focal_alpha` | `label_smoothing` | Mean Test AUROC | Mean Test AUPRC | Mean Test F1 |
|---|---|---|---|---|---|
| 1.0 | 0.3 | 0.25 | 0.857 | 0.690 | 0.148 |
| 1.0 | 0.5 | 0.15 | 0.837 | 0.778 | 0.172 |
| 0.5 | 0.5 | 0.2 | 0.832 | 0.717 | 0.142 |
| 1.0 | 0.5 | 0.1 | 0.831 | 0.730 | 0.118 |
| 1.0 | 0.7 | 0.25 | 0.829 | 0.713 | 0.578 |
| 1.0 | 0.5 | 0.2 | 0.826 | 0.768 | 0.219 |
| 1.0 | 0.3 | 0.2 | 0.818 | 0.721 | 0.052 |
| 0.5 | 0.5 | 0.1 | 0.817 | 0.714 | 0.297 |
| 0.5 | 0.3 | 0.1 | 0.816 | 0.720 | 0.000 |
| 0.5 | 0.5 | 0.25 | 0.811 | 0.705 | 0.164 |
| 0.5 | 0.7 | 0.15 | 0.808 | 0.684 | 0.584 |
| 0.5 | 0.3 | 0.25 | 0.807 | 0.695 | 0.078 |
| 1.0 | 0.7 | 0.2 | 0.807 | 0.676 | 0.504 |
| 0.5 | 0.3 | 0.15 | 0.803 | 0.686 | 0.000 |
| 1.0 | 0.7 | 0.1 | 0.803 | 0.692 | 0.495 |
| 1.0 | 0.3 | 0.15 | 0.802 | 0.705 | 0.106 |
| 1.0 | 0.7 | 0.15 | 0.796 | 0.671 | 0.509 |
| 0.5 | 0.3 | 0.2 | 0.792 | 0.716 | 0.154 |
| 0.5 | 0.5 | 0.15 | 0.791 | 0.668 | 0.412 |
| 0.5 | 0.7 | 0.2 | 0.784 | 0.662 | 0.557 |
| 1.0 | 0.5 | 0.25 | 0.782 | 0.674 | 0.236 |
| 0.5 | 0.7 | 0.1 | 0.777 | 0.652 | 0.584 |
| 0.5 | 0.7 | 0.25 | 0.773 | 0.658 | 0.506 |
| 1.0 | 0.3 | 0.1 | 0.756 | 0.656 | 0.000 |


## Per-Fold Results (Best: focal_gamma=1.0, focal_alpha=0.3, label_smoothing=0.25)

| Fold | Patient | Test AUROC | Test AUPRC | Test F1 |
|------|---------|------------|------------|---------|
| 0 | P1 | — | — | 0.000 |
| 1 | P2 | 0.659 | 0.482 | 0.000 |
| 2 | P3 | — | — | 0.000 |
| 3 | P4 | 0.842 | 0.310 | 0.393 |
| 4 | P5 | — | — | 0.000 |
| 5 | P6 | — | — | 0.000 |
| 6 | P7 | 0.860 | 0.711 | 0.000 |
| 7 | P8 | 0.979 | 0.994 | 0.943 |
| 8 | P9 | 0.945 | 0.953 | 0.000 |
| **Mean** | | **0.857** | **0.690** | **0.148** |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Test AUROC | 0.857 | 0.111 |
| Test AUPRC | 0.690 | 0.264 |
| Test F1 | 0.148 | 0.307 |
| Total runtime | 311s | |
