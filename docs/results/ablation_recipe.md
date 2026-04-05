# Experiment: `ablation_recipe`

Training recipe sweep: optimizer, LR schedule, grad clipping, warmup. Focal loss locked (γ=1.0, α=0.5). 9 folds × 2 opt × 2 sched × 2 clip × 3 warmup = 216 jobs. warmup_epochs ignored when lr_schedule=none (36 redundant combos accepted).

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
| focal_gamma | 1.0 |
| focal_alpha | 0.5 |
| optimizer | ['adam', 'adamw'] (sweep) |
| lr_schedule | ['none', 'cosine_warm'] (sweep) |
| grad_clip | [0.0, 1.0] (sweep) |
| warmup_epochs | [3, 5, 10] (sweep) |


## Sweep Results (by `optimizer`, `lr_schedule`, `grad_clip`, `warmup_epochs`)

| `optimizer` | `lr_schedule` | `grad_clip` | `warmup_epochs` | Mean Test AUROC | Mean Test AUPRC | Mean Test F1 |
|---|---|---|---|---|---|---|
| adam | none | 1.0 | 3 | 0.842 | 0.749 | 0.261 |
| adamw | none | 0.0 | 10 | 0.838 | 0.757 | 0.272 |
| adamw | none | 0.0 | 3 | 0.838 | 0.757 | 0.272 |
| adamw | none | 0.0 | 5 | 0.838 | 0.757 | 0.272 |
| adamw | none | 1.0 | 5 | 0.814 | 0.761 | 0.450 |
| adamw | none | 1.0 | 3 | 0.810 | 0.727 | 0.422 |
| adam | none | 0.0 | 5 | 0.804 | 0.679 | 0.267 |
| adam | none | 0.0 | 10 | 0.802 | 0.701 | 0.339 |
| adam | none | 0.0 | 3 | 0.801 | 0.668 | 0.412 |
| adam | none | 1.0 | 10 | 0.800 | 0.690 | 0.329 |
| adamw | none | 1.0 | 10 | 0.786 | 0.679 | 0.455 |
| adam | cosine_warm | 0.0 | 5 | 0.785 | 0.644 | 0.237 |
| adamw | cosine_warm | 0.0 | 10 | 0.782 | 0.682 | 0.620 |
| adam | none | 1.0 | 5 | 0.780 | 0.698 | 0.203 |
| adam | cosine_warm | 0.0 | 10 | 0.777 | 0.710 | 0.379 |
| adam | cosine_warm | 1.0 | 10 | 0.772 | 0.671 | 0.379 |
| adamw | cosine_warm | 1.0 | 5 | 0.771 | 0.657 | 0.530 |
| adam | cosine_warm | 1.0 | 3 | 0.769 | 0.676 | 0.530 |
| adamw | cosine_warm | 1.0 | 3 | 0.769 | 0.676 | 0.530 |
| adam | cosine_warm | 0.0 | 3 | 0.768 | 0.649 | 0.532 |
| adamw | cosine_warm | 0.0 | 3 | 0.756 | 0.658 | 0.459 |
| adamw | cosine_warm | 0.0 | 5 | 0.755 | 0.616 | 0.480 |
| adamw | cosine_warm | 1.0 | 10 | 0.728 | 0.609 | 0.444 |
| adam | cosine_warm | 1.0 | 5 | 0.728 | 0.609 | 0.504 |


## Per-Fold Results (Best: optimizer=adam, lr_schedule=none, grad_clip=1.0, warmup_epochs=3)

| Fold | Patient | Test AUROC | Test AUPRC | Test F1 |
|------|---------|------------|------------|---------|
| 0 | P1 | 0.787 | 0.737 | 0.000 |
| 1 | P2 | 0.776 | 0.717 | 0.510 |
| 2 | P3 | 0.897 | 0.911 | 0.237 |
| 3 | P4 | 0.878 | 0.383 | 0.000 |
| 4 | P5 | 0.753 | 0.732 | 0.000 |
| 5 | P6 | 0.739 | 0.646 | 0.620 |
| 6 | P7 | 0.838 | 0.634 | 0.480 |
| 7 | P8 | 0.907 | 0.976 | 0.000 |
| 8 | P9 | 1.000 | 1.000 | 0.500 |
| **Mean** | | **0.842** | **0.749** | **0.261** |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Test AUROC | 0.842 | 0.081 |
| Test AUPRC | 0.749 | 0.182 |
| Test F1 | 0.261 | 0.251 |
| Total runtime | 304s | |
