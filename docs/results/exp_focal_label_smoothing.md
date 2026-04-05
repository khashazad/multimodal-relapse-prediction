# Experiment: `exp_focal_label_smoothing`

Label smoothing (eps=0.1) with focal loss to reduce overconfidence and maintain gradient flow

## Configuration

| Parameter | Value |
|---|---|
| model_name | transformer_v1 |
| d_model | 32 |
| nhead | 2 |
| num_encoder_layers | 1 |
| num_fusion_layers | 1 |
| dropout | 0.3 |
| lr | 0.0001 |
| weight_decay | 0.01 |
| batch_size | 32 |
| epochs | 100 |
| patience | 15 |
| loss_fn | focal |
| focal_alpha | 0.75 |
| focal_gamma | 2.0 |
| label_smoothing | 0.1 |
| seed | 42 |
| threshold | 0.5 |
| calibrate_threshold | True |
| data_dir | data/processed/track1 |


## Per-Fold Results

| Fold | Patient | Best Epoch | Val AUROC | Val AUPRC | Val F1 | Test AUROC | Test AUPRC | Test F1 |
|------|---------|-----------|-----------|-----------|--------|------------|------------|---------|
| 0 | P1 | 2 | 0.680 | 0.625 | 0.000 | 0.439 | 0.286 | 0.000 |
| 1 | P2 | 2 | 0.975 | 0.946 | 0.000 | 0.497 | 0.324 | 0.000 |
| 2 | P3 | 5 | 0.733 | 0.680 | 0.049 | 0.857 | 0.891 | 0.349 |
| 3 | P4 | 0 | — | — | 0.000 | 0.725 | 0.195 | 0.263 |
| 4 | P5 | 6 | 0.891 | 0.825 | 0.000 | 0.536 | 0.535 | 0.265 |
| 5 | P6 | 4 | 0.577 | 0.552 | 0.598 | 0.556 | 0.497 | 0.506 |
| 6 | P7 | 1 | 0.347 | 0.221 | 0.000 | 0.716 | 0.613 | 0.000 |
| 7 | P8 | 2 | 0.308 | 0.087 | 0.000 | 0.850 | 0.950 | 0.000 |
| 8 | P9 | 4 | 0.617 | 0.392 | 0.353 | 0.999 | 0.999 | 0.815 |
| **Mean** | | | **0.641** | | | **0.686** | **0.588** | **0.244** |


## Calibrated Threshold Results

| Fold | Threshold | Cal Val F1 | Cal Val Prec | Cal Val Rec | Cal Test F1 | Cal Test Prec | Cal Test Rec |
|------|-----------|------------|--------------|-------------|-------------|---------------|--------------|
| 0 | 0.40 | 0.646 | 0.568 | 0.750 | 0.373 | 0.306 | 0.478 |
| 1 | 0.45 | 0.920 | 0.930 | 0.909 | 0.143 | 0.211 | 0.108 |
| 2 | 0.15 | 0.633 | 0.534 | 0.775 | 0.807 | 0.730 | 0.902 |
| 3 | 0.05 | 0.000 | 0.000 | 0.000 | 0.272 | 0.157 | 1.000 |
| 4 | 0.25 | 0.810 | 0.756 | 0.872 | 0.624 | 0.550 | 0.721 |
| 5 | 0.35 | 0.645 | 0.476 | 1.000 | 0.551 | 0.427 | 0.776 |
| 6 | 0.05 | 0.345 | 0.209 | 1.000 | 0.400 | 0.250 | 1.000 |
| 7 | 0.05 | 0.205 | 0.114 | 1.000 | 0.871 | 0.771 | 1.000 |
| 8 | 0.35 | 0.426 | 0.270 | 1.000 | 0.985 | 0.970 | 1.000 |
| **Mean** | | | | | **0.558** | | |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Val AUROC | 0.641 | 0.220 |
| Test AUROC | 0.686 | 0.180 |
| Test AUPRC | 0.588 | 0.283 |
| Test F1 | 0.244 | 0.268 |
| Cal Test F1 | 0.558 | 0.271 |
| Total runtime | 318s | |
