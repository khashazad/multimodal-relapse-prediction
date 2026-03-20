# Experiment: `train`

## Configuration

| Parameter | Value |
|---|---|
| d_model | 64 |
| nhead | 4 |
| num_encoder_layers | 1 |
| num_fusion_layers | 1 |
| dropout | 0.1 |
| lr | 0.0001 |
| weight_decay | 0.01 |
| batch_size | 32 |
| epochs | 100 |
| patience | 15 |
| loss_fn | weighted_bce |
| focal_alpha | 0.25 |
| focal_gamma | 2.0 |
| seed | 42 |
| threshold | 0.5 |
| data_dir | data/processed/track1 |


## Per-Fold Results

| Fold | Patient | Best Epoch | Val AUROC | Val AUPRC | Val F1 | Test AUROC | Test AUPRC | Test F1 |
|------|---------|-----------|-----------|-----------|--------|------------|------------|---------|
| 0 | P1 | 1 | 0.843 | 0.721 | 0.000 | 0.389 | 0.272 | 0.000 |
| 1 | P2 | 1 | 0.864 | 0.755 | 0.452 | 0.724 | 0.555 | 0.050 |
| 2 | P3 | 5 | 0.838 | 0.774 | 0.182 | 0.578 | 0.692 | 0.254 |
| 3 | P4 | 0 | — | — | 0.000 | 0.149 | 0.075 | 0.000 |
| 4 | P5 | 4 | 0.742 | 0.597 | 0.000 | 0.653 | 0.653 | 0.345 |
| 5 | P6 | 1 | 0.608 | 0.634 | 0.467 | 0.545 | 0.543 | 0.395 |
| 6 | P7 | 1 | 0.331 | 0.158 | 0.122 | 0.773 | 0.416 | 0.542 |
| 7 | P8 | 21 | 0.629 | 0.182 | 0.182 | 0.231 | 0.626 | 0.000 |
| 8 | P9 | 1 | 0.745 | 0.485 | 0.429 | 1.000 | 1.000 | 0.984 |
| **Mean** | | | **0.700** | | | **0.560** | **0.537** | **0.285** |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Val AUROC | 0.700 | 0.166 |
| Test AUROC | 0.560 | 0.254 |
| Test AUPRC | 0.537 | 0.249 |
| Test F1 | 0.285 | 0.311 |
| Total runtime | 425s | |
