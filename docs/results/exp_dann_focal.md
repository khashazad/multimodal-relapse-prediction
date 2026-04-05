# Experiment: `exp_dann_focal`

DANN + focal loss + small model. Warmup reduced to 10 so adversarial signal contributes before early stopping

## Configuration

| Parameter | Value |
|---|---|
| model_name | transformer_v4 |
| d_model | 32 |
| nhead | 2 |
| num_encoder_layers | 1 |
| num_fusion_layers | 1 |
| dropout | 0.3 |
| dann_lambda | 0.1 |
| dann_warmup_epochs | 10 |
| lr | 0.0001 |
| weight_decay | 0.01 |
| batch_size | 32 |
| epochs | 100 |
| patience | 15 |
| loss_fn | focal |
| focal_alpha | 0.75 |
| focal_gamma | 2.0 |
| seed | 42 |
| threshold | 0.5 |
| calibrate_threshold | True |
| data_dir | data/processed/track1 |


## Per-Fold Results

| Fold | Patient | Best Epoch | Val AUROC | Val AUPRC | Val F1 | Test AUROC | Test AUPRC | Test F1 |
|------|---------|-----------|-----------|-----------|--------|------------|------------|---------|
| 0 | P1 | 2 | 0.686 | 0.670 | 0.000 | 0.453 | 0.291 | 0.000 |
| 1 | P2 | 3 | 0.961 | 0.911 | 0.740 | 0.587 | 0.402 | 0.000 |
| 2 | P3 | 11 | 0.811 | 0.740 | 0.093 | 0.812 | 0.833 | 0.364 |
| 3 | P4 | 0 | — | — | 0.000 | 0.738 | 0.212 | 0.240 |
| 4 | P5 | 16 | 0.859 | 0.831 | 0.267 | 0.570 | 0.535 | 0.440 |
| 5 | P6 | 3 | 0.568 | 0.515 | 0.578 | 0.612 | 0.544 | 0.533 |
| 6 | P7 | 1 | 0.317 | 0.162 | 0.000 | 0.645 | 0.477 | 0.000 |
| 7 | P8 | 2 | 0.448 | 0.111 | 0.000 | 0.898 | 0.965 | 0.071 |
| 8 | P9 | 4 | 0.740 | 0.442 | 0.400 | 0.999 | 0.999 | 0.815 |
| **Mean** | | | **0.674** | | | **0.702** | **0.584** | **0.274** |


## Calibrated Threshold Results

| Fold | Threshold | Cal Val F1 | Cal Val Prec | Cal Val Rec | Cal Test F1 | Cal Test Prec | Cal Test Rec |
|------|-----------|------------|--------------|-------------|-------------|---------------|--------------|
| 0 | 0.40 | 0.667 | 0.520 | 0.929 | 0.472 | 0.347 | 0.739 |
| 1 | 0.45 | 0.905 | 0.843 | 0.977 | 0.255 | 0.600 | 0.162 |
| 2 | 0.05 | 0.661 | 0.494 | 1.000 | 0.745 | 0.593 | 1.000 |
| 3 | 0.05 | 0.000 | 0.000 | 0.000 | 0.244 | 0.139 | 1.000 |
| 4 | 0.20 | 0.727 | 0.600 | 0.923 | 0.689 | 0.586 | 0.836 |
| 5 | 0.45 | 0.648 | 0.507 | 0.897 | 0.608 | 0.500 | 0.776 |
| 6 | 0.05 | 0.345 | 0.209 | 1.000 | 0.400 | 0.250 | 1.000 |
| 7 | 0.45 | 0.222 | 0.125 | 1.000 | 0.876 | 0.791 | 0.981 |
| 8 | 0.35 | 0.482 | 0.317 | 1.000 | 0.970 | 0.941 | 1.000 |
| **Mean** | | | | | **0.584** | | |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Val AUROC | 0.674 | 0.203 |
| Test AUROC | 0.702 | 0.165 |
| Test AUPRC | 0.584 | 0.269 |
| Test F1 | 0.274 | 0.271 |
| Cal Test F1 | 0.584 | 0.245 |
| Total runtime | 446s | |
