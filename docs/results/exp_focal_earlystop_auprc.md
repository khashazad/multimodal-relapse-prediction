# Experiment: `exp_focal_earlystop_auprc`

Early stop on val AUPRC instead of AUROC, increased patience to 25

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
| patience | 25 |
| loss_fn | focal |
| focal_alpha | 0.75 |
| focal_gamma | 2.0 |
| early_stop_metric | auprc |
| seed | 42 |
| threshold | 0.5 |
| calibrate_threshold | True |
| data_dir | data/processed/track1 |


## Per-Fold Results

| Fold | Patient | Best Epoch | Val AUROC | Val AUPRC | Val F1 | Test AUROC | Test AUPRC | Test F1 |
|------|---------|-----------|-----------|-----------|--------|------------|------------|---------|
| 0 | P1 | 2 | 0.628 | 0.631 | 0.000 | 0.436 | 0.286 | 0.000 |
| 1 | P2 | 0 | 0.500 | 0.407 | 0.579 | 0.500 | 0.339 | 0.507 |
| 2 | P3 | 5 | 0.732 | 0.679 | 0.095 | 0.861 | 0.890 | 0.349 |
| 3 | P4 | 0 | — | — | 0.000 | 0.646 | 0.160 | 0.279 |
| 4 | P5 | 6 | 0.876 | 0.807 | 0.000 | 0.533 | 0.533 | 0.259 |
| 5 | P6 | 4 | 0.566 | 0.547 | 0.584 | 0.563 | 0.502 | 0.577 |
| 6 | P7 | 1 | 0.303 | 0.186 | 0.000 | 0.705 | 0.568 | 0.000 |
| 7 | P8 | 1 | 0.378 | 0.106 | 0.000 | 0.906 | 0.972 | 0.000 |
| 8 | P9 | 3 | 0.577 | 0.399 | 0.364 | 0.982 | 0.982 | 0.792 |
| **Mean** | | | **0.570** | | | **0.681** | **0.581** | **0.307** |


## Calibrated Threshold Results

| Fold | Threshold | Cal Val F1 | Cal Val Prec | Cal Val Rec | Cal Test F1 | Cal Test Prec | Cal Test Rec |
|------|-----------|------------|--------------|-------------|-------------|---------------|--------------|
| 0 | 0.40 | 0.659 | 0.491 | 1.000 | 0.488 | 0.339 | 0.870 |
| 1 | 0.05 | 0.579 | 0.407 | 1.000 | 0.507 | 0.339 | 1.000 |
| 2 | 0.15 | 0.640 | 0.471 | 1.000 | 0.750 | 0.600 | 1.000 |
| 3 | 0.05 | 0.000 | 0.000 | 0.000 | 0.256 | 0.147 | 1.000 |
| 4 | 0.25 | 0.817 | 0.704 | 0.974 | 0.689 | 0.586 | 0.836 |
| 5 | 0.40 | 0.650 | 0.487 | 0.974 | 0.585 | 0.469 | 0.776 |
| 6 | 0.05 | 0.345 | 0.209 | 1.000 | 0.400 | 0.250 | 1.000 |
| 7 | 0.45 | 0.209 | 0.117 | 1.000 | 0.898 | 0.828 | 0.981 |
| 8 | 0.05 | 0.396 | 0.247 | 1.000 | 0.646 | 0.478 | 1.000 |
| **Mean** | | | | | **0.580** | | |


## Aggregate Summary

| Metric | Mean | Std |
|--------|------|-----|
| Val AUROC | 0.570 | 0.172 |
| Test AUROC | 0.681 | 0.184 |
| Test AUPRC | 0.581 | 0.287 |
| Test F1 | 0.307 | 0.266 |
| Cal Test F1 | 0.580 | 0.182 |
| Total runtime | 432s | |
