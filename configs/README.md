# Config Index

43 experiment configs grouped by phase. Scripts reference `configs/${NAME}.json`.

## Preprocessing

| Config | Purpose |
|--------|---------|
| `preprocessing.json` | Feature extraction + LOSO fold generation |

## Infrastructure

| Config | Purpose |
|--------|---------|
| `train.json` | Single training run config |

## Exploration (V1)

Early architecture and loss function exploration with d_model=32 transformer.

| Config | Key variation |
|--------|--------------|
| `exp_focal.json` | Focal loss (α=0.75, γ=2.0) |
| `exp_focal_alpha_sweep.json` | α sweep with focal |
| `exp_focal_gamma_sweep.json` | γ sweep with focal |
| `exp_focal_augmented.json` | Focal + data augmentation |
| `exp_focal_label_smoothing.json` | Focal + label smoothing |
| `exp_focal_lr_warmup.json` | Focal + LR warmup |
| `exp_focal_earlystop_auprc.json` | Early stop on AUPRC |
| `exp_asymmetric_focal.json` | Asymmetric focal loss |
| `exp_augmented.json` | Data augmentation only |
| `exp_bottleneck.json` | Bottleneck transformer (v2) |
| `exp_gated.json` | Gated fusion transformer (v3) |
| `exp_gated_focal.json` | Gated + focal loss |
| `exp_dann.json` | Domain-adversarial (v4) |
| `exp_dann_focal.json` | DANN + focal loss |
| `exp_heavy_reg.json` | Heavy regularization |
| `exp_ablation_dropout_only.json` | Dropout ablation |
| `exp_ablation_loss_only.json` | Loss ablation |
| `exp_ablation_small_model.json` | Small model ablation |

## Ablation V1

d_model=1024, 3L, 4H, 9-fold LOSO. 621 SLURM jobs. **Winner: focal (0.845 AUROC).**

| Config | Variation | Best AUROC |
|--------|-----------|------------|
| `ablation_baseline.json` | BCE baseline | 0.808 |
| `ablation_focal.json` | Focal loss sweep | **0.845** |
| `ablation_focal_smooth.json` | Focal + label smoothing | 0.838 |
| `ablation_label_smooth.json` | Label smoothing only | 0.833 |
| `ablation_stoch_depth.json` | Stochastic depth | 0.820 |
| `ablation_weight_decay.json` | Weight decay sweep | 0.812 |
| `ablation_rope.json` | RoPE positional encoding | 0.756 |
| `ablation_depth.json` | Depth scaling | — |
| `ablation_recipe.json` | Training recipe sweep | 0.842 |

## Ablation V2

Re-ran all techniques with `union` (24 feat) and `all` (69 feat) feature sets. 810 SLURM jobs.

| Config | Feature set | Variation |
|--------|-------------|-----------|
| `ablv2_all_baseline.json` | all (69) | BCE baseline |
| `ablv2_all_focal.json` | all | Focal loss |
| `ablv2_all_focal_smooth.json` | all | Focal + smoothing |
| `ablv2_all_label_smooth.json` | all | Label smoothing |
| `ablv2_all_stoch_depth.json` | all | Stochastic depth |
| `ablv2_all_weight_decay.json` | all | Weight decay |
| `ablv2_all_rope.json` | all | RoPE |
| `ablv2_union_baseline.json` | union (24) | BCE baseline |
| `ablv2_union_focal.json` | union | Focal loss |
| `ablv2_union_focal_smooth.json` | union | Focal + smoothing |
| `ablv2_union_label_smooth.json` | union | Label smoothing |
| `ablv2_union_stoch_depth.json` | union | Stochastic depth |
| `ablv2_union_weight_decay.json` | union | Weight decay |
| `ablv2_union_rope.json` | union | RoPE |
