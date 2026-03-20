# Results Summary

Auto-generated from experiment output JSONs.


## Baseline

| Experiment | Mean Test AUROC | Mean Test AUPRC | Mean Test F1 | Best Sweep | Notes |
|---|---|---|---|---|---|
| [train](train.md) | 0.560 | 0.537 | 0.285 |  |  |

## Exploration (V1)

| Experiment | Mean Test AUROC | Mean Test AUPRC | Mean Test F1 | Best Sweep | Notes |
|---|---|---|---|---|---|
| [exp_ablation_dropout_only](exp_ablation_dropout_only.md) | 0.614 | 0.555 | 0.385 |  | Ablation: large model with weighted BCE but dropout=0.3 to i |
| [exp_ablation_loss_only](exp_ablation_loss_only.md) | 0.500 | 0.488 | 0.221 |  | Ablation: focal loss with LARGE model (d=64) to isolate loss |
| [exp_ablation_small_model](exp_ablation_small_model.md) | 0.683 | 0.555 | 0.383 |  | Ablation: small model (d=32, dropout=0.3) with weighted BCE  |
| [exp_asymmetric_focal](exp_asymmetric_focal.md) | 0.692 | 0.556 | 0.577 |  | Asymmetric focal loss: gamma_pos=1.0 (strong positive gradie |
| [exp_augmented](exp_augmented.md) | 0.520 | 0.508 | 0.284 |  |  |
| [exp_bottleneck](exp_bottleneck.md) | 0.506 | 0.485 | 0.269 |  |  |
| [exp_dann](exp_dann.md) | 0.626 | 0.552 | 0.350 |  |  |
| [exp_dann_focal](exp_dann_focal.md) | 0.702 | 0.584 | 0.274 |  | DANN + focal loss + small model. Warmup reduced to 10 so adv |
| [exp_focal](exp_focal.md) | 0.670 | 0.553 | 0.333 |  |  |
| [exp_focal_alpha_sweep](exp_focal_alpha_sweep.md) | 0.675 | 0.553 | 0.311 | focal_alpha=0.75 | Sweep focal alpha across 5 values (9 folds x 5 = 45 jobs) |
| [exp_focal_augmented](exp_focal_augmented.md) | 0.688 | 0.557 | 0.319 |  | Focal loss + small model + data augmentation (jitter+scale). |
| [exp_focal_earlystop_auprc](exp_focal_earlystop_auprc.md) | 0.681 | 0.581 | 0.307 |  | Early stop on val AUPRC instead of AUROC, increased patience |
| [exp_focal_gamma_sweep](exp_focal_gamma_sweep.md) | 0.691 | 0.558 | 0.293 | focal_gamma=3.0 | Sweep focal gamma across 5 values (9 folds x 5 = 45 jobs) |
| [exp_focal_label_smoothing](exp_focal_label_smoothing.md) | 0.686 | 0.588 | 0.244 |  | Label smoothing (eps=0.1) with focal loss to reduce overconf |
| [exp_focal_lr_warmup](exp_focal_lr_warmup.md) | 0.596 | 0.480 | 0.173 |  | Lower LR (5e-5) with 5-epoch linear warmup then constant to  |
| [exp_gated](exp_gated.md) | 0.616 | 0.567 | 0.405 |  |  |
| [exp_gated_focal](exp_gated_focal.md) | 0.541 | 0.487 | 0.225 |  | Gated fusion + focal loss + small model. Lightweight per-mod |
| [exp_heavy_reg](exp_heavy_reg.md) | 0.648 | 0.557 | 0.450 |  | Aggressive regularization: dropout=0.5, weight_decay=0.1, lr |

## Ablation V1

| Experiment | Mean Test AUROC | Mean Test AUPRC | Mean Test F1 | Best Sweep | Notes |
|---|---|---|---|---|---|
| [ablation_baseline](ablation_baseline.md) | 0.824 | 0.751 | 0.132 |  | Ablation Stage 0: Baseline control — confirm ablation.py rep |
| [ablation_depth](ablation_depth.md) | 0.824 | 0.732 | 0.336 | n_layers=3 | Ablation: layer depth sweep with focal loss (best from ablat |
| [ablation_focal](ablation_focal.md) | 0.822 | 0.739 | 0.285 | focal_gamma=1.0, focal_alpha=0.5 | Ablation Stage 1: Focal loss. 9 folds x 3 gamma x 3 alpha =  |
| [ablation_focal_smooth](ablation_focal_smooth.md) | 0.857 | 0.690 | 0.148 | focal_gamma=1.0, focal_alpha=0.3, label_smoothing=0.25 | Ablation: Focal loss + label smoothing combined. 9 folds x 2 |
| [ablation_label_smooth](ablation_label_smooth.md) | 0.839 | 0.736 | 0.235 | label_smoothing=0.2 | Ablation Stage 2: Label smoothing. 9 folds x 4 eps = 36 jobs |
| [ablation_recipe](ablation_recipe.md) | 0.842 | 0.749 | 0.261 | optimizer=adam, lr_schedule=none, grad_clip=1.0, warmup_epochs=3 | Training recipe sweep: optimizer, LR schedule, grad clipping |
| [ablation_rope](ablation_rope.md) | 0.758 | 0.645 | 0.308 |  | Ablation Stage 4: RoPE (Rotary Position Embedding). 9 folds  |
| [ablation_stoch_depth](ablation_stoch_depth.md) | 0.824 | 0.717 | 0.318 | stochastic_depth=0.2 | Ablation Stage 3: Stochastic depth. 9 folds x 3 drop_rate =  |
| [ablation_weight_decay](ablation_weight_decay.md) | 0.811 | 0.704 | 0.453 | weight_decay=0.001 | Ablation Stage 5: Weight decay. 9 folds x 3 wd = 27 jobs. |

## Ablation V2

| Experiment | Mean Test AUROC | Mean Test AUPRC | Mean Test F1 | Best Sweep | Notes |
|---|---|---|---|---|---|
| [ablv2_all_baseline](ablv2_all_baseline.md) | 0.813 | 0.728 | 0.318 |  | Ablation v2 — All baseline. 9 jobs. |
| [ablv2_all_focal](ablv2_all_focal.md) | 0.802 | 0.713 | 0.462 | focal_gamma=2.0, focal_alpha=0.5 | Ablation v2 — All focal loss. 9 folds x 3 gamma x 3 alpha =  |
| [ablv2_all_focal_smooth](ablv2_all_focal_smooth.md) | 0.841 | 0.725 | 0.332 | focal_gamma=1.0, focal_alpha=0.5, label_smoothing=0.2 | Ablation v2 — All focal + label smoothing. 9 folds x 2 gamma |
| [ablv2_all_label_smooth](ablv2_all_label_smooth.md) | 0.810 | 0.686 | 0.251 | label_smoothing=0.05 | Ablation v2 — All label smoothing. 9 folds x 4 eps = 36 jobs |
| [ablv2_all_rope](ablv2_all_rope.md) | 0.771 | 0.673 | 0.321 |  | Ablation v2 — All RoPE. 9 folds x 1 = 9 jobs. |
| [ablv2_all_stoch_depth](ablv2_all_stoch_depth.md) | 0.824 | 0.735 | 0.363 | stochastic_depth=0.2 | Ablation v2 — All stochastic depth. 9 folds x 3 drop_rate =  |
| [ablv2_all_weight_decay](ablv2_all_weight_decay.md) | 0.826 | 0.722 | 0.513 | weight_decay=0.001 | Ablation v2 — All weight decay. 9 folds x 3 wd = 27 jobs. |
| [ablv2_union_baseline](ablv2_union_baseline.md) | 0.794 | 0.682 | 0.236 |  | Ablation v2 — Union baseline. 9 jobs. |
| [ablv2_union_focal](ablv2_union_focal.md) | 0.824 | 0.730 | 0.522 | focal_gamma=2.0, focal_alpha=0.75 | Ablation v2 — Union focal loss. 9 folds x 3 gamma x 3 alpha  |
| [ablv2_union_focal_smooth](ablv2_union_focal_smooth.md) | 0.827 | 0.686 | 0.000 | focal_gamma=1.0, focal_alpha=0.3, label_smoothing=0.25 | Ablation v2 — Union focal + label smoothing. 9 folds x 2 gam |
| [ablv2_union_label_smooth](ablv2_union_label_smooth.md) | 0.802 | 0.714 | 0.282 | label_smoothing=0.05 | Ablation v2 — Union label smoothing. 9 folds x 4 eps = 36 jo |
| [ablv2_union_rope](ablv2_union_rope.md) | 0.767 | 0.682 | 0.231 |  | Ablation v2 — Union RoPE. 9 folds x 1 = 9 jobs. |
| [ablv2_union_stoch_depth](ablv2_union_stoch_depth.md) | 0.797 | 0.674 | 0.356 | stochastic_depth=0.1 | Ablation v2 — Union stochastic depth. 9 folds x 3 drop_rate  |
| [ablv2_union_weight_decay](ablv2_union_weight_decay.md) | 0.814 | 0.732 | 0.507 | weight_decay=0.0001 | Ablation v2 — Union weight decay. 9 folds x 3 wd = 27 jobs. |
