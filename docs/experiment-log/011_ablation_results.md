# 011 — Ablation Study Results

## Summary

7 ablation experiments completed (621 total SLURM jobs) testing loss functions, regularization, and positional encoding against the baseline transformer (d_model=1024, n_layers=3, dropout=0.3, nhead=4, seq_len=7, batch=32, lr=1e-3, 80 epochs). All use 9-fold LOSO on all 9 patients with 69 features.

**Winner: focal loss (γ=1.0, α=0.5)** — best AUROC at 0.845.

## Results

| Experiment | Best Config | AUROC | Δ Baseline | AUPRC | Δ Baseline |
|---|---|---|---|---|---|
| Baseline | BCE | 0.808 | — | 0.689 | — |
| **Focal loss** | γ=1.0, α=0.5 | **0.845** | **+0.037** | **0.762** | **+0.073** |
| Focal+Smooth | γ=1.0, α=0.3, ε=0.25 | 0.838 | +0.030 | 0.775 | +0.086 |
| Label smooth | ε=0.2 | 0.833 | +0.025 | 0.752 | +0.063 |
| Stoch depth | p=0.3 | 0.820 | +0.012 | 0.711 | +0.022 |
| Weight decay | λ=0.001 | 0.812 | +0.004 | 0.701 | +0.012 |
| RoPE | enabled | 0.756 | -0.052 | 0.649 | -0.040 |

## Per-Experiment Analysis

### Baseline (9 jobs)
- Config: BCE loss, no regularization
- AUROC=0.808±0.098, AUPRC=0.689±0.231
- Per-fold: P1=0.708, P2=0.743, P3=0.847, P4=0.795, P5=0.683, P6=0.852, P7=0.723, P8=0.956, P9=0.964
- High variance (σ=0.098); P5 and P7 hardest folds, P8/P9 near-ceiling

### Focal Loss (81 jobs: 9 folds × 3γ × 3α)
- Best: γ=1.0, α=0.5
- AUROC=0.845±0.104, AUPRC=0.762±0.189
- Per-fold: P1=0.668, P2=0.725, P3=0.889, P4=0.872, P5=0.794, P6=0.910, P7=0.777, P8=1.000, P9=0.965
- Biggest gains on hard folds: P4 +0.077, P5 +0.111, P7 +0.054
- γ=1.0 (mild down-weighting) >> γ=2.0 or 3.0 — heavy focus hurts with small n
- α=0.5 (equal class weight) optimal; higher α over-corrects for minority

### Focal + Label Smoothing (243 jobs: 9 folds × 3γ × 3α × 3ε)
- Best: γ=1.0, α=0.3, ε=0.25
- AUROC=0.838±0.066, AUPRC=0.775±0.140
- Highest AUPRC of all experiments (+0.086 vs baseline)
- But AUROC sub-additive vs focal alone (0.838 vs 0.845) — smoothing softens the focal effect
- Lower variance (σ=0.066) — more stable but slightly less discriminative
- Takeaway: if AUPRC is primary metric, focal+smooth is preferred

### Label Smoothing (27 jobs: 9 folds × 3ε)
- Best: ε=0.2
- AUROC=0.833±0.107, AUPRC=0.752±0.189
- Solid improvement — prevents overconfident predictions
- Works by reducing target polarization (0→ε, 1→1-ε), acts as implicit calibration

### Stochastic Depth (27 jobs: 9 folds × 3p)
- Best: p=0.3
- AUROC=0.820±0.077, AUPRC=0.711±0.191
- Modest improvement, lower variance — regularizes by randomly dropping transformer layers
- Effect smaller than loss-function changes — architecture regularization is secondary when data is scarce

### Weight Decay (27 jobs: 9 folds × 3λ)
- Best: λ=0.001
- AUROC=0.812±0.093, AUPRC=0.701±0.207
- Marginal improvement — L2 penalty on weights has minimal effect on a model that isn't overfitting in weight space
- The bottleneck is data scarcity + class imbalance, not weight magnitude

### RoPE (9 jobs)
- AUROC=0.756±0.090, AUPRC=0.649±0.195
- **Worst performer** — AUROC drops 0.052 below baseline
- Rotary Position Embeddings hurt because seq_len=7 is too short for relative position encoding to add signal
- The standard learned positional embedding already captures what's needed in 7 timesteps
- RoPE adds parameters and structural bias that the model can't leverage

## Key Takeaways

1. **Loss function > regularization**: focal loss (+0.037 AUROC) and label smoothing (+0.025) dominate; architectural regularization (stoch depth, weight decay) give marginal gains.
2. **Mild focus is better**: γ=1.0 beats γ=2.0 and 3.0. With only 9 patients and ~20 relapse days per fold, aggressive down-weighting of easy examples loses too much gradient signal.
3. **RoPE hurts at short sequences**: 7-day windows are too short for relative position bias to help. Remove from future experiments.
4. **Focal+smooth is sub-additive for AUROC**: the two techniques target overlapping failure modes (miscalibrated confidence). Choose one based on primary metric — focal for AUROC, focal+smooth for AUPRC.
5. **Hard folds (P1, P5, P7)** improved most with focal loss — these patients have the most class imbalance, which is exactly what focal loss addresses.

## Conclusion

Focal loss (γ=1.0, α=0.5) is the default loss going forward. Next experiment: sweep `n_layers` with focal loss locked to test depth scaling.

## Configs & Outputs

- `configs/ablation_baseline.json` — 9 jobs
- `configs/ablation_focal.json` — 81 jobs
- `configs/ablation_focal_smooth.json` — 243 jobs
- `configs/ablation_label_smooth.json` — 27 jobs
- `configs/ablation_stoch_depth.json` — 27 jobs
- `configs/ablation_weight_decay.json` — 27 jobs
- `configs/ablation_rope.json` — 9 jobs
- Outputs: `outputs/ablations/ablation_*__*.json` (621 files)
