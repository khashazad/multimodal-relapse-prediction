# Multimodal Relapse Prediction

Non-psychotic relapse detection from wearable sensor data. Binary per-day classification for 9 psychiatric patients using 5 sensor modalities under Leave-One-Subject-Out (LOSO) cross-validation.

**Best result: 0.846 AUROC** (focal loss + label smoothing + gradient clipping, d=1024, 3L/4H transformer, 9/9 folds).

## Architecture

Two-stage multimodal transformer: independent per-modality encoders compress each sensor's 7-day window into a summary vector, then a fusion transformer attends across modalities for classification.

```
                    ┌─────────────────────────────────────────────────────┐
                    │          Input: 7-day window per modality           │
                    │  accel(38) gyr(38) hr(26) step(10) sleep(9) = 121  │
                    └────┬────────┬───────┬───────┬────────┬─────────────┘
                         │        │       │       │        │
                    ┌────▼──┐ ┌───▼──┐ ┌──▼──┐ ┌──▼──┐ ┌──▼──┐
                    │Modality│ │  Mod │ │ Mod │ │ Mod │ │ Mod │  Stage 1
                    │Encoder │ │ Enc  │ │ Enc │ │ Enc │ │ Enc │  (independent
                    │1L, 4H  │ │      │ │     │ │     │ │     │   weights)
                    └───┬────┘ └──┬───┘ └──┬──┘ └──┬──┘ └──┬──┘
                        │        │       │       │        │
                     [CLS]    [CLS]   [CLS]   [CLS]   [CLS]
                        │        │       │       │        │
                    ┌───▼────────▼───────▼───────▼────────▼───┐
                    │   + Modality Embeddings                  │
                    │   Fusion Transformer (3L, 4H, d=1024)    │  Stage 2
                    │   Mean Pool → (B, 1024)                  │
                    └──────────────────┬───────────────────────┘
                                       │
                    ┌──────────────────▼───────────────────────┐
                    │  Linear → GELU → Dropout → Linear → σ    │  Classifier
                    └──────────────────────────────────────────┘
```

Each **ModalityEncoder** projects raw features to d_model, prepends a learnable [CLS] token with positional embeddings, and runs a 1-layer TransformerEncoder. The [CLS] output is the modality summary. The **FusionTransformer** stacks all 5 summaries, adds learned modality embeddings, and attends across them. ~19M parameters at d=1024.

### Why this works

The two-stage design lets each modality develop its own temporal representation before cross-modal attention. This matters because the 5 sensors have very different dimensionalities (9–38 features) and sampling rates (per-minute steps vs. 20 Hz accelerometer). Fusing raw features directly would force the model to learn alignment and abstraction simultaneously.

## Experiment Trajectory

The project followed a clear escalation path — traditional ML hit a ceiling, transformers broke through, and the remaining gains came from training dynamics rather than architecture changes.

### 1. Traditional ML (notebook)

| Model | AUROC | Notes |
|-------|-------|-------|
| XGBoost | 0.530 | sleep + steps features |
| Logistic Regression | 0.580 | sleep + steps features |
| MLP | 0.550 | 24 engineered features |

All three plateaued well below 0.6. The per-patient heterogeneity and class imbalance (~20 relapse vs ~60 stable days) defeated standard approaches.

### 2. Transformer breakthrough (notebook)

| Config | AUROC | Folds |
|--------|-------|-------|
| d=32 | ~0.51 | 9/9 |
| d=1024 (bipolar-6) | 0.849 | 6/6 |
| d=1024 (all-9) | 0.793 | 9/9 |

Scaling from d=32 to d=1024 was critical — the small model couldn't capture cross-modal interactions. Bipolar patients (6/9) were significantly easier to predict than the full cohort.

### 3. Architecture variants (framework, exploration)

Tested bottleneck fusion (v2), gated fusion (v3), and DANN domain-adversarial (v4). None beat the standard transformer v1. The bottleneck and gated variants scored 0.51–0.62; DANN reached 0.70 but added complexity without surpassing simpler focal loss tuning.

### 4. Ablation V1 (621 SLURM jobs)

Systematic loss and regularization sweep with d=1024, 3L, 4H:

| Technique | AUROC | Folds | vs Baseline |
|-----------|-------|-------|-------------|
| Focal + smooth | 0.857 | 5/9 | — (invalid) |
| Label smoothing (eps=0.05) | 0.839 | 9/9 | +0.015 |
| Stochastic depth (p=0.2) | 0.824 | 8/9 | +0.000 |
| **Baseline (BCE)** | **0.824** | **8/9** | **—** |
| Focal loss (gamma=1, alpha=0.5) | 0.822 | 9/9 | -0.002 |
| Weight decay (1e-3) | 0.811 | 8/9 | -0.013 |
| RoPE | 0.758 | 8/9 | -0.066 |

Focal+smooth showed the highest raw number but only 5/9 folds converged — training instability without gradient clipping made it scientifically invalid.

### 5. Training recipe sweep (216 SLURM jobs)

Swept optimizer (Adam/AdamW), LR schedule (none/cosine), gradient clipping, warmup:

**Winner: Adam + fixed LR + grad_clip=1.0 → 0.842 AUROC (9/9 folds)**

Cosine scheduling actively hurt performance — counter-intuitive, but fixed LR with early stopping via best-epoch checkpointing worked better for this small dataset.

### 6. Ablation V2 + stabilization (810+ SLURM jobs)

Re-ran all ablations with both feature sets (24 "union" vs 69 "all") and added gradient clipping + AdamW to stabilize focal+smooth:

| Config | Features | AUROC | Folds |
|--------|----------|-------|-------|
| **Focal+smooth+grad_clip (stabilized)** | **69 (all)** | **0.846** | **9/9** |
| Focal+smooth+grad_clip (stabilized) | 24 (union) | 0.840 | 9/9 |
| Focal+smooth (unstabilized) | 69 (all) | 0.841 | 6/9 |
| Weight decay | 69 (all) | 0.826 | 9/9 |
| Stochastic depth | 69 (all) | 0.824 | 9/9 |
| Focal loss | 24 (union) | 0.824 | 9/9 |
| Baseline | 69 (all) | 0.813 | 9/9 |

Best config: focal loss (gamma=1.0, alpha=0.7) + label smoothing (eps=0.2) + grad_clip=1.0, AdamW.

## Full Experiment Summary

All experiments ranked by AUROC (9/9-fold results only marked with checkmark):

| Experiment | AUROC | AUPRC | F1 | 9/9 |
|------------|-------|-------|-----|-----|
| ablv2_all_focal_smooth_v2 (stabilized) | 0.846 | 0.725 | 0.522 | yes |
| ablation_recipe (focal+grad_clip) | 0.842 | 0.749 | 0.261 | yes |
| ablv2_union_focal_smooth_v2 (stabilized) | 0.840 | 0.742 | 0.318 | yes |
| ablation_label_smooth | 0.839 | 0.736 | 0.235 | yes |
| ablv2_union_focal_smooth | 0.827 | 0.686 | 0.000 | no (8/9) |
| ablv2_all_weight_decay | 0.826 | 0.722 | 0.513 | yes |
| ablv2_union_focal | 0.824 | 0.730 | 0.522 | yes |
| ablation_baseline (V1) | 0.824 | 0.751 | 0.132 | no (8/9) |
| ablation_depth | 0.824 | 0.732 | 0.336 | yes |
| ablv2_all_stoch_depth | 0.824 | 0.735 | 0.363 | yes |
| ablation_focal | 0.822 | 0.739 | 0.285 | yes |
| ablv2_all_focal_smooth | 0.841 | 0.725 | 0.332 | no (6/9) |
| ablv2_union_weight_decay | 0.814 | 0.732 | 0.507 | yes |
| ablv2_all_baseline | 0.813 | 0.728 | 0.318 | yes |
| ablation_weight_decay | 0.811 | 0.704 | 0.453 | no (8/9) |
| ablv2_all_label_smooth | 0.810 | 0.686 | 0.251 | yes |
| ablv2_all_focal | 0.802 | 0.713 | 0.462 | yes |
| ablv2_union_label_smooth | 0.802 | 0.714 | 0.282 | yes |
| ablv2_union_stoch_depth | 0.797 | 0.674 | 0.356 | yes |
| ablv2_union_baseline | 0.794 | 0.682 | 0.236 | yes |
| ablv2_all_rope | 0.771 | 0.673 | 0.321 | yes |
| ablv2_union_rope | 0.767 | 0.682 | 0.231 | yes |
| ablation_rope | 0.758 | 0.645 | 0.308 | yes |

Exploration-phase experiments (d=32–64, various architectures) ranged from 0.50–0.70 and are documented in `docs/results/summary.md`.

## Key Findings

1. **Loss engineering > architecture engineering.** Bottleneck, gated, and DANN variants all underperformed the standard transformer. The gains came from focal loss tuning and training stabilization.

2. **Mild focal focus is optimal.** gamma=1.0 >> gamma=2.0 or 3.0. With only ~20 relapse days per fold, aggressive down-weighting starves gradient signal.

3. **Gradient clipping unlocks focal+smooth.** Without grad_clip, focal+smooth diverged in 4-5/9 folds. Adding grad_clip=1.0 stabilized all 9 folds and achieved the best result (0.846).

4. **RoPE hurts short sequences.** 7-day windows are too short for relative position encoding to help (-0.066 vs baseline).

5. **24 features ~ 69 features.** The "union" feature set (sleep + steps + HRV + circadian + demographics) captures most signal. Additional features add noise.

6. **Per-patient variance dominates.** Oracle AUROC across all configs is 0.907 — the 0.061 gap from 0.846 requires per-patient adaptation, not better global models.

### Per-patient breakdown (best config)

| Patient | AUROC | Diagnosis |
|---------|-------|-----------|
| P9 | 0.996 | Bipolar |
| P8 | 0.939 | Bipolar |
| P3 | 0.905 | Bipolar |
| P7 | 0.896 | Schizophrenia |
| P1 | 0.808 | Brief psychotic |
| P6 | 0.803 | Bipolar |
| P4 | 0.775 | Bipolar |
| P5 | 0.750 | Bipolar |
| P2 | 0.745 | Schizophreniform |

## Feature Engineering

Five wearable modalities → 121 per-day features (69 "all" set, 24 "union" subset):

- **Sleep** (9 feat): main episode duration, nap count/duration, z-score deviations from personal baseline
- **Steps** (10 feat): daily totals, inverted z-score (fewer steps = higher risk)
- **HRV** (26 feat): nighttime RMSSD/SDNN from accelerometer-filtered heart rate
- **Circadian** (38 feat): hourly activity profiles from gyroscope + accelerometer fusion (relative amplitude, intradaily variability, cosinor, L5/M10)
- **Demographics** (3 feat): age, sex, diagnosis

Patient-specific baselines from non-relapse days enable z-score deviation features — a critical design choice given the per-patient heterogeneity.

## Setup

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Reproduction

```bash
# Preprocess data (expects data in data/original/track1/)
python scripts/preprocess_data.py

# Best result: stabilized focal + smooth + grad_clip
bash scripts/run.sh -n ablv2_all_focal_smooth_v2

# Or submit to SLURM
bash scripts/submit_slurm.sh -n ablv2_all_focal_smooth_v2

# Previous best (focal + grad_clip only)
bash scripts/run.sh -n ablation_recipe
```

Full experiment configs in `configs/`. Per-experiment results in `docs/results/`.

## Repository Structure

```
configs/             ~43 experiment JSON configs
docs/results/        Auto-generated per-experiment reports + summary
notebooks/main.ipynb Feature engineering + notebook-phase modeling
scripts/             Run, submit, preprocess, verify scripts
src/
  models/            Transformer v1-v4
  losses/            Focal loss
  data_loader.py     Raw sensor data loading
  feature_extractor.py  121 per-day features
  preprocess_loso.py LOSO fold generation
  train.py           Training loop
  ablation.py        Ablation study runner
```

## Data

Expects patient data in `data/original/track1/`:

```
track1/
  demographics.csv
  P{1-9}/{train,val,test}_{0,1,2}/
    gyr.parquet       # Gyroscope (20 Hz)
    hrm.parquet       # Heart rate (5 Hz)
    linacc.parquet    # Linear accelerometer (20 Hz)
    step.parquet      # Step count (per minute)
    sleep.parquet     # Sleep episodes
    relapses.csv      # Relapse labels
```
