# Multimodal Relapse Prediction

<p align="center">
  <a href="report/MLHC2026-ResearchTrack-Template/main.tex">📄 Read the Paper</a>
</p>

Benchmarking multi-modal fusion strategies for relapse prediction in bipolar and related disorders using smartwatch sensor data. We tackle the non-psychotic relapse task from the 2nd e-Prevention challenge: binary per-day classification of relapse vs. stable days for 9 patients under Leave-One-Patient-Out (LOPO) cross-validation.

**Headline result: 0.846 ± 0.085 AUROC** (9/9 folds) with a supervised sequence transformer trained with focal loss + label smoothing and AdamW with gradient clipping — a +0.135 AUROC jump over the 0.711 e-Prevention challenge winner.

## Overview

Relapse episodes in bipolar and related disorders are irregular, labels are scarce, and patients differ widely — models trained on one cohort often fail to generalize. We worked the Track 1 dataset from the 2nd e-Prevention challenge (9 patients, 5 sensor modalities, ~3,337 patient-days) across three model families:

- **Traditional ML**: XGBoost and L1/L2 logistic regression as non-sequential baselines
- **Unsupervised anomaly detection**: replication of the challenge-winning transformer autoencoder with iNNE scoring
- **Supervised sequence transformer**: 7-day sliding window classifier trained with SMOTE oversampling, focal loss, and label smoothing

A systematic ablation over loss functions, regularization, and training configurations identified focal loss (γ=1.0, α=0.7) + label smoothing (ε=0.2) as the best configuration, but only after stabilizing with AdamW and gradient clipping.

## Task & Data

We used the publicly available Track 1 dataset from the [2nd e-Prevention challenge](https://robotics.ntua.gr/eprevention-sp-challenge/):

- **9 patients (P1–P9)**: 6 bipolar (P3–P6, P8, P9), 1 brief psychotic (P1), 1 schizophreniform (P2), 1 schizophrenia (P7)
- **5 sensor modalities** recorded on a Samsung Gear S3 smartwatch:
  - Gyroscope (20 Hz), linear accelerometer (20 Hz)
  - Heart rate monitor (5 Hz)
  - Per-minute step counts
  - Sleep episode annotations
- **Splits** (per patient): pre-divided into temporally ordered sequences `train_{0,1}`, `val_{0,1}`, `test_{0,1,2}`. Training splits contain only stable days; validation and test splits mix stable and relapse days (~10–20% relapse prevalence)
- **Demographics**: age, sex, diagnosis, illness duration, treatment compliance

### Expected data layout

```
data/original/track1/
  demographics.csv
  P{1-9}/
    {train_0,train_1,val_0,val_1,test_0,test_1,test_2}/
      gyr.parquet         # Gyroscope (20 Hz)
      hrm.parquet         # Heart rate (5 Hz)
      linacc.parquet      # Linear accelerometer (20 Hz)
      step.parquet        # Step count (per minute)
      sleep.parquet       # Sleep episodes
      relapses.csv        # Daily relapse labels
```

## Feature Engineering

Each sensor stream is reduced to per-day features, then z-scored against each patient's own baseline (computed on their non-relapse training days). PCA and t-SNE both confirmed that without per-patient normalization the feature space clusters by patient rather than by clinical state.

| Group | Description |
|---|---|
| **Sleep (16)** | Main sleep duration, onset/wake times (circular stats), nap count + total nap hours, z-score deviations from personal baseline |
| **Steps** | Daily totals with inverted z-score (more sedentary → higher risk signal) |
| **Nighttime HRV** | RMSSD, SDNN from 5-s windows of heart rate, filtered by accelerometer stationarity (<0.2g), binned into 55 temporal bins over an 8-hour extraction window |
| **Circadian actigraphy (15)** | Fused gyroscope + accelerometer hourly activity profiles: relative amplitude, intradaily variability, cosinor amplitude/acrophase, L5/M10 onset deviations, evening activity proportion |
| **Demographics** | Age, sex, diagnosis, illness duration, treatment compliance |

A 12-candidate sweep over HRV extraction windows selected **W14 (14:00–22:00)** as optimal, replacing the classical nighttime window in all final models — evening physiological signal turned out to be a more reliable prodromal indicator than overnight recordings.

Two feature sets were evaluated: a **24-feature "union"** subset (Boruta ∪ mRMR top-15, tuned on bipolar patients) and a **69-feature "all"** set used when extending to the full 9-patient cohort to preserve diagnostic-specific signals.

## Models

### Traditional ML baselines

XGBoost and logistic regression (L1/L2) trained under 9-fold LOPO on all 9 patients. Class imbalance handled via `scale_pos_weight` (XGBoost) and inverse-frequency class weights (LR). A per-fold sweep selected the number of input features (`K ∈ {3, 5, 8, 10, 15, 20, 25, 30, all}`).

### Unsupervised anomaly detection

Replication of the 2nd e-Prevention challenge winner: a transformer autoencoder trained on 55-bin sequences of non-relapse data, with each bin containing 24 features (6 base physiological signals + causal moving means + daily means + daily standard deviations). Latent representations are scored with isolation Nearest Neighbor Ensembles (iNNE). Evaluated under the same LOPO protocol.

### Supervised sequence transformer (primary model)

A transformer encoder with a linear classification head over a 7-day sliding window. Shorter sequences are left-padded with attention masking so every test day gets a prediction. SMOTE oversampling is applied to the flattened training windows (then reshaped back to 3D); test data is never oversampled.

**Best configuration** (from global hyperparameter grid search on bipolar-6, then extended to all-9):

| Hyperparameter | Value |
|---|---|
| `d_model` | 1024 |
| Layers | 3 |
| Heads | 4 |
| Dropout | 0.3 |
| Optimizer | AdamW, lr = 1e-3, grad clip = 1.0 |
| Batch size | 32 |
| Epochs | 80, best-epoch checkpointing |
| Loss | Focal (γ=1.0, α=0.7) + label smoothing (ε=0.2) |
| Feature set | 69 ("all") |

Per-fold hyperparameter tuning was attempted early but degraded results — individual validation sets contain only ~9 relapse events, which is not enough signal for reliable model selection. Global hyperparameter selection across all LOPO folds consistently outperformed per-fold tuning.

A parallel line of experiments on **multimodal fusion architectures** (bottleneck, gated, DANN) using a separate raw-statistical preprocessing pipeline was also explored but all underperformed the simple sequence transformer (see `src/models/transformer_v{2,3,4}.py`).

## Results

All models evaluated under 9-fold LOPO cross-validation. Each fold holds out one patient, using their validation sequences for early stopping and test sequences for final evaluation; the other 8 patients contribute all their splits to training.

### Model comparison

| Method | Features | AUROC | AUPRC |
|---|---|---|---|
| Challenge winner | raw | 0.711 | 0.620 |
| XGBoost (top-K=3) | 24 | 0.584 ± 0.085 | 0.530 |
| Logistic Regression | 24 | 0.580 | — |
| BumbleBee AE + iNNE (unsupervised) | 24 | 0.559 | — |
| Transformer + BCE | 69 | 0.813 ± 0.092 | 0.728 |
| **Transformer + focal + smooth** (stabilized) | **69** | **0.846 ± 0.085** | **0.725** |

### Ablation study (9 patients, d_model=1024)

| Technique | Best Config | AUROC | Δ vs BCE |
|---|---|---|---|
| Baseline (BCE) | — | 0.813 ± 0.092 | — |
| **Focal + smooth** † | γ=1.0, α=0.7, ε=0.2 | **0.846 ± 0.085** | **+0.033** |
| Weight decay | λ=1e-3 | 0.826 ± 0.104 | +0.013 |
| Stochastic depth | p=0.2 | 0.824 ± 0.106 | +0.011 |
| Label smoothing | ε=0.05 | 0.810 ± 0.093 | −0.003 |
| Focal loss | γ=2.0, α=0.5 | 0.802 ± 0.125 | −0.011 |
| RoPE | enabled | 0.771 | −0.042 |

† Requires AdamW + gradient clipping (max norm 1.0). Every other entry uses Adam without clipping.

### Per-patient breakdown (best config)

| Patient | AUROC | Diagnosis |
|---|---|---|
| P9 | 0.996 | Bipolar |
| P8 | 0.939 | Bipolar |
| P3 | 0.905 | Bipolar |
| P7 | 0.896 | Schizophrenia |
| P1 | 0.808 | Brief psychotic |
| P6 | 0.803 | Bipolar |
| P4 | 0.775 | Bipolar |
| P5 | 0.750 | Bipolar II |
| P2 | 0.745 | Schizophreniform |

Fold variance (σ = 0.085) is lower than the BCE baseline (σ = 0.092), indicating more consistent cross-patient performance under the stabilized optimizer. Under AdamW + grad-clip, the diagnostically atypical patients (P1, P7) perform comparably to several bipolar patients.

## Key Findings

1. **Loss engineering beats architecture engineering.** Bottleneck, gated, and DANN fusion variants all underperformed the plain sequence transformer. The gains came from loss function design and training stabilization, not architectural novelty.
2. **Focal + smooth requires stabilization.** Focal loss alone *degraded* performance (−0.011), and focal + smooth diverged in 4–5/9 folds under Adam. Swapping to AdamW with gradient clipping (max norm 1.0) stabilized all 9 folds and yielded the best result.
3. **Mild focal focus is optimal.** γ=1.0 ≫ γ=2.0 or 3.0. With only ~20 relapse days per fold, aggressive down-weighting of easy negatives starves the gradient signal.
4. **Patient-specific z-scoring is essential.** Raw inter-patient variability dominates the feature space; without per-patient normalization, no model learns to distinguish relapse from non-relapse states.
5. **Evening HRV window > classical nighttime.** A sweep over 12 candidate windows selected W14 (14:00–22:00) as optimal, outperforming overnight recordings.
6. **Global hyperparameter selection > per-fold tuning.** Individual validation sets are too small (~9 relapse events) for reliable model selection.
7. **RoPE hurts short sequences.** 7-day windows are too short for relative position encoding to contribute useful signal (−0.042 AUROC).

## Project Structure

```
├── configs/                    # ~43 experiment JSON configs + README
│   ├── preprocessing.json
│   ├── ablation_*.json          # Initial ablation sweep (V1)
│   ├── ablv2_{union,all}_*.json # Second ablation sweep across both feature sets
│   └── exp_*.json               # Exploration-phase variants (bottleneck, gated, DANN, ...)
├── src/
│   ├── models/                  # transformer_v1..v4 (v1 is the paper's model)
│   ├── losses/                  # Focal loss
│   ├── data_loader.py           # Raw sensor loading
│   ├── feature_extractor.py     # 69 per-day features
│   ├── preprocess_loso.py       # 7-day sliding windows, LOPO fold generation
│   ├── train.py                 # Training loop
│   ├── ablation.py              # Ablation study runner
│   └── experiment.py
├── scripts/
│   ├── preprocess_data.py
│   ├── run.sh                   # Local runner
│   └── submit_slurm.sh          # SLURM array-job submission
├── notebooks/
│   └── main.ipynb               # Feature engineering + notebook-phase modeling
├── docs/
│   ├── TASK_ARCHIVE.md          # Development log
│   └── results/                 # Auto-generated per-experiment reports + summary
├── report/
│   └── MLHC2026-ResearchTrack-Template/   # Paper source (LaTeX)
└── figures/
```

## Setup

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Reproduction

```bash
# 1. Preprocess data (expects data in data/original/track1/)
python scripts/preprocess_data.py

# 2. Best result: stabilized focal + smooth + grad_clip (the paper's headline model)
bash scripts/run.sh -n ablv2_all_focal_smooth_v2

# Or submit as a SLURM array job
bash scripts/submit_slurm.sh -n ablv2_all_focal_smooth_v2
bash scripts/submit_slurm.sh -n ablv2_all_focal_smooth_v2 --dry-run

# BCE baseline (ablation reference point)
bash scripts/run.sh -n ablv2_all_baseline

# Explore the notebook-phase modeling
jupyter notebook notebooks/main.ipynb
```

Per-experiment configs live in `configs/` (see `configs/README.md` for the index). Auto-generated per-experiment reports and the aggregated leaderboard are in `docs/results/`.

## Limitations

- **Small cohort.** Only 9 patients; LOPO results are high-variance, and patients P2 and P5 remain the weakest folds. Some relapse patterns may simply not be detectable from wearable data alone.
- **COVID-19 era data.** Collection overlapped with the pandemic, which may have destabilized baseline recordings.
- **Label noise.** Ground truth was annotated monthly at clinical follow-ups, so exact relapse onset days are inherently uncertain — this motivated label smoothing.
- **No explainability analysis.** We did not perform SHAP or attention-weight analysis to identify which features drive per-patient predictions. This is left for future work.
- **No prospective evaluation.** All results are retrospective LOPO; real-world prospective deployment performance is untested.
