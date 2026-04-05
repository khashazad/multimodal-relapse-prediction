# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Note: This repository is archived after experimentation. Headline result: 0.842 AUROC (focal loss transformer + grad_clip, 9/9 fold LOSO). Stabilized focal+smooth rerun pending.**

## Project Overview

Multimodal relapse prediction from wearable sensor data (Track 1: non-psychotic relapse detection). Binary per-day classification (relapse vs. stable) for 9 patients using 5 sensor modalities. Primary notebook: `notebooks/main.ipynb`. The `src/`/`scripts/`/`configs/` directories form the experiment framework (grid search + SLURM runner) for LOSO cross-validation with transformer fusion architectures.

## Setup & Commands

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt

# Feature engineering notebook
jupyter notebook notebooks/main.ipynb

# Preprocess for experiment framework
python scripts/preprocess_data.py

# Run experiments locally
bash scripts/run.sh -n experiment

# Submit to SLURM
bash scripts/submit_slurm.sh -n experiment
bash scripts/submit_slurm.sh -n experiment --dry-run
```

Preprocessing config: `configs/preprocessing.json`. Experiment configs: `configs/<name>.json`.

## Data Structure

```
data/original/track1/
  demographics.csv
  P{1-9}/
    {train_0,train_1,val_0,val_1,test_0,test_1,test_2}/
      gyr.parquet           # Gyroscope (20 Hz)
      hrm.parquet           # Heart rate (5 Hz)
      linacc.parquet        # Linear accelerometer (20 Hz)
      step.parquet          # Step count (per minute)
      sleep.parquet         # Sleep episodes
      relapses.csv          # Relapse labels
    sleep_files/P{x}/{sequence}/sleep.parquet
```

## Notebook Architecture (notebooks/main.ipynb)

Progressive feature engineering + modeling pipeline:

1. **Data exploration** (cells 0–6): Load and inspect each sensor modality; count relapses.

2. **Feature engineering** (cells 7–21): Patient-specific baselines from non-relapse days, then z-score deviation features:
   - **Sleep**: main episode + nap metrics
   - **Steps**: daily totals, inverted z-score
   - **Nighttime HRV**: RMSSD/SDNN from acc-filtered heart rate
   - **Circadian activity**: 15 actigraphy metrics from gyroscope + accelerometer fusion
   - **Demographics**: age, sex, diagnosis

3. **Models** (cells 8–40): LOPO CV with AUROC/AUPRC. XGBoost → Logistic Regression → MLP → Transformer.

## Caching Convention

Delete a cache file to force recomputation. Version suffixes bumped when logic changes.

- `cache/hrv_features_nighttime_v3.parquet`, `cache/hrv_baselines_nighttime_v3.pkl`
- `cache/hrv_sleep_verified_v2.parquet`
- `cache/circadian_features_fused_v3.parquet`
- `cache/transformer1_seq5.pth`, `cache/transformer2_seq5.pth`
- `cache/lopo_cv_tuning.pkl`

## Experiment Framework

### Data Pipeline

Raw parquet/CSV → `src/feature_extractor.py` (21 per-day features) → `src/preprocess_loso.py` (24-day sliding windows, LOSO splits) → `data/processed/track1/fold_N/`.

### LOSO Fold Structure

For fold k (test patient = P_k):
- **Train**: all sequences from remaining 8 patients
- **Val**: `val_*` sequences from P_k
- **Test**: `test_*` sequences from P_k

### Critical Data Quirks

1. **Extra-day bug**: each `relapses.csv` has one spurious extra row — `LOSOPreprocessor` drops last row
2. **Label availability**: `train_*` = all zeros (stable); `val_*` = real labels; `test_*` = no `relapse` column
3. **Class imbalance**: ~20 relapse vs ~60 stable per validation sequence
4. **Missing sensor days**: some `day_index` values absent from modality parquets → NaN features

### Experiment Runner

`scripts/run.sh` / `scripts/submit_slurm.sh` read JSON configs and dispatch to `src/executor.py`. Array parameters create grid search; `_`-prefixed keys are comments. See `configs/README.md` for config index.

### Directory Layout

```
├── EXPERIMENTS.md              # Full experiment catalog
├── configs/                    # ~43 JSON configs + README
├── docs/
│   ├── TASK_ARCHIVE.md         # Development log (48KB)
│   └── results/                # Auto-generated per-experiment docs + summary
├── figures/                    # Exported plots
├── notebooks/
│   ├── main.ipynb              # Primary notebook (outputs stripped)
│   └── README.md               # TinyTeX workaround
├── scripts/                    # Run/submit/preprocess/verify scripts
└── src/
    ├── models/                 # Transformer v1–v4
    ├── losses/                 # Focal loss
    ├── data_loader.py
    ├── feature_extractor.py
    ├── preprocess_loso.py
    ├── train.py
    ├── ablation.py
    └── experiment.py
```
