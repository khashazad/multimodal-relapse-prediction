# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multimodal relapse prediction from wearable sensor data (Track 1: non-psychotic relapse detection). Binary per-day classification (relapse vs. stable) for 9 patients using 5 sensor modalities. Primary work lives in `main.ipynb`. The `src/`/`scripts/`/`configs/` directories form a separate experiment execution framework (grid search + SLURM runner) targeting a supervised Leave-One-Subject-Out (LOSO) cross-validation setup with a transformer fusion architecture.

## Setup & Commands

```bash
# Create and activate virtual environment
python -m venv env
source env/bin/activate
pip install -r requirements.txt

# Run the notebook
jupyter notebook main.ipynb

# Run preprocessing pipeline
python scripts/preprocess_data.py

# Run experiments locally (all parameter combinations)
bash scripts/run.sh -n experiment

# Submit to SLURM as array job
bash scripts/submit_slurm.sh -n experiment
bash scripts/submit_slurm.sh -n experiment --dry-run  # preview command
```

Preprocessing config is read from `configs/preprocessing.json`. Experiment configs go in `configs/<name>.json` and are referenced by name when running scripts.

## Data Structure

```
track1/
  demographics.csv          # Patient demographics
  P{1-9}/
    {train_0,train_1,val_0,val_1,test_0,test_1,test_2}/
      gyr.parquet           # Gyroscope (20 Hz)
      hrm.parquet           # Heart rate monitor (5 Hz)
      linacc.parquet        # Linear accelerometer (20 Hz)
      step.parquet          # Step count (per minute totals)
      sleep.parquet         # Sleep episodes
      relapses.csv          # Relapse labels
```

Sleep data is also stored separately under `data/original/track1/sleep_files/P{x}/{sequence}/sleep.parquet`.

## Notebook Architecture (main.ipynb)

The notebook follows a progressive feature engineering + modeling pipeline:

1. **Data exploration** (cells 0–6): Load and inspect each sensor modality for a sample patient; count relapses across all patients.

2. **Feature engineering** (cells 7–21): Each modality builds patient-specific baselines from non-relapse days in train+val splits, then computes z-score deviation features for each day:
   - **Sleep features** (cell 7): Main sleep episode + nap metrics; deviation from personal baseline.
   - **Step count features** (cell 11): Daily total steps; `steps_zscore_inv` (inverted so fewer steps = higher risk signal).
   - **Nighttime HRV** (cell 16): RMSSD and SDNN derived from `nighttime_seqs_v4.pkl` (bin-averaged, acc-based sleep filter); cached to `cache/hrv_features_nighttime_v3.parquet` and `cache/hrv_baselines_nighttime_v3.pkl`.
   - **Sleep-verified HRV** (cell 24–25): HRV derived from same `nighttime_seqs_v4.pkl` source (acc filter already applied); cached to `cache/hrv_sleep_verified_v2.parquet`.
   - **Demographics** (cell 20–21): Age, sex, diagnosis loaded from `track1/demographics.csv`.
   - **Circadian activity** (cell 29): Hourly activity profiles fused from gyroscope + linear accelerometer (weighted sum, time-aligned to 1s); 15 features including clinically-validated actigraphy metrics (relative amplitude, intradaily variability, cosinor amplitude/acrophase, L5/M10 onset, evening activity); cached to `cache/circadian_features_fused_v3.parquet`.

3. **Models** (cells 8–40): All models use LOPO (Leave-One-Patient-Out) cross-validation. Primary metrics: AUROC and AUPRC.
   - XGBoost and Logistic Regression on progressively richer feature sets.
   - MLP (cell 36).
   - **Transformer** (cell 38–39): Sequence model over days; 5 input features; trained weights cached to `cache/transformer1_seq5.pth` and `cache/transformer2_seq5.pth`. LOPO CV hyperparameter tuning cached to `cache/lopo_cv_tuning.pkl`.

## Caching Convention

Delete a cache file to force recomputation of that step. Cache filenames include version suffixes (e.g., `_v2`) which are bumped when the computation logic changes.

All cached files:
- `cache/hrv_features_nighttime_v3.parquet`, `cache/hrv_baselines_nighttime_v3.pkl`
- `cache/hrv_sleep_verified_v2.parquet`
- `cache/circadian_features_fused_v3.parquet`
- `cache/transformer1_seq5.pth`, `cache/transformer2_seq5.pth`
- `cache/lopo_cv_tuning.pkl`

## Experiment Framework Architecture

### Data Pipeline

The preprocessing pipeline flows: raw parquet/CSV → feature extraction → sliding windows → LOSO fold splits → saved pickle/numpy.

**`src/data_loader.py`** — `MultimodalDataLoader` reads all modalities into `SequenceData` dataclasses. Patients are `P1`–`P9`; each has multiple sequences (`train_0`, `val_0`, `test_0`, etc.).

**`src/feature_extractor.py`** — `FeatureExtractor` computes 21 per-day scalar features from raw sensor streams: accelerometer norm stats, gyroscope norm stats, HRV metrics (RMSSD, SDNN, Lomb-Scargle HF power), sleep duration/segments, step/distance/calorie aggregates, and sinusoidal time encoding.

**`src/preprocess_loso.py`** — `LOSOPreprocessor` orchestrates the full pipeline: extracts features per sequence, creates 24-day sliding windows (stride=1), organizes supervised LOSO splits, and saves to `data/processed/track1/fold_N/`. Each window dict contains `features (window_size, n_features)`, `labels (window_size,)`, and `label_mask (window_size,)`.

**`scripts/preprocess_data.py`** — Entry point that reads `configs/preprocessing.json` and runs `LOSOPreprocessor.run()`.

### LOSO Fold Structure

For each fold (test patient = P_k):
- **Train**: ALL sequences (`train_*`, `val_*`, `test_*`) from the remaining 8 patients. `train_*` sequences are treated as stable (label=0); `val_*`/`test_*` sequences use actual relapse labels.
- **Val**: `val_*` sequences from P_k only.
- **Test**: `test_*` sequences from P_k only.

This means `test_*` sequences from non-held-out patients contribute labeled data to training, since those labels are known from the challenge dataset.

### Critical Data Quirks

1. **Extra-day bug**: each `relapses.csv` has one spurious extra row at the end. `LOSOPreprocessor` drops the last row before using labels.

2. **Label availability by split**:
   - `train_*`: `relapse` column exists but is all zeros (confirmed stable)
   - `val_*`: `relapse` column has real 0/1 labels
   - `test_*`: **no `relapse` column** — must predict these days

3. **Class imbalance**: relapse days are a minority (e.g., ~20 relapse vs. ~60 stable in a single validation sequence).

4. **Missing sensor days**: a `day_index` in `relapses.csv` may be absent from a modality's parquet (sensor not recording). Features will be NaN for that day.

### Experiment Runner

`scripts/run.sh` / `scripts/submit_slurm.sh` read a JSON config (`configs/<name>.json`) and dispatch to `src/executor.py`. The config's `executor.exec_name` determines which `src/<name>.py` (or `.jl`) is run. SLURM settings (`partition`, `gres`, `time`, etc.) are in the config's `slurm` section. Both Python and Julia sources are supported.

Config lives in `configs/{name}.json`. Parameters listed as arrays create a grid search; scalar parameters are fixed across all runs. Keys prefixed with `_` are treated as comments and ignored by the executor.

### Notebooks

- `notebooks/main.ipynb` — feature engineering exploration (HRV, sleep, steps, accelerometer, gyroscope, demographics)
- `notebooks/experiment.ipynb` — relapse prediction analysis

If using `matplotlib` with `usetex=True` in notebooks, see `notebooks/README.md` for a TinyTeX workaround when LaTeX is unavailable system-wide.

### Data Directory Layout

```
data/
├── original/track1/
│   ├── P1/–P9/
│   │   ├── train_0/, train_1/ [, train_2/]
│   │   ├── val_0/, val_1/ [, val_2/]
│   │   └── test_0/, test_1/ [, test_2/]
│   ├── sleep_files/P1/–P9/{sequence}/sleep.parquet
│   ├── demographics.csv
│   └── demographics.xlsx
└── processed/track1/
    ├── metadata.json
    └── fold_N/train.pkl, val.pkl, test.pkl
```

## Task Tracking

`TASK.md` tracks completed and pending work for this project. Update it when making significant changes.

## Good practices

When you come up with a new plan, save the context and plan in a number file starting with three digits ***_file_name.md and numbering them in order.
At the start of each plan, summarize the purpose of the plan and the output summary of what was found in the cell in a 'Summarize Section'.
Keep track of which plans were completed in the TASK.md file.
