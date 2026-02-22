# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multimodal relapse prediction from wearable sensor data (Track 1: non-psychotic relapse detection). The goal is binary per-day classification (relapse vs. stable) for 9 patients using 5 sensor modalities. This is a challenge submission project targeting a supervised Leave-One-Subject-Out (LOSO) cross-validation setup with a transformer fusion architecture.

## Setup & Commands

```bash
# Create and activate virtual environment
python -m venv env
source env/bin/activate
pip install -r requirements.txt

# Run preprocessing pipeline
python scripts/preprocess_data.py

# Run experiments locally (all parameter combinations)
bash scripts/run.sh -n experiment

# Submit to SLURM as array job
bash scripts/submit_slurm.sh -n experiment
bash scripts/submit_slurm.sh -n experiment --dry-run  # preview command

# Start Jupyter
jupyter notebook
```

Preprocessing config is read from `configs/preprocessing.json`. Experiment configs go in `configs/<name>.json` and are referenced by name when running scripts.

## Architecture

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

1. **Sleep data is stored separately**: not in the main sequence folder but under `data/original/track1/sleep_files/P{x}/{sequence}/sleep.parquet`. The current `data_loader.py` looks for `sleep.parquet` inline — this may need fixing if sleep data isn't loading.

2. **Extra-day bug**: each `relapses.csv` has one spurious extra row at the end. `LOSOPreprocessor` drops the last row before using labels.

3. **Label availability by split**:
   - `train_*`: `relapse` column exists but is all zeros (confirmed stable)
   - `val_*`: `relapse` column has real 0/1 labels
   - `test_*`: **no `relapse` column** — must predict these days

4. **Class imbalance**: relapse days are a minority (e.g., ~20 relapse vs. ~60 stable in a single validation sequence).

5. **Missing sensor days**: a `day_index` in `relapses.csv` may be absent from a modality's parquet (sensor not recording). Features will be NaN for that day.

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

### Experiment Runner

`scripts/run.sh` / `scripts/submit_slurm.sh` read a JSON config (`configs/<name>.json`) and dispatch to `src/executor.py`. The config's `executor.exec_name` determines which `src/<name>.py` (or `.jl`) is run. SLURM settings (`partition`, `gres`, `time`, etc.) are in the config's `slurm` section. Both Python and Julia sources are supported.

### Notebooks

- `notebooks/main.ipynb` — feature engineering exploration (HRV, sleep, steps, accelerometer, gyroscope, demographics)
- `notebooks/experiment.ipynb` — relapse prediction analysis

If using `matplotlib` with `usetex=True` in notebooks, see `notebooks/README.md` for a TinyTeX workaround when LaTeX is unavailable system-wide.
