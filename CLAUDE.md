# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a psychiatric relapse prediction project using wearable sensor data from 9 patients (P1–P9). The primary work lives in `main.ipynb`. The `multimodal-relapse-prediction/` subdirectory is a separate experiment execution framework (grid search + SLURM runner) that is independent of the main notebook.

## Running the Notebook

Open and run `main.ipynb` in Jupyter. Cells are designed to be run top-to-bottom, but many expensive cells are cached and will reload from `./cache/` on rerun.

```bash
jupyter notebook main.ipynb
```

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

## Notebook Architecture (main.ipynb, 41 cells)

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

## Task Tracking

`TASK.md` tracks completed and pending work for this project. Update it when making significant changes.

## multimodal-relapse-prediction/ Framework

A generic experiment runner (unrelated to the main notebook). Requires a Python virtual environment at `multimodal-relapse-prediction/env/`.

```bash
cd multimodal-relapse-prediction
# First-time setup
python -m venv env && source env/bin/activate && pip install -r requirements.txt

# Run all parameter combinations locally
./scripts/run.sh -n experiment

# Submit as SLURM array job
./scripts/submit_slurm.sh -n experiment
./scripts/submit_slurm.sh -n experiment --dry-run  # preview sbatch command
```

Config lives in `configs/{name}.json`. Parameters listed as arrays create a grid search; scalar parameters are fixed across all runs. Keys prefixed with `_` are treated as comments and ignored by the executor. The `executor.exec_name` field names the script in `src/` to invoke per experiment (default: `experiment`). `src/experiment.py` is a placeholder template — replace it with the actual experiment logic. Results go to `outputs/`; logs go to `outputs/logs/`. SLURM settings (partition, memory, time, etc.) are configured in the `slurm` block of the config JSON.

## Good practices
When you come up with a new plan, save the context and plan in a number file starting with three digits ***_file_name.md and numbering them in order.
at the start of each plan, summarize the purpose of the plan and the output summary of what was found in the cell in a 'Summarize Section'
Keep track of which plans were completed in the TASK.md file.


