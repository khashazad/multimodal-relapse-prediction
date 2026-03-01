# Preprocessing Pipeline Reference

This document provides a complete technical reference for the multimodal relapse prediction preprocessing pipeline. It covers every stage from raw sensor parquet/CSV files to the final saved LOSO fold pickles used by downstream model training.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Data Loading (`data_loader.py`)](#2-data-loading)
3. [Feature Extraction (`feature_extractor.py`)](#3-feature-extraction)
4. [Normalization](#4-normalization)
5. [Windowing](#5-windowing)
6. [LOSO Fold Organization](#6-loso-fold-organization)
7. [Configuration](#7-configuration)
8. [Output Format](#8-output-format)
9. [Critical Design Decisions](#9-critical-design-decisions)
10. [Known Quirks](#10-known-quirks)

---

## 1. Overview

The preprocessing pipeline transforms raw wearable sensor data from 9 psychiatric patients (P1--P9) into supervised sliding-window samples organized for Leave-One-Subject-Out (LOSO) cross-validation. The goal is per-day binary classification: **relapse (1) vs. stable (0)**.

### End-to-End Flow

```
Raw parquet/CSV files
        |
        v
[Step 0] Stage supplementary files (sleep + annotations) into sequence dirs
        |
        v
[Step 1] For each patient:
         a. Load all modalities via MultimodalDataLoader
         b. Extract per-day feature vectors via FeatureExtractor (121 features total)
         c. Per-patient z-score normalization (fit on train_* sequences only)
         d. Create 7-day left-padded sliding windows (stride=1)
        |
        v
[Step 2] Organize windows into 9 LOSO folds (one per held-out patient)
        |
        v
[Step 3] Save to disk as pickle (or numpy) + metadata.json
```

### Source Files

| File | Role |
|------|------|
| `src/data_loader.py` | Raw sensor data loading into `SequenceData` dataclasses |
| `src/feature_extractor.py` | Per-day feature extraction for all 5 modalities (121 total features) |
| `src/preprocess_loso.py` | LOSO orchestration: normalization, windowing, fold organization, saving |
| `scripts/preprocess_data.py` | CLI entry point (full / per-patient / merge modes) |
| `configs/preprocessing.json` | All pipeline parameters |

### Feature Dimension Summary

| Modality | Feature Count |
|----------|--------------|
| Accelerometer (`accel`) | 38 |
| Gyroscope (`gyr`) | 38 |
| Heart rate (`hr`) | 26 |
| Step (`step`) | 10 |
| Sleep (`sleep`) | 9 |
| **Total** | **121** |

---

## 2. Data Loading

**Source file:** `src/data_loader.py`

### 2.1 `SequenceData` Dataclass

Every patient sequence is loaded into a `SequenceData` instance:

```python
@dataclass
class SequenceData:
    patient_id:    str                     # e.g. "P1"
    sequence_name: str                     # e.g. "train_0"
    split:         str                     # "train", "val", or "test"

    linacc:   Optional[pd.DataFrame]       # Linear accelerometer (x, y, z, time, day_index)
    gyr:      Optional[pd.DataFrame]       # Gyroscope (x, y, z, time, day_index)
    hrm:      Optional[pd.DataFrame]       # Heart rate monitor (heartRate, rrInterval, time, day_index)
    step:     Optional[pd.DataFrame]       # Step data (totalSteps, stepsWalking, stepsRunning, distance, calories, ...)
    sleep:    Optional[pd.DataFrame]       # Sleep episodes (start_time, end_time, start_date_index, end_date_index)
    relapses: Optional[pd.DataFrame]       # Labels (day_index, relapse)
```

Each field is `None` if the corresponding file does not exist for that sequence.

### 2.2 Patient/Sequence Structure

Patients are identified by directory names matching the pattern `P{N}` (e.g., `P1` through `P9`). Each patient directory contains multiple sequence subdirectories named `{split}_{index}`:

- `train_0`, `train_1`, `train_2`, ...
- `val_0`, `val_1`, `val_2`, ...
- `test_0`, `test_1`, `test_2`, ...

The `split` is derived by splitting on `"_"` and taking the first token.

### 2.3 Modality File Mapping

Each sequence directory is expected to contain the following files:

| Modality Key | Filename | Format |
|-------------|----------|--------|
| `linacc` | `linacc.parquet` | Parquet |
| `gyr` | `gyr.parquet` | Parquet |
| `hrm` | `hrm.parquet` | Parquet |
| `step` | `step.parquet` | Parquet |
| `sleep` | `sleep.parquet` | Parquet |
| (labels) | `relapses.csv` | CSV |

All sensor parquets are read via `pd.read_parquet()`. The labels file `relapses.csv` is read via `pd.read_csv()`. Missing files produce a warning but do not cause failure; the corresponding `SequenceData` attribute remains `None`.

### 2.4 Sleep File Handling

Sleep files are distributed separately from the main sensor parquets. Their original location is:

```
data/original/track1/sleep_files/P{x}/{sequence}/sleep.parquet
```

Before loading, `LOSOPreprocessor._stage_supplementary_files()` copies each sleep parquet into the corresponding inline sequence directory:

```
data/original/track1/P{x}/{sequence}/sleep.parquet
```

This staging is non-destructive: files already present at the destination are left untouched. The same staging mechanism applies to annotation files (`relapses.csv`) from a separate annotations directory.

### 2.5 Discovery Methods

- `get_patients()` -- returns a sorted list of patient IDs by scanning for subdirectories starting with `"P"` under the track path.
- `get_sequences(patient_id)` -- returns a sorted list of sequence names whose first token (before `"_"`) is one of `{"train", "val", "test"}`.
- `load_sequence(patient_id, sequence_name)` -- loads a single sequence.
- `load_all_sequences(patient_id)` -- loads all sequences for one patient.
- `load_all_patients()` -- loads everything, returning `Dict[str, List[SequenceData]]`.

---

## 3. Feature Extraction

**Source file:** `src/feature_extractor.py`

The `FeatureExtractor` class computes per-day feature vectors from raw sensor data. Extraction is performed at the sequence level: all days in a sequence are processed together using vectorized pandas `groupby` operations where possible, falling back to per-group loops only where necessary.

### 3.1 Constructor Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size_minutes` | 5 | Size of intra-day time windows in minutes (used to bin samples within each day) |
| `sample_rate_imu` | 20 | IMU sampling rate in Hz (used for FFT frequency axis) |
| `sample_rate_hr` | 5 | HR sampling rate in Hz |
| `coverage_threshold` | 0.25 | Minimum fraction of time windows per day to consider data sufficient |

Derived attributes:
- `win_secs = window_size_minutes * 60` (300 seconds by default)
- `min_windows = int(coverage_threshold * (24 * 60 / window_size_minutes))` (72 by default, though this threshold is defined but not used for gating in the current code)

### 3.2 Return Format

For each modality, the extractor returns a dict:

```python
{
    "features": np.ndarray,  # shape (N_days, F), dtype float32
    "mask":     np.ndarray,  # shape (N_days,), dtype bool
}
```

The `mask` is `True` for days where the modality had data and `False` for days with no sensor readings (features are all-NaN for those days before normalization).

### 3.3 Time Parsing

The helper `_time_to_seconds()` converts time values to float seconds. It handles two formats:
- String format `"HH:MM:SS.ffffff"` -- parsed via string slicing.
- `datetime.time` objects (as stored in parquet) -- converted via `np.fromiter` over the `.hour`, `.minute`, `.second`, `.microsecond` attributes.

Intra-day time windows are assigned by `floor(seconds / win_secs)`.

---

### 3.4 Accelerometer Features (38 features)

**Source modality:** `linacc.parquet` (columns: `x`, `y`, `z`, `time`, `day_index`)

**Extraction method:** Vectorized pandas groupby over `[day_index, _win]` (intra-day 5-minute windows), then averaged across windows per day.

#### 3.4.1 Statistical Features (28 features)

For each of the 4 signals (`x`, `y`, `z`, `mag`), 7 statistics are computed per intra-day window via `groupby.agg()`:

| Statistic | Description |
|-----------|-------------|
| `mean` | Arithmetic mean |
| `std` | Standard deviation (ddof=1) |
| `min` | Minimum value |
| `max` | Maximum value |
| `median` | Median value |
| `skew` | Skewness (scipy.stats.skew) |
| `kurt` | Excess kurtosis (scipy.stats.kurtosis) |

Windows with fewer than 2 samples are set to NaN. Per-window statistics are then averaged across all windows for each day.

The magnitude signal is computed as: `mag = sqrt(x^2 + y^2 + z^2)`.

**Feature names (28):**
```
x_mean, x_std, x_min, x_max, x_median, x_skew, x_kurt,
y_mean, y_std, y_min, y_max, y_median, y_skew, y_kurt,
z_mean, z_std, z_min, z_max, z_median, z_skew, z_kurt,
mag_mean, mag_std, mag_min, mag_max, mag_median, mag_skew, mag_kurt
```

**Rationale:** Per-axis statistics capture the distribution of accelerometer readings throughout the day. The magnitude collapses the 3D signal into a rotation-invariant scalar that reflects overall motion intensity. Skewness and kurtosis capture asymmetry and tail behavior in movement patterns, which may differ between stable and relapse periods.

#### 3.4.2 Activity Features (4 features)

| Feature | Computation | Description |
|---------|-------------|-------------|
| `total_energy` | `sum(mag^2)` per window, averaged per day | Total kinetic energy proxy |
| `zero_crossing_rate` | Fraction of sign-changes in `(mag - mean(mag))` per window, averaged per day | Oscillation frequency of movement |
| `pct_active` | Fraction of samples where `mag > 1.5` per window, averaged per day | Proportion of active time |
| `pct_sedentary` | Fraction of samples where `mag < 0.2` per window, averaged per day | Proportion of sedentary time |

**Rationale:** Activity-level features provide a coarse behavioral summary. Reduced energy and increased sedentary time are well-established behavioral markers of relapse in mood disorders. The zero-crossing rate captures rhythmicity of movement.

#### 3.4.3 Frequency-Domain Features (6 features)

Computed per intra-day window via FFT on the demeaned magnitude signal, then averaged per day. Requires at least 10 samples per window.

| Feature | Computation | Description |
|---------|-------------|-------------|
| `dominant_freq` | Frequency (Hz) of peak FFT power (excluding DC) | Primary oscillation frequency |
| `spectral_entropy` | `-sum(p_norm * log(p_norm))` where `p_norm` is normalized power spectrum | Complexity/regularity of movement |
| `power_0_1hz` | Band power in [0, 1) Hz | Very low frequency power (postural sway) |
| `power_1_3hz` | Band power in [1, 3) Hz | Walking frequency band |
| `power_3_8hz` | Band power in [3, 8) Hz | Running/vigorous activity band |
| `power_8plus_hz` | Band power in [8, inf) Hz | High-frequency vibration/tremor |

FFT is computed via `np.fft.rfft` on demeaned magnitude with frequency axis derived from `np.fft.rfftfreq(N, d=1/fs_imu)`.

**Rationale:** Spectral features capture the quality of movement. Higher spectral entropy indicates more complex, varied movement. Band powers decompose activity into physiologically meaningful frequency ranges (postural vs. ambulatory vs. vigorous). Changes in these distributions may precede relapse.

---

### 3.5 Gyroscope Features (38 features)

**Source modality:** `gyr.parquet` (columns: `x`, `y`, `z`, `time`, `day_index`)

The gyroscope extraction pipeline is **identical** to the accelerometer pipeline. The same `_extract_imu()` method is called for both modalities. Feature names share the same layout:

```
x_mean, x_std, x_min, x_max, x_median, x_skew, x_kurt,
y_mean, y_std, y_min, y_max, y_median, y_skew, y_kurt,
z_mean, z_std, z_min, z_max, z_median, z_skew, z_kurt,
mag_mean, mag_std, mag_min, mag_max, mag_median, mag_skew, mag_kurt,
total_energy, zero_crossing_rate, pct_active, pct_sedentary,
dominant_freq, spectral_entropy, power_0_1hz, power_1_3hz, power_3_8hz, power_8plus_hz
```

**Rationale:** Gyroscope data captures rotational motion (angular velocity), which complements the linear acceleration signal. The same feature extraction applies because the statistical and spectral properties of angular velocity are equally informative. Together, accelerometer and gyroscope features provide a more complete picture of motor behavior.

---

### 3.6 Heart Rate Features (26 features)

**Source modality:** `hrm.parquet` (columns detected dynamically: heart rate column, RR interval column, `time`, `day_index`)

Column detection:
- HR column: first column with `"heart"` in name (case-insensitive) or named `"hr"`
- RR column: first column with `"rr"` in name or `"interval"` in name (case-insensitive)

**Extraction method:** Per-day loop via `df.groupby("day_index")`. Each day is processed independently.

#### 3.6.1 HR Statistics (7 features)

Standard 7-statistic summary (`_seven_stats`) applied to all non-NaN heart rate values for the day:

```
hr_mean, hr_std, hr_min, hr_max, hr_median, hr_skew, hr_kurt
```

Returns NaN array if fewer than 2 valid samples.

#### 3.6.2 RR Statistics (7 features)

Standard 7-statistic summary applied to non-zero, non-NaN RR interval values:

```
rr_mean, rr_std, rr_min, rr_max, rr_median, rr_skew, rr_kurt
```

**Zero RR filtering:** RR values of exactly 0 are excluded (the sensor reports 0 when no beat is detected).

#### 3.6.3 HRV Time-Domain (4 features)

Computed from cleaned RR intervals (deduplicated consecutive repeated values, then outliers removed via 3-sigma rule). Requires at least 4 clean RR intervals.

| Feature | Computation | Description |
|---------|-------------|-------------|
| `sdnn` | `std(rr_clean, ddof=1)` | Standard deviation of NN intervals (ms) |
| `rmssd` | `sqrt(mean(diff(rr_clean)^2))` | Root mean square of successive differences (ms) |
| `pnn50` | `mean(abs(diff(rr_clean)) > 50) * 100` | Percentage of successive differences > 50 ms |
| `hrv_tri_index` | `N / max(histogram_counts)` with bin width 1000/128 ms | HRV triangular index (geometric measure) |

**RR deduplication:** The wearable device holds the last measured RR interval until a new beat arrives. Consecutive duplicate values are removed before HRV computation to avoid inflating regularity metrics.

**RR outlier removal (`_remove_rr_outliers`):** Intervals more than 3 standard deviations from the mean are removed. If the cleaned array has fewer than 2 elements, the original array is returned.

**Rationale:** HRV time-domain metrics are established biomarkers of autonomic nervous system function. SDNN reflects overall HRV, RMSSD captures vagal (parasympathetic) activity, pNN50 reflects short-term variability, and the triangular index provides a geometric HRV estimate robust to outliers. Reduced HRV is associated with stress and mood disturbance.

#### 3.6.4 HRV Frequency-Domain (4 features)

Computed via the Lomb-Scargle periodogram (`scipy.signal.lombscargle`) on outlier-cleaned RR intervals. Requires at least 10 clean RR intervals.

| Feature | Frequency Band | Description |
|---------|---------------|-------------|
| `lf_power` | 0.04--0.15 Hz (100 evaluation points) | Low-frequency HRV power (sympathetic + parasympathetic) |
| `hf_power` | 0.15--0.40 Hz (200 evaluation points) | High-frequency HRV power (parasympathetic/vagal) |
| `lf_hf_ratio` | `lf_power / (hf_power + 1e-12)` | Sympathovagal balance index |
| `total_power` | 0.003--0.40 Hz (500 evaluation points) | Total spectral HRV power |

**Lomb-Scargle rationale:** Standard FFT requires evenly-spaced samples, but RR intervals are inherently uneven (each interval duration varies). The Lomb-Scargle periodogram handles unevenly-spaced time series natively. The cumulative RR time is computed as `cumsum(rr_clean) / 1000` (converting ms to seconds), and the signal is zero-meaned before periodogram evaluation.

Angular frequencies are passed to `lombscargle()` as `freq_hz * 2 * pi`. Integration uses `np.trapz`.

**Rationale:** The LF/HF ratio reflects the balance between sympathetic and parasympathetic branches of the autonomic nervous system. Shifts in this balance are among the earliest physiological signatures of mood episodes.

#### 3.6.5 Poincare Features (3 features)

Computed from outlier-cleaned RR intervals. Requires at least 4 intervals.

| Feature | Computation | Description |
|---------|-------------|-------------|
| `sd1` | `std((rr[1:] - rr[:-1]) / sqrt(2), ddof=1)` | Short-term variability (perpendicular to identity line) |
| `sd2` | `std((rr[1:] + rr[:-1]) / sqrt(2), ddof=1)` | Long-term variability (along identity line) |
| `sd1_sd2` | `sd1 / (sd2 + 1e-12)` | Ratio of short-term to long-term variability |

**Rationale:** Poincare plot analysis provides a geometric representation of heart rate dynamics that captures nonlinear aspects not visible in linear time/frequency metrics. SD1 relates to RMSSD (parasympathetic), SD2 relates to SDNN (overall variability), and their ratio captures the balance.

#### 3.6.6 Coverage Feature (1 feature)

| Feature | Computation | Description |
|---------|-------------|-------------|
| `coverage_fraction` | `n_unique_windows / 288` | Fraction of 5-minute windows with data in a 24-hour day |

The denominator 288 = 24 hours * 60 minutes / 5-minute windows.

**Rationale:** Sensor wear compliance varies and is itself a behavioral signal. Low coverage may indicate the patient removed the device, which can correlate with behavioral changes preceding relapse.

---

### 3.7 Step Features (10 features)

**Source modality:** `step.parquet` (columns: `totalSteps`, `stepsWalking`, `stepsRunning`, `distance`, `calories`, `start_time`, `end_time`, `start_date_index`, ...)

**Extraction method:** Per-day loop via `df.groupby("start_date_index")`. Each row in the step parquet represents one activity segment (not a single sample).

#### 3.7.1 Aggregate Features (6 features)

| Feature | Computation | Description |
|---------|-------------|-------------|
| `total_steps` | `sum(totalSteps)` | Total steps for the day |
| `walking_steps` | `sum(stepsWalking)` | Steps classified as walking |
| `running_steps` | `sum(stepsRunning)` | Steps classified as running |
| `distance` | `sum(distance)` | Total distance covered |
| `calories` | `sum(calories)` | Total calories burned |
| `n_segments` | `len(day_df)` | Number of activity segments recorded |

#### 3.7.2 Temporal Pattern Features (4 features)

| Feature | Computation | Description |
|---------|-------------|-------------|
| `first_activity_hour` | `min(start_time)` in hours | Earliest activity start (proxy for wake time) |
| `last_activity_hour` | `max(end_time)` in hours | Latest activity end (proxy for bedtime) |
| `longest_gap_hours` | `max(start[i+1] - end[i])` sorted by start time, in hours | Longest inactivity gap |
| `gap_std_hours` | `std(gaps)` in hours | Variability of inactivity gaps |

Gaps are computed by sorting segments by start time and calculating `max(0, start[i+1] - end[i])` for consecutive segments.

**Rationale:** Step count and distance are direct measures of physical activity level, a well-known correlate of mood state. The temporal pattern features (first/last activity, gap structure) capture daily rhythm regularity. Disrupted circadian patterns and prolonged inactivity gaps are prodromal signs of relapse.

---

### 3.8 Sleep Features (9 features)

**Source modality:** `sleep.parquet` (columns: `start_time`, `end_time`, `start_date_index`, `end_date_index`)

**Extraction method:** Two-pass approach:
1. All sleep episodes are parsed and grouped by `end_date_index` (the day the person woke up).
2. Per-day features are computed, followed by a sequential pass for rolling deviation features.

Episode duration handles overnight sleep: if `end_time < start_time`, duration = `(86400 - start_s + end_s) / 3600` hours.

#### 3.8.1 Summary Features (3 features)

| Feature | Computation | Description |
|---------|-------------|-------------|
| `total_sleep_min` | `sum(episode_durations) * 60` | Total sleep duration in minutes |
| `n_episodes` | Number of episodes for the day | Sleep fragmentation indicator |
| `longest_bout_hours` | `max(episode_durations)` in hours | Duration of longest continuous sleep bout |

Days with no sleep episodes get `total_sleep_min = 0`, `n_episodes = 0`, `longest_bout_hours = 0`, with timing features as NaN.

#### 3.8.2 Circular-Encoded Timing Features (4 features)

Timing is taken from the **main sleep episode** (longest duration for the day).

| Feature | Computation | Description |
|---------|-------------|-------------|
| `sleep_onset_sin` | `sin(2 * pi * onset_hour / 24)` | Sine component of sleep onset time |
| `sleep_onset_cos` | `cos(2 * pi * onset_hour / 24)` | Cosine component of sleep onset time |
| `wake_time_sin` | `sin(2 * pi * wake_hour / 24)` | Sine component of wake time |
| `wake_time_cos` | `cos(2 * pi * wake_hour / 24)` | Cosine component of wake time |

**Circular encoding rationale:** Sleep onset times wrap around midnight (e.g., 23:00 and 01:00 are 2 hours apart, not 22). Representing time-of-day as a point on the unit circle via sine/cosine encoding preserves this circular topology and avoids discontinuities at midnight that would confuse downstream models.

#### 3.8.3 Rolling Deviation Features (2 features)

| Feature | Computation | Description |
|---------|-------------|-------------|
| `onset_7day_rolling_deviation` | `circ_diff(today_onset, mean(prior_7_days_onset))` | How much sleep onset deviates from recent average |
| `wake_7day_rolling_deviation` | `circ_diff(today_wake, mean(prior_7_days_wake))` | How much wake time deviates from recent average |

The circular difference function `_circ_diff(a, b, period=24)` computes the minimum of `|a - b| % 24` and `24 - (|a - b| % 24)`, ensuring that the distance between 23:00 and 01:00 is 2 hours, not 22.

The rolling window looks at the **previous** 7 days (indices `[max(0, i-7):i]`), using only days with valid onset/wake data. The deviation is computed against the arithmetic mean of valid prior values.

**Rationale:** Day-to-day variability in sleep timing (social jet lag) is a stronger predictor of mood episodes than absolute sleep duration. The 7-day rolling deviation captures whether the patient's sleep schedule is drifting or becoming erratic compared to their recent baseline.

---

## 4. Normalization

**Source:** `LOSOPreprocessor._normalize_patient()` in `src/preprocess_loso.py`

### 4.1 Strategy: Per-Patient Z-Score

Normalization is applied independently for each patient and each modality. The z-score parameters (mean and standard deviation) are **fit exclusively on `train_*` sequences** for that patient.

```python
z = (x - mean) / std
```

### 4.2 Fitting Procedure

For each modality:
1. Collect all feature rows from `train_*` sequences where the modality mask is `True` (i.e., sensor data was available for that day).
2. Compute `mean = nanmean(all_train, axis=0)` -- shape `(F,)`, dtype `float32`.
3. Compute `std = nanstd(all_train, axis=0)` -- shape `(F,)`, dtype `float32`.
4. Replace any `std < 1e-8` with `1.0` (constant features are not scaled).

If no `train_*` data exists for a modality, an identity normalization is used: `mean = 0`, `std = 1`.

### 4.3 Application

The fitted scaler is applied to **all** sequences (train, val, test) for that patient:

```python
feat = (feat - mean) / std          # z-score
feat = np.clip(feat, -5.0, 5.0)    # clip extreme values
feat = np.where(np.isnan(feat), 0.0, feat)  # NaN -> 0
```

### 4.4 Rationale

| Decision | Rationale |
|----------|-----------|
| Per-patient (not global) | Wearable data has large inter-patient variability (e.g., baseline heart rate differs by 30+ bpm). Per-patient normalization removes this confound. |
| Fit on `train_*` only | `train_*` sequences are confirmed stable periods. Fitting on stable data means z-scores represent deviations from the patient's stable baseline, making relapse deviations more visible. Including val/test data would leak relapse-period statistics into the normalization. |
| NaN to 0 | After z-scoring, NaN features (missing sensor days) become zero, which represents "at the mean." This is a neutral imputation that does not bias the model toward relapse or stable. |
| Clip to [-5, 5] | Prevents extreme outliers from dominating the feature space. Five standard deviations covers 99.99994% of a normal distribution. |

### 4.5 Scaler Storage

Per-patient scalers are stored in `self.patient_scalers[patient_id]`:

```python
{
    "accel": {"mean": np.ndarray(38,), "std": np.ndarray(38,)},
    "gyr":   {"mean": np.ndarray(38,), "std": np.ndarray(38,)},
    "hr":    {"mean": np.ndarray(26,), "std": np.ndarray(26,)},
    "step":  {"mean": np.ndarray(10,), "std": np.ndarray(10,)},
    "sleep": {"mean": np.ndarray(9,),  "std": np.ndarray(9,)},
}
```

These are also saved to `data/processed/track1/patient_scalers.pkl` for reproducibility and inference use.

---

## 5. Windowing

**Source:** `LOSOPreprocessor._create_windows()` in `src/preprocess_loso.py`

### 5.1 Sliding Window With Left Padding

For each sequence of `N` days, the pipeline creates exactly `N` windows (one per day), each of size `W` (default 7 days). The window for day `t` contains days `[t-W+1, ..., t]`.

For early days where `t < W-1`, the window is **left-padded** with zeros:

```
Day 0: [PAD, PAD, PAD, PAD, PAD, PAD, day_0]
Day 1: [PAD, PAD, PAD, PAD, PAD, day_0, day_1]
...
Day 6: [day_0, day_1, day_2, day_3, day_4, day_5, day_6]
Day 7: [day_1, day_2, day_3, day_4, day_5, day_6, day_7]
```

The stride is always 1: every day produces a window.

### 5.2 Window Dict Schema

Each window is a Python dict with the following fields:

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `patient_id` | `str` | -- | Patient identifier, e.g. `"P3"` |
| `sequence_name` | `str` | -- | Sequence name, e.g. `"val_0"` |
| `window_end_day` | `int` | -- | Day index of the last (rightmost) position in the window; this is the day being predicted |
| `window_days` | `List[int]` | `(W,)` | Day indices for each position; `-1` indicates a left-padded position |
| `features` | `Dict[str, np.ndarray]` | `{mod: (W, F_mod)}` | Normalized feature arrays keyed by modality. dtype `float32`. Padded positions are all zeros. |
| `modality_masks` | `Dict[str, np.ndarray]` | `{mod: (W,)}` | Per-modality boolean masks. `True` = modality had sensor data for that day. dtype `bool`. |
| `padding_mask` | `np.ndarray` | `(W,)` | `True` = real day, `False` = left-padded position. dtype `bool`. |
| `labels` | `np.ndarray` | `(W,)` | Per-day labels: `0` = stable, `1` = relapse, `-1` = unknown/unlabeled. dtype `int32`. |
| `label_mask` | `np.ndarray` | `(W,)` | `True` = label is valid and usable for loss computation. dtype `bool`. |

### 5.3 Modality Keys in `features` and `modality_masks`

| Key | Feature Dimension |
|-----|-------------------|
| `"accel"` | 38 |
| `"gyr"` | 38 |
| `"hr"` | 26 |
| `"step"` | 10 |
| `"sleep"` | 9 |

### 5.4 Padding Behavior

- **Padded positions** (`padding_mask[w] == False`):
  - `features[mod][w]` = all zeros (shape `(F_mod,)`)
  - `modality_masks[mod][w]` = `False`
  - `labels[w]` = `-1`
  - `label_mask[w]` = `False`
  - `window_days[w]` = `-1`
- **Real positions** (`padding_mask[w] == True`):
  - Features and masks come from the normalized sequence data.

---

## 6. LOSO Fold Organization

**Source:** `LOSOPreprocessor.organize_loso_splits()` in `src/preprocess_loso.py`

### 6.1 Fold Construction

With 9 patients (P1--P9), 9 LOSO folds are created. For fold `i` where the held-out test patient is `P_k`:

| Split | Source | Content |
|-------|--------|---------|
| **Train** | All patients **except** P_k | ALL windows from ALL sequences (`train_*`, `val_*`, `test_*`) |
| **Val** | P_k only | Windows from `val_*` sequences only |
| **Test** | P_k only | Windows from `test_*` sequences only |

**Important:** The `train_*` sequences from P_k are **not** included in any split for fold `i`. They are used only for normalization (as the z-score fitting source for P_k).

### 6.2 Label Availability by Sequence Type

| Sequence Type | `relapse` Column | Label Behavior |
|---------------|-----------------|----------------|
| `train_*` | Exists, all zeros | Labels set to `0` (stable), `label_mask = True` |
| `val_*` | Exists with real 0/1 labels | Labels read from CSV, `label_mask = True` |
| `test_*` | **Does not exist** (no `relapse` column) | Labels set to `-1`, `label_mask = False` |

This means:
- Training data from non-held-out patients includes both stable (from `train_*`) and labeled relapse/stable (from `val_*` and `test_*`) sequences.
- Validation data (from the held-out patient's `val_*` sequences) has ground-truth labels for evaluation.
- Test data (from the held-out patient's `test_*` sequences) has no labels by default; these are the days to predict for the challenge submission.

### 6.3 The Extra-Day Bug Fix

Each `relapses.csv` has one spurious extra row at the end. The pipeline drops the last row before using labels:

```python
relapses = seq.relapses.iloc[:-1].copy()
```

### 6.4 Fallback Day List

If `relapses.csv` is missing or empty, the day list is constructed from the union of `day_index` values across all sensor modalities (using `start_date_index` for step data and `end_date_index` for sleep data). Only non-negative day indices are included.

### 6.5 Fold Statistics

Each fold records statistics for monitoring:

```python
{
    "n_windows":          int,   # Total number of windows
    "n_stable_days":      int,   # Stable (0) days in real (non-padded) positions with valid labels
    "n_relapse_days":     int,   # Relapse (1) days in real positions with valid labels
    "n_unlabeled_days":   int,   # Real days without valid labels
    "n_padded_positions": int,   # Padded (non-real) positions across all windows
}
```

---

## 7. Configuration

**Source file:** `configs/preprocessing.json`

### 7.1 Top-Level Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_root` | `"data/original"` | Root directory containing the raw data |
| `track` | `1` | Challenge track number (determines `track1` subdirectory) |
| `window_size` | `7` | Number of days per sliding window |
| `stride` | `1` | Stride for sliding window (always 1 in current setup; each day gets a window) |
| `output_dir` | `"data/processed"` | Root directory for processed output |
| `save_format` | `"pickle"` | Output format: `"pickle"` or `"numpy"` |
| `sleep_files_dir` | `"data/original/track1/sleep_files"` | Source directory for sleep parquet files to stage |
| `annotations_dir` | `"data/original/track1/track_1_annotations"` | Source directory for annotation CSVs to stage |

### 7.2 Feature Extraction Parameters

Nested under `"feature_extraction"`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size_minutes` | `5` | Intra-day time window for binning sensor samples (minutes) |
| `sample_rate_imu` | `20` | IMU sampling rate in Hz (for FFT frequency axis) |
| `sample_rate_hr` | `5` | Heart rate sampling rate in Hz |
| `coverage_threshold` | `0.25` | Minimum fraction of 5-min windows per day for coverage gating |

### 7.3 SLURM Parameters

Nested under `"slurm"` (used by `scripts/submit_slurm.sh` for cluster execution):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cores` | `4` | CPUs per task |
| `time` | `"0-04:00"` | Per-patient wall time limit (D-HH:MM) |
| `memory` | `"12G"` | RAM per task |
| `merge_time` | `"0-00:30"` | Wall time for the merge job |
| `merge_memory` | `"8G"` | RAM for merge job |
| `partition` | `"cpunodes"` | SLURM partition |
| `account` | `""` | Billing account |
| `email` | (configured) | Notification email |
| `email_type` | `"FAIL,END"` | When to send email |
| `log_dir` | `"outputs/slurm_logs"` | SLURM log directory |
| `job_name` | `"preprocess"` | Base job name |

---

## 8. Output Format

### 8.1 Directory Structure

```
data/processed/track1/
    metadata.json
    patient_scalers.pkl
    fold_0/
        train.pkl
        val.pkl
        test.pkl
    fold_1/
        train.pkl
        val.pkl
        test.pkl
    ...
    fold_8/
        train.pkl
        val.pkl
        test.pkl
    patients/            (intermediate files, only present for SLURM runs)
        P1.pkl
        P2.pkl
        ...
```

### 8.2 Pickle Format (`save_format = "pickle"`)

Each `{split}.pkl` file contains a Python list of window dicts. Each dict has the schema described in [Section 5.2](#52-window-dict-schema).

```python
import pickle
with open("data/processed/track1/fold_0/train.pkl", "rb") as f:
    windows = pickle.load(f)  # List[Dict]

# Access first window
w = windows[0]
w["features"]["accel"].shape   # (7, 38)
w["features"]["hr"].shape      # (7, 26)
w["labels"].shape               # (7,)
w["padding_mask"].shape         # (7,)
```

### 8.3 Numpy Format (`save_format = "numpy"`)

When using numpy format, each split produces multiple files:

```
fold_0/
    train_accel_features.npy    # shape (N_windows, W, 38), float32
    train_accel_masks.npy       # shape (N_windows, W), bool
    train_gyr_features.npy      # shape (N_windows, W, 38), float32
    train_gyr_masks.npy         # shape (N_windows, W), bool
    train_hr_features.npy       # shape (N_windows, W, 26), float32
    train_hr_masks.npy          # shape (N_windows, W), bool
    train_step_features.npy     # shape (N_windows, W, 10), float32
    train_step_masks.npy        # shape (N_windows, W), bool
    train_sleep_features.npy    # shape (N_windows, W, 9), float32
    train_sleep_masks.npy       # shape (N_windows, W), bool
    train_labels.npy            # shape (N_windows, W), int32
    train_label_masks.npy       # shape (N_windows, W), bool
    train_padding_masks.npy     # shape (N_windows, W), bool
    train_metadata.pkl          # List[Dict] with patient_id, sequence_name, window_end_day, window_days
```

### 8.4 `metadata.json`

Contains pipeline parameters and per-fold statistics:

```json
{
  "track": 1,
  "window_size": 7,
  "stride": 1,
  "n_folds": 9,
  "modality_dims": {"accel": 38, "gyr": 38, "hr": 26, "step": 10, "sleep": 9},
  "modality_feature_names": {
    "accel": ["x_mean", "x_std", ...],
    "gyr": ["x_mean", "x_std", ...],
    "hr": ["hr_mean", "hr_std", ...],
    "step": ["total_steps", ...],
    "sleep": ["total_sleep_min", ...]
  },
  "window_format": "per_modality_dict",
  "normalization": {
    "method": "per_patient_zscore",
    "fit_on": "train_sequences_only",
    "clip": [-5.0, 5.0],
    "nan_fill": 0.0
  },
  "folds": {
    "0": {
      "test_patient": "P1",
      "train_patients": ["P2", "P3", ...],
      "train_stats": {"n_windows": ..., "n_stable_days": ..., "n_relapse_days": ..., ...},
      "val_stats": {...},
      "test_stats": {...}
    },
    ...
  }
}
```

### 8.5 `patient_scalers.pkl`

A pickled `Dict[str, Dict[str, Dict[str, np.ndarray]]]`:

```python
{
    "P1": {
        "accel": {"mean": np.ndarray(38,), "std": np.ndarray(38,)},
        "gyr":   {"mean": np.ndarray(38,), "std": np.ndarray(38,)},
        "hr":    {"mean": np.ndarray(26,), "std": np.ndarray(26,)},
        "step":  {"mean": np.ndarray(10,), "std": np.ndarray(10,)},
        "sleep": {"mean": np.ndarray(9,),  "std": np.ndarray(9,)},
    },
    "P2": { ... },
    ...
}
```

---

## 9. Critical Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Window size** | 7 days | Captures weekly behavioral patterns and provides enough context for temporal models. Short enough to produce sufficient training samples per sequence. |
| **Stride** | 1 (every day) | Maximizes training data volume. Each day gets its own window with full context. |
| **Left-padding** (not skipping early days) | Zero-pad windows for the first W-1 days | Avoids discarding early days of a sequence. The `padding_mask` lets models distinguish real vs. padded positions. |
| **Per-patient normalization** | Z-score fitted on `train_*` only | Removes inter-patient variability. Fitting on stable periods means z-scores represent deviations from the patient's own stable baseline. |
| **Intra-day windowing** (5-minute bins) | Average statistics across 5-min windows per day | Aggregating at the 5-minute level before computing daily stats reduces noise and handles variable sampling rates within a day. |
| **Features per modality** (not concatenated) | Store as `Dict[str, ndarray]` | Enables modality-specific processing (e.g., separate transformer encoders) and modality dropout during training. |
| **NaN to 0** (not interpolation) | Replace NaN with 0.0 after z-scoring | Zero after z-scoring represents the mean value. Combined with modality masks, the model can learn to ignore missing days rather than receiving a possibly misleading interpolation. |
| **Clip to [-5, 5]** | Hard clip after z-score | Bounds the input range for numerical stability without discarding samples. |
| **Lomb-Scargle for HRV** | Use Lomb-Scargle instead of FFT | RR intervals are inherently unevenly spaced (each beat has a different duration). Lomb-Scargle handles this natively without resampling. |
| **Circular sleep encoding** | sin/cos of hour/24 | Preserves the circular topology of time-of-day. Two features (sin, cos) uniquely encode any time on the 24-hour clock. |
| **Sleep onset/wake rolling deviation** | 7-day circular deviation | Captures instability in sleep schedule relative to recent personal baseline, a clinically meaningful feature. |
| **Label handling for `train_*`** | Force label=0, mask=True | Challenge rules confirm `train_*` sequences are stable periods, so all days are labeled as stable even though the CSV only has zeros. |
| **Label handling for `test_*`** | Label=-1, mask=False | No ground-truth labels available at preprocessing time. The model should not compute loss on these. |
| **Extra-day bug fix** | Drop last row of `relapses.csv` | Each CSV has a spurious extra row. Dropping it aligns day indices with actual sensor data. |

---

## 10. Known Quirks

### 10.1 Extra-Day Bug in `relapses.csv`

Every `relapses.csv` file has one spurious extra row at the end that does not correspond to a real study day. The pipeline handles this by dropping the last row:

```python
relapses = seq.relapses.iloc[:-1].copy()
```

This is applied in `LOSOPreprocessor._get_labels_and_days()`.

### 10.2 Sleep Files Stored Separately

Sleep parquet files are not in the main sequence directories. They reside under:

```
data/original/track1/sleep_files/P{x}/{sequence}/sleep.parquet
```

The pipeline stages these into the inline sequence directories before loading. If the `sleep_files_dir` config parameter is not set or points to the wrong location, sleep features will be all-NaN.

### 10.3 Annotations Stored Separately

Similarly, `relapses.csv` files may be distributed in a separate directory:

```
data/original/track1/track_1_annotations/P{x}/{sequence}/relapses.csv
```

The `annotations_dir` config parameter controls staging of these files.

### 10.4 Missing Sensor Days

A `day_index` listed in `relapses.csv` may be absent from one or more modality parquets (e.g., the accelerometer was not worn that day). In such cases:
- The corresponding feature row is all-NaN (before normalization).
- The modality mask for that day is `False`.
- After normalization, NaN values become 0.0.

This means the model receives zero features for that modality on that day, with the modality mask indicating the data is missing.

### 10.5 Class Imbalance

Relapse days are a small minority of the dataset. A typical validation sequence might have approximately 20 relapse days vs. 60 stable days (roughly 3:1). Across the full training set of a LOSO fold, the imbalance can be much more extreme because `train_*` sequences contribute only stable days. The fold statistics in `metadata.json` report the exact stable/relapse counts per fold.

### 10.6 RR Interval Deduplication

The wearable device holds the last measured RR interval value until a new beat is detected. This creates runs of identical consecutive values in the RR time series. Before computing HRV metrics, the extractor removes consecutive duplicates:

```python
keep = np.concatenate([[True], rr_nz[1:] != rr_nz[:-1]])
rr_beats = rr_nz[keep]
```

Without this step, HRV metrics would be artificially deflated (low variability due to repeated values).

### 10.7 Zero RR Values

RR interval values of exactly 0 are excluded before any computation. The sensor reports 0 when no beat is detected.

### 10.8 Step Data Uses Different Day Index Column

Unlike other modalities that use `day_index`, the step parquet uses `start_date_index` as the grouping column. This is handled internally in `_extract_step()`.

### 10.9 Sleep Episodes Assigned by Wake Day

Sleep episodes are assigned to the day the person woke up (`end_date_index`), not the day they fell asleep. This means a sleep episode from 23:00 on day 5 to 07:00 on day 6 is assigned to day 6. Episodes with negative `end_date_index` (pre-study) are skipped.

### 10.10 Scipy Precision Warnings

The entry point script suppresses `RuntimeWarning: Precision loss occurred in moment calculation` from scipy, which occurs when computing skewness/kurtosis on near-constant windows:

```python
warnings.filterwarnings("ignore", message="Precision loss occurred in moment calculation")
```

### 10.11 SLURM Parallel Execution

The `preprocess_data.py` entry point supports three execution modes for cluster environments:

| Mode | Flag | Behavior |
|------|------|----------|
| Full (default) | (none) | Process all patients sequentially in one job |
| Per-patient | `--patient P3` | Process one patient, save intermediate pickle to `patients/P3.pkl` |
| Merge | `--merge` | Load all per-patient intermediates, build LOSO folds, save final output |

The per-patient mode enables SLURM array jobs where each task processes one patient in parallel, followed by a single merge job.
