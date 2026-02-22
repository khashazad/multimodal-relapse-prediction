# Dataset Format — Track 1 (Non-Psychotic Relapse Detection)

## Top-Level Layout

```
data/original/track1/
├── P1/ ... P9/            ← 9 patients
│   ├── train_0/, train_1/ [, train_2/]
│   ├── val_0/, val_1/ [, val_2/]
│   └── test_0/, test_1/ [, test_2/]
│
└── sleep_files/           ← sleep data stored SEPARATELY
    ├── P1/ ... P9/
    │   └── (same sequence structure, each containing sleep.parquet)
```

Each patient has 2–3 sequences per split (train/val/test). Each sequence represents a contiguous segment of many days of recording.

## Sequences Per Patient

| Patient | train | val | test |
|---------|-------|-----|------|
| P1 | train_0, train_1 | val_0, val_1 | test_0, test_1, test_2 |
| P2 | train_0, train_1, train_2 | val_0, val_1 | test_0, test_1 |
| P3 | train_0, train_1 | val_0, val_1 | test_0, test_1 |
| P4 | train_0, train_1 | val_0, val_1 | test_0, test_1 |
| P5 | train_0, train_1 | val_0, val_1 | test_0, test_1 |
| P6 | train_0, train_1, train_2 | val_0, val_1, val_2 | test_0, test_1 |
| P7 | train_0, train_1, train_2 | val_0, val_1 | test_0, test_1 |
| P8 | train_0, train_1, train_2 | val_0, val_1 | test_0, test_1 |
| P9 | train_0, train_1 | val_0 | test_0, test_1 |

## Files Per Sequence

Each sequence folder (e.g., `P1/train_0/`) contains:

| File | Format | Size (typical) | Description |
|------|--------|----------------|-------------|
| `linacc.parquet` | Parquet | ~1.6 GB | Linear accelerometer (high-frequency) |
| `gyr.parquet` | Parquet | ~900 MB | Gyroscope (high-frequency) |
| `hrm.parquet` | Parquet | ~165 MB | Heart rate monitor |
| `step.parquet` | Parquet | ~142 KB | Step/walking segments |
| `relapses.csv` | CSV | ~500 B | Per-day relapse labels |

Sleep data is in a **separate directory tree**: `sleep_files/P{x}/{sequence}/sleep.parquet`. No patient has `sleep.parquet` inline in their main sequence folder.

## Modality Formats

### 1. `linacc.parquet` — Linear Accelerometer

High-frequency 3-axis accelerometer readings (millions of rows per sequence).

| Column | Type | Description |
|--------|------|-------------|
| measurement columns (e.g., `x`, `y`, `z`) | float | Acceleration along each axis |
| `time` | string | Time of day (e.g., `00:00:00.050896`) |
| `day_index` | int | Day number within the sequence (0-indexed) |

### 2. `gyr.parquet` — Gyroscope

Same format as accelerometer. 3-axis angular velocity, millions of rows.

| Column | Type | Description |
|--------|------|-------------|
| measurement columns (e.g., `x`, `y`, `z`) | float | Angular velocity per axis |
| `time` | string | Time of day |
| `day_index` | int | Day number |

### 3. `hrm.parquet` — Heart Rate Monitor

Heart rate and RR-interval measurements.

| Column | Type | Description |
|--------|------|-------------|
| HR column (e.g., `heartRate`) | float | Heart rate in BPM |
| RR column (e.g., `rrInterval`) | float | RR interval in ms |
| `time` | string | Time of day |
| `day_index` | int | Day number |

### 4. `step.parquet` — Walking/Step Segments

One row per walking segment (not per step). A single day can have many segments.

| Column | Type | Description |
|--------|------|-------------|
| `totalSteps` | int | Total steps in this segment |
| `stepsWalking` | int | Steps while walking |
| `stepsRunning` | int | Steps while running |
| `distance` | float | Distance traveled (meters) |
| `calories` | float | Calories burned |
| `start_time` | string | Segment start time |
| `end_time` | string | Segment end time |
| `start_date_index` | int | Day index of start |
| `end_date_index` | int | Day index of end |

### 5. `sleep.parquet` — Sleep Segments (in `sleep_files/`)

One row per sleep episode. Sleep can span midnight (start on one day, end on the next).

| Column | Type | Description |
|--------|------|-------------|
| `start_time` | string | Sleep start time (e.g., `22:55:10.309725`) |
| `end_time` | string | Sleep end time (e.g., `07:14:36.473329`) |
| `start_date_index` | int | Day index of start (-1 means previous day) |
| `end_date_index` | int | Day index of end |

## Label Format: `relapses.csv`

There are **three distinct formats** depending on the split:

### Train sequences — has labels, all zeros

```csv
relapse,day_index
0,0
0,1
0,2
...
```

Columns: `relapse`, `day_index`. All `relapse` values are `0` (confirmed stable — no relapse episodes occur during training periods).

### Validation sequences — has labels with 0s and 1s

```csv
relapse,day_index
0,0
0,1
...
1,61
1,62
...
```

Columns: `relapse`, `day_index`. Contains both `0` (stable) and `1` (relapse) days. Example: P9/val_0 has 83 days, with days 61–81 marked as relapse.

### Test sequences — NO relapse column (labels withheld)

```csv
day_index
0
1
2
...
```

Column: `day_index` **only**. No `relapse` column exists. These are the days you must predict for. The labels were withheld for challenge evaluation.

## Important Notes

1. **Sleep is stored separately** in `sleep_files/` rather than alongside the other modalities. Any data loading code must check `sleep_files/P{x}/{sequence}/sleep.parquet` in addition to the main sequence folder.

2. **Test labels are missing entirely** — not just set to NaN, but the `relapse` column does not exist in test `relapses.csv` files. For supervised LOSO training, only `train_*` (label=0) and `val_*` (labels 0/1) sequences from other patients provide supervised signal. Test sequences from other patients cannot be used for supervised training since they have no labels.

3. **The extra-day bug**: per the challenge notes, each `relapses.csv` has one extra row at the bottom that should be ignored. Quote: *"The relapses.csv falsely contains one extra day at the bottom of the file. This is a mistake — participants should not predict the final day in test datasets."*

4. **Missing days are possible**: some `day_index` values may be absent from a modality's parquet file (the sensor was not recording), but you still need to predict for every day listed in `relapses.csv`.

5. **Imbalanced classes**: relapse days are a minority. For example, P9/val_0 has ~60 stable days and ~20 relapse days.

6. **High-frequency vs. summary data**: `linacc` and `gyr` are raw sensor streams at high sampling rates (millions of rows). `hrm` is also high-frequency but smaller. In contrast, `step` and `sleep` are pre-aggregated summary data with one row per segment.
