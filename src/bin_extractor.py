"""
Bin extractor for CNN+LSTM architecture.

Converts raw high-frequency sensor signals into fixed-size 5-minute bin
representations suitable for 1D CNN processing.

Produces per-day arrays of shape (n_bins, C) where n_bins=288 (24h / 5min):
  linacc/gyr: 8 channels (X_mean, X_std, Y_mean, Y_std, Z_mean, Z_std, mag_mean, mag_std)
  hrm:        4 channels (hr_mean, hr_std, rr_mean, rr_std)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .feature_extractor import _time_to_seconds

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMU_CHANNELS = 8  # X_mean, X_std, Y_mean, Y_std, Z_mean, Z_std, mag_mean, mag_std
HR_CHANNELS = 4  # hr_mean, hr_std, rr_mean, rr_std
MIN_BIN_COVERAGE = 0.10  # at least 10% of bins must have data for a valid day


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class BinExtractor:
    """Extract fixed-size bin representations from raw sensor signals."""

    def __init__(self, bin_minutes: int = 5) -> None:
        self.bin_minutes = bin_minutes
        self.secs_per_bin = bin_minutes * 60
        self.n_bins = 24 * 60 // bin_minutes  # 288
        self.min_bins = int(MIN_BIN_COVERAGE * self.n_bins)

    def extract_imu_bins(
        self, df: pd.DataFrame, day_list: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract binned IMU features from a linacc or gyr DataFrame.

        Parameters
        ----------
        df : DataFrame with columns [X, Y, Z, time, day_index]
        day_list : sorted list of day indices

        Returns
        -------
        bins : (n_days, n_bins, 8) float32
        mask : (n_days,) bool — True if day has >= 10% bin coverage
        """
        n_days = len(day_list)
        bins = np.zeros((n_days, self.n_bins, IMU_CHANNELS), dtype=np.float32)
        mask = np.zeros(n_days, dtype=bool)

        day_set = set(day_list)
        day_pos = {d: i for i, d in enumerate(day_list)}

        df = df[df["day_index"].isin(day_set)].copy()
        if df.empty:
            return bins, mask

        # Compute bin index and magnitude
        secs = _time_to_seconds(df["time"])
        df["_bin"] = (secs // self.secs_per_bin).astype(np.int32).clip(0, self.n_bins - 1)
        df["_mag"] = np.sqrt(df["X"] ** 2 + df["Y"] ** 2 + df["Z"] ** 2)

        # Aggregate per (day, bin): mean + std for each of X, Y, Z, mag
        stats = (
            df.groupby(["day_index", "_bin"])[["X", "Y", "Z", "_mag"]]
            .agg(["mean", "std"])
            .fillna(0)
            .astype(np.float32)
        )
        # stats columns: (X,mean), (X,std), (Y,mean), (Y,std), ... (8 total)

        day_indices = stats.index.get_level_values("day_index").values
        bin_indices = stats.index.get_level_values("_bin").values
        positions = np.array([day_pos.get(d, -1) for d in day_indices])
        valid = positions >= 0

        bins[positions[valid], bin_indices[valid]] = stats.values[valid]

        # Mask: count unique bins with data per day
        bin_counts = df.groupby("day_index")["_bin"].nunique()
        for day_idx, count in bin_counts.items():
            if day_idx in day_pos and count >= self.min_bins:
                mask[day_pos[day_idx]] = True

        return bins, mask

    def extract_hr_bins(
        self, df: pd.DataFrame, day_list: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract binned heart rate features.

        Parameters
        ----------
        df : DataFrame with columns [heartRate, rRInterval, time, day_index]
        day_list : sorted list of day indices

        Returns
        -------
        bins : (n_days, n_bins, 4) float32
        mask : (n_days,) bool
        """
        n_days = len(day_list)
        bins = np.zeros((n_days, self.n_bins, HR_CHANNELS), dtype=np.float32)
        mask = np.zeros(n_days, dtype=bool)

        day_set = set(day_list)
        day_pos = {d: i for i, d in enumerate(day_list)}

        df = df[df["day_index"].isin(day_set)].copy()
        if df.empty:
            return bins, mask

        # Detect column names (same pattern as FeatureExtractor)
        hr_col: Optional[str] = next(
            (c for c in df.columns if "heart" in c.lower() or c.lower() == "hr"), None
        )
        rr_col: Optional[str] = next(
            (c for c in df.columns if "rr" in c.lower() or "interval" in c.lower()), None
        )

        cols = [c for c in [hr_col, rr_col] if c is not None]
        if not cols:
            return bins, mask

        secs = _time_to_seconds(df["time"])
        df["_bin"] = (secs // self.secs_per_bin).astype(np.int32).clip(0, self.n_bins - 1)

        stats = (
            df.groupby(["day_index", "_bin"])[cols]
            .agg(["mean", "std"])
            .fillna(0)
            .astype(np.float32)
        )

        day_indices = stats.index.get_level_values("day_index").values
        bin_indices = stats.index.get_level_values("_bin").values
        positions = np.array([day_pos.get(d, -1) for d in day_indices])
        valid = positions >= 0

        # Map stat columns to channel indices
        # hr_mean=0, hr_std=1, rr_mean=2, rr_std=3
        stat_cols = stats.columns.tolist()
        for i, (col_name, agg_name) in enumerate(stat_cols):
            ch_idx = None
            if col_name == hr_col:
                ch_idx = 0 if agg_name == "mean" else 1
            elif col_name == rr_col:
                ch_idx = 2 if agg_name == "mean" else 3
            if ch_idx is not None:
                bins[positions[valid], bin_indices[valid], ch_idx] = stats.values[
                    valid, i
                ]

        # Mask
        bin_counts = df.groupby("day_index")["_bin"].nunique()
        for day_idx, count in bin_counts.items():
            if day_idx in day_pos and count >= self.min_bins:
                mask[day_pos[day_idx]] = True

        return bins, mask
