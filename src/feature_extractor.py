"""
Feature extractor for multimodal relapse prediction.

Extracts per-day feature vectors for each of the five sensor modalities:
  accel  — 38 features (statistical + activity + frequency-domain)
  gyr    — 38 features (identical pipeline to accel)
  hr     — 26 features (HR/RR stats + HRV time/freq domain + Poincaré + coverage)
  step   — 10 features (daily aggregates + temporal pattern)
  sleep  —  9 features (summary + circular-encoded timing + rolling deviation)

All extraction operates at sequence level (all days at once) using vectorised
pandas groupby to avoid row-level Python loops over large parquet files.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import lombscargle
from scipy.stats import kurtosis as sp_kurtosis
from scipy.stats import skew as sp_skew

from .data_loader import SequenceData

# ---------------------------------------------------------------------------
# Feature dimension registry
# ---------------------------------------------------------------------------

MODALITY_DIMS: Dict[str, int] = {
    "accel": 38,
    "gyr":   38,
    "hr":    26,
    "step":  10,
    "sleep":  9,
}

ACCEL_FEATURE_NAMES: List[str] = [
    # Per-axis stats: 7 stats × 4 signals (x, y, z, mag) = 28
    "x_mean", "x_std", "x_min", "x_max", "x_median", "x_skew", "x_kurt",
    "y_mean", "y_std", "y_min", "y_max", "y_median", "y_skew", "y_kurt",
    "z_mean", "z_std", "z_min", "z_max", "z_median", "z_skew", "z_kurt",
    "mag_mean", "mag_std", "mag_min", "mag_max", "mag_median", "mag_skew", "mag_kurt",
    # Activity-level (4)
    "total_energy", "zero_crossing_rate", "pct_active", "pct_sedentary",
    # Frequency-domain (6)
    "dominant_freq", "spectral_entropy",
    "power_0_1hz", "power_1_3hz", "power_3_8hz", "power_8plus_hz",
]  # 38 total

GYR_FEATURE_NAMES: List[str] = ACCEL_FEATURE_NAMES  # same layout

HR_FEATURE_NAMES: List[str] = [
    # HR stats (7)
    "hr_mean", "hr_std", "hr_min", "hr_max", "hr_median", "hr_skew", "hr_kurt",
    # RR stats (7)
    "rr_mean", "rr_std", "rr_min", "rr_max", "rr_median", "rr_skew", "rr_kurt",
    # HRV time-domain (4)
    "sdnn", "rmssd", "pnn50", "hrv_tri_index",
    # HRV frequency-domain (4)
    "lf_power", "hf_power", "lf_hf_ratio", "total_power",
    # Poincaré (3)
    "sd1", "sd2", "sd1_sd2",
    # Coverage (1)
    "coverage_fraction",
]  # 26 total

STEP_FEATURE_NAMES: List[str] = [
    "total_steps", "walking_steps", "running_steps", "distance", "calories",
    "n_segments",
    "first_activity_hour", "last_activity_hour", "longest_gap_hours", "gap_std_hours",
]  # 10 total

SLEEP_FEATURE_NAMES: List[str] = [
    "total_sleep_min", "n_episodes",
    "sleep_onset_sin", "sleep_onset_cos",
    "wake_time_sin",   "wake_time_cos",
    "longest_bout_hours",
    "onset_7day_rolling_deviation", "wake_7day_rolling_deviation",
]  # 9 total

MODALITY_FEATURE_NAMES: Dict[str, List[str]] = {
    "accel": ACCEL_FEATURE_NAMES,
    "gyr":   GYR_FEATURE_NAMES,
    "hr":    HR_FEATURE_NAMES,
    "step":  STEP_FEATURE_NAMES,
    "sleep": SLEEP_FEATURE_NAMES,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _time_to_seconds(s: pd.Series) -> pd.Series:
    """Parse time column → float seconds.

    Handles both 'HH:MM:SS.ffffff' strings and Python datetime.time objects
    (the latter is how parquet files store time columns).
    """
    if len(s) == 0:
        return pd.Series([], dtype=np.float32)
    first = s.iloc[0]
    if isinstance(first, str):
        h   = s.str[:2].astype(np.int32)
        m   = s.str[3:5].astype(np.int32)
        sec = s.str[6:].astype(np.float32)
        return (h * 3600 + m * 60 + sec).astype(np.float32)
    # datetime.time objects stored in object-dtype columns.
    # np.fromiter with pre-allocated count is significantly faster than .apply().
    arr = s.to_numpy()
    secs = np.fromiter(
        (t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1_000_000.0
         for t in arr),
        dtype=np.float64, count=len(arr),
    )
    return pd.Series(secs.astype(np.float32), index=s.index)


def _seven_stats(values: np.ndarray) -> List[float]:
    """Return [mean, std, min, max, median, skewness, kurtosis] or NaNs."""
    if len(values) < 2:
        return [np.nan] * 7
    return [
        float(np.mean(values)),
        float(np.std(values, ddof=1)),
        float(np.min(values)),
        float(np.max(values)),
        float(np.median(values)),
        float(sp_skew(values)),
        float(sp_kurtosis(values)),
    ]


def _remove_rr_outliers(rr: np.ndarray) -> np.ndarray:
    """Remove RR intervals more than 3 standard deviations from the mean."""
    if len(rr) < 3:
        return rr
    mu, sigma = np.mean(rr), np.std(rr)
    cleaned = rr[np.abs(rr - mu) <= 3 * sigma]
    return cleaned if len(cleaned) >= 2 else rr


def _circ_diff(a: float, b: float, period: float = 24.0) -> float:
    """Circular absolute difference on a period-hour clock."""
    d = abs(a - b) % period
    return min(d, period - d)


def _empty_features(n_days: int, n_feats: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return an all-NaN feature array and an all-False mask."""
    return (
        np.full((n_days, n_feats), np.nan, dtype=np.float32),
        np.zeros(n_days, dtype=bool),
    )


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """Extract per-day features for all five modalities of a patient sequence.

    Returns a dict keyed by modality name, where each value is::

        {
            'features': np.ndarray of shape (N_days, F),
            'mask':     np.ndarray of shape (N_days,) bool
                        True  → modality has data for that day
                        False → all-zeros after normalisation
        }
    """

    MODALITY_SIZES = MODALITY_DIMS

    def __init__(
        self,
        window_size_minutes: int = 5,
        sample_rate_imu: int = 20,
        sample_rate_hr: int = 5,
        coverage_threshold: float = 0.25,
    ) -> None:
        self.win_secs    = window_size_minutes * 60
        self.fs_imu      = sample_rate_imu
        self.fs_hr       = sample_rate_hr
        # Minimum windows per day to consider coverage sufficient
        self.min_windows = int(coverage_threshold * (24 * 60 / window_size_minutes))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_all_modalities_for_sequence(
        self,
        seq: SequenceData,
        day_list: List[int],
    ) -> Dict[str, Dict]:
        """Extract features for every day in *day_list* from *seq*.

        Parameters
        ----------
        seq:      loaded SequenceData (modalities may be None if unavailable)
        day_list: sorted list of day indices to extract (from relapses.csv)

        Returns
        -------
        Dict with keys 'accel', 'gyr', 'hr', 'step', 'sleep'.
        """
        n = len(day_list)
        results: Dict[str, Dict] = {}

        # --- Accelerometer ------------------------------------------------
        if seq.linacc is not None and len(seq.linacc):
            f, m = self._extract_imu(seq.linacc, day_list)
        else:
            f, m = _empty_features(n, MODALITY_DIMS["accel"])
        results["accel"] = {"features": f, "mask": m}

        # --- Gyroscope ----------------------------------------------------
        if seq.gyr is not None and len(seq.gyr):
            f, m = self._extract_imu(seq.gyr, day_list)
        else:
            f, m = _empty_features(n, MODALITY_DIMS["gyr"])
        results["gyr"] = {"features": f, "mask": m}

        # --- Heart rate ---------------------------------------------------
        if seq.hrm is not None and len(seq.hrm):
            f, m = self._extract_hr(seq.hrm, day_list)
        else:
            f, m = _empty_features(n, MODALITY_DIMS["hr"])
        results["hr"] = {"features": f, "mask": m}

        # --- Steps --------------------------------------------------------
        if seq.step is not None and len(seq.step):
            f, m = self._extract_step(seq.step, day_list)
        else:
            f, m = _empty_features(n, MODALITY_DIMS["step"])
        results["step"] = {"features": f, "mask": m}

        # --- Sleep --------------------------------------------------------
        if seq.sleep is not None and len(seq.sleep):
            f, m = self._extract_sleep(seq.sleep, day_list)
        else:
            f, m = _empty_features(n, MODALITY_DIMS["sleep"])
        results["sleep"] = {"features": f, "mask": m}

        return results

    # ------------------------------------------------------------------
    # IMU (accel / gyr)
    # ------------------------------------------------------------------

    def _extract_imu(
        self, df: pd.DataFrame, day_list: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract 38 features per day from an IMU parquet DataFrame.

        Uses vectorised pandas groupby for the 28 statistical features and 4
        activity features, falling back to per-window loops only for the 6
        FFT features (which are already fast).
        """
        n = len(day_list)
        features = np.full((n, MODALITY_DIMS["accel"]), np.nan, dtype=np.float32)
        mask = np.zeros(n, dtype=bool)

        meas_cols = [c for c in df.columns if c not in ("time", "day_index")]
        if not meas_cols:
            return features, mask

        day_index_set = set(day_list)
        day_pos = {d: i for i, d in enumerate(day_list)}

        # Pre-filter to only relevant days before any heavy work
        df = df[df["day_index"].isin(day_index_set)].copy()
        if df.empty:
            return features, mask

        # Compute magnitude and window bin in one pass
        df["_mag"] = np.sqrt((df[meas_cols] ** 2).sum(axis=1))
        df["_win"] = (
            _time_to_seconds(df["time"]) // self.win_secs
        ).astype(np.int32)

        all_cols = meas_cols + ["_mag"]
        grp_keys = ["day_index", "_win"]

        # --- Statistical features (28): vectorised groupby agg ----------
        grp = df.groupby(grp_keys)[all_cols]
        stat_win = grp.agg(
            ["mean", "std", "min", "max", "median", sp_skew, sp_kurtosis]
        )
        # Mask windows with fewer than 2 samples (matching _seven_stats guard)
        counts = grp.size()  # Series indexed by (day_index, _win)
        small_mask = counts < 2
        if small_mask.any():
            stat_win.loc[small_mask] = np.nan

        # Reorder columns: for each signal col, the 7 stats in order
        # stat_win columns are MultiIndex (col, agg_name)
        ordered_stat_cols = []
        for col in all_cols:
            for agg in ["mean", "std", "min", "max", "median", "skew", "kurtosis"]:
                ordered_stat_cols.append((col, agg))
        stat_win = stat_win[ordered_stat_cols]

        # Average stats across windows per day
        stat_day = stat_win.groupby(level="day_index").mean()

        # --- Activity features (4): vectorised --------------------------
        df["_mag2"] = df["_mag"] ** 2
        df["_active"] = (df["_mag"] > 1.5).astype(np.float32)
        df["_sedentary"] = (df["_mag"] < 0.2).astype(np.float32)

        act_grp = df.groupby(grp_keys)
        energy_win = act_grp["_mag2"].sum()
        pct_act_win = act_grp["_active"].mean()
        pct_sed_win = act_grp["_sedentary"].mean()

        # ZCR per window — needs apply (sign-change counting)
        def _zcr(mag_series: pd.Series) -> float:
            mag = mag_series.values
            if len(mag) < 2:
                return np.nan
            return float(
                np.sum(np.diff(np.sign(mag - mag.mean())) != 0)
                / (len(mag) - 1)
            )

        zcr_win = act_grp["_mag"].apply(_zcr)

        # Combine activity features and average per day
        act_win = pd.DataFrame({
            "energy": energy_win,
            "zcr": zcr_win,
            "pct_active": pct_act_win,
            "pct_sedentary": pct_sed_win,
        })
        act_day = act_win.groupby(level="day_index").mean()

        # --- FFT features (6): per-window loop (fast enough) ------------
        fft_rows: Dict[Tuple, List[float]] = {}
        for (day_idx, win_id), win_df in df.groupby(grp_keys):
            fft_rows[(day_idx, win_id)] = self._fft_features(win_df["_mag"].values)

        fft_df = pd.DataFrame.from_dict(fft_rows, orient="index",
                                         columns=["dom_freq", "spec_entropy",
                                                   "p01", "p13", "p38", "p8plus"])
        fft_df.index = pd.MultiIndex.from_tuples(fft_df.index,
                                                   names=["day_index", "_win"])
        fft_day = fft_df.groupby(level="day_index").mean()

        # --- Assemble per-day feature vectors ---------------------------
        for day_idx in stat_day.index:
            if day_idx not in day_pos:
                continue
            pos = day_pos[day_idx]
            mask[pos] = True

            stat_vals = stat_day.loc[day_idx].values.astype(np.float32)
            act_vals = (act_day.loc[day_idx].values.astype(np.float32)
                        if day_idx in act_day.index
                        else np.full(4, np.nan, dtype=np.float32))
            fft_vals = (fft_day.loc[day_idx].values.astype(np.float32)
                        if day_idx in fft_day.index
                        else np.full(6, np.nan, dtype=np.float32))

            features[pos] = np.concatenate([stat_vals, act_vals, fft_vals])

        return features, mask

    def _fft_features(self, mag: np.ndarray) -> List[float]:
        """Compute 6 frequency-domain features from a magnitude window."""
        if len(mag) < 10:
            return [np.nan] * 6

        mag_demeaned = mag - mag.mean()
        fft_power = np.abs(np.fft.rfft(mag_demeaned)) ** 2
        freqs = np.fft.rfftfreq(len(mag), d=1.0 / self.fs_imu)

        # Spectral entropy
        pnorm = fft_power / (fft_power.sum() + 1e-12)
        entropy = float(-np.sum(pnorm * np.log(pnorm + 1e-12)))

        # Dominant frequency (skip DC bin 0)
        nz_power = fft_power[1:]
        nz_freqs = freqs[1:]
        dom_freq = float(nz_freqs[np.argmax(nz_power)]) if len(nz_power) > 0 else np.nan

        def band_power(lo: float, hi: float) -> float:
            return float(fft_power[(freqs >= lo) & (freqs < hi)].sum())

        return [
            dom_freq,
            entropy,
            band_power(0.0, 1.0),
            band_power(1.0, 3.0),
            band_power(3.0, 8.0),
            float(fft_power[freqs >= 8.0].sum()),
        ]

    # ------------------------------------------------------------------
    # Heart rate
    # ------------------------------------------------------------------

    def _extract_hr(
        self, df: pd.DataFrame, day_list: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract 26 features per day from an HRM parquet DataFrame."""
        n = len(day_list)
        features = np.full((n, MODALITY_DIMS["hr"]), np.nan, dtype=np.float32)
        mask = np.zeros(n, dtype=bool)

        # Detect column names
        cols = df.columns.tolist()
        hr_col: Optional[str] = next(
            (c for c in cols if "heart" in c.lower() or c.lower() == "hr"), None
        )
        rr_col: Optional[str] = next(
            (c for c in cols if "rr" in c.lower() or "interval" in c.lower()), None
        )

        df = df.copy()
        df["_win"] = (
            _time_to_seconds(df["time"]) // self.win_secs
        ).astype(np.int32)

        day_index_set = set(day_list)
        day_pos = {d: i for i, d in enumerate(day_list)}

        for day_idx, day_df in df.groupby("day_index"):
            if day_idx not in day_index_set:
                continue

            pos = day_pos[day_idx]
            mask[pos] = True

            n_windows = day_df["_win"].nunique()
            coverage = n_windows / 288.0

            # HR stats (7)
            if hr_col is not None:
                hr_vals = day_df[hr_col].dropna().values.astype(np.float64)
                hr_stats = _seven_stats(hr_vals)
            else:
                hr_stats = [np.nan] * 7

            # RR stats (7) — use all non-zero RR values
            if rr_col is not None:
                rr_all = day_df[rr_col].dropna().values.astype(np.float64)
                rr_nz  = rr_all[rr_all > 0]
                rr_stats = _seven_stats(rr_nz)
            else:
                rr_nz = np.array([], dtype=np.float64)
                rr_stats = [np.nan] * 7

            # HRV uses beat-to-beat transitions only: deduplicate consecutive
            # repeated values (the device holds the last RR until a new beat).
            if len(rr_nz) >= 2:
                keep = np.concatenate([[True], rr_nz[1:] != rr_nz[:-1]])
                rr_beats = rr_nz[keep]
            else:
                rr_beats = rr_nz

            # HRV time-domain (4)
            hrv_td = self._hrv_time_domain(rr_beats)

            # HRV frequency-domain (4)
            rr_clean = _remove_rr_outliers(rr_beats) if len(rr_beats) >= 4 else rr_beats
            hrv_fd = self._hrv_freq_domain(rr_clean)

            # Poincaré (3)
            poincare = self._poincare(rr_clean)

            day_feat = np.array(
                hr_stats + rr_stats + hrv_td + hrv_fd + poincare + [coverage],
                dtype=np.float32,
            )
            features[pos] = day_feat

        return features, mask

    def _hrv_time_domain(self, rr: np.ndarray) -> List[float]:
        """SDNN, RMSSD, pNN50, HRV triangular index."""
        rr_clean = _remove_rr_outliers(rr) if len(rr) >= 4 else rr
        if len(rr_clean) < 4:
            return [np.nan] * 4

        diff = np.diff(rr_clean)
        sdnn  = float(np.std(rr_clean, ddof=1))
        rmssd = float(np.sqrt(np.mean(diff ** 2)))
        pnn50 = float(np.mean(np.abs(diff) > 50.0) * 100.0)

        bw = 1000.0 / 128.0  # 7.8125 ms bins
        bins = np.arange(rr_clean.min(), rr_clean.max() + bw, bw)
        counts, _ = np.histogram(rr_clean, bins=bins)
        tri = float(len(rr_clean) / (counts.max() + 1e-12))

        return [sdnn, rmssd, pnn50, tri]

    def _hrv_freq_domain(self, rr_clean: np.ndarray) -> List[float]:
        """LF power, HF power, LF/HF ratio, total power via Lomb-Scargle."""
        if len(rr_clean) < 10:
            return [np.nan] * 4

        t = np.cumsum(rr_clean) / 1000.0  # ms → seconds
        t -= t[0]
        sig = (rr_clean - rr_clean.mean()).astype(np.float64)

        lf_ang = np.linspace(0.04,  0.15,  100) * 2 * np.pi
        hf_ang = np.linspace(0.15,  0.40,  200) * 2 * np.pi
        tp_ang = np.linspace(0.003, 0.40,  500) * 2 * np.pi

        try:
            lf_p = float(np.trapz(lombscargle(t, sig, lf_ang), lf_ang / (2 * np.pi)))
            hf_p = float(np.trapz(lombscargle(t, sig, hf_ang), hf_ang / (2 * np.pi)))
            tp_p = float(np.trapz(lombscargle(t, sig, tp_ang), tp_ang / (2 * np.pi)))
            lf_hf = lf_p / (hf_p + 1e-12)
        except Exception:  # noqa: BLE001
            return [np.nan] * 4

        return [lf_p, hf_p, lf_hf, tp_p]

    def _poincare(self, rr_clean: np.ndarray) -> List[float]:
        """SD1, SD2, SD1/SD2 from Poincaré plot."""
        if len(rr_clean) < 4:
            return [np.nan] * 3

        rr1, rr2 = rr_clean[:-1], rr_clean[1:]
        sd1 = float(np.std((rr2 - rr1) / np.sqrt(2), ddof=1))
        sd2 = float(np.std((rr2 + rr1) / np.sqrt(2), ddof=1))
        return [sd1, sd2, sd1 / (sd2 + 1e-12)]

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _extract_step(
        self, df: pd.DataFrame, day_list: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract 10 features per day from a step parquet DataFrame."""
        n = len(day_list)
        features = np.full((n, MODALITY_DIMS["step"]), np.nan, dtype=np.float32)
        mask = np.zeros(n, dtype=bool)

        day_index_set = set(day_list)
        day_pos = {d: i for i, d in enumerate(day_list)}

        for day_idx, day_df in df.groupby("start_date_index"):
            if day_idx not in day_index_set:
                continue

            pos = day_pos[day_idx]
            mask[pos] = True

            # Aggregate features
            total_steps   = float(day_df["totalSteps"].sum())   if "totalSteps"   in day_df else np.nan
            walking_steps = float(day_df["stepsWalking"].sum()) if "stepsWalking" in day_df else np.nan
            running_steps = float(day_df["stepsRunning"].sum()) if "stepsRunning" in day_df else np.nan
            distance      = float(day_df["distance"].sum())     if "distance"     in day_df else np.nan
            calories      = float(day_df["calories"].sum())     if "calories"     in day_df else np.nan
            n_segments    = float(len(day_df))

            # Temporal features
            if "start_time" in day_df.columns and "end_time" in day_df.columns:
                start_hrs = _time_to_seconds(day_df["start_time"]) / 3600.0
                end_hrs   = _time_to_seconds(day_df["end_time"])   / 3600.0

                first_hour = float(start_hrs.min())
                last_hour  = float(end_hrs.max())

                order     = start_hrs.argsort().values
                s_sorted  = start_hrs.values[order]
                e_sorted  = end_hrs.values[order]
                gaps      = np.maximum(s_sorted[1:] - e_sorted[:-1], 0.0)
                longest_gap = float(gaps.max()) if len(gaps) else 0.0
                gap_std     = float(gaps.std()) if len(gaps) else 0.0
            else:
                first_hour = last_hour = longest_gap = gap_std = np.nan

            features[pos] = np.array(
                [total_steps, walking_steps, running_steps, distance, calories,
                 n_segments, first_hour, last_hour, longest_gap, gap_std],
                dtype=np.float32,
            )

        return features, mask

    # ------------------------------------------------------------------
    # Sleep
    # ------------------------------------------------------------------

    def _extract_sleep(
        self, df: pd.DataFrame, day_list: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract 9 features per day from a sleep parquet DataFrame."""
        n = len(day_list)
        features = np.full((n, MODALITY_DIMS["sleep"]), np.nan, dtype=np.float32)
        mask = np.zeros(n, dtype=bool)

        # Parse all episodes once
        # Assign each episode to the day the person woke up (end_date_index)
        episodes_by_day: Dict[int, List[Dict]] = {}

        for _, row in df.iterrows():
            end_day = int(row["end_date_index"])
            if end_day < 0:
                continue  # pre-study episode

            try:
                start_s = self._sleep_time_to_secs(str(row["start_time"]))
                end_s   = self._sleep_time_to_secs(str(row["end_time"]))
            except Exception:  # noqa: BLE001
                continue

            if end_s < start_s:
                dur_hrs = (86400.0 - start_s + end_s) / 3600.0
            else:
                dur_hrs = (end_s - start_s) / 3600.0

            onset_hr = start_s / 3600.0
            wake_hr  = end_s   / 3600.0

            if end_day not in episodes_by_day:
                episodes_by_day[end_day] = []
            episodes_by_day[end_day].append(
                {"dur": dur_hrs, "onset": onset_hr, "wake": wake_hr}
            )

        # Per-day summary features + raw onset/wake for rolling stats
        onset_hours = np.full(n, np.nan)
        wake_hours  = np.full(n, np.nan)

        day_pos = {d: i for i, d in enumerate(day_list)}

        for day_idx in day_list:
            eps = episodes_by_day.get(day_idx, [])
            pos = day_pos[day_idx]

            if not eps:
                features[pos, 0] = 0.0   # total_sleep_min
                features[pos, 1] = 0.0   # n_episodes
                features[pos, 6] = 0.0   # longest_bout_hours
                # timing features stay NaN; rolling deviations computed later
                continue

            mask[pos] = True
            durs = [e["dur"] for e in eps]
            total_min     = sum(durs) * 60.0
            n_episodes    = len(eps)
            longest_bout  = max(durs)

            # Main sleep = longest episode
            main_idx  = int(np.argmax(durs))
            onset_hr  = eps[main_idx]["onset"]
            wake_hr   = eps[main_idx]["wake"]

            onset_hours[pos] = onset_hr
            wake_hours[pos]  = wake_hr

            onset_sin = np.sin(2 * np.pi * onset_hr / 24.0)
            onset_cos = np.cos(2 * np.pi * onset_hr / 24.0)
            wake_sin  = np.sin(2 * np.pi * wake_hr  / 24.0)
            wake_cos  = np.cos(2 * np.pi * wake_hr  / 24.0)

            features[pos, :7] = [
                total_min, n_episodes,
                onset_sin, onset_cos,
                wake_sin,  wake_cos,
                longest_bout,
            ]

        # 7-day rolling deviation (circular, computed in chronological order)
        for i in range(n):
            # onset deviation
            if not np.isnan(onset_hours[i]):
                prior = onset_hours[max(0, i - 7):i]
                valid = prior[~np.isnan(prior)]
                if len(valid):
                    features[i, 7] = _circ_diff(onset_hours[i], float(valid.mean()))

            # wake deviation
            if not np.isnan(wake_hours[i]):
                prior = wake_hours[max(0, i - 7):i]
                valid = prior[~np.isnan(prior)]
                if len(valid):
                    features[i, 8] = _circ_diff(wake_hours[i], float(valid.mean()))

        return features, mask

    @staticmethod
    def _sleep_time_to_secs(t: str) -> float:
        """Parse 'HH:MM:SS.ffffff' to float seconds (single string, not vectorised)."""
        parts = t.split(":")
        h = int(parts[0])
        m = int(parts[1])
        s = float(parts[2])
        return h * 3600.0 + m * 60.0 + s
