"""
Preprocessing pipeline for CNN+LSTM architecture with LOSO cross-validation.

Similar to preprocess_loso.py but produces binned signal representations
for CNN modalities (linacc, gyr, hrm) and hand-crafted features for
episodic modalities (step, sleep).

Window dict format
------------------
{
    'patient_id':     str,
    'sequence_name':  str,
    'window_end_day': int,
    'window_days':    List[int],
    # CNN modalities: binned signals
    'linacc_bins':    np.ndarray (W, 288, 8),
    'gyr_bins':       np.ndarray (W, 288, 8),
    'hrm_bins':       np.ndarray (W, 288, 4),
    # FC modalities: hand-crafted features
    'step_features':  np.ndarray (W, 10),
    'sleep_features': np.ndarray (W, 9),
    # Per-modality masks
    'linacc_mask':    np.ndarray (W,) bool,
    'gyr_mask':       np.ndarray (W,) bool,
    'hrm_mask':       np.ndarray (W,) bool,
    'step_mask':      np.ndarray (W,) bool,
    'sleep_mask':     np.ndarray (W,) bool,
    # General
    'padding_mask':   np.ndarray (W,) bool,
    'labels':         np.ndarray (W,) int32,
    'label_mask':     np.ndarray (W,) bool,
}
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .bin_extractor import HR_CHANNELS, IMU_CHANNELS, BinExtractor
from .data_loader import MultimodalDataLoader, SequenceData
from .feature_extractor import MODALITY_DIMS, FeatureExtractor
from .preprocess_loso import LOSOPreprocessor

# Modalities handled by CNN (bins) vs FC (features)
CNN_MODALITIES = {"linacc": IMU_CHANNELS, "gyr": IMU_CHANNELS, "hrm": HR_CHANNELS}
FC_MODALITIES = {"step": MODALITY_DIMS["step"], "sleep": MODALITY_DIMS["sleep"]}


class CNNLOSOPreprocessor:
    """LOSO preprocessor for CNN+LSTM architecture.

    Reuses the data loader and LOSO fold logic from LOSOPreprocessor,
    but replaces feature extraction with bin extraction for continuous
    signal modalities.
    """

    def __init__(
        self,
        data_root: str,
        track: int = 1,
        window_size: int = 7,
        stride: int = 1,
        output_dir: str = "data/processed_cnn",
        bin_extractor: Optional[BinExtractor] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        sleep_files_dir: Optional[str] = None,
        annotations_dir: Optional[str] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.track = track
        self.window_size = window_size
        self.stride = stride
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.loader = MultimodalDataLoader(data_root, track)
        self.bin_extractor = bin_extractor or BinExtractor()
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.patient_scalers: Dict[str, Dict] = {}

        # Reuse LOSOPreprocessor for supplementary file staging and label extraction
        self._loso = LOSOPreprocessor(
            data_root=data_root,
            track=track,
            window_size=window_size,
            stride=stride,
            output_dir=output_dir,
            feature_extractor=self.feature_extractor,
            sleep_files_dir=sleep_files_dir,
            annotations_dir=annotations_dir,
        )

    # ------------------------------------------------------------------
    # Per-patient raw extraction
    # ------------------------------------------------------------------

    def _extract_patient_raw(self, patient_id: str) -> Dict[str, Dict]:
        """Extract raw bins + features for all sequences of a patient."""
        sequences = self.loader.get_sequences(patient_id)
        patient_raw: Dict[str, Dict] = {}

        for i, seq_name in enumerate(sequences, 1):
            t0 = time.time()
            print(f"  [{i}/{len(sequences)}] {seq_name}...", end=" ", flush=True)

            seq = self.loader.load_sequence(patient_id, seq_name)
            day_list, labels, label_mask = self._loso._get_labels_and_days(seq)

            if not day_list:
                print("no days found, skipping.")
                continue

            # CNN modalities: binned signals
            bins_data: Dict[str, np.ndarray] = {}
            bins_masks: Dict[str, np.ndarray] = {}

            if seq.linacc is not None and len(seq.linacc):
                b, m = self.bin_extractor.extract_imu_bins(seq.linacc, day_list)
            else:
                b = np.zeros(
                    (len(day_list), self.bin_extractor.n_bins, IMU_CHANNELS),
                    dtype=np.float32,
                )
                m = np.zeros(len(day_list), dtype=bool)
            bins_data["linacc"] = b
            bins_masks["linacc"] = m
            seq.linacc = None  # free raw data

            if seq.gyr is not None and len(seq.gyr):
                b, m = self.bin_extractor.extract_imu_bins(seq.gyr, day_list)
            else:
                b = np.zeros(
                    (len(day_list), self.bin_extractor.n_bins, IMU_CHANNELS),
                    dtype=np.float32,
                )
                m = np.zeros(len(day_list), dtype=bool)
            bins_data["gyr"] = b
            bins_masks["gyr"] = m
            seq.gyr = None  # free raw data

            if seq.hrm is not None and len(seq.hrm):
                b, m = self.bin_extractor.extract_hr_bins(seq.hrm, day_list)
            else:
                b = np.zeros(
                    (len(day_list), self.bin_extractor.n_bins, HR_CHANNELS),
                    dtype=np.float32,
                )
                m = np.zeros(len(day_list), dtype=bool)
            bins_data["hrm"] = b
            bins_masks["hrm"] = m
            seq.hrm = None  # free raw data

            # FC modalities: hand-crafted features (call extractors directly
            # to avoid computing expensive accel/gyr/hr features)
            feat_data: Dict[str, np.ndarray] = {}
            feat_masks: Dict[str, np.ndarray] = {}

            fe = self.feature_extractor
            n = len(day_list)

            if seq.step is not None and len(seq.step):
                f, m = fe._extract_step(seq.step, day_list)
            else:
                f = np.full((n, FC_MODALITIES["step"]), np.nan, dtype=np.float32)
                m = np.zeros(n, dtype=bool)
            feat_data["step"] = f
            feat_masks["step"] = m

            if seq.sleep is not None and len(seq.sleep):
                f, m = fe._extract_sleep(seq.sleep, day_list)
            else:
                f = np.full((n, FC_MODALITIES["sleep"]), np.nan, dtype=np.float32)
                m = np.zeros(n, dtype=bool)
            feat_data["sleep"] = f
            feat_masks["sleep"] = m

            patient_raw[seq_name] = {
                "bins_data": bins_data,
                "bins_masks": bins_masks,
                "feat_data": feat_data,
                "feat_masks": feat_masks,
                "day_list": day_list,
                "labels": labels,
                "label_mask": label_mask,
                "split": seq.split,
            }
            print(f"done ({time.time() - t0:.1f}s, {len(day_list)} days)")

        return patient_raw

    # ------------------------------------------------------------------
    # Per-patient normalisation
    # ------------------------------------------------------------------

    def _normalize_patient(
        self, patient_raw: Dict[str, Dict]
    ) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
        """Z-score normalisation fitted on train_* days.

        CNN bins: per-channel z-score across all bins from train_* days.
        FC features: per-feature z-score (same as existing pipeline).

        NaN → 0 after normalisation. Clips to [-5, 5].
        """
        scalers: Dict[str, Dict] = {}

        # --- CNN modalities: normalise per channel across (days × bins) ---
        for mod, n_ch in CNN_MODALITIES.items():
            train_chunks: List[np.ndarray] = []
            for seq_name, seq_data in patient_raw.items():
                if not seq_name.startswith("train_"):
                    continue
                data = seq_data["bins_data"][mod]  # (N, n_bins, C)
                msk = seq_data["bins_masks"][mod]  # (N,) bool
                if msk.any():
                    # Reshape valid days to (N_valid * n_bins, C)
                    train_chunks.append(data[msk].reshape(-1, n_ch))

            if train_chunks:
                all_train = np.concatenate(train_chunks, axis=0)
                # Exclude zero-bins (missing data) from stats
                nonzero_mask = np.any(all_train != 0, axis=1)
                if nonzero_mask.any():
                    active = all_train[nonzero_mask]
                    mean = np.mean(active, axis=0).astype(np.float32)
                    std = np.std(active, axis=0).astype(np.float32)
                else:
                    mean = np.zeros(n_ch, dtype=np.float32)
                    std = np.ones(n_ch, dtype=np.float32)
            else:
                mean = np.zeros(n_ch, dtype=np.float32)
                std = np.ones(n_ch, dtype=np.float32)

            std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
            scalers[mod] = {"mean": mean, "std": std}

            # Apply
            for seq_data in patient_raw.values():
                data = seq_data["bins_data"][mod].copy()
                data = (data - mean) / std
                data = np.clip(data, -5.0, 5.0)
                data = np.where(np.isnan(data), 0.0, data).astype(np.float32)
                seq_data["bins_data"][mod] = data

        # --- FC modalities: per-feature z-score ---
        for mod, dim in FC_MODALITIES.items():
            train_chunks: List[np.ndarray] = []
            for seq_name, seq_data in patient_raw.items():
                if not seq_name.startswith("train_"):
                    continue
                feat = seq_data["feat_data"][mod]
                msk = seq_data["feat_masks"][mod]
                if msk.any():
                    train_chunks.append(feat[msk])

            if train_chunks:
                all_train = np.concatenate(train_chunks, axis=0)
                mean = np.nanmean(all_train, axis=0).astype(np.float32)
                std = np.nanstd(all_train, axis=0).astype(np.float32)
            else:
                mean = np.zeros(dim, dtype=np.float32)
                std = np.ones(dim, dtype=np.float32)

            std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
            scalers[mod] = {"mean": mean, "std": std}

            for seq_data in patient_raw.values():
                feat = seq_data["feat_data"][mod].copy()
                feat = (feat - mean) / std
                feat = np.clip(feat, -5.0, 5.0)
                feat = np.where(np.isnan(feat), 0.0, feat).astype(np.float32)
                seq_data["feat_data"][mod] = feat

        return patient_raw, scalers

    # ------------------------------------------------------------------
    # Window creation
    # ------------------------------------------------------------------

    def _create_windows(
        self,
        seq_data: Dict,
        patient_id: str,
        seq_name: str,
    ) -> List[Dict]:
        """Create W-day left-padded windows with CNN bins + FC features."""
        W = self.window_size
        n_bins = self.bin_extractor.n_bins
        day_list = seq_data["day_list"]
        N = len(day_list)

        windows: List[Dict] = []

        for t in range(N):
            pad_len = max(0, W - 1 - t)
            real_len = W - pad_len
            real_start = t - real_len + 1

            # CNN bins
            win_bins = {
                mod: np.zeros((W, n_bins, ch), dtype=np.float32)
                for mod, ch in CNN_MODALITIES.items()
            }
            # FC features
            win_feats = {
                mod: np.zeros((W, dim), dtype=np.float32)
                for mod, dim in FC_MODALITIES.items()
            }
            # Masks
            win_masks = {
                mod: np.zeros(W, dtype=bool)
                for mod in list(CNN_MODALITIES) + list(FC_MODALITIES)
            }
            pmask = np.zeros(W, dtype=bool)
            lbls = np.full(W, -1, dtype=np.int32)
            lmask = np.zeros(W, dtype=bool)
            days_w = [-1] * W

            for w in range(pad_len, W):
                src = real_start + (w - pad_len)

                for mod in CNN_MODALITIES:
                    win_bins[mod][w] = seq_data["bins_data"][mod][src]
                    win_masks[mod][w] = seq_data["bins_masks"][mod][src]

                for mod in FC_MODALITIES:
                    win_feats[mod][w] = seq_data["feat_data"][mod][src]
                    win_masks[mod][w] = seq_data["feat_masks"][mod][src]

                pmask[w] = True
                lbls[w] = seq_data["labels"][src]
                lmask[w] = seq_data["label_mask"][src]
                days_w[w] = day_list[src]

            window = {
                "patient_id": patient_id,
                "sequence_name": seq_name,
                "window_end_day": day_list[t],
                "window_days": days_w,
                "padding_mask": pmask,
                "labels": lbls,
                "label_mask": lmask,
            }

            # Add bins and features
            for mod in CNN_MODALITIES:
                window[f"{mod}_bins"] = win_bins[mod]
                window[f"{mod}_mask"] = win_masks[mod]
            for mod in FC_MODALITIES:
                window[f"{mod}_features"] = win_feats[mod]
                window[f"{mod}_mask"] = win_masks[mod]

            windows.append(window)

        return windows

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def _run_patient_pipeline(self, patient_id: str) -> Dict[str, List[Dict]]:
        t_patient = time.time()

        t0 = time.time()
        print("  Extracting bins + features...")
        patient_raw = self._extract_patient_raw(patient_id)
        if not patient_raw:
            print(f"  Skipping {patient_id}: no sequences found.")
            return {}
        print(f"  Extraction done ({time.time() - t0:.1f}s)")

        t0 = time.time()
        print("  Normalizing...")
        patient_norm, scalers = self._normalize_patient(patient_raw)
        self.patient_scalers[patient_id] = scalers
        print(f"  Normalization done ({time.time() - t0:.1f}s)")

        t0 = time.time()
        print("  Creating windows...")
        patient_windows: Dict[str, List[Dict]] = {}
        for seq_name, seq_data in patient_norm.items():
            wins = self._create_windows(seq_data, patient_id, seq_name)
            patient_windows[seq_name] = wins
            lbl = seq_data["labels"]
            lmk = seq_data["label_mask"]
            print(
                f"    {seq_name}: {len(seq_data['day_list'])} days "
                f"({int((lbl[lmk] == 0).sum())} stable, "
                f"{int((lbl[lmk] == 1).sum())} relapse) "
                f"→ {len(wins)} windows"
            )
        print(f"  Windowing done ({time.time() - t0:.1f}s)")
        print(f"  Patient {patient_id} total: {time.time() - t_patient:.1f}s")

        return patient_windows

    def process_single_patient(self, patient_id: str) -> Dict[str, List[Dict]]:
        print(f"\nStaging supplementary files for {patient_id}...")
        self._loso._stage_supplementary_files(patient_id=patient_id)
        print(f"\nProcessing patient {patient_id}...")
        return self._run_patient_pipeline(patient_id)

    def process_all_patients(self) -> Dict[str, Dict[str, List[Dict]]]:
        all_data: Dict[str, Dict[str, List[Dict]]] = {}
        patients = self.loader.get_patients()

        for pi, patient_id in enumerate(patients, 1):
            print(f"\n[{pi}/{len(patients)}] Processing patient {patient_id}...")
            patient_windows = self._run_patient_pipeline(patient_id)
            if patient_windows:
                all_data[patient_id] = patient_windows

        return all_data

    # ------------------------------------------------------------------
    # LOSO fold organisation (reuse from LOSOPreprocessor)
    # ------------------------------------------------------------------

    def organize_loso_splits(
        self, all_data: Dict[str, Dict[str, List[Dict]]]
    ) -> Dict[int, Dict]:
        """Same LOSO fold logic as LOSOPreprocessor."""
        return self._loso.organize_loso_splits(all_data)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_processed_data(
        self,
        loso_splits: Dict[int, Dict],
        save_format: str = "pickle",
    ) -> None:
        track_dir = self.output_dir / f"track{self.track}"
        track_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "track": self.track,
            "window_size": self.window_size,
            "stride": self.stride,
            "n_folds": len(loso_splits),
            "cnn_modality_channels": CNN_MODALITIES,
            "fc_modality_dims": FC_MODALITIES,
            "n_bins": self.bin_extractor.n_bins,
            "bin_minutes": self.bin_extractor.bin_minutes,
            "window_format": "cnn_bins_plus_fc_features",
            "normalization": {
                "method": "per_patient_zscore",
                "fit_on": "train_sequences_only",
                "clip": [-5.0, 5.0],
                "nan_fill": 0.0,
            },
            "folds": {
                fold_id: {
                    "test_patient": fd["test_patient"],
                    "train_patients": fd["train_patients"],
                    "train_stats": fd["train_stats"],
                    "val_stats": fd["val_stats"],
                    "test_stats": fd["test_stats"],
                }
                for fold_id, fd in loso_splits.items()
            },
        }

        with open(track_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        for fold_id, fold_data in loso_splits.items():
            fold_dir = track_dir / f"fold_{fold_id}"
            fold_dir.mkdir(exist_ok=True)

            for split_name in ("train", "val", "test"):
                wins = fold_data[split_name]
                with open(fold_dir / f"{split_name}.pkl", "wb") as f:
                    pickle.dump(wins, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Patient scalers
        with open(track_dir / "patient_scalers.pkl", "wb") as f:
            pickle.dump(self.patient_scalers, f)

        print(f"\nProcessed CNN data saved to {track_dir}")

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, save_format: str = "pickle") -> Dict[int, Dict]:
        print("=" * 60)
        print("CNN+LSTM LOSO Preprocessing")
        print("=" * 60)
        print(f"Track:       {self.track}")
        print(f"Window size: {self.window_size} days (left-padded)")
        print(f"Bin size:    {self.bin_extractor.bin_minutes} min ({self.bin_extractor.n_bins} bins/day)")
        print(f"Output:      {self.output_dir}")
        print()

        print("Step 0: Staging supplementary files...")
        self._loso._stage_supplementary_files()
        print()

        print("Step 1: Bin extraction + feature extraction + normalisation...")
        all_data = self.process_all_patients()
        print()

        print("Step 2: Organising LOSO splits...")
        loso_splits = self.organize_loso_splits(all_data)
        print()

        print("Step 3: Saving processed data...")
        self.save_processed_data(loso_splits, save_format=save_format)

        print("\n" + "=" * 60)
        print("CNN preprocessing complete!")
        print("=" * 60)

        return loso_splits
