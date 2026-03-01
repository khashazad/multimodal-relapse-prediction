"""
Preprocessing pipeline for supervised leave-one-subject-out (LOSO) cross-validation.

Pipeline for each patient:
  1. Extract raw per-day modality features for every sequence.
  2. Per-patient z-score normalisation (fit on train_* stable days only).
  3. Create 7-day sliding windows with left-padding (one window per day).
  4. Organise into LOSO folds and save.

Window dict format
------------------
{
    'patient_id':     str,
    'sequence_name':  str,
    'window_end_day': int,          # the day being predicted (last in window)
    'window_days':    List[int],    # day indices; -1 = left-padded position
    'features':       Dict[str, np.ndarray (W, F)],
    'modality_masks': Dict[str, np.ndarray (W,) bool],
    'padding_mask':   np.ndarray (W,) bool,   # True = real day, False = padding
    'labels':         np.ndarray (W,) int32,  # 0=stable, 1=relapse, -1=unknown
    'label_mask':     np.ndarray (W,) bool,   # True = label valid
}
"""

from __future__ import annotations

import json
import pickle
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_loader import MultimodalDataLoader, SequenceData
from .feature_extractor import MODALITY_DIMS, MODALITY_FEATURE_NAMES, FeatureExtractor


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class LOSOPreprocessor:

    def __init__(
        self,
        data_root: str,
        track: int = 1,
        window_size: int = 7,
        stride: int = 1,
        output_dir: str = "data/processed",
        feature_extractor: Optional[FeatureExtractor] = None,
        sleep_files_dir: Optional[str] = None,
        annotations_dir: Optional[str] = None,
    ) -> None:
        self.data_root    = Path(data_root)
        self.track        = track
        self.window_size  = window_size
        self.stride       = stride
        self.output_dir   = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Optional external source directories for supplementary files.
        # If set, _stage_supplementary_files() will copy from these into
        # each sequence directory before feature extraction.
        self.sleep_files_dir = Path(sleep_files_dir) if sleep_files_dir else None
        self.annotations_dir = Path(annotations_dir) if annotations_dir else None

        self.loader            = MultimodalDataLoader(data_root, track)
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.patient_scalers:  Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Supplementary file staging
    # ------------------------------------------------------------------

    def _stage_supplementary_files(
        self, patient_id: Optional[str] = None
    ) -> None:
        """Copy sleep files and annotation CSVs into their sequence directories.

        Sleep files and relapse annotations are distributed separately from the
        main sensor parquets:
          - sleep:       <sleep_files_dir>/P{x}/{seq}/sleep.parquet
          - annotations: <annotations_dir>/P{x}/{seq}/relapses.csv

        This step copies them to their inline locations expected by
        MultimodalDataLoader:
          - <track_path>/P{x}/{seq}/sleep.parquet
          - <track_path>/P{x}/{seq}/relapses.csv

        Files already present at the destination are left untouched.

        Parameters
        ----------
        patient_id:
            If given, stage only for that patient; otherwise stage for all
            patients discovered in the track directory.
        """
        if self.sleep_files_dir is None and self.annotations_dir is None:
            return

        track_path = self.loader.track_path
        patients = [patient_id] if patient_id else self.loader.get_patients()

        n_sleep = 0
        n_ann   = 0

        for pid in patients:
            for seq_name in self.loader.get_sequences(pid):
                seq_dir = track_path / pid / seq_name

                if self.sleep_files_dir is not None:
                    src = self.sleep_files_dir / pid / seq_name / "sleep.parquet"
                    dst = seq_dir / "sleep.parquet"
                    if src.exists() and not dst.exists():
                        shutil.copy2(src, dst)
                        n_sleep += 1

                if self.annotations_dir is not None:
                    src = self.annotations_dir / pid / seq_name / "relapses.csv"
                    dst = seq_dir / "relapses.csv"
                    if src.exists() and not dst.exists():
                        shutil.copy2(src, dst)
                        n_ann += 1

        scope = patient_id if patient_id else "all patients"
        print(f"  Staged {n_sleep} sleep file(s), {n_ann} annotation file(s) "
              f"for {scope}.")

    # ------------------------------------------------------------------
    # Label / day-list extraction
    # ------------------------------------------------------------------

    def _get_labels_and_days(
        self, seq: SequenceData
    ) -> Tuple[List[int], np.ndarray, np.ndarray]:
        """Extract canonical day list and per-day labels from a sequence.

        Applies the extra-day bug fix (drops the last row of relapses.csv).

        Returns
        -------
        day_list:   sorted list of day indices
        labels:     (N,) int32 — 0, 1, or -1 (unknown)
        label_mask: (N,) bool  — True where label is valid
        """
        if seq.relapses is not None and len(seq.relapses):
            relapses = seq.relapses.iloc[:-1].copy()   # drop extra-day bug row
            day_list = sorted(relapses["day_index"].tolist())
        else:
            # Fall back to union of day_index values across sensor modalities
            day_set: set = set()
            for attr in ("linacc", "gyr", "hrm", "step"):
                df = getattr(seq, attr)
                if df is not None and "day_index" in df.columns:
                    day_set.update(df["day_index"].unique())
            if seq.sleep is not None and "end_date_index" in seq.sleep.columns:
                day_set.update(seq.sleep["end_date_index"].unique())
            day_list = sorted(d for d in day_set if d >= 0)
            relapses = None

        N = len(day_list)
        labels     = np.full(N, -1, dtype=np.int32)
        label_mask = np.zeros(N, dtype=bool)

        for i, day_idx in enumerate(day_list):
            if seq.split == "train":
                labels[i]     = 0
                label_mask[i] = True
            elif relapses is not None and "relapse" in relapses.columns:
                row = relapses[relapses["day_index"] == day_idx]
                if len(row):
                    labels[i]     = int(row.iloc[0]["relapse"])
                    label_mask[i] = True
            # test_* with no relapse column: labels stay -1, mask stays False
            # (after prepare_test_labels.py, test_* sequences have real labels
            #  and are handled by the elif branch above)

        return day_list, labels, label_mask

    # ------------------------------------------------------------------
    # Per-patient raw feature extraction
    # ------------------------------------------------------------------

    def _extract_patient_raw(
        self, patient_id: str
    ) -> Dict[str, Dict]:
        """Extract raw (un-normalised) features for all sequences of a patient.

        Returns
        -------
        {sequence_name: {
            'modality_features': {mod: (N, F) float32},
            'modality_masks':    {mod: (N,) bool},
            'day_list':  List[int],
            'labels':    (N,) int32,
            'label_mask':(N,) bool,
            'split':     str,
        }}
        """
        sequences = self.loader.get_sequences(patient_id)
        patient_raw: Dict[str, Dict] = {}

        for i, seq_name in enumerate(sequences, 1):
            t0 = time.time()
            print(f"  [{i}/{len(sequences)}] {seq_name}...", end=" ", flush=True)

            seq = self.loader.load_sequence(patient_id, seq_name)
            day_list, labels, label_mask = self._get_labels_and_days(seq)

            if not day_list:
                print(f"no days found, skipping.")
                continue

            mod_results = self.feature_extractor.extract_all_modalities_for_sequence(
                seq, day_list
            )

            patient_raw[seq_name] = {
                "modality_features": {m: r["features"] for m, r in mod_results.items()},
                "modality_masks":    {m: r["mask"]     for m, r in mod_results.items()},
                "day_list":   day_list,
                "labels":     labels,
                "label_mask": label_mask,
                "split":      seq.split,
            }
            print(f"done ({time.time() - t0:.1f}s, {len(day_list)} days)")

        return patient_raw

    # ------------------------------------------------------------------
    # Per-patient normalisation
    # ------------------------------------------------------------------

    def _normalize_patient(
        self, patient_raw: Dict[str, Dict]
    ) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
        """Fit z-score normalisation on train_* days; apply to all sequences.

        NaN → 0 after normalisation. Clips to [-5, 5].

        Returns
        -------
        (normalised patient_raw, scalers {mod: {'mean': ..., 'std': ...}})
        """
        scalers: Dict[str, Dict] = {}

        for mod, dim in MODALITY_DIMS.items():
            # Collect masked rows from train_* sequences
            train_chunks: List[np.ndarray] = []
            for seq_name, seq_data in patient_raw.items():
                if not seq_name.startswith("train_"):
                    continue
                feat = seq_data["modality_features"][mod]
                msk  = seq_data["modality_masks"][mod]
                if msk.any():
                    train_chunks.append(feat[msk])

            if train_chunks:
                all_train = np.concatenate(train_chunks, axis=0)  # (M, F)
                mean = np.nanmean(all_train, axis=0).astype(np.float32)
                std  = np.nanstd(all_train,  axis=0).astype(np.float32)
            else:
                print(f"  Warning: no train_* data for modality '{mod}'. Using identity normalization.")
                mean = np.zeros(dim, dtype=np.float32)
                std  = np.ones(dim,  dtype=np.float32)

            # Replace zero std (constant features) with 1 to avoid division by zero
            std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
            scalers[mod] = {"mean": mean, "std": std}

            # Apply normalisation to every sequence
            for seq_data in patient_raw.values():
                feat = seq_data["modality_features"][mod].copy()
                feat = (feat - mean) / std
                feat = np.clip(feat, -5.0, 5.0)
                feat = np.where(np.isnan(feat), 0.0, feat).astype(np.float32)
                seq_data["modality_features"][mod] = feat

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
        """Create one W-day left-padded window per day.

        Parameters
        ----------
        seq_data:   normalised sequence dict from _extract_patient_raw
        patient_id: e.g. 'P1'
        seq_name:   e.g. 'train_0'

        Returns
        -------
        List of window dicts.
        """
        W          = self.window_size
        day_list   = seq_data["day_list"]
        N          = len(day_list)
        mod_feats  = seq_data["modality_features"]
        mod_masks  = seq_data["modality_masks"]
        labels     = seq_data["labels"]
        label_mask = seq_data["label_mask"]

        windows: List[Dict] = []

        for t in range(N):
            # Number of zero-padded positions on the left
            pad_len    = max(0, W - 1 - t)
            real_len   = W - pad_len
            real_start = t - real_len + 1  # first real index into day_list

            feats  = {mod: np.zeros((W, MODALITY_DIMS[mod]), dtype=np.float32)
                      for mod in MODALITY_DIMS}
            mmasks = {mod: np.zeros(W, dtype=bool) for mod in MODALITY_DIMS}
            pmask  = np.zeros(W, dtype=bool)
            lbls   = np.full(W, -1,  dtype=np.int32)
            lmask  = np.zeros(W, dtype=bool)
            days_w = [-1] * W

            for w in range(pad_len, W):
                src = real_start + (w - pad_len)
                for mod in MODALITY_DIMS:
                    feats[mod][w]  = mod_feats[mod][src]
                    mmasks[mod][w] = mod_masks[mod][src]
                pmask[w]  = True
                lbls[w]   = labels[src]
                lmask[w]  = label_mask[src]
                days_w[w] = day_list[src]

            windows.append({
                "patient_id":     patient_id,
                "sequence_name":  seq_name,
                "window_end_day": day_list[t],
                "window_days":    days_w,
                "features":       feats,
                "modality_masks": mmasks,
                "padding_mask":   pmask,
                "labels":         lbls,
                "label_mask":     lmask,
            })

        return windows

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def process_single_patient(
        self, patient_id: str
    ) -> Dict[str, List[Dict]]:
        """Extract, normalise, and window one patient.

        Returns
        -------
        {sequence_name: [window_dicts]}
        Scalers are stored in ``self.patient_scalers[patient_id]``.
        """
        print(f"\nStaging supplementary files for {patient_id}...")
        self._stage_supplementary_files(patient_id=patient_id)

        t_patient = time.time()
        print(f"\nProcessing patient {patient_id}...")

        t0 = time.time()
        print(f"  Extracting features...")
        patient_raw = self._extract_patient_raw(patient_id)
        if not patient_raw:
            print(f"  Skipping {patient_id}: no sequences found.")
            return {}
        print(f"  Feature extraction done ({time.time() - t0:.1f}s)")

        t0 = time.time()
        print(f"  Normalizing...")
        patient_norm, scalers = self._normalize_patient(patient_raw)
        self.patient_scalers[patient_id] = scalers
        print(f"  Normalization done ({time.time() - t0:.1f}s)")

        t0 = time.time()
        print(f"  Creating windows...")
        patient_windows: Dict[str, List[Dict]] = {}
        for seq_name, seq_data in patient_norm.items():
            wins = self._create_windows(seq_data, patient_id, seq_name)
            patient_windows[seq_name] = wins
            lbl = seq_data["labels"]
            lmk = seq_data["label_mask"]
            print(
                f"  {seq_name}: {len(seq_data['day_list'])} days "
                f"({int((lbl[lmk] == 0).sum())} stable, "
                f"{int((lbl[lmk] == 1).sum())} relapse) "
                f"→ {len(wins)} windows"
            )
        print(f"  Windowing done ({time.time() - t0:.1f}s)")
        print(f"  Patient {patient_id} total: {time.time() - t_patient:.1f}s")

        return patient_windows

    def process_all_patients(self) -> Dict[str, Dict[str, List[Dict]]]:
        """Extract, normalise, and window all patients.

        Returns
        -------
        {patient_id: {sequence_name: [window_dicts]}}
        """
        all_data: Dict[str, Dict[str, List[Dict]]] = {}
        patients = self.loader.get_patients()

        for pi, patient_id in enumerate(patients, 1):
            t_patient = time.time()
            print(f"\n[{pi}/{len(patients)}] Processing patient {patient_id}...")

            # 1. Raw feature extraction
            t0 = time.time()
            print(f"  Extracting features...")
            patient_raw = self._extract_patient_raw(patient_id)
            if not patient_raw:
                print(f"  Skipping {patient_id}: no sequences found.")
                continue
            print(f"  Feature extraction done ({time.time() - t0:.1f}s)")

            # 2. Per-patient normalisation
            t0 = time.time()
            print(f"  Normalizing...")
            patient_norm, scalers = self._normalize_patient(patient_raw)
            self.patient_scalers[patient_id] = scalers
            print(f"  Normalization done ({time.time() - t0:.1f}s)")

            # 3. Create windows per sequence
            t0 = time.time()
            print(f"  Creating windows...")
            patient_windows: Dict[str, List[Dict]] = {}
            for seq_name, seq_data in patient_norm.items():
                wins = self._create_windows(seq_data, patient_id, seq_name)
                patient_windows[seq_name] = wins
                lbl = seq_data["labels"]
                lmk = seq_data["label_mask"]
                print(
                    f"  {seq_name}: {len(seq_data['day_list'])} days "
                    f"({int((lbl[lmk] == 0).sum())} stable, "
                    f"{int((lbl[lmk] == 1).sum())} relapse) "
                    f"→ {len(wins)} windows"
                )
            print(f"  Windowing done ({time.time() - t0:.1f}s)")
            print(f"  Patient {patient_id} total: {time.time() - t_patient:.1f}s")

            all_data[patient_id] = patient_windows

        return all_data

    # ------------------------------------------------------------------
    # LOSO fold organisation
    # ------------------------------------------------------------------

    def organize_loso_splits(
        self, all_data: Dict[str, Dict[str, List[Dict]]]
    ) -> Dict[int, Dict]:
        """Organise windows into supervised LOSO folds.

        For fold i (test patient = P_k):
          train: ALL sequences from all other patients
          val:   val_* sequences from P_k
          test:  test_* sequences from P_k
        """
        patients = sorted(all_data.keys())
        splits: Dict[int, Dict] = {}

        for fold_id, test_patient in enumerate(patients):
            train_wins: List[Dict] = []
            val_wins:   List[Dict] = []
            test_wins:  List[Dict] = []

            for pid, seqs in all_data.items():
                for seq_name, wins in seqs.items():
                    if pid != test_patient:
                        train_wins.extend(wins)
                    elif seq_name.startswith("val_"):
                        val_wins.extend(wins)
                    elif seq_name.startswith("test_"):
                        test_wins.extend(wins)

            splits[fold_id] = {
                "test_patient":  test_patient,
                "train_patients": [p for p in patients if p != test_patient],
                "train": train_wins,
                "val":   val_wins,
                "test":  test_wins,
                "train_stats": self._compute_stats(train_wins),
                "val_stats":   self._compute_stats(val_wins),
                "test_stats":  self._compute_stats(test_wins),
            }

            ts, vs, tes = (splits[fold_id][k] for k in
                           ("train_stats", "val_stats", "test_stats"))
            print(
                f"Fold {fold_id} (test: {test_patient}): "
                f"train {ts['n_windows']} win "
                f"({ts['n_stable_days']}S/{ts['n_relapse_days']}R) | "
                f"val {vs['n_windows']} win "
                f"({vs['n_stable_days']}S/{vs['n_relapse_days']}R) | "
                f"test {tes['n_windows']} win"
            )

        return splits

    @staticmethod
    def _compute_stats(windows: List[Dict]) -> Dict:
        if not windows:
            return dict(n_windows=0, n_stable_days=0, n_relapse_days=0,
                        n_unlabeled_days=0, n_padded_positions=0)

        all_l  = np.concatenate([w["labels"]       for w in windows])
        all_lm = np.concatenate([w["label_mask"]   for w in windows])
        all_pm = np.concatenate([w["padding_mask"]  for w in windows])

        return {
            "n_windows":          len(windows),
            "n_stable_days":      int(((all_l == 0) & all_lm & all_pm).sum()),
            "n_relapse_days":     int(((all_l == 1) & all_lm & all_pm).sum()),
            "n_unlabeled_days":   int((~all_lm & all_pm).sum()),
            "n_padded_positions": int((~all_pm).sum()),
        }

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_processed_data(
        self,
        loso_splits: Dict[int, Dict],
        save_format: str = "pickle",
    ) -> None:
        """Save LOSO splits to disk.

        Parameters
        ----------
        loso_splits:  output of organize_loso_splits()
        save_format:  'pickle' (default) or 'numpy'
        """
        track_dir = self.output_dir / f"track{self.track}"
        track_dir.mkdir(parents=True, exist_ok=True)

        # Metadata
        metadata = {
            "track":       self.track,
            "window_size": self.window_size,
            "stride":      self.stride,
            "n_folds":     len(loso_splits),
            "modality_dims": MODALITY_DIMS,
            "modality_feature_names": MODALITY_FEATURE_NAMES,
            "window_format": "per_modality_dict",
            "normalization": {
                "method":   "per_patient_zscore",
                "fit_on":   "train_sequences_only",
                "clip":     [-5.0, 5.0],
                "nan_fill": 0.0,
            },
            "folds": {
                fold_id: {
                    "test_patient":   fd["test_patient"],
                    "train_patients": fd["train_patients"],
                    "train_stats":    fd["train_stats"],
                    "val_stats":      fd["val_stats"],
                    "test_stats":     fd["test_stats"],
                }
                for fold_id, fd in loso_splits.items()
            },
        }

        with open(track_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Per-fold data
        for fold_id, fold_data in loso_splits.items():
            fold_dir = track_dir / f"fold_{fold_id}"
            fold_dir.mkdir(exist_ok=True)

            for split_name in ("train", "val", "test"):
                wins = fold_data[split_name]

                if save_format == "pickle":
                    with open(fold_dir / f"{split_name}.pkl", "wb") as f:
                        pickle.dump(wins, f)

                elif save_format == "numpy":
                    if not wins:
                        continue
                    for mod in MODALITY_DIMS:
                        np.save(
                            fold_dir / f"{split_name}_{mod}_features.npy",
                            np.stack([w["features"][mod] for w in wins]),
                        )
                        np.save(
                            fold_dir / f"{split_name}_{mod}_masks.npy",
                            np.stack([w["modality_masks"][mod] for w in wins]),
                        )
                    np.save(fold_dir / f"{split_name}_labels.npy",
                            np.stack([w["labels"]       for w in wins]))
                    np.save(fold_dir / f"{split_name}_label_masks.npy",
                            np.stack([w["label_mask"]   for w in wins]))
                    np.save(fold_dir / f"{split_name}_padding_masks.npy",
                            np.stack([w["padding_mask"] for w in wins]))

                    win_meta = [
                        {
                            "patient_id":    w["patient_id"],
                            "sequence_name": w["sequence_name"],
                            "window_end_day": w["window_end_day"],
                            "window_days":   w["window_days"],
                        }
                        for w in wins
                    ]
                    with open(fold_dir / f"{split_name}_metadata.pkl", "wb") as f:
                        pickle.dump(win_meta, f)

        # Patient scalers
        with open(track_dir / "patient_scalers.pkl", "wb") as f:
            pickle.dump(self.patient_scalers, f)

        print(f"\nProcessed data saved to {track_dir}")

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self, save_format: str = "pickle") -> Dict[int, Dict]:
        """Run the complete preprocessing pipeline.

        Returns
        -------
        loso_splits dict (also saved to disk).
        """
        print("=" * 60)
        print("Supervised LOSO Preprocessing")
        print("=" * 60)
        print(f"Track:       {self.track}")
        print(f"Window size: {self.window_size} days (left-padded)")
        print(f"Output:      {self.output_dir}")
        print()

        print("Step 0: Staging supplementary files (sleep + annotations)...")
        self._stage_supplementary_files()
        print()

        print("Step 1: Feature extraction + per-patient normalisation...")
        all_data = self.process_all_patients()
        print()

        print("Step 2: Organising LOSO splits...")
        loso_splits = self.organize_loso_splits(all_data)
        print()

        print("Step 3: Saving processed data...")
        self.save_processed_data(loso_splits, save_format=save_format)

        print("\n" + "=" * 60)
        print("Preprocessing complete!")
        print("=" * 60)

        return loso_splits
