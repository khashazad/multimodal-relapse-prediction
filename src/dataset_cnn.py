"""
PyTorch Dataset for CNN+LSTM fold data.

Loads preprocessed window pickles containing binned signals (for CNN branches)
and hand-crafted features (for FC branches), returning per-sample dicts with
tensors ready for CNNLSTMEnsemble.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

# Modality definitions (must match preprocess_cnn_loso.py)
CNN_MODALITIES = {"linacc": 8, "gyr": 8, "hrm": 4}
FC_MODALITIES = {"step": 10, "sleep": 9}

JITTER_STD = 0.02
SCALE_RANGE = (0.9, 1.1)


class CNNRelapseDataset(Dataset):
    """Dataset for CNN+LSTM architecture.

    Each sample is a window dict from CNNLOSOPreprocessor.
    Returns tensors for both CNN branches (bins) and FC branches (features).
    """

    PATIENT_IDX: Dict[str, int] = {f"P{i}": i - 1 for i in range(1, 10)}

    def __init__(
        self,
        pkl_path: str | Path,
        augmentation: str = "none",
    ) -> None:
        with open(pkl_path, "rb") as f:
            self.windows: List[Dict] = pickle.load(f)
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        w = self.windows[idx]
        sample: Dict[str, torch.Tensor] = {}

        # CNN modalities: bins of shape (W, n_bins, C)
        for mod in CNN_MODALITIES:
            bins = w[f"{mod}_bins"].astype(np.float32)
            if self.augmentation != "none":
                bins = self._augment_bins(bins)
            sample[f"{mod}_bins"] = torch.from_numpy(bins)
            sample[f"{mod}_mask"] = torch.from_numpy(w[f"{mod}_mask"].astype(np.bool_))

        # FC modalities: features of shape (W, F)
        for mod in FC_MODALITIES:
            feats = w[f"{mod}_features"].astype(np.float32)
            if self.augmentation != "none":
                feats = self._augment_feats(feats)
            sample[f"{mod}_features"] = torch.from_numpy(feats)
            sample[f"{mod}_mask"] = torch.from_numpy(w[f"{mod}_mask"].astype(np.bool_))

        # Padding mask
        sample["padding_mask"] = torch.from_numpy(w["padding_mask"].astype(np.bool_))

        # Last-day label
        last_label = int(w["labels"][-1])
        sample["label"] = torch.tensor(max(last_label, 0), dtype=torch.float32)
        sample["label_valid"] = torch.tensor(
            bool(w["label_mask"][-1]), dtype=torch.bool
        )

        # Patient index
        patient_id = w.get("patient_id", None)
        if patient_id is not None and patient_id in self.PATIENT_IDX:
            sample["patient_idx"] = torch.tensor(
                self.PATIENT_IDX[patient_id], dtype=torch.long
            )
        else:
            sample["patient_idx"] = torch.tensor(-1, dtype=torch.long)

        return sample

    def _augment_bins(self, bins: np.ndarray) -> np.ndarray:
        """Augment bin data (W, n_bins, C)."""
        if self.augmentation in ("jitter", "all"):
            noise = np.random.normal(0, JITTER_STD, size=bins.shape).astype(np.float32)
            bins = bins + noise
        if self.augmentation in ("scale", "all"):
            # Scale per channel
            scale = np.random.uniform(*SCALE_RANGE, size=(1, 1, bins.shape[2])).astype(
                np.float32
            )
            bins = bins * scale
        return bins

    def _augment_feats(self, feats: np.ndarray) -> np.ndarray:
        """Augment feature data (W, F)."""
        if self.augmentation in ("jitter", "all"):
            noise = np.random.normal(0, JITTER_STD, size=feats.shape).astype(np.float32)
            feats = feats + noise
        if self.augmentation in ("scale", "all"):
            scale = np.random.uniform(*SCALE_RANGE, size=(1, feats.shape[1])).astype(
                np.float32
            )
            feats = feats * scale
        return feats


def cnn_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate for CNN dataset."""
    collated: Dict[str, torch.Tensor] = {}
    keys = batch[0].keys()
    for key in keys:
        collated[key] = torch.stack([sample[key] for sample in batch])
    return collated
