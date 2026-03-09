"""
PyTorch Dataset for LOSO fold data.

Loads preprocessed window pickles and returns per-sample dicts with
per-modality feature tensors, masks, and last-day label for binary
relapse prediction.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


# Canonical modality ordering used throughout the model
MODALITY_ORDER: List[str] = ["accel", "gyr", "hr", "step", "sleep"]

JITTER_STD = 0.03
SCALE_RANGE = (0.9, 1.1)


class RelapseDataset(Dataset):
    """Dataset that wraps a single LOSO fold split (train/val/test pickle).

    Each sample is a window dict produced by ``LOSOPreprocessor``.  The dataset
    converts numpy arrays to tensors and extracts the **last day's label** as
    the prediction target.

    Parameters
    ----------
    pkl_path : str or Path
        Path to a fold pickle file (e.g. ``data/processed/track1/fold_0/train.pkl``).
    augmentation : str
        Augmentation strategy: ``"none"``, ``"jitter"``, ``"scale"``, or
        ``"all"`` (jitter + scale).  Only applied during training.
    """

    # Patient ID to integer index mapping
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

        # Per-modality features and masks
        for mod in MODALITY_ORDER:
            # features: (W, F_mod) float32
            feats = w["features"][mod].astype(np.float32)

            # Apply augmentation to features
            if self.augmentation != "none":
                feats = self._augment(feats)

            sample[f"{mod}_features"] = torch.from_numpy(feats)
            # modality_mask: (W,) bool — True where sensor had data that day
            sample[f"{mod}_mask"] = torch.from_numpy(
                w["modality_masks"][mod].astype(np.bool_)
            )

        # Padding mask: (W,) bool — True = real day, False = left-padding
        sample["padding_mask"] = torch.from_numpy(w["padding_mask"].astype(np.bool_))

        # Last-day label (the prediction target)
        last_label = int(w["labels"][-1])
        # Clamp unknown labels (-1) to 0; track validity separately
        sample["label"] = torch.tensor(max(last_label, 0), dtype=torch.float32)
        sample["label_valid"] = torch.tensor(
            bool(w["label_mask"][-1]), dtype=torch.bool
        )

        # Patient index for DANN (if patient_id is available)
        patient_id = w.get("patient_id", None)
        if patient_id is not None and patient_id in self.PATIENT_IDX:
            sample["patient_idx"] = torch.tensor(
                self.PATIENT_IDX[patient_id], dtype=torch.long
            )
        else:
            sample["patient_idx"] = torch.tensor(-1, dtype=torch.long)

        return sample

    def _augment(self, feats: np.ndarray) -> np.ndarray:
        """Apply augmentation transforms to feature array (W, F)."""
        if self.augmentation in ("jitter", "all"):
            noise = np.random.normal(0, JITTER_STD, size=feats.shape).astype(np.float32)
            feats = feats + noise
        if self.augmentation in ("scale", "all"):
            scale = np.random.uniform(*SCALE_RANGE, size=(1, feats.shape[1])).astype(
                np.float32
            )
            feats = feats * scale
        return feats


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate that stacks dict-based samples into batched tensors."""
    collated: Dict[str, torch.Tensor] = {}

    keys = batch[0].keys()
    for key in keys:
        collated[key] = torch.stack([sample[key] for sample in batch])

    return collated
