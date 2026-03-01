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

from .feature_extractor import MODALITY_DIMS

# Canonical modality ordering used throughout the model
MODALITY_ORDER: List[str] = ["accel", "gyr", "hr", "step", "sleep"]


class RelapseDataset(Dataset):
    """Dataset that wraps a single LOSO fold split (train/val/test pickle).

    Each sample is a window dict produced by ``LOSOPreprocessor``.  The dataset
    converts numpy arrays to tensors and extracts the **last day's label** as
    the prediction target.

    Parameters
    ----------
    pkl_path : str or Path
        Path to a fold pickle file (e.g. ``data/processed/track1/fold_0/train.pkl``).
    """

    def __init__(self, pkl_path: str | Path) -> None:
        with open(pkl_path, "rb") as f:
            self.windows: List[Dict] = pickle.load(f)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        w = self.windows[idx]

        sample: Dict[str, torch.Tensor] = {}

        # Per-modality features and masks
        for mod in MODALITY_ORDER:
            # features: (W, F_mod) float32
            sample[f"{mod}_features"] = torch.from_numpy(
                w["features"][mod].astype(np.float32)
            )
            # modality_mask: (W,) bool — True where sensor had data that day
            sample[f"{mod}_mask"] = torch.from_numpy(
                w["modality_masks"][mod].astype(np.bool_)
            )

        # Padding mask: (W,) bool — True = real day, False = left-padding
        sample["padding_mask"] = torch.from_numpy(
            w["padding_mask"].astype(np.bool_)
        )

        # Last-day label (the prediction target)
        last_label = int(w["labels"][-1])
        # Clamp unknown labels (-1) to 0; track validity separately
        sample["label"] = torch.tensor(max(last_label, 0), dtype=torch.float32)
        sample["label_valid"] = torch.tensor(
            bool(w["label_mask"][-1]), dtype=torch.bool
        )

        return sample


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate that stacks dict-based samples into batched tensors."""
    collated: Dict[str, torch.Tensor] = {}

    keys = batch[0].keys()
    for key in keys:
        collated[key] = torch.stack([sample[key] for sample in batch])

    return collated
