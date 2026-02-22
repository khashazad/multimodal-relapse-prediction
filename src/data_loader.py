"""
Data loader for multimodal relapse prediction dataset.

All five modalities (linacc, gyr, hrm, step, sleep) as well as relapses.csv
are expected to reside in the same sequence directory:
    data/original/track{N}/{patient_id}/{sequence_name}/

Sleep files and annotation CSVs are distributed separately from the main
sensor parquets and must be staged into the sequence directories before using
this loader.  LOSOPreprocessor._stage_supplementary_files() performs this
step automatically at the start of every preprocessing run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Modality filename mapping
# ---------------------------------------------------------------------------

MODALITY_FILES: Dict[str, str] = {
    "linacc": "linacc.parquet",
    "gyr":    "gyr.parquet",
    "hrm":    "hrm.parquet",
    "step":   "step.parquet",
    "sleep":  "sleep.parquet",
}


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class SequenceData:
    """All data for a single patient sequence."""

    patient_id:    str
    sequence_name: str
    split:         str  # 'train', 'val', or 'test'

    linacc:   Optional[pd.DataFrame] = None
    gyr:      Optional[pd.DataFrame] = None
    hrm:      Optional[pd.DataFrame] = None
    step:     Optional[pd.DataFrame] = None
    sleep:    Optional[pd.DataFrame] = None
    relapses: Optional[pd.DataFrame] = None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class MultimodalDataLoader:
    """Load multimodal wearable sensor data for relapse prediction."""

    def __init__(self, data_root: str, track: int = 1) -> None:
        self.data_root  = Path(data_root)
        self.track      = track
        self.track_path = self.data_root / f"track{track}"

        if not self.track_path.exists():
            raise ValueError(f"Track path does not exist: {self.track_path}")

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def get_patients(self) -> List[str]:
        """Return sorted list of patient IDs (directories starting with 'P')."""
        return sorted(
            p.name for p in self.track_path.iterdir()
            if p.is_dir() and p.name.startswith("P")
        )

    def get_sequences(self, patient_id: str) -> List[str]:
        """Return sorted list of sequence names for a patient."""
        patient_path = self.track_path / patient_id
        if not patient_path.exists():
            return []
        return sorted(
            item.name
            for item in patient_path.iterdir()
            if item.is_dir() and item.name.split("_")[0] in {"train", "val", "test"}
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_sequence(self, patient_id: str, sequence_name: str) -> SequenceData:
        """Load all modalities for one sequence.

        Parameters
        ----------
        patient_id:     e.g. 'P1'
        sequence_name:  e.g. 'train_0'

        Returns
        -------
        SequenceData with all available modalities populated.
        """
        seq_path = self.track_path / patient_id / sequence_name

        if not seq_path.exists():
            raise ValueError(f"Sequence path does not exist: {seq_path}")

        split = sequence_name.split("_")[0]  # 'train', 'val', or 'test'

        data = SequenceData(
            patient_id=patient_id,
            sequence_name=sequence_name,
            split=split,
        )

        # Load each modality if the file is present
        for attr, filename in MODALITY_FILES.items():
            filepath = seq_path / filename
            if filepath.exists():
                try:
                    setattr(data, attr, pd.read_parquet(filepath))
                except Exception as exc:  # noqa: BLE001
                    print(f"Warning: could not load {filepath}: {exc}")

        # Load relapse labels
        relapses_path = seq_path / "relapses.csv"
        if relapses_path.exists():
            try:
                data.relapses = pd.read_csv(relapses_path)
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: could not load {relapses_path}: {exc}")

        return data

    def load_all_sequences(self, patient_id: str) -> List[SequenceData]:
        """Load all sequences for a patient."""
        return [
            self.load_sequence(patient_id, seq)
            for seq in self.get_sequences(patient_id)
        ]

    def load_all_patients(self) -> Dict[str, List[SequenceData]]:
        """Load all sequences for all patients."""
        return {p: self.load_all_sequences(p) for p in self.get_patients()}
