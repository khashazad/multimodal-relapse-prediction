#!/usr/bin/env python3
"""
Run the CNN+LSTM preprocessing pipeline.

Three modes (same as preprocess_data.py):
  Default:          Full sequential pipeline on all patients.
  --patient PID:    Single patient → intermediate pickle.
  --merge:          Load intermediates → LOSO folds → final output.
"""

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings(
    "ignore", message="Precision loss occurred in moment calculation"
)

from src.bin_extractor import BinExtractor
from src.feature_extractor import FeatureExtractor
from src.preprocess_cnn_loso import CNN_MODALITIES, FC_MODALITIES, CNNLOSOPreprocessor


def _build_preprocessor(cfg: dict) -> CNNLOSOPreprocessor:
    bin_cfg = cfg.get("bin_extraction", {})
    bin_ext = BinExtractor(bin_minutes=bin_cfg.get("bin_minutes", 5))

    fe_cfg = cfg.get("feature_extraction", {})
    feat_ext = FeatureExtractor(
        window_size_minutes=fe_cfg.get("window_size_minutes", 5),
        sample_rate_imu=fe_cfg.get("sample_rate_imu", 20),
        sample_rate_hr=fe_cfg.get("sample_rate_hr", 5),
        coverage_threshold=fe_cfg.get("coverage_threshold", 0.25),
    )

    return CNNLOSOPreprocessor(
        data_root=cfg["data_root"],
        track=cfg["track"],
        window_size=cfg["window_size"],
        stride=cfg.get("stride", 1),
        output_dir=cfg["output_dir"],
        bin_extractor=bin_ext,
        feature_extractor=feat_ext,
        sleep_files_dir=cfg.get("sleep_files_dir"),
        annotations_dir=cfg.get("annotations_dir"),
    )


def _intermediate_dir(cfg: dict) -> Path:
    return Path(cfg["output_dir"]) / f"track{cfg['track']}" / "patients"


def _print_summary(cfg: dict) -> None:
    bin_min = cfg.get("bin_extraction", {}).get("bin_minutes", 5)
    n_bins = 24 * 60 // bin_min
    print("CNN modality channels:")
    for mod, ch in CNN_MODALITIES.items():
        print(f"  {mod:6s}: {ch:3d} channels × {n_bins} bins/day")
    print("FC modality features:")
    for mod, dim in FC_MODALITIES.items():
        print(f"  {mod:6s}: {dim:3d} features")
    print(f"Window: {cfg['window_size']} days (left-padded)")
    print()


def run_patient(cfg: dict, patient_id: str) -> None:
    _print_summary(cfg)
    preprocessor = _build_preprocessor(cfg)

    patients = preprocessor.loader.get_patients()
    if patient_id not in patients:
        print(f"Error: patient '{patient_id}' not found. Available: {patients}", file=sys.stderr)
        sys.exit(1)

    windows = preprocessor.process_single_patient(patient_id)

    inter_dir = _intermediate_dir(cfg)
    inter_dir.mkdir(parents=True, exist_ok=True)
    out_path = inter_dir / f"{patient_id}.pkl"

    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "windows": windows,
                "scalers": preprocessor.patient_scalers.get(patient_id, {}),
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    n_wins = sum(len(v) for v in windows.values())
    print(f"\nSaved {n_wins} windows for {patient_id} → {out_path}")


def run_merge(cfg: dict) -> None:
    inter_dir = _intermediate_dir(cfg)
    inter_files = sorted(inter_dir.glob("P*.pkl"))

    if not inter_files:
        print(f"Error: no patient intermediates found in {inter_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {len(inter_files)} patient intermediate files...")
    all_data: dict = {}
    preprocessor = _build_preprocessor(cfg)

    for pf in inter_files:
        patient_id = pf.stem
        with open(pf, "rb") as f:
            data = pickle.load(f)
        all_data[patient_id] = data["windows"]
        preprocessor.patient_scalers[patient_id] = data["scalers"]
        n_wins = sum(len(v) for v in data["windows"].values())
        print(f"  {patient_id}: {n_wins} windows")

    print("\nOrganising LOSO splits...")
    splits = preprocessor.organize_loso_splits(all_data)

    print("\nSaving processed data...")
    preprocessor.save_processed_data(splits, save_format=cfg.get("save_format", "pickle"))


def run_full(cfg: dict) -> None:
    _print_summary(cfg)
    preprocessor = _build_preprocessor(cfg)
    preprocessor.run(save_format=cfg.get("save_format", "pickle"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CNN+LSTM preprocessing pipeline."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--patient", metavar="PATIENT_ID")
    group.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    config_path = Path(__file__).parent.parent / "configs" / "preprocessing_cnn.json"
    with open(config_path) as f:
        cfg = json.load(f)

    if args.patient:
        run_patient(cfg, args.patient)
    elif args.merge:
        run_merge(cfg)
    else:
        run_full(cfg)


if __name__ == "__main__":
    main()
