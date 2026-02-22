#!/usr/bin/env python3
"""
Run the full dataset preprocessing pipeline.

Three operating modes
---------------------
Default (no flags):
    Run the full pipeline sequentially on all patients.

--patient PATIENT_ID:
    Extract, normalise, and window a single patient, then save an
    intermediate result to <output_dir>/track<N>/patients/<PATIENT_ID>.pkl.
    Used by the SLURM array job (one task per patient).

--merge:
    Load all per-patient intermediate files and organise them into LOSO
    folds, then write the final pickle/numpy output.  Run this after all
    --patient tasks have finished.

Usage
-----
    # Local (sequential)
    python scripts/preprocess_data.py

    # SLURM per-patient task (called by submit_preprocess.sh)
    python scripts/preprocess_data.py --patient P3

    # SLURM merge task (called by submit_preprocess.sh after array completes)
    python scripts/preprocess_data.py --merge
"""

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress noisy scipy RuntimeWarning for nearly-constant windows
warnings.filterwarnings("ignore", message="Precision loss occurred in moment calculation")

from src.feature_extractor import MODALITY_DIMS, FeatureExtractor
from src.preprocess_loso import LOSOPreprocessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_preprocessor(cfg: dict) -> LOSOPreprocessor:
    fe_cfg = cfg.get("feature_extraction", {})
    extractor = FeatureExtractor(
        window_size_minutes=fe_cfg.get("window_size_minutes", 5),
        sample_rate_imu    =fe_cfg.get("sample_rate_imu",     20),
        sample_rate_hr     =fe_cfg.get("sample_rate_hr",       5),
        coverage_threshold =fe_cfg.get("coverage_threshold", 0.25),
    )
    return LOSOPreprocessor(
        data_root        =cfg["data_root"],
        track            =cfg["track"],
        window_size      =cfg["window_size"],
        stride           =cfg.get("stride", 1),
        output_dir       =cfg["output_dir"],
        feature_extractor=extractor,
        sleep_files_dir  =cfg.get("sleep_files_dir"),
        annotations_dir  =cfg.get("annotations_dir"),
    )


def _intermediate_dir(cfg: dict) -> Path:
    return Path(cfg["output_dir"]) / f"track{cfg['track']}" / "patients"


def _print_feature_summary(cfg: dict) -> None:
    print("Feature dimensions per modality:")
    for mod, dim in MODALITY_DIMS.items():
        print(f"  {mod:6s}: {dim:3d} features")
    total = sum(MODALITY_DIMS.values())
    print(f"  {'total':6s}: {total:3d} features")
    print(f"Window: {cfg['window_size']} days (left-padded), stride={cfg.get('stride', 1)}")
    print()


def _print_fold_summary(splits: dict) -> None:
    print("\n" + "=" * 70)
    print(f"{'Fold':<6} {'Test':>4}  {'Train wins':>10}  "
          f"{'S/R (train)':>12}  {'Val wins':>9}  {'S/R (val)':>10}")
    print("-" * 70)
    total_relapse = 0
    total_stable  = 0
    for fold_id, fd in splits.items():
        ts, vs = fd["train_stats"], fd["val_stats"]
        total_stable  += ts["n_stable_days"]
        total_relapse += ts["n_relapse_days"]
        ratio = ts["n_stable_days"] / max(ts["n_relapse_days"], 1)
        print(
            f"{fold_id:<6} {fd['test_patient']:>4}  "
            f"{ts['n_windows']:>10}  "
            f"{ts['n_stable_days']:>5}/{ts['n_relapse_days']:<5} ({ratio:4.0f}:1)  "
            f"{vs['n_windows']:>9}  "
            f"{vs['n_stable_days']:>4}/{vs['n_relapse_days']:<5}"
        )
    print("-" * 70)
    overall_ratio = total_stable / max(total_relapse, 1)
    print(f"Overall class imbalance: {total_stable} stable / "
          f"{total_relapse} relapse ({overall_ratio:.0f}:1)")


# ---------------------------------------------------------------------------
# Mode: --patient
# ---------------------------------------------------------------------------

def run_patient(cfg: dict, patient_id: str) -> None:
    """Process one patient and save an intermediate pickle."""
    _print_feature_summary(cfg)

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
        pickle.dump({
            "windows": windows,
            "scalers": preprocessor.patient_scalers.get(patient_id, {}),
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    n_wins = sum(len(v) for v in windows.values())
    print(f"\nSaved {n_wins} windows for {patient_id} → {out_path}")


# ---------------------------------------------------------------------------
# Mode: --merge
# ---------------------------------------------------------------------------

def run_merge(cfg: dict) -> None:
    """Load all per-patient intermediates, build LOSO folds, save output."""
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

    _print_fold_summary(splits)


# ---------------------------------------------------------------------------
# Mode: full pipeline (default)
# ---------------------------------------------------------------------------

def run_full(cfg: dict) -> None:
    _print_feature_summary(cfg)
    preprocessor = _build_preprocessor(cfg)
    splits = preprocessor.run(save_format=cfg.get("save_format", "pickle"))
    _print_fold_summary(splits)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocessing pipeline for multimodal relapse prediction."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--patient", metavar="PATIENT_ID",
        help="Process a single patient and save an intermediate result "
             "(used by SLURM array tasks).",
    )
    group.add_argument(
        "--merge", action="store_true",
        help="Load per-patient intermediates and create final LOSO folds "
             "(run after all --patient tasks complete).",
    )
    args = parser.parse_args()

    config_path = Path(__file__).parent.parent / "configs" / "preprocessing.json"
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
