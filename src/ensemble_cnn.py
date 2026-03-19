"""
Post-hoc ensemble of independently-trained CNN+LSTM per-modality models.

After training 5 independent single-modality models, this script:
  1. Loads each modality's checkpoint
  2. Runs inference on val/test sets
  3. Computes per-modality val F1 for weights (paper method)
  4. Produces F1-weighted and rank-average ensemble predictions

Usage:
    python -m src.ensemble_cnn --data_dir data/processed_cnn/track1 \
        --results_dir outputs --exp_name cnn_lstm_independent
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import click
import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.dataset_cnn import CNNRelapseDataset, cnn_collate_fn
from src.models.cnn_lstm import ALL_MODALITIES, CNNLSTMEnsemble

DEFAULT_MODALITIES = ALL_MODALITIES


def _collect_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (probs, labels, valid_mask) arrays for the full dataset."""
    model.eval()
    all_probs = []
    all_labels = []
    all_valid = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch["label"].cpu().numpy())
            all_valid.append(batch["label_valid"].cpu().numpy())

    return (
        np.concatenate(all_probs),
        np.concatenate(all_labels).astype(int),
        np.concatenate(all_valid).astype(bool),
    )


def _eval_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    preds = (probs >= threshold).astype(int)

    if n_pos == 0 or n_neg == 0:
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = float(roc_auc_score(labels, probs))
        auprc = float(average_precision_score(labels, probs))

    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "n_samples": len(labels),
    }


@click.command()
@click.option("--data_dir", type=str, default="data/processed_cnn/track1")
@click.option("--results_dir", type=str, default="outputs")
@click.option("--exp_name", type=str, default="cnn_lstm_independent")
@click.option("--output_file", type=str, default="outputs/cnn_ensemble_results.json")
@click.option("--batch_size", type=int, default=32)
@click.option("--seed", type=int, default=42)
# Model architecture (must match trained models)
@click.option("--cnn_hidden_1", type=int, default=32)
@click.option("--cnn_hidden_2", type=int, default=16)
@click.option("--lstm_hidden", type=int, default=16)
@click.option("--fc_hidden", type=int, default=16)
@click.option("--cnn_kernel", type=int, default=5)
@click.option("--cnn_dropout", type=float, default=0.5)
@click.option("--lstm_dropout", type=float, default=0.3)
@click.option("--fc_dropout", type=float, default=0.3)
def ensemble(
    data_dir: str,
    results_dir: str,
    exp_name: str,
    output_file: str,
    batch_size: int,
    seed: int,
    cnn_hidden_1: int,
    cnn_hidden_2: int,
    lstm_hidden: int,
    fc_hidden: int,
    cnn_kernel: int,
    cnn_dropout: float,
    lstm_dropout: float,
    fc_dropout: float,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_path = Path(data_dir)
    results_path = Path(results_dir)

    # Discover folds
    fold_dirs = sorted(data_path.glob("fold_*"))
    n_folds = len(fold_dirs)
    print(f"Found {n_folds} folds")

    all_fold_results = []

    for fold_id in range(n_folds):
        print(f"\n{'='*50}")
        print(f"Fold {fold_id}")
        print(f"{'='*50}")

        # Load data
        fold_path = data_path / f"fold_{fold_id}"
        val_ds = CNNRelapseDataset(fold_path / "val.pkl")
        test_ds = CNNRelapseDataset(fold_path / "test.pkl")

        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            collate_fn=cnn_collate_fn,
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            collate_fn=cnn_collate_fn,
        )

        # Collect per-modality predictions
        mod_val_probs = {}
        mod_test_probs = {}
        mod_val_f1 = {}
        val_labels = None
        test_labels = None

        for mod in DEFAULT_MODALITIES:
            # Find checkpoint for this modality + fold
            # Naming convention from executor: {exp_name}__fold={fold}__modality={mod}.pt
            ckpt_pattern = f"{exp_name}__fold={fold_id}__modality={mod}.pt"
            ckpt_path = results_path / ckpt_pattern

            if not ckpt_path.exists():
                # Try searching in subdirectories
                matches = list(results_path.rglob(ckpt_pattern))
                if not matches:
                    print(f"  {mod}: checkpoint not found ({ckpt_pattern}), skipping")
                    continue
                ckpt_path = matches[0]

            print(f"  {mod}: loading {ckpt_path}")

            # Build single-modality model
            model = CNNLSTMEnsemble(
                cnn_hidden_1=cnn_hidden_1,
                cnn_hidden_2=cnn_hidden_2,
                lstm_hidden=lstm_hidden,
                fc_hidden=fc_hidden,
                cnn_kernel=cnn_kernel,
                cnn_dropout=cnn_dropout,
                lstm_dropout=lstm_dropout,
                fc_dropout=fc_dropout,
                modality=mod,
            ).to(device)

            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(state)

            # Get predictions
            vp, vl, vv = _collect_probs(model, val_loader, device)
            tp, tl, tv = _collect_probs(model, test_loader, device)

            # Store only valid-label predictions
            mod_val_probs[mod] = vp[vv]
            mod_test_probs[mod] = tp[tv]

            if val_labels is None:
                val_labels = vl[vv]
            if test_labels is None:
                test_labels = tl[tv]

            # Compute val F1 at optimal threshold
            best_f1 = 0.0
            for t in np.arange(0.05, 1.0, 0.05):
                f1_val = float(f1_score(val_labels, (vp[vv] >= t).astype(int), zero_division=0))
                best_f1 = max(best_f1, f1_val)
            mod_val_f1[mod] = best_f1

            val_metrics = _eval_metrics(vp[vv], val_labels)
            test_metrics = _eval_metrics(tp[tv], test_labels)
            print(
                f"    Val AUROC: {val_metrics['auroc']:.4f} | "
                f"Test AUROC: {test_metrics['auroc']:.4f} | "
                f"Best val F1: {best_f1:.4f}"
            )

        if not mod_val_probs:
            print(f"  No modality checkpoints found for fold {fold_id}, skipping")
            continue

        available_mods = list(mod_val_probs.keys())
        n_available = len(available_mods)
        print(f"\n  Ensemble over {n_available} modalities: {available_mods}")

        # --- F1-weighted ensemble (paper method) ---
        f1_weights = np.array([mod_val_f1[m] for m in available_mods])
        f1_sum = f1_weights.sum()
        if f1_sum > 0:
            f1_weights = f1_weights / f1_sum
        else:
            f1_weights = np.ones(n_available) / n_available

        val_f1_weighted = sum(
            mod_val_probs[m] * w for m, w in zip(available_mods, f1_weights)
        )
        test_f1_weighted = sum(
            mod_test_probs[m] * w for m, w in zip(available_mods, f1_weights)
        )

        f1w_val = _eval_metrics(val_f1_weighted, val_labels)
        f1w_test = _eval_metrics(test_f1_weighted, test_labels)

        # --- Rank-average ensemble ---
        val_ranks = np.stack(
            [rankdata(mod_val_probs[m]) / len(val_labels) for m in available_mods]
        )
        test_ranks = np.stack(
            [rankdata(mod_test_probs[m]) / len(test_labels) for m in available_mods]
        )
        val_rank_avg = val_ranks.mean(axis=0)
        test_rank_avg = test_ranks.mean(axis=0)

        ra_val = _eval_metrics(val_rank_avg, val_labels)
        ra_test = _eval_metrics(test_rank_avg, test_labels)

        # --- Simple mean ensemble ---
        val_mean = np.stack([mod_val_probs[m] for m in available_mods]).mean(axis=0)
        test_mean = np.stack([mod_test_probs[m] for m in available_mods]).mean(axis=0)

        mean_val = _eval_metrics(val_mean, val_labels)
        mean_test = _eval_metrics(test_mean, test_labels)

        print(f"\n  F1-weighted: val AUROC {f1w_val['auroc']:.4f} | test AUROC {f1w_test['auroc']:.4f}")
        print(f"  Rank-avg:    val AUROC {ra_val['auroc']:.4f} | test AUROC {ra_test['auroc']:.4f}")
        print(f"  Mean:        val AUROC {mean_val['auroc']:.4f} | test AUROC {mean_test['auroc']:.4f}")

        fold_result = {
            "fold": fold_id,
            "available_modalities": available_mods,
            "per_modality_val_f1": {m: float(mod_val_f1[m]) for m in available_mods},
            "f1_weights": {m: float(w) for m, w in zip(available_mods, f1_weights)},
            "f1_weighted_val": f1w_val,
            "f1_weighted_test": f1w_test,
            "rank_avg_val": ra_val,
            "rank_avg_test": ra_test,
            "mean_val": mean_val,
            "mean_test": mean_test,
        }
        all_fold_results.append(fold_result)

    # --- Summary across folds ---
    if all_fold_results:
        print(f"\n{'='*60}")
        print("Summary across folds")
        print(f"{'='*60}")

        for method in ["f1_weighted", "rank_avg", "mean"]:
            val_aurocs = [r[f"{method}_val"]["auroc"] for r in all_fold_results
                         if not np.isnan(r[f"{method}_val"]["auroc"])]
            test_aurocs = [r[f"{method}_test"]["auroc"] for r in all_fold_results
                          if not np.isnan(r[f"{method}_test"]["auroc"])]

            if val_aurocs:
                print(
                    f"  {method:12s}: val AUROC {np.mean(val_aurocs):.4f} ± {np.std(val_aurocs):.4f} | "
                    f"test AUROC {np.mean(test_aurocs):.4f} ± {np.std(test_aurocs):.4f}"
                )

    # Save
    output = {
        "exp_name": exp_name,
        "ensemble_type": "post_hoc",
        "fold_results": all_fold_results,
    }

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    ensemble()
