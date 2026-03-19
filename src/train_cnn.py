"""
Training script for the CNN+LSTM per-modality architecture.

Supports two training modes:
  - Joint: all 5 branches trained end-to-end with fusion
  - Independent: single modality branch (for post-hoc ensemble)

Click CLI for executor integration (same pattern as train.py).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import click
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from src.dataset_cnn import CNNRelapseDataset, cnn_collate_fn
from src.losses import FocalLoss
from src.models.cnn_lstm import CNNLSTMEnsemble


# ---------------------------------------------------------------------------
# Helpers (adapted from train.py)
# ---------------------------------------------------------------------------


def compute_pos_weight(dataset: CNNRelapseDataset) -> float:
    n_pos = 0
    n_neg = 0
    for w in dataset.windows:
        if w["label_mask"][-1]:
            if w["labels"][-1] == 1:
                n_pos += 1
            else:
                n_neg += 1
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)
            probs = torch.sigmoid(logits)

            valid = batch["label_valid"]
            if valid.any():
                all_probs.append(probs[valid].cpu().numpy())
                all_labels.append(batch["label"][valid].cpu().numpy())

    if not all_probs:
        return np.array([]), np.array([])

    return np.concatenate(all_probs), np.concatenate(all_labels).astype(int)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    probs, labels = _collect_predictions(model, loader, device)

    if len(probs) == 0:
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "f1": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "accuracy": float("nan"),
            "n_samples": 0,
        }

    nan_mask = np.isnan(probs)
    if nan_mask.any():
        probs = np.where(nan_mask, 0.5, probs)

    preds = (probs >= threshold).astype(int)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)

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
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
        "n_samples": len(labels),
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


def find_optimal_threshold(
    model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> dict:
    val_probs, val_labels = _collect_predictions(model, val_loader, device)

    if len(val_probs) == 0:
        return {"calibrated_threshold": 0.5}

    thresholds = np.arange(0.05, 1.0, 0.05)
    best_f1 = -1.0
    best_thresh = 0.5

    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        f1 = float(f1_score(val_labels, preds, zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(round(t, 2))

    cal_val = evaluate(model, val_loader, device, threshold=best_thresh)
    cal_test = evaluate(model, test_loader, device, threshold=best_thresh)

    return {
        "calibrated_threshold": best_thresh,
        "calibrated_val_f1": best_f1,
        "calibrated_val_metrics": cal_val,
        "calibrated_test_metrics": cal_test,
    }


GRAD_CLIP_NORM = 1.0
LOG_EVERY = 10


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
# Executor-injected params
@click.option("--exp_name", type=str, required=True)
@click.option("--output_dir", type=str, required=True)
@click.option("--output_filename", type=str, required=True)
# Fold
@click.option("--fold", type=int, required=True)
# CNN architecture
@click.option("--cnn_hidden_1", type=int, default=32)
@click.option("--cnn_hidden_2", type=int, default=16)
@click.option("--lstm_hidden", type=int, default=16)
@click.option("--fc_hidden", type=int, default=16)
@click.option("--cnn_kernel", type=int, default=5)
@click.option("--cnn_dropout", type=float, default=0.5)
@click.option("--lstm_dropout", type=float, default=0.3)
@click.option("--fc_dropout", type=float, default=0.3)
# Fusion
@click.option(
    "--fusion",
    type=click.Choice(["learned", "mean", "concat"]),
    default="learned",
)
@click.option("--modality", type=str, default=None, help="Single modality for independent training")
# Training
@click.option("--lr", type=float, default=1e-3)
@click.option("--weight_decay", type=float, default=0.01)
@click.option("--batch_size", type=int, default=32)
@click.option("--epochs", type=int, default=100)
@click.option("--patience", type=int, default=15)
@click.option(
    "--loss_fn", type=click.Choice(["weighted_bce", "focal"]), default="focal"
)
@click.option("--focal_alpha", type=float, default=0.7)
@click.option("--focal_gamma", type=float, default=1.0)
@click.option("--seed", type=int, default=42)
@click.option("--threshold", type=float, default=0.5)
@click.option("--data_dir", type=str, default="data/processed_cnn/track1")
@click.option("--calibrate_threshold", type=str, default="true")
@click.option("--augmentation", type=str, default="none")
@click.option(
    "--scheduler",
    type=click.Choice(["cosine", "warmup_cosine"]),
    default="cosine",
)
@click.option("--warmup_epochs", type=int, default=0)
@click.option(
    "--early_stop_metric", type=click.Choice(["auroc", "auprc"]), default="auroc"
)
@click.option("--label_smoothing", type=float, default=0.0)
def train(
    exp_name: str,
    output_dir: str,
    output_filename: str,
    fold: int,
    cnn_hidden_1: int,
    cnn_hidden_2: int,
    lstm_hidden: int,
    fc_hidden: int,
    cnn_kernel: int,
    cnn_dropout: float,
    lstm_dropout: float,
    fc_dropout: float,
    fusion: str,
    modality: str | None,
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    patience: int,
    loss_fn: str,
    focal_alpha: float,
    focal_gamma: float,
    seed: int,
    threshold: float,
    data_dir: str,
    calibrate_threshold: str,
    augmentation: str,
    scheduler: str,
    warmup_epochs: int,
    early_stop_metric: str,
    label_smoothing: float,
) -> None:
    t_start = time.time()
    calibrate_threshold = calibrate_threshold.lower() in ("true", "1", "yes")

    # ---- Reproducibility ----
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Data ----
    data_path = Path(data_dir) / f"fold_{fold}"
    print(f"Loading fold {fold} from {data_path}...")

    train_ds = CNNRelapseDataset(data_path / "train.pkl", augmentation=augmentation)
    val_ds = CNNRelapseDataset(data_path / "val.pkl")
    test_ds = CNNRelapseDataset(data_path / "test.pkl")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=cnn_collate_fn, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=cnn_collate_fn, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=cnn_collate_fn, drop_last=False,
    )
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ---- Model ----
    model = CNNLSTMEnsemble(
        cnn_hidden_1=cnn_hidden_1,
        cnn_hidden_2=cnn_hidden_2,
        lstm_hidden=lstm_hidden,
        fc_hidden=fc_hidden,
        cnn_kernel=cnn_kernel,
        cnn_dropout=cnn_dropout,
        lstm_dropout=lstm_dropout,
        fc_dropout=fc_dropout,
        fusion=fusion,
        modality=modality,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mode_str = f"single:{modality}" if modality else f"ensemble:{fusion}"
    print(f"CNN+LSTM ({mode_str}) | Parameters: {n_params:,}")

    # ---- Loss ----
    if loss_fn == "weighted_bce":
        pw = compute_pos_weight(train_ds)
        print(f"Weighted BCE — pos_weight: {pw:.2f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))
    else:
        print(f"Focal loss — alpha: {focal_alpha}, gamma: {focal_gamma}")
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    # ---- Optimizer & Scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler == "warmup_cosine" and warmup_epochs > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-2, total_iters=warmup_epochs
        )
        main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs - warmup_epochs, 1)
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, main_sched],
            milestones=[warmup_epochs],
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )

    # ---- Training ----
    best_val_score = -1.0
    best_epoch = 0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_valid = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            logits = model(batch)
            labels = batch["label"]
            valid = batch["label_valid"]

            if not valid.any():
                continue

            train_labels = labels[valid]
            if label_smoothing > 0:
                train_labels = train_labels * (1 - label_smoothing) + 0.5 * label_smoothing

            if loss_fn == "weighted_bce":
                loss = criterion(logits[valid], train_labels)
            else:
                per_sample = criterion(logits[valid], train_labels)
                loss = per_sample.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()

            epoch_loss += loss.item() * valid.sum().item()
            n_valid += valid.sum().item()

        lr_scheduler.step()
        avg_loss = epoch_loss / max(n_valid, 1)

        if np.isnan(avg_loss):
            print(f"Training diverged at epoch {epoch} (NaN loss). Stopping.")
            break

        val_metrics = evaluate(model, val_loader, device, threshold)
        val_score = val_metrics[early_stop_metric]

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
            "val_f1": val_metrics["f1"],
            "lr": optimizer.param_groups[0]["lr"],
        })

        if not np.isnan(val_score) and val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | loss {avg_loss:.4f} | "
                f"val {early_stop_metric} {val_score:.4f} | "
                f"best {best_val_score:.4f} (ep {best_epoch})"
            )

        if epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    # ---- Restore best & evaluate ----
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    val_final = evaluate(model, val_loader, device, threshold)
    test_final = evaluate(model, test_loader, device, threshold)

    print(f"\nBest epoch: {best_epoch}")
    print(
        f"Val  — AUROC: {val_final['auroc']:.4f} | AUPRC: {val_final['auprc']:.4f} | F1: {val_final['f1']:.4f}"
    )
    print(
        f"Test — AUROC: {test_final['auroc']:.4f} | AUPRC: {test_final['auprc']:.4f} | F1: {test_final['f1']:.4f}"
    )

    # ---- Threshold calibration ----
    calibration_results = {}
    if calibrate_threshold:
        calibration_results = find_optimal_threshold(
            model, val_loader, test_loader, device
        )
        ct = calibration_results["calibrated_threshold"]
        cf1 = calibration_results.get("calibrated_val_f1", float("nan"))
        print(f"\nCalibrated threshold: {ct:.2f} (val F1: {cf1:.4f})")
        cal_test = calibration_results.get("calibrated_test_metrics", {})
        print(f"Calibrated test — F1: {cal_test.get('f1', float('nan')):.4f}")

    # ---- Save output ----
    output = {
        "exp_name": exp_name,
        "fold": fold,
        "model_name": "cnn_lstm",
        "mode": mode_str,
        "hyperparameters": {
            "cnn_hidden_1": cnn_hidden_1,
            "cnn_hidden_2": cnn_hidden_2,
            "lstm_hidden": lstm_hidden,
            "fc_hidden": fc_hidden,
            "cnn_kernel": cnn_kernel,
            "cnn_dropout": cnn_dropout,
            "lstm_dropout": lstm_dropout,
            "fc_dropout": fc_dropout,
            "fusion": fusion,
            "modality": modality,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience,
            "loss_fn": loss_fn,
            "focal_alpha": focal_alpha,
            "focal_gamma": focal_gamma,
            "seed": seed,
            "threshold": threshold,
            "augmentation": augmentation,
            "scheduler": scheduler,
            "warmup_epochs": warmup_epochs,
            "early_stop_metric": early_stop_metric,
            "label_smoothing": label_smoothing,
        },
        "model_params": n_params,
        "device": str(device),
        "best_epoch": best_epoch,
        "val_metrics": val_final,
        "test_metrics": test_final,
        "history": history,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "elapsed_seconds": time.time() - t_start,
    }

    if calibration_results:
        output.update(calibration_results)

    # Save model checkpoint alongside results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path / output_filename, "w") as f:
        json.dump(output, f, indent=2)

    # Save model weights for ensemble
    ckpt_filename = output_filename.replace(".json", ".pt")
    if best_state is not None:
        torch.save(best_state, out_path / ckpt_filename)

    print(f"\nResults saved to {out_path / output_filename}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    train()
