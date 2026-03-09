"""
Training script for the Multimodal Transformer Fusion model.

Follows the Click CLI pattern from experiment.py for executor integration.
Trains a per-fold model with early stopping on validation AUROC.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when invoked as `python src/train.py`
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

from src.dataset import RelapseDataset, collate_fn
from src.losses import FocalLoss
from src.models import get_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_pos_weight(dataset: RelapseDataset) -> float:
    """Compute pos_weight = n_neg / n_pos from training data (valid labels only)."""
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
    """Collect all (probs, labels) from a loader (only valid-label samples)."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)
            probs = torch.sigmoid(logits)

            valid = batch["label_valid"]  # (B,)
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
    """Compute evaluation metrics on a dataloader (only valid-label samples)."""
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

    # Guard against NaN predictions (training diverged)
    nan_mask = np.isnan(probs)
    if nan_mask.any():
        probs = np.where(nan_mask, 0.5, probs)

    preds = (probs >= threshold).astype(int)

    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)

    # AUROC/AUPRC require both classes present and finite probabilities
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
    """Find optimal threshold on val set and evaluate on test set.

    Sweeps thresholds [0.05, 0.10, ..., 0.95], picks the one maximizing
    val F1, then reports val and test metrics at both 0.5 and the calibrated
    threshold.
    """
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

    # Evaluate val & test at calibrated threshold
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
# Setup helpers
# ---------------------------------------------------------------------------


def _setup_reproducibility(seed: int) -> torch.device:
    """Set random seeds and return the compute device."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    return device


def _load_fold_data(
    data_dir: str, fold: int, batch_size: int, augmentation: str
) -> tuple[
    RelapseDataset, RelapseDataset, RelapseDataset, DataLoader, DataLoader, DataLoader
]:
    """Load datasets and create dataloaders for a fold."""
    data_path = Path(data_dir) / f"fold_{fold}"
    print(f"Loading fold {fold} from {data_path}...")

    train_ds = RelapseDataset(data_path / "train.pkl", augmentation=augmentation)
    val_ds = RelapseDataset(data_path / "val.pkl")
    test_ds = RelapseDataset(data_path / "test.pkl")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    print(f"Train: {len(train_ds)} windows | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def _build_model(
    model_name: str,
    data_dir: str,
    d_model: int,
    nhead: int,
    num_encoder_layers: int,
    num_fusion_layers: int,
    dropout: float,
    num_bottleneck_tokens: int,
    dann_lambda: float,
    device: torch.device,
) -> tuple[nn.Module, int, int]:
    """Instantiate model and return (model, n_params, window_size)."""
    metadata_path = Path(data_dir) / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)
    window_size = metadata["window_size"]

    model = get_model(
        model_name,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_fusion_layers=num_fusion_layers,
        dropout=dropout,
        window_size=window_size,
        num_bottleneck_tokens=num_bottleneck_tokens,
        dann_lambda=dann_lambda,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name} | Parameters: {n_params:,}")
    return model, n_params, window_size


def _build_loss(
    loss_fn: str,
    train_ds: RelapseDataset,
    focal_alpha: float,
    focal_gamma: float,
    focal_gamma_pos: float | None,
    focal_gamma_neg: float | None,
    device: torch.device,
) -> nn.Module:
    """Configure and return the loss function."""
    if loss_fn == "weighted_bce":
        pw = compute_pos_weight(train_ds)
        print(f"Weighted BCE — pos_weight: {pw:.2f}")
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))

    focal_kwargs = {"alpha": focal_alpha, "gamma": focal_gamma}
    if focal_gamma_pos is not None and focal_gamma_neg is not None:
        focal_kwargs["gamma_pos"] = focal_gamma_pos
        focal_kwargs["gamma_neg"] = focal_gamma_neg
        print(
            f"Asymmetric focal loss — alpha: {focal_alpha}, gamma_pos: {focal_gamma_pos}, gamma_neg: {focal_gamma_neg}"
        )
    else:
        print(f"Focal loss — alpha: {focal_alpha}, gamma: {focal_gamma}")
    return FocalLoss(**focal_kwargs)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler: str,
    epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Configure and return the LR scheduler."""
    if scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if scheduler in ("warmup_constant", "warmup_cosine"):
        warmup_steps = max(warmup_epochs, 1)
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-2, total_iters=warmup_steps
        )
        if scheduler == "warmup_constant":
            main_sched = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=1.0, total_iters=epochs - warmup_steps
            )
        else:  # warmup_cosine
            main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(epochs - warmup_steps, 1)
            )
        lr_sched = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, main_sched],
            milestones=[warmup_steps],
        )
        print(f"Scheduler: {scheduler} (warmup={warmup_steps} epochs)")
        return lr_sched

    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


@click.command()
# Executor-injected params
@click.option("--exp_name", type=str, required=True)
@click.option("--output_dir", type=str, required=True)
@click.option("--output_filename", type=str, required=True)
# Experiment params
@click.option("--fold", type=int, required=True)
@click.option("--d_model", type=int, default=64)
@click.option("--nhead", type=int, default=4)
@click.option("--num_encoder_layers", type=int, default=1)
@click.option("--num_fusion_layers", type=int, default=1)
@click.option("--dropout", type=float, default=0.1)
@click.option("--lr", type=float, default=1e-4)
@click.option("--weight_decay", type=float, default=0.01)
@click.option("--batch_size", type=int, default=32)
@click.option("--epochs", type=int, default=100)
@click.option("--patience", type=int, default=15)
@click.option(
    "--loss_fn", type=click.Choice(["weighted_bce", "focal"]), default="weighted_bce"
)
@click.option("--focal_alpha", type=float, default=0.25)
@click.option("--focal_gamma", type=float, default=2.0)
@click.option("--seed", type=int, default=42)
@click.option("--threshold", type=float, default=0.5)
@click.option("--data_dir", type=str, default="data/processed/track1")
# New experiment options
@click.option("--model_name", type=str, default="transformer_v1")
@click.option("--num_bottleneck_tokens", type=int, default=4)
@click.option("--dann_lambda", type=float, default=0.0)
@click.option("--dann_warmup_epochs", type=int, default=20)
@click.option("--calibrate_threshold", type=bool, default=False)
@click.option("--augmentation", type=str, default="none")
@click.option("--mixup", type=bool, default=False)
@click.option("--mixup_alpha", type=float, default=0.2)
# Focal loss experiment extensions
@click.option(
    "--scheduler",
    type=click.Choice(["cosine", "warmup_constant", "warmup_cosine"]),
    default="cosine",
)
@click.option("--warmup_epochs", type=int, default=0)
@click.option(
    "--early_stop_metric", type=click.Choice(["auroc", "auprc"]), default="auroc"
)
@click.option("--label_smoothing", type=float, default=0.0)
@click.option("--focal_gamma_pos", type=float, default=None)
@click.option("--focal_gamma_neg", type=float, default=None)
def train(
    exp_name: str,
    output_dir: str,
    output_filename: str,
    fold: int,
    d_model: int,
    nhead: int,
    num_encoder_layers: int,
    num_fusion_layers: int,
    dropout: float,
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
    model_name: str,
    num_bottleneck_tokens: int,
    dann_lambda: float,
    dann_warmup_epochs: int,
    calibrate_threshold: bool,
    augmentation: str,
    mixup: bool,
    mixup_alpha: float,
    scheduler: str,
    warmup_epochs: int,
    early_stop_metric: str,
    label_smoothing: float,
    focal_gamma_pos: float | None,
    focal_gamma_neg: float | None,
) -> None:
    t_start = time.time()

    device = _setup_reproducibility(seed)

    # ---- Data ----
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = _load_fold_data(
        data_dir, fold, batch_size, augmentation
    )

    # ---- Model ----
    model, n_params, window_size = _build_model(
        model_name,
        data_dir,
        d_model,
        nhead,
        num_encoder_layers,
        num_fusion_layers,
        dropout,
        num_bottleneck_tokens,
        dann_lambda,
        device,
    )

    # ---- Loss ----
    criterion = _build_loss(
        loss_fn,
        train_ds,
        focal_alpha,
        focal_gamma,
        focal_gamma_pos,
        focal_gamma_neg,
        device,
    )

    # ---- Optimizer & Scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = _build_scheduler(optimizer, scheduler, epochs, warmup_epochs)

    # ---- DANN setup ----
    has_dann = dann_lambda > 0 and hasattr(model, "discriminator")
    if has_dann:
        print(
            f"DANN enabled — lambda: {dann_lambda}, warmup: {dann_warmup_epochs} epochs"
        )

    # ---- Label smoothing ----
    if label_smoothing > 0:
        print(f"Label smoothing: {label_smoothing}")

    # ---- Training ----
    best_val_score = -1.0
    best_epoch = 0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_valid = 0
        epoch_dann_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            # Mixup augmentation
            if mixup and np.random.random() > 0.5:
                batch, lam = _apply_mixup(batch, mixup_alpha)
            else:
                lam = None

            logits = model(batch)  # (B,)
            labels = batch["label"]  # (B,)
            valid = batch["label_valid"]  # (B,)

            if not valid.any():
                continue

            # Apply label smoothing: shift labels away from hard 0/1
            train_labels = labels[valid]
            if label_smoothing > 0:
                train_labels = (
                    train_labels * (1 - label_smoothing) + 0.5 * label_smoothing
                )

            # Compute loss only on valid-label samples
            if loss_fn == "weighted_bce":
                loss = criterion(logits[valid], train_labels)
            else:
                per_sample = criterion(logits[valid], train_labels)
                loss = per_sample.mean()

            # DANN adversarial loss
            if has_dann:
                dann_progress = min(1.0, epoch / max(dann_warmup_epochs, 1))
                current_lambda = dann_lambda * dann_progress
                aux_loss = model.get_auxiliary_loss()
                loss = loss + current_lambda * aux_loss
                epoch_dann_loss += aux_loss.item() * valid.sum().item()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()

            epoch_loss += loss.item() * valid.sum().item()
            n_valid += valid.sum().item()

        lr_scheduler.step()
        avg_loss = epoch_loss / max(n_valid, 1)

        # Detect training divergence
        if np.isnan(avg_loss):
            print(f"Training diverged at epoch {epoch} (NaN loss). Stopping.")
            break

        # Validation
        val_metrics = evaluate(model, val_loader, device, threshold)
        val_score = val_metrics[early_stop_metric]

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
            "val_f1": val_metrics["f1"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        if has_dann:
            epoch_record["dann_loss"] = epoch_dann_loss / max(n_valid, 1)
        history.append(epoch_record)

        # Early stopping check (on configurable metric)
        if not np.isnan(val_score) and val_score > best_val_score:
            best_val_score = val_score
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | loss {avg_loss:.4f} | "
                f"val {early_stop_metric} {val_score:.4f} | best {best_val_score:.4f} (ep {best_epoch})"
            )

        if epoch - best_epoch >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    # ---- Restore best model & evaluate ----
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

    # ---- Threshold calibration (optional) ----
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
        "model_name": model_name,
        "hyperparameters": {
            "d_model": d_model,
            "nhead": nhead,
            "num_encoder_layers": num_encoder_layers,
            "num_fusion_layers": num_fusion_layers,
            "dropout": dropout,
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
            "window_size": window_size,
            "augmentation": augmentation,
            "mixup": mixup,
            "mixup_alpha": mixup_alpha,
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

    if dann_lambda > 0:
        output["hyperparameters"]["dann_lambda"] = dann_lambda
        output["hyperparameters"]["dann_warmup_epochs"] = dann_warmup_epochs

    if num_bottleneck_tokens != 4:
        output["hyperparameters"]["num_bottleneck_tokens"] = num_bottleneck_tokens

    if focal_gamma_pos is not None and focal_gamma_neg is not None:
        output["hyperparameters"]["focal_gamma_pos"] = focal_gamma_pos
        output["hyperparameters"]["focal_gamma_neg"] = focal_gamma_neg

    if calibration_results:
        output.update(calibration_results)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / output_filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {out_path / output_filename}")
    print(f"Total time: {time.time() - t_start:.1f}s")


def _apply_mixup(
    batch: dict[str, torch.Tensor],
    alpha: float,
) -> tuple[dict[str, torch.Tensor], float]:
    """Apply mixup augmentation to a batch.

    Shuffles samples within the batch and interpolates features/labels.
    Uses class-balanced partner selection: relapse samples are paired with
    other relapse samples 50% of the time.
    """
    B = batch["label"].size(0)
    device = batch["label"].device

    lam = float(np.random.beta(alpha, alpha))

    # Class-balanced partner selection
    labels = batch["label"]
    pos_mask = labels == 1
    neg_mask = ~pos_mask

    perm = torch.randperm(B, device=device)
    if pos_mask.sum() > 1 and neg_mask.sum() > 0:
        # For positive samples, 50% chance to pair with another positive
        pos_indices = torch.where(pos_mask)[0]
        for idx in pos_indices:
            if np.random.random() < 0.5 and len(pos_indices) > 1:
                candidates = pos_indices[pos_indices != idx]
                perm[idx] = candidates[torch.randint(len(candidates), (1,))]

    mixed_batch = {}
    for k, v in batch.items():
        if k in ("label", "label_valid"):
            continue
        if v.is_floating_point():
            mixed_batch[k] = lam * v + (1 - lam) * v[perm]
        else:
            mixed_batch[k] = v  # keep masks unchanged
    mixed_batch["label"] = lam * batch["label"] + (1 - lam) * batch["label"][perm]
    mixed_batch["label_valid"] = batch["label_valid"] & batch["label_valid"][perm]

    # Preserve non-mixed keys
    for k in batch:
        if k not in mixed_batch:
            mixed_batch[k] = batch[k]

    return mixed_batch, lam


if __name__ == "__main__":
    train()
