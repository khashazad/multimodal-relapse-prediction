"""
Training script for the Multimodal Transformer Fusion model.

Follows the Click CLI pattern from experiment.py for executor integration.
Trains a per-fold model with early stopping on validation AUROC.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

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

from .dataset import MODALITY_ORDER, RelapseDataset, collate_fn
from .model import MultimodalRelapseTransformer


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Binary focal loss for imbalanced classification."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        ce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * ((1 - p_t) ** self.gamma) * ce
        return loss


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


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """Compute evaluation metrics on a dataloader (only valid-label samples)."""
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
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            "f1": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "accuracy": float("nan"),
            "n_samples": 0,
        }

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels).astype(int)
    preds = (probs >= threshold).astype(int)

    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)

    # AUROC/AUPRC require both classes present
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
@click.option("--loss_fn", type=click.Choice(["weighted_bce", "focal"]), default="weighted_bce")
@click.option("--focal_alpha", type=float, default=0.25)
@click.option("--focal_gamma", type=float, default=2.0)
@click.option("--seed", type=int, default=42)
@click.option("--threshold", type=float, default=0.5)
@click.option("--data_dir", type=str, default="data/processed/track1")
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
) -> None:
    t_start = time.time()

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Data ----
    data_path = Path(data_dir) / f"fold_{fold}"
    print(f"Loading fold {fold} from {data_path}...")

    train_ds = RelapseDataset(data_path / "train.pkl")
    val_ds = RelapseDataset(data_path / "val.pkl")
    test_ds = RelapseDataset(data_path / "test.pkl")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, drop_last=False,
    )

    print(f"Train: {len(train_ds)} windows | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ---- Model ----
    # Read window_size from metadata
    metadata_path = Path(data_dir) / "metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)
    window_size = metadata["window_size"]

    model = MultimodalRelapseTransformer(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_fusion_layers=num_fusion_layers,
        dropout=dropout,
        window_size=window_size,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ---- Loss ----
    if loss_fn == "weighted_bce":
        pw = compute_pos_weight(train_ds)
        print(f"Weighted BCE — pos_weight: {pw:.2f}")
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pw, device=device)
        )
    else:
        print(f"Focal loss — alpha: {focal_alpha}, gamma: {focal_gamma}")
        criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    # ---- Optimizer & Scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    # ---- Training ----
    best_val_auroc = -1.0
    best_epoch = 0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_valid = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            logits = model(batch)  # (B,)
            labels = batch["label"]  # (B,)
            valid = batch["label_valid"]  # (B,)

            if not valid.any():
                continue

            # Compute loss only on valid-label samples
            if loss_fn == "weighted_bce":
                loss = criterion(logits[valid], labels[valid])
            else:
                per_sample = criterion(logits[valid], labels[valid])
                loss = per_sample.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * valid.sum().item()
            n_valid += valid.sum().item()

        scheduler.step()
        avg_loss = epoch_loss / max(n_valid, 1)

        # Validation
        val_metrics = evaluate(model, val_loader, device, threshold)
        val_auroc = val_metrics["auroc"]

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_auroc": val_auroc,
            "val_auprc": val_metrics["auprc"],
            "val_f1": val_metrics["f1"],
            "lr": optimizer.param_groups[0]["lr"],
        })

        # Early stopping check
        if not np.isnan(val_auroc) and val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | loss {avg_loss:.4f} | "
                f"val AUROC {val_auroc:.4f} | best {best_val_auroc:.4f} (ep {best_epoch})"
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
    print(f"Val  — AUROC: {val_final['auroc']:.4f} | AUPRC: {val_final['auprc']:.4f} | F1: {val_final['f1']:.4f}")
    print(f"Test — AUROC: {test_final['auroc']:.4f} | AUPRC: {test_final['auprc']:.4f} | F1: {test_final['f1']:.4f}")

    # ---- Save output ----
    output = {
        "exp_name": exp_name,
        "fold": fold,
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

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / output_filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {out_path / output_filename}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    train()
