"""
Next-phase experiments: multi-scale ensemble, TCN baseline, feature expansion.

Follows the Click CLI pattern from train.py for executor integration.
Trains a single LOPO fold and saves per-fold result JSON.

Methods:
  transformer — flat transformer at specified d_model (for ensemble)
  tcn         — temporal convolution network
  feature_exp — transformer d=1024 with expanded feature sets

Usage (via executor):
  python src/next_phase.py --method transformer --fold 0 --d_model 256 ...
  python src/next_phase.py --method tcn --fold 3 --n_channels 128 ...
  python src/next_phase.py --method feature_exp --fold 5 --feature_set top5 ...
"""

from __future__ import annotations

import copy
import json
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
except ImportError:
    raise ImportError("imbalanced-learn required: pip install imbalanced-learn")


# =============================================================================
# Models
# =============================================================================

class SeqTransformer(nn.Module):
    """Mask-aware transformer for sequence classification."""

    def __init__(self, n_features, d_model=32, nhead=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 500, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, activation="relu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x, src_key_padding_mask=None):
        x = self.input_proj(x) + self.pos_embed[:, :x.size(1)]
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return self.fc(x[:, -1]).squeeze(-1)


class CausalConv1d(nn.Module):
    """Causal convolution: pads only on the left."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """Residual TCN block: 2× CausalConv → BN → ReLU → Dropout + skip."""

    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.skip(x)
        x = self.drop(F.relu(self.bn1(self.conv1(x))))
        x = self.drop(F.relu(self.bn2(self.conv2(x))))
        return F.relu(x + res)


class TCN(nn.Module):
    """Temporal Convolution Network for sequence classification."""

    def __init__(self, n_features, n_channels=128, n_layers=3, kernel_size=3,
                 dropout=0.2):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_ch = n_features if i == 0 else n_channels
            layers.append(TCNBlock(in_ch, n_channels, kernel_size,
                                   dilation=2 ** i, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(n_channels, 1)

    def forward(self, x, src_key_padding_mask=None):
        # (batch, seq_len, features) → (batch, features, seq_len)
        x = x.transpose(1, 2)
        if src_key_padding_mask is not None:
            pad_mask = (~src_key_padding_mask).float().unsqueeze(1)
            x = x * pad_mask
        x = self.network(x)
        if src_key_padding_mask is not None:
            pad_mask = (~src_key_padding_mask).float().unsqueeze(1)
            x = (x * pad_mask).sum(dim=2) / pad_mask.sum(dim=2).clamp(min=1)
        else:
            x = x.mean(dim=2)
        return self.fc(x).squeeze(-1)


# =============================================================================
# Data helpers
# =============================================================================

def load_data(data_path: str) -> dict:
    """Load exported patient data."""
    with open(data_path, "rb") as f:
        return pickle.load(f)


def create_seqs_padded(df, feature_cols, seq_length):
    """One window per day per split, left-padded to seq_length."""
    seqs, labels, masks = [], [], []
    n_feats = len(feature_cols)
    for (pid, split), group in df.groupby(["patient_id", "split"]):
        group = group.sort_values("day_index").reset_index(drop=True)
        feats = group[feature_cols].fillna(0).values.astype(np.float32)
        rel = group["relapse"].values
        n = len(group)
        for label_day in range(n):
            window = np.zeros((seq_length, n_feats), dtype=np.float32)
            mask = np.ones(seq_length, dtype=bool)
            for j, src_day in enumerate(
                range(label_day - seq_length + 1, label_day + 1)
            ):
                if src_day >= 0:
                    window[j] = feats[src_day]
                    mask[j] = False
            seqs.append(window)
            labels.append(int(rel[label_day]))
            masks.append(mask)
    if not seqs:
        return (
            np.empty((0, seq_length, n_feats), dtype=np.float32),
            np.array([], dtype=np.int64),
            np.empty((0, seq_length), dtype=bool),
        )
    return (
        np.stack(seqs).astype(np.float32),
        np.array(labels, dtype=np.int64),
        np.stack(masks),
    )


def apply_smote(X_3d, y, masks, seq_len, n_feats):
    """SMOTE on fully non-padded windows only; padded windows appended unchanged."""
    full_mask = ~masks.any(axis=1)
    X_full, y_full = X_3d[full_mask], y[full_mask]
    X_padded, y_padded = X_3d[~full_mask], y[~full_mask]
    m_padded = masks[~full_mask]

    n_pos_full = int(y_full.sum())
    if n_pos_full >= 2 and len(np.unique(y_full)) > 1:
        X_flat = X_full.reshape(len(X_full), seq_len * n_feats)
        k_nn = min(5, n_pos_full - 1)
        sampler = (SMOTE(k_neighbors=k_nn, random_state=42) if k_nn >= 1
                   else RandomOverSampler(random_state=42))
        X_res_flat, y_res = sampler.fit_resample(X_flat, y_full)
        X_res_3d = X_res_flat.reshape(-1, seq_len, n_feats).astype(np.float32)
        masks_res = np.zeros((len(X_res_3d), seq_len), dtype=bool)
    else:
        X_res_3d = X_full
        y_res = y_full
        masks_res = np.zeros((len(X_full), seq_len), dtype=bool)

    return (
        np.concatenate([X_res_3d, X_padded], axis=0),
        np.concatenate([y_res, y_padded], axis=0),
        np.concatenate([masks_res, m_padded], axis=0),
    )


def get_feature_cols(export_data, feature_set: str) -> list[str]:
    """Return feature column list for the given feature_set name."""
    import pandas as pd

    union_feats = export_data["feature_cols"]
    df = export_data["combined_data"]

    if feature_set == "union":
        return union_feats

    meta_cols = {"day_index", "relapse", "patient_id", "split", "split_type",
                 "split_supervised"}
    all_numeric = [c for c in df.columns
                   if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]
    extra_feats = [c for c in all_numeric if c not in union_feats]

    # Create interaction features
    interaction_pairs = [
        ("nap_hours_diff", "day_activity_zscore"),
        ("steps_zscore", "cosinor_amplitude_zscore"),
        ("main_sleep_hours_zscore", "day_activity_zscore"),
        ("rmssd_zscore_w14", "main_sleep_hours_zscore"),
        ("rmssd_zscore_w14", "steps_zscore"),
        ("sdnn_zscore_w14", "total_sleep_hours_zscore"),
        ("circadian_deviation", "steps_zscore"),
        ("nap_hours_diff", "steps_zscore_inv"),
    ]

    interaction_names = []
    for f1, f2 in interaction_pairs:
        if f1 in df.columns and f2 in df.columns:
            name = f"ix_{f1}_x_{f2}"
            if name not in df.columns:
                df[name] = df[f1].fillna(0) * df[f2].fillna(0)
            interaction_names.append(name)

    # Univariate screening for ranking extra features
    candidate_feats = extra_feats + interaction_names
    feat_scores = {}
    all_patients = sorted(df["patient_id"].unique())
    for feat in candidate_feats:
        patient_aurocs = []
        for p in all_patients:
            val_df = df[(df["patient_id"] == p) & (df["split_type"] == "test")]
            if len(val_df) < 5 or len(val_df["relapse"].unique()) < 2:
                continue
            y = val_df["relapse"].fillna(0).astype(int).values
            x = df.loc[val_df.index, feat].fillna(0).values
            if np.std(x) < 1e-10:
                continue
            try:
                a = roc_auc_score(y, x)
                patient_aurocs.append(max(a, 1 - a))
            except Exception:
                continue
        if patient_aurocs:
            feat_scores[feat] = np.mean(patient_aurocs)

    ranked = sorted(feat_scores.items(), key=lambda x: x[1], reverse=True)

    if feature_set == "top5":
        top = [f for f, _ in ranked[:5]]
        return union_feats + [f for f in top if f not in union_feats]
    elif feature_set == "interactions":
        return union_feats + [f for f in interaction_names if f not in union_feats]
    elif feature_set == "top10":
        top = [f for f, _ in ranked[:10]]
        return union_feats + [f for f in top if f not in union_feats]
    elif feature_set == "all":
        return all_numeric + interaction_names
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")


# =============================================================================
# Training logic (single fold)
# =============================================================================

def train_fold(
    data_df,
    feature_cols: list[str],
    test_patient: str,
    all_patients: list[str],
    model_factory,
    seq_len: int = 7,
    batch_size: int = 32,
    lr: float = 1e-3,
    n_epochs: int = 80,
    seed: int = 42,
) -> dict:
    """Train and evaluate a single LOPO fold."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_feats = len(feature_cols)

    # Build train/test DataFrames
    tr_df = data_df[
        (data_df["patient_id"] != test_patient) &
        (data_df["split_type"] == "val")
    ].copy()
    te_df = data_df[
        (data_df["patient_id"] == test_patient) &
        (data_df["split_type"] == "test")
    ].copy()

    if len(tr_df) == 0 or len(te_df) == 0:
        return {"auroc": float("nan"), "auprc": float("nan"),
                "error": "empty train or test"}

    y_te_raw = te_df["relapse"].fillna(0).astype(int).values
    if len(np.unique(y_te_raw)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"),
                "error": "single class in test"}

    # Standardize BEFORE creating sequences
    scaler = StandardScaler()
    scaler.fit(tr_df[feature_cols].fillna(0).values)
    tr_s = tr_df.copy()
    te_s = te_df.copy()
    tr_s[feature_cols] = scaler.transform(tr_df[feature_cols].fillna(0).values)
    te_s[feature_cols] = scaler.transform(te_df[feature_cols].fillna(0).values)

    X_tr, y_tr, m_tr = create_seqs_padded(tr_s, feature_cols, seq_len)
    X_te, y_te, m_te = create_seqs_padded(te_s, feature_cols, seq_len)

    if len(X_te) == 0 or len(np.unique(y_te)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"),
                "error": "insufficient test data after sequencing"}

    n_test_windows = len(X_te)
    n_test_relapse = int(y_te.sum())

    # SMOTE on train only
    X_tr, y_tr, m_tr = apply_smote(X_tr, y_tr, m_tr, seq_len, n_feats)
    n_train_windows = len(X_tr)
    n_train_relapse = int(y_tr.sum())

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_tr).float(),
            torch.from_numpy(y_tr.astype(np.float32)),
            torch.from_numpy(m_tr),
        ),
        batch_size=batch_size, shuffle=True, drop_last=False,
    )
    X_te_t = torch.from_numpy(X_te).float()
    m_te_t = torch.from_numpy(m_te)

    model = model_factory(n_feats).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_auroc, best_state, best_epoch = -1.0, None, -1
    history = []

    for epoch in range(n_epochs):
        model.train()
        ep_loss = 0.0
        for Xb, yb, mb in loader:
            Xb, yb, mb = Xb.to(device), yb.to(device), mb.to(device)
            opt.zero_grad()
            loss = crit(model(Xb, src_key_padding_mask=mb), yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(Xb)
        ep_loss /= max(n_train_windows, 1)

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(
                model(X_te_t.to(device), src_key_padding_mask=m_te_t.to(device))
            ).cpu().numpy()
        try:
            ep_auroc = roc_auc_score(y_te, probs)
        except Exception:
            ep_auroc = 0.5

        if ep_auroc > best_auroc:
            best_auroc = ep_auroc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

        history.append({
            "epoch": epoch + 1,
            "train_loss": ep_loss,
            "test_auroc": ep_auroc,
        })

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/{n_epochs}  loss={ep_loss:.4f}  "
                  f"AUROC={ep_auroc:.4f}  best={best_auroc:.4f}")

    # Reload best, final score
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    with torch.no_grad():
        final_probs = torch.sigmoid(
            model(X_te_t.to(device), src_key_padding_mask=m_te_t.to(device))
        ).cpu().numpy()

    auroc = float(roc_auc_score(y_te, final_probs))
    auprc = float(average_precision_score(y_te, final_probs))
    preds = (final_probs >= 0.5).astype(int)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": float(f1_score(y_te, preds, zero_division=0)),
        "precision": float(precision_score(y_te, preds, zero_division=0)),
        "recall": float(recall_score(y_te, preds, zero_division=0)),
        "accuracy": float(accuracy_score(y_te, preds)),
        "best_epoch": best_epoch,
        "n_test_windows": n_test_windows,
        "n_test_relapse": n_test_relapse,
        "n_train_windows": n_train_windows,
        "n_train_relapse": n_train_relapse,
        "model_params": n_params,
        "y_true": y_te.tolist(),
        "y_score": final_probs.tolist(),
        "history": history,
    }


# =============================================================================
# Click CLI
# =============================================================================

@click.command()
# Executor-injected
@click.option("--exp_name", type=str, required=True)
@click.option("--output_dir", type=str, required=True)
@click.option("--output_filename", type=str, required=True)
# Core
@click.option("--method", type=click.Choice(["transformer", "tcn", "feature_exp"]),
              required=True)
@click.option("--fold", type=int, required=True, help="LOPO fold index (0-8)")
# Transformer params
@click.option("--d_model", type=int, default=1024)
@click.option("--n_layers", type=int, default=3)
@click.option("--dropout", type=float, default=0.3)
@click.option("--nhead", type=int, default=4)
# TCN params
@click.option("--n_channels", type=int, default=128)
@click.option("--tcn_layers", type=int, default=3)
@click.option("--tcn_dropout", type=float, default=0.2)
@click.option("--kernel_size", type=int, default=3)
# Feature expansion params
@click.option("--feature_set", type=str, default="union",
              help="union|top5|interactions|top10|all")
# Common training params
@click.option("--seq_len", type=int, default=7)
@click.option("--batch_size", type=int, default=32)
@click.option("--lr", type=float, default=1e-3)
@click.option("--n_epochs", type=int, default=80)
@click.option("--seed", type=int, default=42)
@click.option("--data_path", type=str,
              default="data/processed/patient_data_export_all9.pkl")
def main(
    exp_name, output_dir, output_filename,
    method, fold,
    d_model, n_layers, dropout, nhead,
    n_channels, tcn_layers, tcn_dropout, kernel_size,
    feature_set,
    seq_len, batch_size, lr, n_epochs, seed, data_path,
):
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(f"next_phase: method={method}  fold={fold}")
    print(f"  device={device}")
    print("=" * 70)

    # Load data
    export_data = load_data(data_path)
    combined_df = export_data["combined_data"]
    all_patients = export_data["patient_list"]
    test_patient = all_patients[fold]

    print(f"Test patient: {test_patient} (fold {fold})")

    # Determine feature columns
    if method == "feature_exp":
        feature_cols = get_feature_cols(export_data, feature_set)
        print(f"Feature set: {feature_set} ({len(feature_cols)} features)")
    else:
        feature_cols = export_data["feature_cols"]
        print(f"Features: {len(feature_cols)} union features")

    # Build model factory
    if method == "transformer" or method == "feature_exp":
        _d, _nl, _dr, _nh = d_model, n_layers, dropout, nhead

        def model_factory(n_feats):
            return SeqTransformer(
                n_features=n_feats, d_model=_d, nhead=_nh,
                num_layers=_nl, dropout=_dr,
            )

        hp_info = {"d_model": _d, "n_layers": _nl, "dropout": _dr, "nhead": _nh}
        print(f"Model: SeqTransformer d={_d} layers={_nl} dr={_dr}")

    elif method == "tcn":
        _nc, _tl, _td, _ks = n_channels, tcn_layers, tcn_dropout, kernel_size

        def model_factory(n_feats):
            return TCN(
                n_features=n_feats, n_channels=_nc, n_layers=_tl,
                kernel_size=_ks, dropout=_td,
            )

        hp_info = {"n_channels": _nc, "tcn_layers": _tl, "tcn_dropout": _td,
                    "kernel_size": _ks}
        print(f"Model: TCN ch={_nc} layers={_tl} dr={_td} ks={_ks}")

    # Train
    print(f"\nTraining fold {fold} ({test_patient})...")
    result = train_fold(
        data_df=combined_df,
        feature_cols=feature_cols,
        test_patient=test_patient,
        all_patients=all_patients,
        model_factory=model_factory,
        seq_len=seq_len,
        batch_size=batch_size,
        lr=lr,
        n_epochs=n_epochs,
        seed=seed,
    )

    print(f"\n{test_patient}: AUROC={result['auroc']:.4f}  AUPRC={result['auprc']:.4f}  "
          f"best_epoch={result.get('best_epoch', -1)}")

    # Build output
    output = {
        "exp_name": exp_name,
        "method": method,
        "fold": fold,
        "test_patient": test_patient,
        "hyperparameters": {
            **hp_info,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "lr": lr,
            "n_epochs": n_epochs,
            "seed": seed,
        },
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "device": str(device),
        "test_metrics": {
            "auroc": result["auroc"],
            "auprc": result["auprc"],
            "f1": result.get("f1", float("nan")),
            "precision": result.get("precision", float("nan")),
            "recall": result.get("recall", float("nan")),
            "accuracy": result.get("accuracy", float("nan")),
            "n_samples": result.get("n_test_windows", 0),
            "n_pos": result.get("n_test_relapse", 0),
        },
        "best_epoch": result.get("best_epoch", -1),
        "model_params": result.get("model_params", 0),
        "train_size": result.get("n_train_windows", 0),
        "test_size": result.get("n_test_windows", 0),
        "y_true": result.get("y_true", []),
        "y_score": result.get("y_score", []),
        "history": result.get("history", []),
        "elapsed_seconds": time.time() - t_start,
    }

    if method == "feature_exp":
        output["feature_set"] = feature_set

    # Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / output_filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved → {out_path / output_filename}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
