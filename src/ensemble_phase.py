"""
Ensemble phase experiments: heterogeneous stacking, FiLM conditioning,
and test-time adaptation for relapse prediction.

Methods (--method):
  base_model — train transformer/TCN/XGBoost, save predictions + checkpoint
  film       — FiLM-conditioned transformer (diagnosis-aware)
  tta        — test-time adaptation (entropy minimization / feature renorm)

Usage (via executor):
  python src/ensemble_phase.py --method base_model --model_type transformer --fold 0 ...
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

from src.losses.focal import FocalLoss

# Reuse models + data helpers from ablation
from src.ablation import (
    SeqTransformer,
    TCN,
    create_seqs_padded,
    apply_smote,
    get_feature_cols,
    load_data,
)


# =============================================================================
# Diagnosis mapping for FiLM
# =============================================================================

DIAGNOSIS_MAP = {
    "P3": 0, "P4": 0, "P5": 0, "P6": 0, "P8": 0, "P9": 0,  # bipolar
    "P2": 1, "P7": 1,                                         # schiz-spectrum
    "P1": 2,                                                   # brief-psychotic
}
UNKNOWN_DIAGNOSIS_ID = 3
N_DIAGNOSIS_GROUPS = 4


# =============================================================================
# FiLM-conditioned Transformer
# =============================================================================

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: gamma * x + beta, per diagnosis group."""

    def __init__(self, d_model: int, n_conditions: int = N_DIAGNOSIS_GROUPS):
        super().__init__()
        self.gamma = nn.Embedding(n_conditions, d_model)
        self.beta = nn.Embedding(n_conditions, d_model)
        # Init to identity transform
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, x: torch.Tensor, condition_ids: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D), condition_ids: (B,)
        g = self.gamma(condition_ids).unsqueeze(1)  # (B, 1, D)
        b = self.beta(condition_ids).unsqueeze(1)
        return g * x + b


class FiLMTransformer(nn.Module):
    """Transformer with FiLM conditioning after each encoder layer."""

    def __init__(self, n_features, d_model=1024, nhead=4, num_layers=3,
                 dropout=0.3, n_conditions=N_DIAGNOSIS_GROUPS):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 500, d_model))

        self.layers = nn.ModuleList()
        self.film_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead,
                    dim_feedforward=d_model * 4,
                    dropout=dropout, activation="relu", batch_first=True,
                )
            )
            self.film_layers.append(FiLMLayer(d_model, n_conditions))

        self.fc = nn.Linear(d_model, 1)

    def forward(self, x, src_key_padding_mask=None, condition_ids=None):
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :x.size(1)]

        for layer, film in zip(self.layers, self.film_layers):
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
            if condition_ids is not None:
                x = film(x, condition_ids)

        return self.fc(x[:, -1]).squeeze(-1)


# =============================================================================
# Data helpers with patient tracking
# =============================================================================

def create_seqs_padded_with_patient(df, feature_cols, seq_length):
    """Like create_seqs_padded but also returns patient_id per window."""
    seqs, labels, masks, patient_ids = [], [], [], []
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
            patient_ids.append(pid)
    if not seqs:
        return (
            np.empty((0, seq_length, n_feats), dtype=np.float32),
            np.array([], dtype=np.int64),
            np.empty((0, seq_length), dtype=bool),
            [],
        )
    return (
        np.stack(seqs).astype(np.float32),
        np.array(labels, dtype=np.int64),
        np.stack(masks),
        patient_ids,
    )


# =============================================================================
# Training: base_model (transformer / TCN / XGBoost)
# =============================================================================

def train_fold_with_checkpoint(
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
    loss_type: str = "focal",
    focal_gamma: float = 1.0,
    focal_alpha: float = 0.7,
    label_smoothing: float = 0.2,
    weight_decay: float = 0.0,
    save_checkpoint_path: str | None = None,
) -> dict:
    """Train single LOPO fold, optionally save checkpoint + scaler."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_feats = len(feature_cols)

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

    if loss_type == "focal":
        crit = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    else:
        crit = nn.BCEWithLogitsLoss()

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_auroc, best_state, best_epoch = -1.0, None, -1
    history = []

    for epoch in range(n_epochs):
        model.train()
        ep_loss = 0.0
        for Xb, yb, mb in loader:
            Xb, yb, mb = Xb.to(device), yb.to(device), mb.to(device)
            opt.zero_grad()
            targets = yb
            if label_smoothing > 0:
                targets = yb * (1 - label_smoothing) + 0.5 * label_smoothing
            loss = crit(model(Xb, src_key_padding_mask=mb), targets)
            if loss.dim() > 0:
                loss = loss.mean()
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

        history.append({"epoch": epoch + 1, "train_loss": ep_loss, "test_auroc": ep_auroc})
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/{n_epochs}  loss={ep_loss:.4f}  "
                  f"AUROC={ep_auroc:.4f}  best={best_auroc:.4f}")

    if best_state is None:
        return {
            "auroc": float("nan"), "auprc": float("nan"),
            "f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0,
            "best_epoch": -1,
            "n_test_windows": n_test_windows, "n_test_relapse": n_test_relapse,
            "n_train_windows": n_train_windows, "n_train_relapse": n_train_relapse,
            "model_params": n_params,
            "y_true": y_te.tolist(), "y_score": [],
            "history": history, "error": "training diverged",
        }

    model.load_state_dict(best_state)
    model.to(device).eval()
    with torch.no_grad():
        final_probs = torch.sigmoid(
            model(X_te_t.to(device), src_key_padding_mask=m_te_t.to(device))
        ).cpu().numpy()

    if np.any(np.isnan(final_probs)):
        return {
            "auroc": float("nan"), "auprc": float("nan"),
            "f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0,
            "best_epoch": best_epoch,
            "n_test_windows": n_test_windows, "n_test_relapse": n_test_relapse,
            "n_train_windows": n_train_windows, "n_train_relapse": n_train_relapse,
            "model_params": n_params,
            "y_true": y_te.tolist(), "y_score": [],
            "history": history, "error": "NaN in predictions",
        }

    # Save checkpoint
    if save_checkpoint_path:
        Path(save_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": best_state,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "n_features": n_feats,
            "feature_cols": feature_cols,
        }, save_checkpoint_path)
        print(f"  Checkpoint saved → {save_checkpoint_path}")

    auroc = float(roc_auc_score(y_te, final_probs))
    auprc = float(average_precision_score(y_te, final_probs))
    preds = (final_probs >= 0.5).astype(int)

    return {
        "auroc": auroc, "auprc": auprc,
        "f1": float(f1_score(y_te, preds, zero_division=0)),
        "precision": float(precision_score(y_te, preds, zero_division=0)),
        "recall": float(recall_score(y_te, preds, zero_division=0)),
        "accuracy": float(accuracy_score(y_te, preds)),
        "best_epoch": best_epoch,
        "n_test_windows": n_test_windows, "n_test_relapse": n_test_relapse,
        "n_train_windows": n_train_windows, "n_train_relapse": n_train_relapse,
        "model_params": n_params,
        "y_true": y_te.tolist(), "y_score": final_probs.tolist(),
        "history": history,
    }


def train_xgboost_fold(
    data_df,
    feature_cols: list[str],
    test_patient: str,
    seed: int = 42,
) -> dict:
    """Train XGBoost on last-day features (no sequence modeling)."""
    import xgboost as xgb

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

    X_tr = tr_df[feature_cols].fillna(0).values
    y_tr = tr_df["relapse"].fillna(0).astype(int).values
    X_te = te_df[feature_cols].fillna(0).values
    y_te = te_df["relapse"].fillna(0).astype(int).values

    if len(np.unique(y_te)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"),
                "error": "single class in test"}

    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    spw = n_neg / max(n_pos, 1)

    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, scale_pos_weight=spw,
        eval_metric="logloss", random_state=seed, verbosity=0,
    )
    clf.fit(X_tr, y_tr)
    probs = clf.predict_proba(X_te)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auroc = float(roc_auc_score(y_te, probs))
    auprc = float(average_precision_score(y_te, probs))

    return {
        "auroc": auroc, "auprc": auprc,
        "f1": float(f1_score(y_te, preds, zero_division=0)),
        "precision": float(precision_score(y_te, preds, zero_division=0)),
        "recall": float(recall_score(y_te, preds, zero_division=0)),
        "accuracy": float(accuracy_score(y_te, preds)),
        "best_epoch": -1,
        "n_test_windows": len(y_te), "n_test_relapse": int(y_te.sum()),
        "n_train_windows": len(y_tr), "n_train_relapse": int(y_tr.sum()),
        "model_params": 0,
        "y_true": y_te.tolist(), "y_score": probs.tolist(),
        "history": [],
    }


# =============================================================================
# Training: FiLM-conditioned transformer
# =============================================================================

def train_film_fold(
    data_df,
    feature_cols: list[str],
    test_patient: str,
    all_patients: list[str],
    d_model: int = 1024,
    nhead: int = 4,
    n_layers: int = 3,
    dropout: float = 0.3,
    seq_len: int = 7,
    batch_size: int = 32,
    lr: float = 1e-3,
    n_epochs: int = 80,
    seed: int = 42,
    loss_type: str = "focal",
    focal_gamma: float = 1.0,
    focal_alpha: float = 0.7,
    label_smoothing: float = 0.2,
) -> dict:
    """Train FiLM-conditioned transformer for a single fold."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_feats = len(feature_cols)

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

    scaler = StandardScaler()
    scaler.fit(tr_df[feature_cols].fillna(0).values)
    tr_s = tr_df.copy()
    te_s = te_df.copy()
    tr_s[feature_cols] = scaler.transform(tr_df[feature_cols].fillna(0).values)
    te_s[feature_cols] = scaler.transform(te_df[feature_cols].fillna(0).values)

    # Create sequences with patient tracking
    X_tr, y_tr, m_tr, pids_tr = create_seqs_padded_with_patient(tr_s, feature_cols, seq_len)
    X_te, y_te, m_te, pids_te = create_seqs_padded_with_patient(te_s, feature_cols, seq_len)

    if len(X_te) == 0 or len(np.unique(y_te)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"),
                "error": "insufficient test data after sequencing"}

    n_test_windows = len(X_te)
    n_test_relapse = int(y_te.sum())

    # Map patient IDs to diagnosis IDs
    diag_tr = np.array([DIAGNOSIS_MAP.get(p, UNKNOWN_DIAGNOSIS_ID) for p in pids_tr],
                       dtype=np.int64)
    diag_te = np.array([DIAGNOSIS_MAP.get(p, UNKNOWN_DIAGNOSIS_ID) for p in pids_te],
                       dtype=np.int64)

    # Check: if test patient's diagnosis group has 0 training examples, use unknown
    test_diag = DIAGNOSIS_MAP.get(test_patient, UNKNOWN_DIAGNOSIS_ID)
    train_diag_set = set(diag_tr)
    if test_diag not in train_diag_set:
        print(f"  Warning: test diagnosis group {test_diag} not in training. Using identity FiLM (id={UNKNOWN_DIAGNOSIS_ID}).")
        diag_te[:] = UNKNOWN_DIAGNOSIS_ID

    # SMOTE on features (diagnosis IDs not augmented — recomputed from patient IDs)
    X_tr_aug, y_tr_aug, m_tr_aug = apply_smote(X_tr, y_tr, m_tr, seq_len, n_feats)
    # For augmented samples, replicate diagnosis IDs (SMOTE adds to the end)
    n_original = len(diag_tr)
    n_augmented = len(X_tr_aug)
    if n_augmented > n_original:
        # Augmented samples inherit the majority of positive-class diagnosis distribution
        pos_diags = diag_tr[y_tr == 1]
        if len(pos_diags) > 0:
            extra_diags = np.random.choice(pos_diags, size=n_augmented - n_original, replace=True)
        else:
            extra_diags = np.full(n_augmented - n_original, UNKNOWN_DIAGNOSIS_ID, dtype=np.int64)
        diag_tr_aug = np.concatenate([diag_tr, extra_diags])
    else:
        diag_tr_aug = diag_tr[:n_augmented]

    n_train_windows = len(X_tr_aug)
    n_train_relapse = int(y_tr_aug.sum())

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_tr_aug).float(),
            torch.from_numpy(y_tr_aug.astype(np.float32)),
            torch.from_numpy(m_tr_aug),
            torch.from_numpy(diag_tr_aug),
        ),
        batch_size=batch_size, shuffle=True, drop_last=False,
    )
    X_te_t = torch.from_numpy(X_te).float()
    m_te_t = torch.from_numpy(m_te)
    d_te_t = torch.from_numpy(diag_te)

    model = FiLMTransformer(
        n_features=n_feats, d_model=d_model, nhead=nhead,
        num_layers=n_layers, dropout=dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if loss_type == "focal":
        crit = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    else:
        crit = nn.BCEWithLogitsLoss()

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_auroc, best_state, best_epoch = -1.0, None, -1
    history = []

    for epoch in range(n_epochs):
        model.train()
        ep_loss = 0.0
        for Xb, yb, mb, db in loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            mb = mb.to(device)
            db = db.to(device)
            opt.zero_grad()
            targets = yb
            if label_smoothing > 0:
                targets = yb * (1 - label_smoothing) + 0.5 * label_smoothing
            logits = model(Xb, src_key_padding_mask=mb, condition_ids=db)
            loss = crit(logits, targets)
            if loss.dim() > 0:
                loss = loss.mean()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(Xb)
        ep_loss /= max(n_train_windows, 1)

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(
                model(X_te_t.to(device), src_key_padding_mask=m_te_t.to(device),
                      condition_ids=d_te_t.to(device))
            ).cpu().numpy()
        try:
            ep_auroc = roc_auc_score(y_te, probs)
        except Exception:
            ep_auroc = 0.5

        if ep_auroc > best_auroc:
            best_auroc = ep_auroc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

        history.append({"epoch": epoch + 1, "train_loss": ep_loss, "test_auroc": ep_auroc})
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/{n_epochs}  loss={ep_loss:.4f}  "
                  f"AUROC={ep_auroc:.4f}  best={best_auroc:.4f}")

    if best_state is None:
        return {
            "auroc": float("nan"), "auprc": float("nan"),
            "f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0,
            "best_epoch": -1,
            "n_test_windows": n_test_windows, "n_test_relapse": n_test_relapse,
            "n_train_windows": n_train_windows, "n_train_relapse": n_train_relapse,
            "model_params": n_params,
            "y_true": y_te.tolist(), "y_score": [],
            "history": history, "error": "training diverged",
        }

    model.load_state_dict(best_state)
    model.to(device).eval()
    with torch.no_grad():
        final_probs = torch.sigmoid(
            model(X_te_t.to(device), src_key_padding_mask=m_te_t.to(device),
                  condition_ids=d_te_t.to(device))
        ).cpu().numpy()

    if np.any(np.isnan(final_probs)):
        return {
            "auroc": float("nan"), "auprc": float("nan"),
            "f1": 0.0, "precision": 0.0, "recall": 0.0, "accuracy": 0.0,
            "best_epoch": best_epoch,
            "n_test_windows": n_test_windows, "n_test_relapse": n_test_relapse,
            "n_train_windows": n_train_windows, "n_train_relapse": n_train_relapse,
            "model_params": n_params,
            "y_true": y_te.tolist(), "y_score": [],
            "history": history, "error": "NaN in predictions",
        }

    auroc = float(roc_auc_score(y_te, final_probs))
    auprc = float(average_precision_score(y_te, final_probs))
    preds = (final_probs >= 0.5).astype(int)

    return {
        "auroc": auroc, "auprc": auprc,
        "f1": float(f1_score(y_te, preds, zero_division=0)),
        "precision": float(precision_score(y_te, preds, zero_division=0)),
        "recall": float(recall_score(y_te, preds, zero_division=0)),
        "accuracy": float(accuracy_score(y_te, preds)),
        "best_epoch": best_epoch,
        "n_test_windows": n_test_windows, "n_test_relapse": n_test_relapse,
        "n_train_windows": n_train_windows, "n_train_relapse": n_train_relapse,
        "model_params": n_params,
        "y_true": y_te.tolist(), "y_score": final_probs.tolist(),
        "history": history,
    }


# =============================================================================
# TTA: Test-Time Adaptation
# =============================================================================

def apply_tta_entropy(model, X_te_t, m_te_t, n_steps=10, lr=1e-4, device="cpu"):
    """Adapt LayerNorm params by minimizing prediction entropy on test data."""
    model = copy.deepcopy(model).to(device)
    # Freeze all params
    for p in model.parameters():
        p.requires_grad_(False)
    # Unfreeze LayerNorm params only
    ln_params = []
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            for p in m.parameters():
                p.requires_grad_(True)
                ln_params.append(p)

    if not ln_params:
        print("  Warning: no LayerNorm params found for TTA")
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(
                model(X_te_t.to(device), src_key_padding_mask=m_te_t.to(device))
            ).cpu().numpy()
        return probs

    opt = torch.optim.Adam(ln_params, lr=lr)

    for step in range(n_steps):
        model.train()
        logits = model(X_te_t.to(device), src_key_padding_mask=m_te_t.to(device))
        p = torch.sigmoid(logits)
        # Binary entropy: H = -(p*log(p) + (1-p)*log(1-p))
        entropy = -(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8)).mean()
        opt.zero_grad()
        entropy.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(
            model(X_te_t.to(device), src_key_padding_mask=m_te_t.to(device))
        ).cpu().numpy()
    return probs


def apply_tta_renorm(checkpoint, test_df, feature_cols, seq_len, alpha=0.3):
    """Blend train/test scaler stats, then run inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mu_train = checkpoint["scaler_mean"]
    scale_train = checkpoint["scaler_scale"]
    n_feats = checkpoint["n_features"]

    test_vals = test_df[feature_cols].fillna(0).values
    mu_test = test_vals.mean(axis=0)
    scale_test = test_vals.std(axis=0) + 1e-8

    mu_blend = (1 - alpha) * mu_train + alpha * mu_test
    scale_blend = (1 - alpha) * scale_train + alpha * scale_test

    # Apply blended scaler
    te_s = test_df.copy()
    te_s[feature_cols] = (test_vals - mu_blend) / scale_blend

    X_te, y_te, m_te = create_seqs_padded(te_s, feature_cols, seq_len)

    # Rebuild model and load weights
    model = SeqTransformer(n_features=n_feats, d_model=checkpoint.get("d_model", 1024),
                           nhead=4, num_layers=3, dropout=0.3)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    with torch.no_grad():
        probs = torch.sigmoid(
            model(torch.from_numpy(X_te).float().to(device),
                  src_key_padding_mask=torch.from_numpy(m_te).to(device))
        ).cpu().numpy()

    return probs, y_te


def run_tta(
    data_df,
    feature_cols: list[str],
    test_patient: str,
    checkpoint_path: str,
    tta_variant: str = "entropy_ln",
    tta_lr: float = 1e-4,
    tta_steps: int = 10,
    tta_alpha: float = 0.3,
    seq_len: int = 7,
    d_model: int = 1024,
) -> dict:
    """Run test-time adaptation on a pre-trained checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        return {"auroc": float("nan"), "auprc": float("nan"),
                "error": f"checkpoint not found: {checkpoint_path}"}

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    n_feats = checkpoint["n_features"]
    saved_feature_cols = checkpoint["feature_cols"]

    te_df = data_df[
        (data_df["patient_id"] == test_patient) &
        (data_df["split_type"] == "test")
    ].copy()

    if len(te_df) == 0:
        return {"auroc": float("nan"), "auprc": float("nan"), "error": "empty test"}

    y_te_raw = te_df["relapse"].fillna(0).astype(int).values
    if len(np.unique(y_te_raw)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"),
                "error": "single class in test"}

    if tta_variant == "feature_renorm":
        # Feature renormalization uses raw test data + blended scaler
        checkpoint["d_model"] = d_model
        probs, y_te = apply_tta_renorm(
            checkpoint, te_df, saved_feature_cols, seq_len, alpha=tta_alpha,
        )
    elif tta_variant == "entropy_ln":
        # Standard scaling first (using checkpoint scaler)
        te_s = te_df.copy()
        te_s[saved_feature_cols] = (
            (te_df[saved_feature_cols].fillna(0).values - checkpoint["scaler_mean"])
            / checkpoint["scaler_scale"]
        )
        X_te, y_te, m_te = create_seqs_padded(te_s, saved_feature_cols, seq_len)

        model = SeqTransformer(n_features=n_feats, d_model=d_model,
                               nhead=4, num_layers=3, dropout=0.3)
        model.load_state_dict(checkpoint["model_state_dict"])

        X_te_t = torch.from_numpy(X_te).float()
        m_te_t = torch.from_numpy(m_te)

        probs = apply_tta_entropy(model, X_te_t, m_te_t,
                                  n_steps=tta_steps, lr=tta_lr, device=device)
    else:
        return {"auroc": float("nan"), "auprc": float("nan"),
                "error": f"unknown tta_variant: {tta_variant}"}

    if len(probs) == 0 or len(y_te) == 0 or len(np.unique(y_te)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"),
                "error": "insufficient test data"}

    if np.any(np.isnan(probs)):
        return {"auroc": float("nan"), "auprc": float("nan"),
                "y_true": y_te.tolist(), "y_score": [],
                "error": "NaN in TTA predictions"}

    auroc = float(roc_auc_score(y_te, probs))
    auprc = float(average_precision_score(y_te, probs))
    preds = (probs >= 0.5).astype(int)

    return {
        "auroc": auroc, "auprc": auprc,
        "f1": float(f1_score(y_te, preds, zero_division=0)),
        "precision": float(precision_score(y_te, preds, zero_division=0)),
        "recall": float(recall_score(y_te, preds, zero_division=0)),
        "accuracy": float(accuracy_score(y_te, preds)),
        "best_epoch": -1,
        "n_test_windows": len(y_te), "n_test_relapse": int(y_te.sum()),
        "n_train_windows": 0, "n_train_relapse": 0,
        "model_params": 0,
        "y_true": y_te.tolist(), "y_score": probs.tolist(),
        "history": [],
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
@click.option("--method", type=click.Choice(["base_model", "film", "tta"]),
              required=True)
@click.option("--fold", type=int, required=True, help="LOPO fold index (0-8)")
# Model type for base_model
@click.option("--model_type", type=str, default="transformer",
              help="transformer|tcn|xgboost")
# Transformer params
@click.option("--d_model", type=int, default=1024)
@click.option("--n_layers", type=int, default=3)
@click.option("--dropout", type=float, default=0.3)
@click.option("--nhead", type=int, default=4)
# TCN params
@click.option("--n_channels", type=int, default=128)
@click.option("--tcn_layers", type=int, default=4)
@click.option("--tcn_dropout", type=float, default=0.3)
# Feature set
@click.option("--feature_set", type=str, default="union",
              help="union|all")
# Training params
@click.option("--seq_len", type=int, default=7)
@click.option("--batch_size", type=int, default=32)
@click.option("--lr", type=float, default=1e-3)
@click.option("--n_epochs", type=int, default=80)
@click.option("--seed", type=int, default=42)
@click.option("--data_path", type=str,
              default="data/processed/patient_data_export_all9.pkl")
# Loss params
@click.option("--loss_type", type=str, default="focal")
@click.option("--focal_gamma", type=float, default=1.0)
@click.option("--focal_alpha", type=float, default=0.7)
@click.option("--label_smoothing", type=float, default=0.2)
# Checkpoint
@click.option("--save_checkpoint", type=bool, default=False)
# TTA params
@click.option("--tta_variant", type=str, default="entropy_ln",
              help="entropy_ln|feature_renorm")
@click.option("--tta_lr", type=float, default=1e-4)
@click.option("--tta_steps", type=int, default=10)
@click.option("--tta_alpha", type=float, default=0.3)
@click.option("--base_d_model", type=int, default=1024)
def main(
    exp_name, output_dir, output_filename,
    method, fold, model_type,
    d_model, n_layers, dropout, nhead,
    n_channels, tcn_layers, tcn_dropout,
    feature_set,
    seq_len, batch_size, lr, n_epochs, seed, data_path,
    loss_type, focal_gamma, focal_alpha, label_smoothing,
    save_checkpoint,
    tta_variant, tta_lr, tta_steps, tta_alpha, base_d_model,
):
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print(f"ensemble_phase: method={method}  fold={fold}  device={device}")
    if method == "base_model":
        print(f"  model_type={model_type}  feature_set={feature_set}")
    elif method == "film":
        print(f"  d_model={d_model}  feature_set={feature_set}")
    elif method == "tta":
        print(f"  tta_variant={tta_variant}  tta_lr={tta_lr}  tta_steps={tta_steps}")
    print("=" * 70)

    export_data = load_data(data_path)
    combined_df = export_data["combined_data"]
    all_patients = export_data["patient_list"]
    test_patient = all_patients[fold]
    print(f"Test patient: {test_patient} (fold {fold})")

    feature_cols = get_feature_cols(export_data, feature_set)
    print(f"Feature set: {feature_set} ({len(feature_cols)} features)")

    hp_info = {}

    if method == "base_model":
        if model_type == "xgboost":
            result = train_xgboost_fold(
                data_df=combined_df,
                feature_cols=feature_cols,
                test_patient=test_patient,
                seed=seed,
            )
            hp_info = {"model_type": "xgboost", "n_estimators": 200, "max_depth": 6}
        else:
            # Transformer or TCN
            if model_type == "transformer":
                _d, _nl, _dr, _nh = d_model, n_layers, dropout, nhead

                def model_factory(n_feats):
                    return SeqTransformer(
                        n_features=n_feats, d_model=_d, nhead=_nh,
                        num_layers=_nl, dropout=_dr,
                    )
                hp_info = {"model_type": "transformer", "d_model": _d,
                           "n_layers": _nl, "dropout": _dr, "nhead": _nh}
            elif model_type == "tcn":
                _nc, _tl, _td = n_channels, tcn_layers, tcn_dropout

                def model_factory(n_feats):
                    return TCN(
                        n_features=n_feats, n_channels=_nc,
                        n_layers=_tl, dropout=_td,
                    )
                hp_info = {"model_type": "tcn", "n_channels": _nc,
                           "tcn_layers": _tl, "tcn_dropout": _td}

            ckpt_path = None
            if save_checkpoint:
                ckpt_dir = Path(output_dir) / "checkpoints"
                ckpt_path = str(ckpt_dir / f"{model_type}_{d_model}_{feature_set}_fold{fold}.pt")

            result = train_fold_with_checkpoint(
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
                loss_type=loss_type,
                focal_gamma=focal_gamma,
                focal_alpha=focal_alpha,
                label_smoothing=label_smoothing,
                save_checkpoint_path=ckpt_path,
            )

    elif method == "film":
        result = train_film_fold(
            data_df=combined_df,
            feature_cols=feature_cols,
            test_patient=test_patient,
            all_patients=all_patients,
            d_model=d_model,
            nhead=nhead,
            n_layers=n_layers,
            dropout=dropout,
            seq_len=seq_len,
            batch_size=batch_size,
            lr=lr,
            n_epochs=n_epochs,
            seed=seed,
            loss_type=loss_type,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            label_smoothing=label_smoothing,
        )
        hp_info = {"model_type": "film_transformer", "d_model": d_model,
                   "n_layers": n_layers, "dropout": dropout, "nhead": nhead}

    elif method == "tta":
        ckpt_dir = Path(output_dir) / "checkpoints"
        # Always use union checkpoint (all-feature checkpoints removed to save quota)
        ckpt_path = str(ckpt_dir / f"transformer_{base_d_model}_union_fold{fold}.pt")
        if not os.path.exists(ckpt_path):
            # Fallback: try feature_set-specific checkpoint
            ckpt_path = str(ckpt_dir / f"transformer_{base_d_model}_{feature_set}_fold{fold}.pt")
        result = run_tta(
            data_df=combined_df,
            feature_cols=feature_cols,
            test_patient=test_patient,
            checkpoint_path=ckpt_path,
            tta_variant=tta_variant,
            tta_lr=tta_lr,
            tta_steps=tta_steps,
            tta_alpha=tta_alpha,
            seq_len=seq_len,
            d_model=base_d_model,
        )
        hp_info = {"tta_variant": tta_variant, "tta_lr": tta_lr,
                   "tta_steps": tta_steps, "tta_alpha": tta_alpha,
                   "base_d_model": base_d_model}

    print(f"\n{test_patient}: AUROC={result.get('auroc', 'nan'):.4f}  "
          f"AUPRC={result.get('auprc', 'nan'):.4f}")

    output = {
        "exp_name": exp_name,
        "method": method,
        "fold": fold,
        "test_patient": test_patient,
        "feature_set": feature_set,
        "hyperparameters": {
            **hp_info,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "lr": lr,
            "n_epochs": n_epochs,
            "seed": seed,
        },
        "loss_params": {
            "loss_type": loss_type,
            "focal_gamma": focal_gamma,
            "focal_alpha": focal_alpha,
            "label_smoothing": label_smoothing,
        },
        "n_features": len(feature_cols),
        "device": str(device),
        "test_metrics": {
            "auroc": result.get("auroc", float("nan")),
            "auprc": result.get("auprc", float("nan")),
            "f1": result.get("f1", 0.0),
            "precision": result.get("precision", 0.0),
            "recall": result.get("recall", 0.0),
            "accuracy": result.get("accuracy", 0.0),
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

    if result.get("error"):
        output["error"] = result["error"]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / output_filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved → {out_path / output_filename}")
    print(f"Total time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
