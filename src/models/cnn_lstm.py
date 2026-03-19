"""
CNN+LSTM per-modality architecture for relapse prediction.

Architecture adapted from cardiac arrest prediction paper:
  - Per-signal 1D CNN extracts day-level latent representations from binned data
  - LSTM models temporal dependencies across daily latent vectors
  - Ensemble aggregates per-modality predictions

Three component types:
  SignalCNN:         1D CNN for a single day's binned signal → latent vector
  ModalityCNNLSTM:   CNN + LSTM for continuous signal modalities (linacc, gyr, hrm)
  EpisodicBranch:    FC + LSTM for episodic modalities (step, sleep)
  CNNLSTMEnsemble:   Combines all 5 branches with learned or mean fusion
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import BaseRelapseModel

# Modality channel counts (must match dataset_cnn.py)
CNN_MODALITY_CHANNELS = {"linacc": 8, "gyr": 8, "hrm": 4}
FC_MODALITY_DIMS = {"step": 10, "sleep": 9}
ALL_MODALITIES = list(CNN_MODALITY_CHANNELS) + list(FC_MODALITY_DIMS)
N_MODALITIES = len(ALL_MODALITIES)


class SignalCNN(nn.Module):
    """1D CNN that extracts a latent vector from one day's binned signal.

    Architecture: Conv1d → Tanh → MaxPool → Dropout (×2) → GlobalAvgPool

    Input:  (B, C, T)  where C=channels, T=n_bins (288)
    Output: (B, hidden_2)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_1: int = 32,
        hidden_2: int = 16,
        kernel_size: int = 5,
        pool_size: int = 2,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2  # same padding
        self.conv1 = nn.Conv1d(in_channels, hidden_1, kernel_size, padding=pad)
        self.pool1 = nn.MaxPool1d(pool_size)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(hidden_1, hidden_2, kernel_size, padding=pad)
        self.pool2 = nn.MaxPool1d(pool_size)
        self.drop2 = nn.Dropout(dropout)
        self.out_dim = hidden_2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = self.drop1(self.pool1(torch.tanh(self.conv1(x))))
        x = self.drop2(self.pool2(torch.tanh(self.conv2(x))))
        # Global average pool over time dimension
        x = x.mean(dim=2)  # (B, hidden_2)
        return x


class ModalityCNNLSTM(nn.Module):
    """CNN + LSTM branch for a continuous signal modality.

    For each day in the window, CNN extracts a latent vector.
    LSTM processes the sequence of daily latents.
    Final hidden state → linear → logit.

    Input:  (B, W, T, C)  where W=window days, T=n_bins, C=channels
    Output: (B,) logits
    """

    def __init__(
        self,
        in_channels: int,
        cnn_hidden_1: int = 32,
        cnn_hidden_2: int = 16,
        lstm_hidden: int = 16,
        cnn_kernel: int = 5,
        cnn_dropout: float = 0.5,
        lstm_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.cnn = SignalCNN(
            in_channels, cnn_hidden_1, cnn_hidden_2, cnn_kernel, dropout=cnn_dropout
        )
        self.lstm = nn.LSTM(
            input_size=cnn_hidden_2,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0,
        )
        self.drop = nn.Dropout(lstm_dropout)
        self.head = nn.Linear(lstm_hidden, 1)
        self.lstm_hidden = lstm_hidden

    def forward(
        self,
        bins: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        bins: (B, W, T, C)
        padding_mask: (B, W) bool — True for real days
        Returns: (B,) logits
        """
        B, W, T, C = bins.shape

        # Process each day through CNN
        # Reshape to (B*W, C, T) for Conv1d
        x = bins.reshape(B * W, T, C).permute(0, 2, 1)  # (B*W, C, T)
        latents = self.cnn(x)  # (B*W, cnn_out)
        latents = latents.reshape(B, W, -1)  # (B, W, cnn_out)

        # Zero out padded positions
        if padding_mask is not None:
            latents = latents * padding_mask.unsqueeze(-1).float()

        # LSTM over daily latents
        lstm_out, _ = self.lstm(latents)  # (B, W, lstm_hidden)
        # Take the last real day's hidden state
        last_hidden = lstm_out[:, -1, :]  # (B, lstm_hidden)
        logits = self.head(self.drop(last_hidden)).squeeze(-1)  # (B,)
        return logits

    def get_latent(
        self,
        bins: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return LSTM output for fusion (no classification head)."""
        B, W, T, C = bins.shape
        x = bins.reshape(B * W, T, C).permute(0, 2, 1)
        latents = self.cnn(x).reshape(B, W, -1)
        if padding_mask is not None:
            latents = latents * padding_mask.unsqueeze(-1).float()
        lstm_out, _ = self.lstm(latents)
        return self.drop(lstm_out[:, -1, :])  # (B, lstm_hidden)


class EpisodicBranch(nn.Module):
    """FC + LSTM branch for episodic modalities (step, sleep).

    Projects hand-crafted features through FC, then LSTM across days.

    Input:  (B, W, F)  where F=feature dims
    Output: (B,) logits
    """

    def __init__(
        self,
        input_dim: int,
        fc_hidden: int = 16,
        lstm_hidden: int = 16,
        fc_dropout: float = 0.3,
        lstm_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
        )
        self.lstm = nn.LSTM(
            input_size=fc_hidden,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0,
        )
        self.drop = nn.Dropout(lstm_dropout)
        self.head = nn.Linear(lstm_hidden, 1)
        self.lstm_hidden = lstm_hidden

    def forward(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        features: (B, W, F)
        padding_mask: (B, W) bool
        Returns: (B,) logits
        """
        B, W, F = features.shape
        projected = self.fc(features)  # (B, W, fc_hidden)

        if padding_mask is not None:
            projected = projected * padding_mask.unsqueeze(-1).float()

        lstm_out, _ = self.lstm(projected)  # (B, W, lstm_hidden)
        last_hidden = lstm_out[:, -1, :]
        logits = self.head(self.drop(last_hidden)).squeeze(-1)
        return logits

    def get_latent(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return LSTM output for fusion (no classification head)."""
        projected = self.fc(features)
        if padding_mask is not None:
            projected = projected * padding_mask.unsqueeze(-1).float()
        lstm_out, _ = self.lstm(projected)
        return self.drop(lstm_out[:, -1, :])


class CNNLSTMEnsemble(BaseRelapseModel):
    """Ensemble of CNN+LSTM (signal) and FC+LSTM (episodic) branches.

    Fusion modes:
      - "learned": softmax-weighted sum of per-branch logits
      - "mean": simple average of available branch logits

    Supports single-modality mode (modality=...) for independent training.
    """

    def __init__(
        self,
        cnn_hidden_1: int = 32,
        cnn_hidden_2: int = 16,
        lstm_hidden: int = 16,
        fc_hidden: int = 16,
        cnn_kernel: int = 5,
        cnn_dropout: float = 0.5,
        lstm_dropout: float = 0.3,
        fc_dropout: float = 0.3,
        fusion: str = "learned",
        modality: Optional[str] = None,
        # Ignored kwargs for compatibility with get_model
        **kwargs,
    ) -> None:
        super().__init__()
        self.fusion_mode = fusion
        self.single_modality = modality

        if modality is not None:
            # Single-modality mode: only build one branch
            if modality in CNN_MODALITY_CHANNELS:
                self.branch = ModalityCNNLSTM(
                    in_channels=CNN_MODALITY_CHANNELS[modality],
                    cnn_hidden_1=cnn_hidden_1,
                    cnn_hidden_2=cnn_hidden_2,
                    lstm_hidden=lstm_hidden,
                    cnn_kernel=cnn_kernel,
                    cnn_dropout=cnn_dropout,
                    lstm_dropout=lstm_dropout,
                )
            elif modality in FC_MODALITY_DIMS:
                self.branch = EpisodicBranch(
                    input_dim=FC_MODALITY_DIMS[modality],
                    fc_hidden=fc_hidden,
                    lstm_hidden=lstm_hidden,
                    fc_dropout=fc_dropout,
                    lstm_dropout=lstm_dropout,
                )
            else:
                raise ValueError(f"Unknown modality: {modality}")
            return

        # Full ensemble: all 5 branches
        self.cnn_branches = nn.ModuleDict()
        for mod, n_ch in CNN_MODALITY_CHANNELS.items():
            self.cnn_branches[mod] = ModalityCNNLSTM(
                in_channels=n_ch,
                cnn_hidden_1=cnn_hidden_1,
                cnn_hidden_2=cnn_hidden_2,
                lstm_hidden=lstm_hidden,
                cnn_kernel=cnn_kernel,
                cnn_dropout=cnn_dropout,
                lstm_dropout=lstm_dropout,
            )

        self.fc_branches = nn.ModuleDict()
        for mod, dim in FC_MODALITY_DIMS.items():
            self.fc_branches[mod] = EpisodicBranch(
                input_dim=dim,
                fc_hidden=fc_hidden,
                lstm_hidden=lstm_hidden,
                fc_dropout=fc_dropout,
                lstm_dropout=lstm_dropout,
            )

        if fusion == "learned":
            # Learnable logit weights (one per modality)
            self.fusion_weights = nn.Parameter(torch.zeros(N_MODALITIES))
        elif fusion == "concat":
            # Concatenate LSTM hidden states and classify jointly
            total_hidden = lstm_hidden * N_MODALITIES
            self.fusion_head = nn.Sequential(
                nn.Linear(total_hidden, lstm_hidden),
                nn.ReLU(),
                nn.Dropout(lstm_dropout),
                nn.Linear(lstm_hidden, 1),
            )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the model on a collated batch.

        Returns: (B,) logits
        """
        padding_mask = batch["padding_mask"]  # (B, W)

        if self.single_modality is not None:
            return self._forward_single(batch, padding_mask)

        if self.fusion_mode == "concat":
            return self._forward_concat(batch, padding_mask)

        return self._forward_ensemble(batch, padding_mask)

    def _forward_single(
        self, batch: Dict[str, torch.Tensor], padding_mask: torch.Tensor
    ) -> torch.Tensor:
        mod = self.single_modality
        if mod in CNN_MODALITY_CHANNELS:
            return self.branch(batch[f"{mod}_bins"], padding_mask)
        else:
            return self.branch(batch[f"{mod}_features"], padding_mask)

    def _forward_ensemble(
        self, batch: Dict[str, torch.Tensor], padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Weighted average of per-branch logits."""
        B = padding_mask.shape[0]
        device = padding_mask.device

        all_logits = []
        all_avail = []

        for i, mod in enumerate(ALL_MODALITIES):
            avail = batch[f"{mod}_mask"].any(dim=1)  # (B,) bool
            all_avail.append(avail)

            if mod in CNN_MODALITY_CHANNELS:
                logit = self.cnn_branches[mod](batch[f"{mod}_bins"], padding_mask)
            else:
                logit = self.fc_branches[mod](batch[f"{mod}_features"], padding_mask)
            all_logits.append(logit)

        logits = torch.stack(all_logits, dim=1)  # (B, M)
        avail_mask = torch.stack(all_avail, dim=1).float()  # (B, M)

        if self.fusion_mode == "learned":
            weights = torch.softmax(self.fusion_weights, dim=0)  # (M,)
            # Mask unavailable modalities and renormalize
            masked_weights = weights.unsqueeze(0) * avail_mask  # (B, M)
            masked_weights = masked_weights / (masked_weights.sum(dim=1, keepdim=True) + 1e-8)
            fused = (logits * masked_weights).sum(dim=1)  # (B,)
        else:
            # Simple mean over available modalities
            fused = (logits * avail_mask).sum(dim=1) / (avail_mask.sum(dim=1) + 1e-8)

        return fused

    def _forward_concat(
        self, batch: Dict[str, torch.Tensor], padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """Concatenate LSTM hidden states from all branches."""
        latents = []

        for mod in ALL_MODALITIES:
            if mod in CNN_MODALITY_CHANNELS:
                h = self.cnn_branches[mod].get_latent(batch[f"{mod}_bins"], padding_mask)
            else:
                h = self.fc_branches[mod].get_latent(
                    batch[f"{mod}_features"], padding_mask
                )

            # Zero out if modality unavailable
            avail = batch[f"{mod}_mask"].any(dim=1, keepdim=True).float()
            latents.append(h * avail)

        concat = torch.cat(latents, dim=1)  # (B, M * lstm_hidden)
        return self.fusion_head(concat).squeeze(-1)  # (B,)
