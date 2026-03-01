"""
Multimodal Transformer Fusion model for relapse prediction.

Three-stage architecture:
  1. ModalityEncoder  — per-modality transformer (independent weights per sensor)
  2. FusionTransformer — cross-modal attention over modality summary tokens
  3. MultimodalRelapseTransformer — top-level module with classification head
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .dataset import MODALITY_ORDER
from .feature_extractor import MODALITY_DIMS


class ModalityEncoder(nn.Module):
    """Encodes a single modality's temporal features into a fixed-size summary.

    Projects raw features to d_model, prepends a learnable CLS token, adds
    positional embeddings, and runs a small TransformerEncoder.  Returns the
    CLS token output as the modality summary.

    Parameters
    ----------
    input_dim : int
        Number of raw features per day for this modality.
    d_model : int
        Transformer hidden dimension.
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of TransformerEncoder layers.
    dropout : float
        Dropout rate.
    max_seq_len : int
        Maximum window size (number of days + 1 for CLS token).
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
        max_seq_len: int = 8,  # 7 days + 1 CLS
    ) -> None:
        super().__init__()

        # Input projection
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Learnable CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        # Transformer encoder (pre-norm for stable training)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, W, F) — raw features for this modality
        padding_mask : (B, W) bool — True = real day, False = padding
        modality_mask : (B, W) bool — True = sensor had data

        Returns
        -------
        cls_out : (B, d_model) — modality summary vector
        """
        B, W, _ = x.shape

        # Project features
        x = self.proj(x)  # (B, W, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)  # (B, W+1, d_model)

        # Add positional embeddings
        x = x + self.pos_embed[:, : W + 1, :]

        # Build attention mask: combine padding + modality masks
        # CLS token is always valid (True)
        cls_valid = torch.ones(B, 1, dtype=torch.bool, device=x.device)
        combined_mask = padding_mask & modality_mask  # (B, W)
        combined_mask = torch.cat([cls_valid, combined_mask], dim=1)  # (B, W+1)

        # PyTorch convention: True = IGNORE in src_key_padding_mask
        attn_mask = ~combined_mask  # (B, W+1)

        x = self.encoder(x, src_key_padding_mask=attn_mask)

        # Return CLS token output
        return x[:, 0, :]  # (B, d_model)


class FusionTransformer(nn.Module):
    """Fuses modality summary tokens via cross-modal attention.

    Receives one token per modality, adds learnable modality-type embeddings,
    runs a TransformerEncoder, and mean-pools over available modalities.

    Parameters
    ----------
    d_model : int
        Hidden dimension (must match ModalityEncoder output).
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of TransformerEncoder layers.
    num_modalities : int
        Number of modalities (default: 5).
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 1,
        num_modalities: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Learnable modality-type embeddings
        self.modality_embed = nn.Parameter(
            torch.randn(1, num_modalities, d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(
        self,
        tokens: torch.Tensor,
        modality_available: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens : (B, M, d_model) — one summary token per modality
        modality_available : (B, M) bool — True if modality has data
                             for ANY day in the window

        Returns
        -------
        fused : (B, d_model) — mean-pooled over available modalities
        """
        # Add modality-type embeddings
        tokens = tokens + self.modality_embed

        # PyTorch convention: True = IGNORE
        attn_mask = ~modality_available  # (B, M)

        tokens = self.encoder(tokens, src_key_padding_mask=attn_mask)

        # Mean-pool over available modalities
        # Expand mask for broadcasting: (B, M, 1)
        avail = modality_available.unsqueeze(-1).float()
        n_avail = avail.sum(dim=1).clamp(min=1.0)  # (B, 1)
        fused = (tokens * avail).sum(dim=1) / n_avail  # (B, d_model)

        return fused


class MultimodalRelapseTransformer(nn.Module):
    """Top-level multimodal transformer for binary relapse prediction.

    Combines per-modality encoders, a fusion transformer, and a classification
    head that outputs a single logit per sample.

    Parameters
    ----------
    d_model : int
        Hidden dimension throughout the model.
    nhead : int
        Number of attention heads.
    num_encoder_layers : int
        Layers in each per-modality TransformerEncoder.
    num_fusion_layers : int
        Layers in the fusion TransformerEncoder.
    dropout : float
        Dropout rate.
    window_size : int
        Number of days per window (for positional embeddings).
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 1,
        num_fusion_layers: int = 1,
        dropout: float = 0.1,
        window_size: int = 7,
    ) -> None:
        super().__init__()

        # Per-modality encoders (independent weights)
        self.encoders = nn.ModuleDict({
            mod: ModalityEncoder(
                input_dim=MODALITY_DIMS[mod],
                d_model=d_model,
                nhead=nhead,
                num_layers=num_encoder_layers,
                dropout=dropout,
                max_seq_len=window_size + 1,  # +1 for CLS token
            )
            for mod in MODALITY_ORDER
        })

        # Fusion transformer
        self.fusion = FusionTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_fusion_layers,
            num_modalities=len(MODALITY_ORDER),
            dropout=dropout,
        )

        # Classification head: logit output
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        batch : dict from collate_fn with keys like ``accel_features``,
                ``accel_mask``, ``padding_mask``, etc.

        Returns
        -------
        logits : (B,) — raw logits (apply sigmoid for probabilities)
        """
        padding_mask = batch["padding_mask"]  # (B, W)

        modality_tokens = []
        modality_available = []

        for mod in MODALITY_ORDER:
            features = batch[f"{mod}_features"]    # (B, W, F_mod)
            mod_mask = batch[f"{mod}_mask"]         # (B, W)

            # Encode this modality
            cls_out = self.encoders[mod](features, padding_mask, mod_mask)
            modality_tokens.append(cls_out)

            # A modality is "available" if it has data for ANY day in the window
            avail = mod_mask.any(dim=1)  # (B,)
            modality_available.append(avail)

        # Stack modality tokens: (B, M, d_model)
        tokens = torch.stack(modality_tokens, dim=1)
        # Stack availability: (B, M)
        avail_mask = torch.stack(modality_available, dim=1)

        # Fuse modalities
        fused = self.fusion(tokens, avail_mask)  # (B, d_model)

        # Classify
        logits = self.classifier(fused).squeeze(-1)  # (B,)

        return logits
