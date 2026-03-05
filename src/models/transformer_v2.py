"""Multimodal Transformer Fusion model for relapse prediction (v2 — Perceiver bottleneck).

Three-stage architecture:
  1. ModalityEncoder  — per-modality transformer (reused from transformer_v1)
  2. BottleneckFusion — learnable latent tokens compress modality tokens via cross-attention
  3. BottleneckRelapseTransformer — top-level module with classification head
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..dataset import MODALITY_ORDER
from ..feature_extractor import MODALITY_DIMS
from .base import BaseRelapseModel
from .transformer_v1 import ModalityEncoder


class BottleneckFusion(nn.Module):
    """Perceiver-style bottleneck fusion over modality summary tokens.

    A small set of learnable latent tokens compress information from all
    modality tokens via cross-attention (latent as Q, modalities as K/V),
    then refine via self-attention.  The final representation is the
    mean-pool of the latent tokens.

    Parameters
    ----------
    d_model : int
        Hidden dimension (must match ModalityEncoder output).
    nhead : int
        Number of attention heads.
    num_bottleneck : int
        Number of learnable latent tokens.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_bottleneck: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Learnable latent tokens (the bottleneck)
        self.latent = nn.Parameter(torch.randn(1, num_bottleneck, d_model) * 0.02)

        # Cross-attention: latent (Q) attends to modality tokens (K, V)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_model)

        # Feedforward on latent tokens after cross-attention
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.ff_norm = nn.LayerNorm(d_model)

        # Self-attention: latent tokens attend to each other (pre-norm)
        self_attn_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.self_attn = nn.TransformerEncoder(self_attn_layer, num_layers=1)

    def forward(
        self,
        modality_tokens: torch.Tensor,
        modality_available: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        modality_tokens : (B, M, d_model) — one summary token per modality
        modality_available : (B, M) bool — True if modality has data for
                             any day in the window

        Returns
        -------
        fused : (B, d_model) — mean-pooled latent tokens
        """
        B = modality_tokens.size(0)

        # Expand latent tokens to batch size
        latent = self.latent.expand(B, -1, -1)  # (B, num_bottleneck, d_model)

        # Cross-attention: latent (Q) <- modality_tokens (K, V)
        # key_padding_mask uses PyTorch convention: True = IGNORE
        key_padding_mask = ~modality_available  # (B, M)
        attn_out, _ = self.cross_attn(
            query=latent,
            key=modality_tokens,
            value=modality_tokens,
            key_padding_mask=key_padding_mask,
        )

        # Residual + LayerNorm
        latent = self.cross_norm(latent + attn_out)

        # Feedforward + residual + LayerNorm
        latent = self.ff_norm(latent + self.ff(latent))

        # Self-attention over latent tokens
        latent = self.self_attn(latent)  # (B, num_bottleneck, d_model)

        # Mean-pool latent tokens
        fused = latent.mean(dim=1)  # (B, d_model)

        return fused


class BottleneckRelapseTransformer(BaseRelapseModel):
    """Top-level multimodal Perceiver-bottleneck transformer for relapse prediction.

    Combines per-modality encoders (from v1), a Perceiver-style bottleneck
    fusion module, and a classification head that outputs a single logit per
    sample.

    Parameters
    ----------
    d_model : int
        Hidden dimension throughout the model.
    nhead : int
        Number of attention heads.
    num_encoder_layers : int
        Layers in each per-modality TransformerEncoder.
    dropout : float
        Dropout rate.
    window_size : int
        Number of days per window (for positional embeddings).
    num_bottleneck_tokens : int
        Number of learnable latent tokens in the bottleneck fusion.
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 1,
        dropout: float = 0.1,
        window_size: int = 7,
        num_bottleneck_tokens: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()

        # Per-modality encoders (independent weights, reused from v1)
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

        # Perceiver-style bottleneck fusion
        self.fusion = BottleneckFusion(
            d_model=d_model,
            nhead=nhead,
            num_bottleneck=num_bottleneck_tokens,
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
            features = batch[f"{mod}_features"]  # (B, W, F_mod)
            mod_mask = batch[f"{mod}_mask"]       # (B, W)

            # Encode this modality to a CLS summary token
            cls_out = self.encoders[mod](features, padding_mask, mod_mask)
            modality_tokens.append(cls_out)

            # A modality is "available" if it has data for ANY day in the window
            avail = mod_mask.any(dim=1)  # (B,)
            modality_available.append(avail)

        # Stack modality tokens: (B, M, d_model)
        tokens = torch.stack(modality_tokens, dim=1)
        # Stack availability: (B, M)
        avail_mask = torch.stack(modality_available, dim=1)

        # Bottleneck fusion: compress modality tokens into latent representation
        fused = self.fusion(tokens, avail_mask)  # (B, d_model)

        # Classify
        logits = self.classifier(fused).squeeze(-1)  # (B,)

        return logits
