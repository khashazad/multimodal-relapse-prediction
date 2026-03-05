"""Multimodal Transformer with Gated Fusion for relapse prediction (v3).

Three-stage architecture:
  1. ModalityEncoder  — per-modality transformer (imported from v1)
  2. GatedFusion      — learned sigmoid gate per modality, weighted sum
  3. GatedRelapseTransformer — top-level module with classification head
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..dataset import MODALITY_ORDER
from ..feature_extractor import MODALITY_DIMS
from .base import BaseRelapseModel
from .transformer_v1 import ModalityEncoder


class GatedFusion(nn.Module):
    """Fuses modality summary tokens via learned sigmoid gates.

    Each modality has an independent linear gate that maps its d_model token
    to a scalar gate value in (0, 1).  Missing modalities are zeroed out via
    the ``modality_available`` mask before normalisation, so the weighted sum
    is always well-defined.

    Parameters
    ----------
    d_model : int
        Hidden dimension (must match ModalityEncoder output).
    num_modalities : int
        Number of modalities (default: 5).
    """

    def __init__(
        self,
        d_model: int = 64,
        num_modalities: int = 5,
    ) -> None:
        super().__init__()

        # One scalar gate per modality: Linear(d_model, 1)
        self.gates = nn.ModuleList(
            [nn.Linear(d_model, 1) for _ in range(num_modalities)]
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
        fused : (B, d_model) — gated weighted sum over available modalities
        """
        B, M, d_model = tokens.shape

        # Collect per-modality tokens as a list for gate application
        tokens_per_mod = [tokens[:, i, :] for i in range(M)]  # list of (B, d_model)

        gates = []
        for token, gate_fn in zip(tokens_per_mod, self.gates):
            g = torch.sigmoid(gate_fn(token))  # (B, 1)
            gates.append(g)

        gates = torch.stack(gates, dim=1)  # (B, M, 1)
        gates = gates * modality_available.unsqueeze(-1).float()  # zero out missing
        gate_sum = gates.sum(dim=1).clamp(min=1e-8)  # (B, 1)
        fused = (tokens * gates).sum(dim=1) / gate_sum  # (B, d_model)

        return fused


class GatedRelapseTransformer(BaseRelapseModel):
    """Top-level multimodal transformer with gated fusion for binary relapse prediction.

    Combines per-modality encoders (same as v1), a GatedFusion module, and a
    classification head that outputs a single logit per sample.

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
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 1,
        dropout: float = 0.1,
        window_size: int = 7,
        **kwargs,
    ) -> None:
        super().__init__()

        # Per-modality encoders (independent weights, same as v1)
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

        # Gated fusion
        self.fusion = GatedFusion(
            d_model=d_model,
            num_modalities=len(MODALITY_ORDER),
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

        # Fuse modalities via learned gates
        fused = self.fusion(tokens, avail_mask)  # (B, d_model)

        # Classify
        logits = self.classifier(fused).squeeze(-1)  # (B,)

        return logits
