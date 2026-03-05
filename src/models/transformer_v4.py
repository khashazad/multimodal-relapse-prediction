"""
DANN (Domain-Adversarial Neural Network) Transformer for relapse prediction (v4).

Extends the standard FusionTransformer (v1) with a domain-adversarial head that
encourages the fused representation to be patient-invariant.  During training the
gradient reversal layer negates gradients flowing into the shared encoder from the
discriminator, pushing the encoder toward patient-agnostic features.

Architecture:
  1. ModalityEncoder   — per-modality transformer (imported from v1)
  2. FusionTransformer — cross-modal attention over modality tokens (imported from v1)
  3. classifier head   — same binary logit head as v1
  4. GradientReversal  — negates gradients with scale lambda_
  5. PatientDiscriminator — small MLP that predicts which of the 9 patients a
                            sample came from (adversarial objective)
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from ..dataset import MODALITY_ORDER
from ..feature_extractor import MODALITY_DIMS
from .base import BaseRelapseModel
from .transformer_v1 import FusionTransformer, ModalityEncoder


# ---------------------------------------------------------------------------
# Gradient reversal
# ---------------------------------------------------------------------------

class GradientReversalLayer(torch.autograd.Function):
    """Reverses gradients during backward pass, scaled by lambda_."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.save_for_backward(torch.tensor(lambda_))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (lambda_,) = ctx.saved_tensors
        return -lambda_.item() * grad_output, None


class GradientReversal(nn.Module):
    """Wrapper module around :class:`GradientReversalLayer`.

    Parameters
    ----------
    lambda_ : float
        Scale factor applied to the reversed gradient.
    """

    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalLayer.apply(x, self.lambda_)


# ---------------------------------------------------------------------------
# Patient discriminator
# ---------------------------------------------------------------------------

class PatientDiscriminator(nn.Module):
    """Small MLP that predicts patient identity from the fused representation.

    Used adversarially via :class:`GradientReversal` to make the encoder
    patient-invariant.

    Parameters
    ----------
    d_model : int
        Input dimension (must match FusionTransformer output size).
    dropout : float
        Dropout rate applied between the two linear layers.
    """

    def __init__(self, d_model: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 9),  # 9 patients (P1–P9)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, d_model)

        Returns
        -------
        logits : (B, 9)
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# DANN relapse transformer
# ---------------------------------------------------------------------------

class DANNRelapseTransformer(BaseRelapseModel):
    """Domain-Adversarial Multimodal Transformer for binary relapse prediction.

    Identical to v1 for the main prediction path (ModalityEncoder →
    FusionTransformer → classifier).  Additionally trains a
    PatientDiscriminator through a GradientReversal layer, encouraging the
    shared encoder to learn patient-invariant representations.

    The adversarial loss is exposed via :meth:`get_auxiliary_loss` so the
    training loop can add it (optionally weighted) to the main loss.

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
    dann_lambda : float
        Gradient reversal scale.  Larger values apply more domain pressure.
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 1,
        num_fusion_layers: int = 1,
        dropout: float = 0.1,
        window_size: int = 7,
        dann_lambda: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__()

        self.dann_lambda = dann_lambda

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

        # Fusion transformer (same as v1)
        self.fusion = FusionTransformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_fusion_layers,
            num_modalities=len(MODALITY_ORDER),
            dropout=dropout,
        )

        # Classification head (same as v1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # Domain-adversarial components
        self.grad_reversal = GradientReversal(lambda_=dann_lambda)
        self.discriminator = PatientDiscriminator(d_model=d_model, dropout=dropout)

        # Storage for auxiliary loss computation (populated in forward)
        self._dann_logits: Optional[torch.Tensor] = None
        self._dann_targets: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_and_fuse(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode all modalities and return the fused representation (B, d_model)."""
        padding_mask = batch["padding_mask"]  # (B, W)

        modality_tokens = []
        modality_available = []

        for mod in MODALITY_ORDER:
            features = batch[f"{mod}_features"]  # (B, W, F_mod)
            mod_mask = batch[f"{mod}_mask"]       # (B, W)

            cls_out = self.encoders[mod](features, padding_mask, mod_mask)
            modality_tokens.append(cls_out)

            avail = mod_mask.any(dim=1)  # (B,)
            modality_available.append(avail)

        tokens = torch.stack(modality_tokens, dim=1)     # (B, M, d_model)
        avail_mask = torch.stack(modality_available, dim=1)  # (B, M)

        fused = self.fusion(tokens, avail_mask)  # (B, d_model)
        return fused

    # ------------------------------------------------------------------
    # BaseRelapseModel interface
    # ------------------------------------------------------------------

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        batch : dict
            Collated batch from ``collate_fn``.  Must contain per-modality
            ``{mod}_features`` and ``{mod}_mask`` tensors, ``padding_mask``,
            and (during training) ``patient_idx`` (B,) with values 0–8 or -1
            for unknown patients.

        Returns
        -------
        logits : (B,) — raw classification logits
        """
        # Reset stored DANN outputs
        self._dann_logits = None
        self._dann_targets = None

        # Encode and fuse
        fused = self._encode_and_fuse(batch)  # (B, d_model)

        # Classification
        logits = self.classifier(fused).squeeze(-1)  # (B,)

        # Domain-adversarial branch (training only)
        if self.training:
            reversed_fused = self.grad_reversal(fused)
            disc_logits = self.discriminator(reversed_fused)  # (B, 9)
            self._dann_logits = disc_logits
            self._dann_targets = batch["patient_idx"]  # (B,)

        return logits

    def get_auxiliary_loss(self) -> torch.Tensor:
        """Compute the adversarial domain loss.

        Filters out samples whose ``patient_idx`` is -1 (unknown / no label).

        Returns
        -------
        loss : scalar tensor — CrossEntropy over valid patient indices, or 0
               if no valid patients are present.
        """
        if self._dann_logits is None:
            return torch.tensor(0.0)

        patient_idx = self._dann_targets  # (B,)
        valid = patient_idx != -1          # (B,) bool mask

        if not valid.any():
            return torch.tensor(0.0, device=self._dann_logits.device)

        logits = self._dann_logits[valid]   # (N_valid, 9)
        targets = patient_idx[valid]        # (N_valid,)

        return nn.functional.cross_entropy(logits, targets)
