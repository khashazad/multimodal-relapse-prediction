"""Base class for all relapse prediction models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class BaseRelapseModel(ABC, nn.Module):
    """Abstract base for multimodal relapse prediction models.

    All model variants must implement ``forward()`` and may optionally
    override ``get_auxiliary_loss()`` to add regularisation terms
    (e.g. DANN adversarial loss).

    Subclasses that define ``self.encoders`` (ModuleDict) can use
    ``_encode_modalities()`` to avoid duplicating the encoding loop.
    """

    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the model on a collated batch.

        Parameters
        ----------
        batch : dict
            Collated batch from ``collate_fn`` with per-modality features,
            masks, labels, etc.

        Returns
        -------
        logits : (B,) — raw logits (apply sigmoid for probabilities)
        """

    def get_auxiliary_loss(self) -> torch.Tensor:
        """Return an auxiliary loss term (default: 0).

        Override in subclasses that need additional losses
        (e.g. domain-adversarial training).
        """
        return torch.tensor(0.0)

    def _encode_modalities(
        self,
        batch: Dict[str, torch.Tensor],
        modality_order: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode all modalities and return stacked tokens + availability mask.

        Parameters
        ----------
        batch : collated batch dict
        modality_order : list of modality names matching encoder keys

        Returns
        -------
        tokens : (B, M, d_model) — one summary token per modality
        avail_mask : (B, M) bool — True if modality has data for any day
        """
        padding_mask = batch["padding_mask"]
        modality_tokens = []
        modality_available = []

        for mod in modality_order:
            cls_out = self.encoders[mod](
                batch[f"{mod}_features"], padding_mask, batch[f"{mod}_mask"]
            )
            modality_tokens.append(cls_out)
            modality_available.append(batch[f"{mod}_mask"].any(dim=1))

        return (
            torch.stack(modality_tokens, dim=1),
            torch.stack(modality_available, dim=1),
        )
