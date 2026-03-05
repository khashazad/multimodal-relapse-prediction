"""Base class for all relapse prediction models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn


class BaseRelapseModel(ABC, nn.Module):
    """Abstract base for multimodal relapse prediction models.

    All model variants must implement ``forward()`` and may optionally
    override ``get_auxiliary_loss()`` to add regularisation terms
    (e.g. DANN adversarial loss).
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
