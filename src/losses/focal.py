"""Binary focal loss for imbalanced classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """Focal loss down-weights well-classified examples.

    Parameters
    ----------
    alpha : float
        Weighting factor for the positive class (1-alpha for negative).
    gamma : float
        Focusing parameter — higher values focus more on hard examples.
    """

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
