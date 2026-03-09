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
        Used for both classes unless ``gamma_pos``/``gamma_neg`` are set.
    gamma_pos : float or None
        Focusing parameter for the positive class only.  When set together
        with ``gamma_neg``, enables asymmetric focal loss (Ridnik et al.,
        2021).  Overrides ``gamma`` for positive targets.
    gamma_neg : float or None
        Focusing parameter for the negative class only.  Overrides ``gamma``
        for negative targets.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        gamma_pos: float | None = None,
        gamma_neg: float | None = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        ce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Asymmetric gamma: use per-class gamma when both are provided
        if self.gamma_pos is not None and self.gamma_neg is not None:
            gamma_t = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        else:
            gamma_t = self.gamma

        loss = alpha_t * ((1 - p_t) ** gamma_t) * ce
        return loss
