"""Loss function registry."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .focal import FocalLoss

LOSS_REGISTRY: dict[str, type[nn.Module]] = {
    "focal": FocalLoss,
}


def get_loss(name: str, **kwargs: Any) -> nn.Module:
    """Instantiate a loss by registry name."""
    if name not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY.keys())}"
        )
    return LOSS_REGISTRY[name](**kwargs)


__all__ = ["FocalLoss", "LOSS_REGISTRY", "get_loss"]
