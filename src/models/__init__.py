"""Model registry for relapse prediction architectures."""

from __future__ import annotations

from typing import Any

from .base import BaseRelapseModel
from .transformer_v1 import (
    FusionTransformer,
    ModalityEncoder,
    MultimodalRelapseTransformer,
)
from .transformer_v2 import BottleneckRelapseTransformer
from .transformer_v3 import GatedRelapseTransformer
from .transformer_v4 import DANNRelapseTransformer

MODEL_REGISTRY: dict[str, type[BaseRelapseModel]] = {
    "transformer_v1": MultimodalRelapseTransformer,
    "transformer_v2": BottleneckRelapseTransformer,
    "transformer_v3": GatedRelapseTransformer,
    "transformer_v4": DANNRelapseTransformer,
}


def get_model(name: str, **kwargs: Any) -> BaseRelapseModel:
    """Instantiate a model by registry name, forwarding all kwargs."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](**kwargs)


__all__ = [
    "BaseRelapseModel",
    "FusionTransformer",
    "ModalityEncoder",
    "MultimodalRelapseTransformer",
    "MODEL_REGISTRY",
    "get_model",
]
