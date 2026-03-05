"""Backward-compatibility shim — re-exports from src.models.transformer_v1."""

from .models.transformer_v1 import (  # noqa: F401
    FusionTransformer,
    ModalityEncoder,
    MultimodalRelapseTransformer,
)

__all__ = [
    "ModalityEncoder",
    "FusionTransformer",
    "MultimodalRelapseTransformer",
]
