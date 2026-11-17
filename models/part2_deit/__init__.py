"""Part 2: training-recipe and DeiT-style distillation variants."""

from .vision_transformer import (
    DistilledVisionTransformer,
    build_distilled_vision_transformer,
    build_vision_transformer_single_classifier,
)

__all__ = [
    "DistilledVisionTransformer",
    "build_distilled_vision_transformer",
    "build_vision_transformer_single_classifier",
]
