"""Saliency sources for the explicit gaze loop.

All sources are callables ``(B, 3, H, W) -> (B, H, W)`` with non-negative,
sum-normalised per-image output; :class:`SaliencySource` enforces this.
"""

from gazejepa.saliency.base import SaliencySource
from gazejepa.saliency.center_bias import (
    CenterBiasDataFit,
    CenterBiasParametric,
    CenterBiasSaliency,
)
from gazejepa.saliency.itti_koch import IttiKochSaliency
from gazejepa.saliency.learned import LearnedSaliency
from gazejepa.saliency.local_contrast import LocalContrastSaliency
from gazejepa.saliency.random import RandomSaliency
from gazejepa.saliency.resnet_saliency import ResNetSaliency

__all__ = [
    "SaliencySource",
    "CenterBiasDataFit",
    "CenterBiasParametric",
    "CenterBiasSaliency",
    "IttiKochSaliency",
    "LearnedSaliency",
    "LocalContrastSaliency",
    "RandomSaliency",
    "ResNetSaliency",
]
