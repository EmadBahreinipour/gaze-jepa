"""Saliency sources for the explicit gaze loop.

All sources are callables ``(B, 3, H, W) -> (B, H, W)`` with non-negative,
sum-normalised per-image output; :class:`SaliencySource` enforces this.
"""

from gazejepa.saliency.base import SaliencySource
from gazejepa.saliency.center_bias import (
    CenterBiasParametric,
    CenterBiasSaliency,
)
from gazejepa.saliency.spectral_residual import SpectralResidualSaliency
from gazejepa.saliency.random import RandomSaliency
from gazejepa.saliency.resnet_saliency import ResNetSaliency
from gazejepa.saliency.ijepa_saliency import IJepaSaliency

__all__ = [
    "SaliencySource",
    "CenterBiasParametric",
    "CenterBiasSaliency",
    "SpectralResidualSaliency",
    "RandomSaliency",
    "ResNetSaliency",
    "IJepaSaliency",
]
