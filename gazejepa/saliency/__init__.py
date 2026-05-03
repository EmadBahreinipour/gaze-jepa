"""Saliency sources for the explicit gaze loop.

All sources are callables ``(B, 3, H, W) -> (B, H, W)`` with non-negative,
sum-normalised per-image output; :class:`SaliencySource` enforces this.
"""

from gazejepa.saliency.base import SaliencySource
from gazejepa.saliency.center_bias import CenterBiasSaliency
from gazejepa.saliency.itti_koch import IttiKochSaliency
from gazejepa.saliency.random import RandomSaliency

__all__ = [
    "SaliencySource",
    "CenterBiasSaliency",
    "IttiKochSaliency",
    "RandomSaliency",
]
