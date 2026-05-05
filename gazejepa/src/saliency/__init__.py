"""Saliency sources for gaze target selection.

All sources implement SaliencySource and share the same interface:
    predict(images: Tensor[B, 3, H, W]) -> Tensor[B, H_out, W_out]

The output is non-negative and sums to 1 per sample (probability distribution
over spatial locations). All sources evaluate at 224×224.
"""

import torch
from abc import ABC, abstractmethod


class SaliencySource(ABC):
    """Abstract base for all saliency predictors.

    Subclasses implement predict() returning a (B, H, W) tensor of
    non-negative values that sum to 1 per sample.
    """

    name: str  # short identifier used in result tables

    @abstractmethod
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Predict saliency maps.

        Args:
            images: (B, 3, H, W) float tensor, ImageNet-normalized.

        Returns:
            (B, H_out, W_out) float tensor, non-negative, sums to 1 per sample.
        """
        ...

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        out = self.predict(images)
        assert out.ndim == 3, f"expected (B, H, W), got {out.shape}"
        assert (out >= 0).all(), "saliency values must be non-negative"
        return out
