"""Abstract base class for saliency sources.

Subclasses implement ``_compute`` (raw, possibly low-res, possibly un-normalised);
``__call__`` resamples to image resolution and sum-normalises per image so any
source plugs in regardless of its native output size.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class SaliencySource(ABC):
    """Abstract base class for saliency sources."""

    name: str = "abstract"

    @abstractmethod
    def _compute(self, images: torch.Tensor) -> torch.Tensor:
        """Raw saliency map ``(B, H_out, W_out)``; need not be normalised."""

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Contract-compliant saliency: ``(B, H, W)`` float32, non-negative, sum-to-1 per image."""
        if not isinstance(images, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(images).__name__}")
        if images.dim() != 4 or images.shape[1] != 3:
            raise ValueError(
                f"Expected (B, 3, H, W), got {tuple(images.shape)}"
            )
        b, _, h, w = images.shape

        raw = self._compute(images)

        if raw.dim() != 3 or raw.shape[0] != b:
            raise ValueError(
                f"_compute must return (B, H_out, W_out), got {tuple(raw.shape)}"
            )

        raw = raw.float()
        if raw.shape[-2:] != (h, w):
            raw = F.interpolate(
                raw.unsqueeze(1),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        raw = raw.clamp(min=0.0)
        flat = raw.reshape(b, -1)
        denom = flat.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        flat = flat / denom
        return flat.reshape(b, h, w)


def assert_saliency_contract(saliency: torch.Tensor, batch_size: int) -> None:
    """Assert saliency tensor satisfies the SaliencySource contract."""
    assert saliency.dim() == 3, f"Expected 3D, got {saliency.dim()}D"
    assert saliency.shape[0] == batch_size, (
        f"Expected batch={batch_size}, got {saliency.shape[0]}"
    )
    assert saliency.dtype == torch.float32, f"Expected float32, got {saliency.dtype}"
    assert (saliency >= 0).all(), "Saliency must be non-negative"
    sums = saliency.reshape(batch_size, -1).sum(dim=-1)
    assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5), (
        f"Per-image sums must equal 1; got {sums.tolist()}"
    )


if __name__ == "__main__":
    class _Const(SaliencySource):
        name = "_const_test"

        def _compute(self, images: torch.Tensor) -> torch.Tensor:
            b, _, h, w = images.shape
            return torch.ones(b, h // 2, w // 2)  # half-res to exercise upsampling

    src = _Const()
    img = torch.zeros(2, 3, 64, 96)
    sal = src(img)
    assert_saliency_contract(sal, batch_size=2)
    assert sal.shape == (2, 64, 96), f"Expected (2, 64, 96), got {tuple(sal.shape)}"
    print(f"SaliencySource OK — uniform map upsampled+normalized: shape={tuple(sal.shape)}")
