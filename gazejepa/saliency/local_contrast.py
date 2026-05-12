"""Local contrast saliency — simple multi-scale luminance difference.

Pure-PyTorch fallback for the "classical" saliency slot. Use this when
``cv2.saliency`` (i.e. ``opencv-contrib-python``) is unavailable, or when a
deterministic, no-OpenCV-dependency baseline is desired. Arash's
"local_contrast" row in ``report_arash/data/saliency_comparison.csv`` was
produced by this exact algorithm.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from gazejepa.saliency.base import SaliencySource


class LocalContrastSaliency(SaliencySource):
    """Sum of |blur_k − blur_2k| over multiple scales of the luminance channel."""

    name = "local_contrast"

    def __init__(self, sigmas: tuple[int, ...] = (2, 4, 8, 16)):
        if not sigmas:
            raise ValueError("sigmas must be non-empty")
        if any(s <= 0 for s in sigmas):
            raise ValueError(f"All sigmas must be positive, got {sigmas}")
        self.sigmas = tuple(int(s) for s in sigmas)

    def _compute(self, images: torch.Tensor) -> torch.Tensor:
        b, _, h, w = images.shape
        r = images[:, 0]
        g = images[:, 1]
        bl = images[:, 2]
        lum = (0.2126 * r + 0.7152 * g + 0.0722 * bl).unsqueeze(1)
        out = torch.zeros(b, h, w, device=images.device, dtype=lum.dtype)
        for sigma in self.sigmas:
            k = max(int(4 * sigma) | 1, 3)
            blur = F.avg_pool2d(lum, kernel_size=k, stride=1, padding=k // 2)
            blur2 = F.avg_pool2d(lum, kernel_size=k * 2 + 1, stride=1, padding=k)
            blur = blur[..., :h, :w]
            blur2 = blur2[..., :h, :w]
            out = out + (blur - blur2).abs().squeeze(1)
        return out


if __name__ == "__main__":
    from gazejepa.saliency.base import assert_saliency_contract

    src = LocalContrastSaliency()
    img = torch.full((2, 3, 64, 96), 0.4)
    img[0, :, 20:35, 30:50] = 1.0
    sal = src(img)
    assert_saliency_contract(sal, batch_size=2)
    print(f"LocalContrastSaliency OK — shape={tuple(sal.shape)}")
