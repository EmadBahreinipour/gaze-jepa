"""Small CNN saliency predictor (~50K params) trained on FIND fixation heatmaps.

Lightweight alternative to ResNetSaliency with no ImageNet prior.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from gazejepa.saliency.base import SaliencySource


_IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


class LearnedSaliency(nn.Module, SaliencySource):
    """4-layer CNN with one stride-2 downsample; output is H/2 × W/2."""

    name = "learned"

    def __init__(self, in_channels: int = 3, hidden: int = 32):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden * 2, 3, padding=1, stride=2),
            nn.GroupNorm(8, hidden * 2), nn.GELU(),
            nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1),
            nn.GroupNorm(8, hidden * 2), nn.GELU(),
            nn.Conv2d(hidden * 2, 1, 1),
        )
        self.register_buffer(
            "_imagenet_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "_imagenet_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1),
        )

    def forward_logits(self, images: torch.Tensor) -> torch.Tensor:
        normed = (images - self._imagenet_mean) / self._imagenet_std
        return self.net(normed).squeeze(1)  # (B, H/2, W/2)

    def _compute(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.forward_logits(images)
        b = logits.shape[0]
        return logits.view(b, -1).softmax(dim=1).view_as(logits)

    # nn.Module routes calls through forward(); override to use SaliencySource.__call__.
    def forward(self, images: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return SaliencySource.__call__(self, images)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls, path: str, device: str | torch.device = "cpu",
    ) -> "LearnedSaliency":
        model = cls()
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model.to(device)


if __name__ == "__main__":
    from gazejepa.saliency.base import assert_saliency_contract

    model = LearnedSaliency()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"LearnedSaliency: {n_params:,} params")

    img = torch.rand(2, 3, 200, 200)
    sal = model(img)
    assert_saliency_contract(sal, batch_size=2)
    assert sal.shape == (2, 200, 200), sal.shape
    print(f"LearnedSaliency OK — input {tuple(img.shape)} -> saliency {tuple(sal.shape)}")
