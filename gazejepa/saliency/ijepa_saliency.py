"""I-JEPA ViT-H/14 backbone + lightweight saliency head.

Pretrained I-JEPA encoder (frozen) as backbone; only the small conv head is
trained on FIND fixation data.

Architecture:
    I-JEPA ViT-H/14 → (B, 256, 1280) patch embeddings (16×16 grid, no CLS)
    → reshape (B, 1280, 16, 16)
    → Conv3×3 1280→64 → GELU → Conv1×1 64→1
    → (B, 16, 16) logits

Internally rescales input to 224×224 and applies ImageNet normalisation;
callers pass raw [0, 1] images at any resolution.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import IJepaModel

from gazejepa.saliency.base import SaliencySource


_IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
_IJEPA_INPUT_SIZE: int = 224
_PRETRAINED_ID: str = "facebook/ijepa_vith14_1k"


class IJepaSaliency(nn.Module, SaliencySource):
    """I-JEPA ViT-H/14 backbone (frozen) + 3×3 conv head producing a 16×16 saliency grid."""

    name = "ijepa_saliency"

    def __init__(self, pretrained: bool = True):
        nn.Module.__init__(self)

        self.backbone = (
            IJepaModel.from_pretrained(_PRETRAINED_ID)
            if pretrained
            else IJepaModel(IJepaModel.config_class())
        )

        self.head = nn.Sequential(
            nn.Conv2d(1280, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

        self._backbone_frozen = False
        self.freeze_backbone()

        self.register_buffer(
            "_imagenet_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "_imagenet_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1),
        )

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        self._backbone_frozen = True

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True
        self._backbone_frozen = False

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._backbone_frozen:
            self.backbone.eval()
        return self

    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Run backbone; returns (B, 1280, 16, 16) spatial feature map."""
        out = self.backbone(pixel_values=images)
        patches = out.last_hidden_state  # (B, 256, 1280) — no CLS token
        b = patches.shape[0]
        return patches.permute(0, 2, 1).reshape(b, 1280, 16, 16)

    def forward_logits(self, images: torch.Tensor) -> torch.Tensor:
        """Backbone + head; expects ImageNet-normalised 224×224 input."""
        feat = self._extract_features(images)  # (B, 1280, 16, 16)
        return self.head(feat).squeeze(1)       # (B, 16, 16)

    def _compute(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-2:] != (_IJEPA_INPUT_SIZE, _IJEPA_INPUT_SIZE):
            images = F.interpolate(
                images, size=(_IJEPA_INPUT_SIZE, _IJEPA_INPUT_SIZE),
                mode="bilinear", align_corners=False,
            )
        normed = (images - self._imagenet_mean) / self._imagenet_std
        return self.forward_logits(normed)  # (B, 16, 16)

    # nn.Module routes calls through forward(); override to use SaliencySource.__call__.
    def forward(self, images: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return SaliencySource.__call__(self, images)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls, path: str, device: str | torch.device = "cpu",
    ) -> "IJepaSaliency":
        model = cls(pretrained=True)  # backbone architecture must match checkpoint
        result = model.load_state_dict(
            torch.load(path, map_location=device), strict=False,
        )
        allowed_missing = {"_imagenet_mean", "_imagenet_std"}
        unexpected_missing = set(result.missing_keys) - allowed_missing
        if unexpected_missing or result.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint mismatch loading {path}: "
                f"missing={sorted(unexpected_missing)}, "
                f"unexpected={sorted(result.unexpected_keys)}"
            )
        model.eval()
        return model.to(device)


if __name__ == "__main__":
    from gazejepa.saliency.base import assert_saliency_contract

    model = IJepaSaliency(pretrained=True)
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {n_total:,}")
    print(f"Trainable params: {n_trainable:,}  (head only)")

    img = torch.rand(2, 3, 200, 200)
    with torch.no_grad():
        sal = model(img)
    assert_saliency_contract(sal, batch_size=2)
    assert sal.shape == (2, 200, 200), sal.shape
    print(f"IJepaSaliency OK — input {tuple(img.shape)} -> saliency {tuple(sal.shape)}")
