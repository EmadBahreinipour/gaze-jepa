"""ResNet-18 backbone + lightweight head saliency predictor.

Internally rescales any input to the ResNet-pretrained 224×224 grid and applies
ImageNet normalisation, so callers can pass raw ``[0, 1]`` images at any
resolution. ``SaliencySource.__call__`` then resamples back to the input size
and re-normalises to sum-to-1.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from gazejepa.saliency.base import SaliencySource


_IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
_RESNET_INPUT_SIZE: int = 224


class ResNetSaliency(nn.Module, SaliencySource):
    """ResNet-18 backbone + 1×1 conv head producing a 14×14 saliency grid."""

    name = "resnet_saliency"

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        nn.Module.__init__(self)
        backbone = tvm.resnet18(
            weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None,
        )
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
        )
        self.layer1 = backbone.layer1   # 56×56,  64ch
        self.layer2 = backbone.layer2   # 28×28, 128ch
        self.layer3 = backbone.layer3   # 14×14, 256ch
        self.head = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

        self._backbone_frozen = False
        if freeze_backbone:
            self.freeze_backbone()

        self.register_buffer(
            "_imagenet_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "_imagenet_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1),
        )

    def freeze_backbone(self) -> None:
        for module in (self.stem, self.layer1, self.layer2, self.layer3):
            for p in module.parameters():
                p.requires_grad = False
        self._backbone_frozen = True

    def unfreeze_backbone(self) -> None:
        for module in (self.stem, self.layer1, self.layer2, self.layer3):
            for p in module.parameters():
                p.requires_grad = True
        self._backbone_frozen = False

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._backbone_frozen:
            for module in (self.stem, self.layer1, self.layer2, self.layer3):
                module.eval()
        return self

    def forward_logits(self, images: torch.Tensor) -> torch.Tensor:
        """Backbone + head; expects ImageNet-normalised 224×224 input."""
        x = self.stem(images)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.head(x).squeeze(1)  # (B, 14, 14)

    def _compute(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-2:] != (_RESNET_INPUT_SIZE, _RESNET_INPUT_SIZE):
            images = F.interpolate(
                images, size=(_RESNET_INPUT_SIZE, _RESNET_INPUT_SIZE),
                mode="bilinear", align_corners=False,
            )
        normed = (images - self._imagenet_mean) / self._imagenet_std
        logits = self.forward_logits(normed)
        b = logits.shape[0]
        return logits.view(b, -1).softmax(dim=1).view_as(logits)

    # ``nn.Module.__call__`` dispatches to ``forward``; route it through the
    # ``SaliencySource`` contract so callers get the validated, sum-to-1 map.
    def forward(self, images: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return SaliencySource.__call__(self, images)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls, path: str, device: str | torch.device = "cpu",
    ) -> "ResNetSaliency":
        model = cls(pretrained=False, freeze_backbone=False)
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

    model = ResNetSaliency(pretrained=True, freeze_backbone=True)
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {n_total:,}")
    print(f"Trainable params: {n_trainable:,}  (head only)")

    img = torch.rand(2, 3, 200, 200)
    sal = model(img)
    assert_saliency_contract(sal, batch_size=2)
    assert sal.shape == (2, 200, 200), sal.shape
    print(f"ResNetSaliency OK — input {tuple(img.shape)} -> saliency {tuple(sal.shape)}")
