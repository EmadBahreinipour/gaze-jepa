"""ResNet-18 based saliency predictor.

Uses a pretrained ResNet-18 backbone to extract features, then a lightweight
head to produce a saliency map. The backbone already knows faces, objects and
textures from ImageNet — we only need to teach the head to combine those
features into a saliency distribution.

Training strategy:
  1. Freeze backbone, train only head for first N epochs (fast convergence)
  2. Optionally unfreeze and fine-tune end-to-end

Output resolution: 14x14 for loss computation (matches ResNet layer3 stride),
upsampled to 224x224 for evaluation. KL divergence over 196 bins is very
tractable vs 50176 bins at full resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from . import SaliencySource


class ResNetSaliency(nn.Module, SaliencySource):
    """ResNet-18 backbone + lightweight saliency head.

    Input:  (B, 3, 224, 224) ImageNet-normalized
    Output: (B, 224, 224) softmax-normalized saliency map

    Architecture:
        ResNet-18 pretrained → layer3 features (B, 256, 14, 14)
        → Conv 256→64 → GELU → Conv 64→1
        → (B, 1, 14, 14) logits
        → upsample to (B, 224, 224) → softmax
    """

    name = "resnet_saliency"

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        nn.Module.__init__(self)

        backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)

        # Keep everything up to layer3 — gives 14x14 feature maps at 224x224 input
        self.stem    = nn.Sequential(backbone.conv1, backbone.bn1,
                                     backbone.relu, backbone.maxpool)
        self.layer1  = backbone.layer1   # 56x56, 64ch
        self.layer2  = backbone.layer2   # 28x28, 128ch
        self.layer3  = backbone.layer3   # 14x14, 256ch

        self.head = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        for m in [self.stem, self.layer1, self.layer2, self.layer3]:
            for p in m.parameters():
                p.requires_grad = False
        self._backbone_frozen = True

    def unfreeze_backbone(self):
        for m in [self.stem, self.layer1, self.layer2, self.layer3]:
            for p in m.parameters():
                p.requires_grad = True
        self._backbone_frozen = False

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep frozen backbone in eval mode so BatchNorm uses pretrained
        # running statistics instead of noisy per-batch statistics.
        if mode and getattr(self, "_backbone_frozen", False):
            for m in [self.stem, self.layer1, self.layer2, self.layer3]:
                m.eval()
        return self

    def forward_logits(self, images: torch.Tensor) -> torch.Tensor:
        """Returns raw logits at 14x14, shape (B, 14, 14)."""
        x = self.stem(images)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.head(x).squeeze(1)  # (B, 14, 14)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.predict(images)

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.forward_logits(images)                          # (B, 14, 14)
        logits_up = F.interpolate(logits.unsqueeze(1), size=(224, 224),
                                  mode="bilinear", align_corners=False).squeeze(1)
        B = logits_up.shape[0]
        return logits_up.view(B, -1).softmax(dim=1).view(B, 224, 224)

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "ResNetSaliency":
        model = cls(pretrained=False)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model


if __name__ == "__main__":
    model = ResNetSaliency(pretrained=True, freeze_backbone=True)
    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {n_total:,}")
    print(f"Trainable params: {n_trainable:,}  (head only)")

    x = torch.rand(2, 3, 224, 224)
    out = model.predict(x)
    print(f"Output shape: {out.shape}")
    print(f"Sum per sample: {out.sum(dim=(1,2))}")
    assert out.shape == (2, 224, 224)
    assert torch.allclose(out.sum(dim=(1, 2)), torch.ones(2), atol=1e-4)
    print("ResNetSaliency OK")
