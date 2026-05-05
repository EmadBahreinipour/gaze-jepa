"""Saliency-guided saccade generator.

Answers the professor's question: "Can SaccadeJEPA be reused for explicit
gaze simulation?"

SaccadeJEPA's original sampling (saccade.py line 66):
    translations = torch.empty(b, 2).uniform_(-max_frac, max_frac)  # RANDOM

This module replaces that with:
    translations = sample_from_saliency(saliency_map)  # CONTENT-DRIVEN

The rest of SaccadeJEPA's machinery (affine warp, NeRF embedding, prediction)
is unchanged — we only swap out the one line that decides WHERE to look.
"""

import torch
import torch.nn.functional as F
import torchvision
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.saliency import SaliencySource


def _rotate_and_translate(images, translation, theta, interp_mode="nearest"):
    """Affine warp — same as SaccadeCropper's rotate_and_translate_batch."""
    translation = translation.unsqueeze(-1)
    theta = theta.unsqueeze(-1)
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    rot = torch.cat([cos_t, -sin_t, sin_t, cos_t], dim=-1).view(-1, 2, 2)
    rot = torch.cat([rot, translation], dim=-1)
    grid = F.affine_grid(rot, images.size(), align_corners=False)
    return F.grid_sample(images, grid, mode=interp_mode, align_corners=False)


def sample_fixations(saliency_map: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
    """Sample pixel locations weighted by saliency probability.

    Args:
        saliency_map: (B, H, W) — must sum to 1 per sample (SaliencySource output)
        n_samples:    how many fixations to draw per image

    Returns:
        (B, n_samples, 2) float tensor of (x, y) pixel coordinates
    """
    B, H, W = saliency_map.shape
    flat = saliency_map.view(B, -1)                          # (B, H*W)
    indices = torch.multinomial(flat, num_samples=n_samples) # (B, n_samples)
    y = (indices // W).float()
    x = (indices  % W).float()
    return torch.stack([x, y], dim=-1)                       # (B, n_samples, 2)


def fixations_to_translations(fixations: torch.Tensor,
                               image_h: int, image_w: int,
                               max_translation_frac: float) -> torch.Tensor:
    """Convert pixel fixation coords to SaccadeJEPA-style NDC translations.

    SaccadeJEPA's affine_grid expects translations in [-1, 1] normalized device
    coordinates. We map fixation pixel (px, py) to the offset from image center,
    then clamp to the max allowed shift.

    Args:
        fixations:            (B, 2) float tensor of (x, y) pixel coords
        image_h / image_w:    original image dimensions
        max_translation_frac: SaccadeCropper's max_translation / input_h

    Returns:
        (B, 2) translations in NDC, clipped to [-max_translation_frac, max_translation_frac]
    """
    tx = (2.0 * fixations[:, 0] / image_w - 1.0)
    ty = (2.0 * fixations[:, 1] / image_h - 1.0)
    frac = max_translation_frac
    return torch.stack([tx, ty], dim=1).clamp(-frac, frac)


def generate_scanpath(saliency_source: SaliencySource,
                      images: torch.Tensor,
                      n_fixations: int = 8) -> torch.Tensor:
    """Generate a scanpath for a batch of images using saliency-weighted sampling.

    This is the minimal replacement for SaccadeJEPA's random gaze target:

        # SaccadeJEPA (saccade.py:66) — what we replace:
        translations = torch.empty(b, 2).uniform_(-max_frac, max_frac)

        # Our version — saliency-driven:
        saliency     = saliency_source(images)           # (B, H, W)
        fixations    = sample_fixations(saliency, 1)     # (B, 1, 2)
        translations = fixations_to_translations(...)    # (B, 2)

    Args:
        saliency_source: any SaliencySource — swap RandomSaliency ↔ ResNetSaliency
                         to reproduce the ablation from the report
        images:          (B, 3, H, W) ImageNet-normalized tensor
        n_fixations:     length of scanpath to generate

    Returns:
        (B, n_fixations, 2) float tensor of (x, y) fixation pixel coordinates
    """
    with torch.no_grad():
        saliency = saliency_source(images)                        # (B, H, W)
        return sample_fixations(saliency, n_fixations)            # (B, n_fixations, 2)


class SaliencyGuidedSaccadeCropper(torch.nn.Module):
    """Drop-in replacement for SaccadeJEPA's SaccadeCropper.

    Identical interface and output format. The only change is that translations
    are sampled from the saliency distribution instead of from torch.uniform_.

    Usage:
        # Original SaccadeJEPA:
        cropper = SaccadeCropper(input_h=224, input_w=224)

        # Saliency-guided replacement:
        saliency = ResNetSaliency.load("checkpoints/resnet_saliency_best.pt")
        cropper  = SaliencyGuidedSaccadeCropper(saliency, input_h=224, input_w=224)

        x1, x2, transform_values = cropper(images)  # same API
    """

    def __init__(self,
                 saliency_source: SaliencySource,
                 input_h: int = 224,
                 input_w: int = 224,
                 target_h: int = 224,
                 target_w: int = 224,
                 max_translation: int = 40,
                 max_rotation: float = 2.0,
                 embed_affines: bool = True,
                 translation_L: int = 4,
                 rotation_L: int = 10):
        super().__init__()
        self.saliency_source = saliency_source
        self.input_h = input_h
        self.input_w = input_w
        self.target_h = target_h
        self.target_w = target_w
        self.max_translation_frac = max_translation / input_h
        self.max_rotation_rad = max_rotation * torch.pi / 180
        self.embed_affines = embed_affines
        self.translation_L = translation_L
        self.rotation_L = rotation_L
        self.affine_embed_dim = translation_L * 4 + rotation_L * 2
        self._coeff = {}
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor):
        """Same return signature as SaccadeCropper: (x1, x2, transform_values)."""
        with torch.no_grad():
            b, c, h, w = x.shape
            device = x.device

            # --- ROTATION: still random (gaze doesn't rotate the retina) ---
            rotations = torch.empty(b, 1, device=device).uniform_(
                -self.max_rotation_rad, self.max_rotation_rad)

            # --- TRANSLATION: saliency-guided (replaces uniform random) ---
            saliency = self.saliency_source(x)                    # (B, H, W)
            fix = sample_fixations(saliency, n_samples=1)         # (B, 1, 2)
            fix = fix.squeeze(1).to(device)                       # (B, 2)
            translations = fixations_to_translations(
                fix, h, w, self.max_translation_frac)             # (B, 2)

            # Affine warp + center crop — identical to SaccadeCropper
            x_affined = _rotate_and_translate(x, translations, rotations)
            if self.embed_affines:
                rotations    = self._nerf(rotations,    self.rotation_L)
                translations = self._nerf(translations, self.translation_L)
            transform_values = torch.cat([rotations, translations], dim=-1)

            x1 = torchvision.transforms.functional.center_crop(x_affined, (self.target_h, self.target_w))
            x2 = torchvision.transforms.functional.center_crop(x,         (self.target_h, self.target_w))

        return x1, x2, transform_values

    def _nerf(self, t: torch.Tensor, L: int) -> torch.Tensor:
        b = t.shape[0]
        if L not in self._coeff:
            self._coeff[L] = (torch.pi * (2 ** torch.arange(L, dtype=torch.float32))).unsqueeze(0).unsqueeze(0)
        coeff = self._coeff[L].to(t.device)
        emb = torch.cat([torch.sin(t.unsqueeze(-1) * coeff),
                         torch.cos(t.unsqueeze(-1) * coeff)], dim=-1)
        return emb.view(b, -1)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from src.find_data import load_frame, load_fixations, get_frame_fixations, rescale_fixations, make_heatmap
    from src.saliency.random_saliency import RandomSaliency
    from src.saliency.resnet_saliency import ResNetSaliency
    from src.saliency.learned import IMAGE_TRANSFORM

    DATA_ROOT = "f:/university/milan/NI/info/Gaze_Jepa/find_dataset"
    CHECKPOINT = "checkpoints/resnet_saliency_best.pt"

    SAMPLES = [
        ("070", 100),
        ("071", 80),
    ]

    random_source  = RandomSaliency()
    learned_source = ResNetSaliency.load(CHECKPOINT)

    fig, axes = plt.subplots(len(SAMPLES), 4, figsize=(20, 5 * len(SAMPLES)))
    col_titles = ["Input frame", "Random (SaccadeJEPA)", "ResNetSaliency (ours)", "Human fixations (GT)"]

    for row, (video_id, frame_idx) in enumerate(SAMPLES):
        frame        = load_frame(DATA_ROOT, video_id, frame_idx, size=224)
        image_tensor = IMAGE_TRANSFORM(frame).unsqueeze(0)

        fix_array  = load_fixations(DATA_ROOT, video_id)
        raw_fix    = get_frame_fixations(fix_array, frame_idx)
        gt_fix     = rescale_fixations(raw_fix, (1280, 720), (224, 224))
        gt_heatmap = make_heatmap(gt_fix, 224, sigma_pixels=20.0)

        with torch.no_grad():
            random_sal  = random_source(image_tensor)[0].numpy()
            learned_sal = learned_source(image_tensor)[0].numpy()

        maps = [None, random_sal, learned_sal, gt_heatmap]
        for col, (ax, sal, title) in enumerate(zip(axes[row], maps, col_titles)):
            ax.imshow(frame)
            if sal is not None:
                ax.imshow(sal, cmap="hot", alpha=0.55)
            if row == 0:
                ax.set_title(title, fontsize=11)
            ax.set_ylabel(f"Video {video_id}, frame {frame_idx}", fontsize=9)
            ax.axis("off")

    plt.tight_layout()
    os.makedirs("report_arash/figures", exist_ok=True)
    plt.savefig("report_arash/figures/saliency_comparison.png", dpi=150)
    print("Saved report_arash/figures/saliency_comparison.png")
    plt.show()
