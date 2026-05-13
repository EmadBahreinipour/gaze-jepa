"""Saliency × IOR-driven sequential cropper — drop-in for ``SaccadeCropper``.

Yields ``T-1`` ``(view_t, view_{t+1}, displacement)`` triples per image.
Displacement is NeRF-encoded with zero rotation so the upstream ``affine_embedder``
accepts it unchanged.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from gazejepa.gaze_loop.fixation_sampler import FixationSampler
from gazejepa.gaze_loop.ior import IORMask


def _crop_at(
    image: torch.Tensor, fixation: torch.Tensor, target_h: int, target_w: int,
) -> torch.Tensor:
    """Reflect-padded crop centered on ``fixation``."""
    pad_h, pad_w = target_h, target_w
    padded = F.pad(
        image.unsqueeze(0),
        [pad_w, pad_w, pad_h, pad_h],
        mode="reflect",
    ).squeeze(0)
    x = int(round(fixation[0].item())) + pad_w
    y = int(round(fixation[1].item())) + pad_h
    half_h, half_w = target_h // 2, target_w // 2
    crop = padded[:, y - half_h : y + half_h, x - half_w : x + half_w]
    # Odd target dims leave the slice one short on a side; resample to fix.
    if crop.shape[1] != target_h or crop.shape[2] != target_w:
        crop = F.interpolate(
            crop.unsqueeze(0), size=(target_h, target_w),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
    return crop


class _NeRFEncoder:
    """NeRF-style sinusoidal embedding, matching ``SaccadeCropper.nerf_like_encoder``."""

    def __init__(self) -> None:
        self._coeff: dict[int, torch.Tensor] = {}

    def __call__(self, in_tensor: torch.Tensor, L: int) -> torch.Tensor:
        batch_size = in_tensor.shape[0]
        if L in self._coeff:
            coeff = self._coeff[L].to(in_tensor.device)
        else:
            coeff = (
                torch.pi * (2 ** torch.arange(L, dtype=torch.float32))
            ).unsqueeze(0).unsqueeze(0)
            self._coeff[L] = coeff
            coeff = coeff.to(in_tensor.device)
        embedding = torch.cat(
            [
                torch.sin(in_tensor.unsqueeze(-1) * coeff),
                torch.cos(in_tensor.unsqueeze(-1) * coeff),
            ],
            dim=-1,
        )
        return embedding.view(batch_size, -1)


class GazeCropper(torch.nn.Module):
    """Saliency × IOR sequential cropper, drop-in for ``SaccadeCropper``.

    ``target_h/w`` must match SaccadeJepa's ``model_input_size``; ``translation_L`` and
    ``rotation_L`` must match the upstream NeRF dims (defaults: 128×128, 36-dim embed).
    """

    def __init__(
        self,
        saliency_source: Any,
        input_h: int,
        input_w: int,
        target_h: int = 128,
        target_w: int = 128,
        n_fixations: int = 5,
        ior_sigma: float = 20.0,
        ior_decay: float = 0.7,
        sampling_mode: str = "stochastic",
        translation_L: int = 4,
        rotation_L: int = 10,
    ):
        super().__init__()
        if n_fixations < 2:
            raise ValueError(
                f"n_fixations must be >= 2 to yield transitions, got {n_fixations}"
            )
        if input_h <= 0 or input_w <= 0:
            raise ValueError(
                f"input_h, input_w must be positive, got ({input_h}, {input_w})"
            )
        self.saliency_source = saliency_source
        self.input_h = int(input_h)
        self.input_w = int(input_w)
        self.target_h = int(target_h)
        self.target_w = int(target_w)
        self.n_fixations = int(n_fixations)

        self.ior_sigma = float(ior_sigma)
        self.ior_decay = float(ior_decay)
        self.sampling_mode = sampling_mode

        self.translation_L = int(translation_L)
        self.rotation_L = int(rotation_L)
        self.affine_embed_dim = self.translation_L * 4 + self.rotation_L * 2

        self._sampler = FixationSampler(mode=sampling_mode)
        self._nerf = _NeRFEncoder()
        self.requires_grad_(False)

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(view_1, view_2, affines)`` — ``B*(T-1)`` transitions, NeRF-encoded displacements, on ``x``'s device."""
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected (B, 3, H, W), got {tuple(x.shape)}")
        B, _, H, W = x.shape
        if H != self.input_h or W != self.input_w:
            raise ValueError(
                f"Image size ({H},{W}) must match cropper init "
                f"({self.input_h},{self.input_w})"
            )

        with torch.no_grad():
            # static scene: one saliency call per image, reused across the walk
            saliency = self.saliency_source(x.float()).cpu()
            x_cpu = x.detach().cpu()

            views_1: list[torch.Tensor] = []
            views_2: list[torch.Tensor] = []
            displacements: list[torch.Tensor] = []

            for b in range(B):
                ior = IORMask(
                    image_size=(H, W), sigma=self.ior_sigma, decay=self.ior_decay,
                )
                fixations: list[torch.Tensor] = []
                for _ in range(self.n_fixations):
                    fix = self._sampler(saliency[b], ior())
                    fixations.append(fix)
                    ior.update(fix)

                for t in range(self.n_fixations - 1):
                    fix_t, fix_tp1 = fixations[t], fixations[t + 1]
                    views_1.append(_crop_at(x_cpu[b], fix_t, self.target_h, self.target_w))
                    views_2.append(_crop_at(x_cpu[b], fix_tp1, self.target_h, self.target_w))
                    # affine_grid's [-1, 1] coords, matching SaccadeCropper.max_translation_frac
                    dx_norm = (fix_tp1[0] - fix_t[0]) / (self.input_w / 2.0)
                    dy_norm = (fix_tp1[1] - fix_t[1]) / (self.input_h / 2.0)
                    displacements.append(torch.stack([dx_norm, dy_norm]))

            view_1_batch = torch.stack(views_1, dim=0).to(x.device)
            view_2_batch = torch.stack(views_2, dim=0).to(x.device)
            disp_batch = torch.stack(displacements, dim=0).to(x.device)

            zero_rot = torch.zeros(disp_batch.shape[0], 1, device=disp_batch.device)
            rot_emb = self._nerf(zero_rot, self.rotation_L)
            trans_emb = self._nerf(disp_batch, self.translation_L)
            transform_values = torch.cat([rot_emb, trans_emb], dim=-1)

            assert transform_values.shape[-1] == self.affine_embed_dim, (
                f"NeRF embed dim {transform_values.shape[-1]} != "
                f"affine_embed_dim {self.affine_embed_dim}"
            )

        return view_1_batch, view_2_batch, transform_values


if __name__ == "__main__":
    H, W = 224, 224
    T = 5
    B = 2

    class _BlobSaliency:
        name = "_blob"

        def __call__(self, images: torch.Tensor) -> torch.Tensor:
            b, _, h, w = images.shape
            yy, xx = torch.meshgrid(
                torch.arange(h, dtype=torch.float32),
                torch.arange(w, dtype=torch.float32),
                indexing="ij",
            )
            blob = torch.exp(-((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (2 * 30.0**2))
            blob = blob / blob.sum()
            return blob.unsqueeze(0).expand(b, -1, -1).contiguous()

    torch.manual_seed(0)
    cropper = GazeCropper(
        saliency_source=_BlobSaliency(),
        input_h=H, input_w=W,
        target_h=128, target_w=128,
        n_fixations=T,
    )
    x = torch.rand(B, 3, H, W)
    v1, v2, aff = cropper(x)

    n_transitions = B * (T - 1)
    assert v1.shape == (n_transitions, 3, 128, 128), v1.shape
    assert v2.shape == (n_transitions, 3, 128, 128), v2.shape
    assert aff.shape == (n_transitions, cropper.affine_embed_dim), aff.shape
    assert cropper.affine_embed_dim == 36, cropper.affine_embed_dim

    print(
        f"GazeCropper OK — view_1 {tuple(v1.shape)}, "
        f"view_2 {tuple(v2.shape)}, affines {tuple(aff.shape)}, "
        f"affine_embed_dim={cropper.affine_embed_dim} (matches SaccadeCropper default)"
    )
