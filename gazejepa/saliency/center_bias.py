"""Image-centered Gaussian saliency (FIND-style center prior)."""

from __future__ import annotations

import torch

from gazejepa.saliency.base import SaliencySource, assert_saliency_contract


class CenterBiasSaliency(SaliencySource):
    """Image-centered Gaussian saliency. ``sigma_frac`` scales sigma per axis."""

    name = "center_bias"

    def __init__(self, sigma_frac: float = 0.25):
        if sigma_frac <= 0:
            raise ValueError(f"sigma_frac must be positive, got {sigma_frac}")
        self.sigma_frac = float(sigma_frac)

    def _compute(self, images: torch.Tensor) -> torch.Tensor:
        b, _, h, w = images.shape
        device = images.device
        cy, cx = h / 2.0, w / 2.0
        sy, sx = h * self.sigma_frac, w * self.sigma_frac
        ys = torch.arange(h, device=device, dtype=torch.float32).view(-1, 1)
        xs = torch.arange(w, device=device, dtype=torch.float32).view(1, -1)
        gauss = torch.exp(
            -((xs - cx) ** 2 / (2.0 * sx ** 2) + (ys - cy) ** 2 / (2.0 * sy ** 2))
        )
        return gauss.unsqueeze(0).expand(b, -1, -1).contiguous()


if __name__ == "__main__":
    src = CenterBiasSaliency(sigma_frac=0.25)
    img = torch.zeros(2, 3, 64, 96)
    sal = src(img)
    assert_saliency_contract(sal, batch_size=2)

    flat = sal[0].flatten()
    idx = int(flat.argmax().item())
    h, w = sal.shape[-2:]
    py, px = idx // w, idx % w
    assert abs(py - h // 2) <= 1 and abs(px - w // 2) <= 1, (
        f"Argmax {(px, py)} not at center {(w // 2, h // 2)}"
    )
    print(f"CenterBiasSaliency OK — shape={tuple(sal.shape)}, "
          f"argmax at (x={px}, y={py}) ≈ center ({w // 2}, {h // 2})")
