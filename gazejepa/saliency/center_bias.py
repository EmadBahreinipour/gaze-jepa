"""Center bias saliency — fixed 2D Gaussian centered on the image."""

from __future__ import annotations

import torch

from gazejepa.saliency.base import SaliencySource, assert_saliency_contract


class CenterBiasSaliency(SaliencySource):
    """Fixed 2D Gaussian centered on the image, σ = ``image_size / 4``."""

    name = "center_bias_parametric"

    def __init__(self, image_size: int = 224):
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        self.image_size = int(image_size)
        self._map = self._build_map(self.image_size)

    @staticmethod
    def _build_map(size: int) -> torch.Tensor:
        cx = cy = size / 2.0
        sigma = size / 4.0
        ys, xs = torch.meshgrid(
            torch.arange(size, dtype=torch.float32),
            torch.arange(size, dtype=torch.float32),
            indexing="ij",
        )
        g = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma ** 2))
        return g / g.sum()

    def _compute(self, images: torch.Tensor) -> torch.Tensor:
        b = images.shape[0]
        m = self._map.to(images.device)
        return m.unsqueeze(0).expand(b, -1, -1).contiguous()


CenterBiasParametric = CenterBiasSaliency


if __name__ == "__main__":
    src = CenterBiasSaliency(image_size=64)
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
    print(
        f"CenterBiasSaliency OK — shape={tuple(sal.shape)}, "
        f"argmax at (x={px}, y={py}) ≈ center ({w // 2}, {h // 2}); "
        f"alias CenterBiasParametric={CenterBiasParametric is CenterBiasSaliency}"
    )
