"""Sequential scanpath generator: saliency × IOR × FixationSampler for ``T`` steps."""

from __future__ import annotations

from typing import Any

import torch

from gazejepa.gaze_loop.fixation_sampler import FixationSampler
from gazejepa.gaze_loop.ior import IORMask


class GazeLoop:
    """Sequential scanpath generator. ``saliency_source`` is any ``(B, 3, H, W) → (B, H, W)`` callable (sum-normalised)."""

    def __init__(
        self,
        saliency_source: Any,
        n_fixations: int = 5,
        image_size: int | tuple[int, int] = 200,
        ior_sigma: float = 20.0,
        ior_decay: float = 0.7,
        sampling_mode: str = "stochastic",
    ):
        if n_fixations < 1:
            raise ValueError(f"n_fixations must be >= 1, got {n_fixations}")
        self.saliency_source = saliency_source
        self.n_fixations = int(n_fixations)
        self.image_size = image_size
        self.ior = IORMask(image_size=image_size, sigma=ior_sigma, decay=ior_decay)
        self.sampler = FixationSampler(mode=sampling_mode)

    def __call__(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run on one ``(3, H, W)`` image → ``scanpath`` ``(T, 2)``, ``saliency_map``, ``ior_history`` (mask *before* each update)."""
        if image.dim() != 3:
            raise ValueError(f"Expected (3, H, W) image, got {tuple(image.shape)}")

        self.ior.reset()

        # One saliency call per image (static-scene assumption).
        with torch.no_grad():
            sal_batch = self.saliency_source(image.unsqueeze(0).float())
            saliency = sal_batch[0]

        if saliency.shape != self.ior().shape:
            raise RuntimeError(
                f"Saliency shape {tuple(saliency.shape)} must match IOR "
                f"shape {tuple(self.ior().shape)}; check saliency_source "
                "and image_size are consistent."
            )

        scanpath: list[torch.Tensor] = []
        ior_history: list[torch.Tensor] = []

        for _ in range(self.n_fixations):
            ior_history.append(self.ior().clone())
            fixation = self.sampler(saliency, self.ior())
            scanpath.append(fixation)
            self.ior.update(fixation)

        return {
            "scanpath":     torch.stack(scanpath, dim=0),
            "saliency_map": saliency,
            "ior_history":  torch.stack(ior_history, dim=0),
        }


if __name__ == "__main__":
    H = W = 200

    class _GaussianBlobSaliency:
        name = "_blob"

        def __init__(self, center=(100, 100), sigma=20.0):
            self.center = center
            self.sigma = sigma

        def __call__(self, images: torch.Tensor) -> torch.Tensor:
            b, _, h, w = images.shape
            cx, cy = self.center
            yy, xx = torch.meshgrid(
                torch.arange(h, dtype=torch.float32),
                torch.arange(w, dtype=torch.float32),
                indexing="ij",
            )
            blob = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * self.sigma ** 2))
            blob = blob / blob.sum()
            return blob.unsqueeze(0).expand(b, -1, -1).contiguous()

    torch.manual_seed(0)
    image = torch.zeros(3, H, W)
    loop = GazeLoop(
        saliency_source=_GaussianBlobSaliency(center=(100, 100), sigma=20.0),
        n_fixations=5,
        image_size=(H, W),
        ior_sigma=20.0,
        ior_decay=0.7,
        sampling_mode="argmax",
    )
    result = loop(image)

    T = 5
    assert result["scanpath"].shape == (T, 2)
    assert result["saliency_map"].shape == (H, W)
    assert result["ior_history"].shape == (T, H, W)

    sp = result["scanpath"]
    print(
        f"GazeLoop OK — scanpath: {tuple(sp.shape)}, "
        f"saliency: {tuple(result['saliency_map'].shape)}"
    )
    print(
        "  scanpath = "
        + ", ".join(f"({x.item():.0f},{y.item():.0f})" for x, y in sp)
    )

    d0 = (sp[0] - torch.tensor([100.0, 100.0])).norm().item()
    assert d0 < 1.5, (
        f"Argmax fixation 1 = {sp[0].tolist()} should be at peak (100, 100)"
    )

    for i in range(T):
        for j in range(i + 1, T):
            d = (sp[i] - sp[j]).norm().item()
            assert d >= 1.0, (
                f"Fixations {i+1} and {j+1} both at {sp[i].tolist()} — "
                "IOR is not preventing revisits"
            )

    pairwise = torch.cdist(sp.unsqueeze(0), sp.unsqueeze(0)).squeeze(0)
    max_pair = pairwise.max().item()
    print(
        f"  max pairwise distance across the scanpath: {max_pair:.1f} "
        f"(>= ior_sigma = 20.0)"
    )
    assert max_pair >= 20.0, (
        f"Scanpath max pairwise distance {max_pair:.1f} < ior_sigma; IOR "
        "may not be pushing fixations outward"
    )
    print("IOR non-revisit + spread check PASSED")
