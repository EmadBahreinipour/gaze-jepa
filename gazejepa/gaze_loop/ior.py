"""Stateful inhibition-of-return mask. Fixations are ``(x, y)``; mask indexed ``[y, x]``."""

from __future__ import annotations

import torch


class IORMask:
    """Stateful Gaussian IOR at image resolution. ``decay``: 1.0 = permanent, 0.0 = forget."""

    def __init__(
        self,
        image_size: int | tuple[int, int],
        sigma: float = 20.0,
        decay: float = 0.7,
        device: torch.device | str | None = None,
    ):
        if isinstance(image_size, int):
            self.h, self.w = int(image_size), int(image_size)
        else:
            self.h, self.w = int(image_size[0]), int(image_size[1])
        if self.h <= 0 or self.w <= 0:
            raise ValueError(f"image_size must be positive, got ({self.h}, {self.w})")
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if not (0.0 <= decay <= 1.0):
            raise ValueError(f"decay must be in [0, 1], got {decay}")
        self.sigma = float(sigma)
        self.decay = float(decay)
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        ys = torch.arange(self.h, dtype=torch.float32, device=self.device).view(-1, 1)
        xs = torch.arange(self.w, dtype=torch.float32, device=self.device).view(1, -1)
        self._ys = ys
        self._xs = xs

        self.reset()

    def reset(self) -> None:
        """Reset the mask to all 1s."""
        self.mask = torch.ones(self.h, self.w, dtype=torch.float32, device=self.device)

    def update(self, fixation: torch.Tensor) -> None:
        """Decay the mask and subtract a Gaussian well at ``fixation``."""
        if fixation.shape != (2,):
            raise ValueError(
                f"Expected (2,) fixation, got {tuple(fixation.shape)}"
            )

        self.mask = 1.0 - self.decay * (1.0 - self.mask)

        x = float(fixation[0].item())
        y = float(fixation[1].item())
        well = torch.exp(
            -((self._xs - x) ** 2 + (self._ys - y) ** 2) / (2.0 * self.sigma ** 2)
        )
        self.mask = self.mask * (1.0 - well)

        self.mask = self.mask.clamp(min=0.0)

    def __call__(self) -> torch.Tensor:
        """Return the current mask (no copy; caller clones if storing)."""
        return self.mask


if __name__ == "__main__":
    ior = IORMask(image_size=(64, 96), sigma=8.0, decay=0.7)
    assert (ior() == 1.0).all(), "Fresh mask should be all 1s"

    fixations = [
        torch.tensor([20.0, 30.0]),
        torch.tensor([60.0, 30.0]),
        torch.tensor([80.0, 50.0]),
    ]
    masks_after_each: list[torch.Tensor] = []
    for f in fixations:
        ior.update(f)
        masks_after_each.append(ior().clone())

    final = masks_after_each[-1]
    h, w = final.shape
    for i, f in enumerate(fixations):
        x, y = int(f[0].item()), int(f[1].item())
        assert final[y, x] < 0.95, (
            f"Well {i} at ({x}, {y}) not suppressed (mask={final[y, x]:.3f})"
        )

    oldest_depth = 1.0 - final[int(fixations[0][1]), int(fixations[0][0])].item()
    newest_depth = 1.0 - final[int(fixations[2][1]), int(fixations[2][0])].item()
    assert newest_depth > oldest_depth, (
        f"Newest well should be deeper than oldest: "
        f"newest={newest_depth:.3f}, oldest={oldest_depth:.3f}"
    )

    print(f"IORMask OK — shape={tuple(final.shape)}, "
          f"oldest-well depth={oldest_depth:.3f}, "
          f"newest-well depth={newest_depth:.3f} (newest > oldest as expected)")
