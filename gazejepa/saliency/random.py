"""Uniform random saliency — baseline matching SaccadeJEPA's random crop selection."""

from __future__ import annotations

import torch

from gazejepa.saliency.base import SaliencySource, assert_saliency_contract


class RandomSaliency(SaliencySource):
    """Uniform random saliency map. Serves as a lower-bound baseline."""

    name = "random"

    def _compute(self, images: torch.Tensor) -> torch.Tensor:
        b, _, h, w = images.shape
        return torch.rand(b, h, w, device=images.device)


if __name__ == "__main__":
    torch.manual_seed(0)
    src = RandomSaliency()
    img = torch.zeros(2, 3, 64, 96)
    sal = src(img)
    assert_saliency_contract(sal, batch_size=2)
    flat = sal.reshape(2, -1)
    n = flat.shape[1]
    max_per_image = flat.max(dim=-1).values
    print(f"RandomSaliency OK — shape={tuple(sal.shape)}, "
          f"max per image={[f'{v:.5f}' for v in max_per_image.tolist()]}, "
          f"uniform expected ~={2.0/n:.5f}")
