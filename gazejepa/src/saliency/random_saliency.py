"""Random saliency baseline — equivalent to what SaccadeJEPA actually does.

SaccadeJEPA's SaccadeCropper samples translation uniformly at random:
    torch.empty(b, 2).uniform_(-max_translation_frac, max_translation_frac)
This RandomSaliency class makes that behavior explicit and evaluable.
"""

import torch
from . import SaliencySource


class RandomSaliency(SaliencySource):
    name = "random"

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        B, _, H, W = images.shape
        out = torch.rand(B, H, W, device=images.device)
        out = out / out.sum(dim=(1, 2), keepdim=True)
        return out


if __name__ == "__main__":
    source = RandomSaliency()
    x = torch.rand(4, 3, 224, 224)
    out = source(x)
    print(f"Shape: {out.shape}")
    print(f"Sum per sample: {out.sum(dim=(1,2))}")
    assert out.shape == (4, 224, 224)
    assert torch.allclose(out.sum(dim=(1, 2)), torch.ones(4), atol=1e-5)
    print("RandomSaliency OK")
