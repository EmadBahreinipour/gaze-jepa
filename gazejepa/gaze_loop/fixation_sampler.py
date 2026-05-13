"""Pick one ``(x, y)`` pixel from a saliency × IOR map.

Two modes: ``"stochastic"`` draws from ``Categorical(saliency * ior + ε)``;
``"argmax"`` picks the top-scoring location deterministically.
"""

from __future__ import annotations

import torch


class FixationSampler:
    """Sample one fixation location from ``saliency × ior_mask``.

    ``epsilon`` is a numerical floor added to scores before normalisation
    so a fully-suppressed map still yields a valid sample.
    """

    def __init__(self, mode: str = "stochastic", epsilon: float = 1e-8):
        if mode not in ("stochastic", "argmax"):
            raise ValueError(
                f"mode must be 'stochastic' or 'argmax', got {mode!r}"
            )
        if epsilon < 0:
            raise ValueError(f"epsilon must be >= 0, got {epsilon}")
        self.mode = mode
        self.epsilon = float(epsilon)

    def __call__(
        self, saliency_map: torch.Tensor, ior_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return a ``(2,)`` ``(x, y)`` pixel-coordinate fixation."""
        if saliency_map.dim() != 2 or ior_mask.dim() != 2:
            raise ValueError(
                f"Expected 2D maps, got saliency {saliency_map.shape} and "
                f"ior {ior_mask.shape}"
            )
        if saliency_map.shape != ior_mask.shape:
            raise ValueError(
                f"Shape mismatch: saliency {tuple(saliency_map.shape)} vs "
                f"ior {tuple(ior_mask.shape)}"
            )

        scores = saliency_map * ior_mask + self.epsilon
        h, w = scores.shape

        if self.mode == "argmax":
            flat_idx = int(scores.flatten().argmax().item())
        else:
            probs = scores.flatten()
            probs = probs / probs.sum()
            flat_idx = int(torch.distributions.Categorical(probs=probs).sample().item())

        y = float(flat_idx // w)
        x = float(flat_idx % w)
        return torch.tensor([x, y], dtype=torch.float32, device=scores.device)


if __name__ == "__main__":
    H = W = 200
    sigma = 5.0
    cx, cy = 100, 100
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    saliency = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    ior = torch.ones(H, W)

    sampler_a = FixationSampler(mode="argmax")
    fix_a = sampler_a(saliency, ior)
    assert fix_a.shape == (2,)
    err_a = abs(fix_a[0].item() - cx) + abs(fix_a[1].item() - cy)
    assert err_a < 1.0, f"argmax fix {fix_a.tolist()} too far from peak ({cx}, {cy})"

    torch.manual_seed(0)
    sampler_s = FixationSampler(mode="stochastic")
    samples = torch.stack([sampler_s(saliency, ior) for _ in range(1000)])
    mean = samples.mean(dim=0)
    err_s = abs(mean[0].item() - cx) + abs(mean[1].item() - cy)
    assert err_s < 1.5, (
        f"stochastic empirical mean {mean.tolist()} too far from peak "
        f"({cx}, {cy})"
    )

    print(f"FixationSampler OK — argmax fix={fix_a.tolist()}, "
          f"stochastic 1000-sample mean=({mean[0]:.2f}, {mean[1]:.2f}) "
          f"near peak ({cx}, {cy})")
