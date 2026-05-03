"""Compact reimplementation of Itti, Koch & Niebur (1998).

Included: intensity, RG/BY colour opponents (with the 10%-of-max intensity
mask), Sobel-gradient orientation channel, 5-level Gaussian pyramid,
center-surround at (c, s) in {(0, 2), (1, 3), (2, 4)}, an N(.)-style
``(1 - mean)^2`` across-scale normalisation, and a final Gaussian smooth.

Deliberately omitted: the four-Gabor orientation bank (replaced by one
Sobel channel), the paper's six center-surround pairs, the iterative
DoG-based N(.) with local maxima, and winner-take-all dynamics.
"""

from __future__ import annotations

import cv2
import numpy as np
import scipy.ndimage as ndi
import torch

from gazejepa.saliency.base import SaliencySource, assert_saliency_contract


def _build_pyramid(arr: np.ndarray, n_levels: int) -> list[np.ndarray]:
    """Gaussian pyramid via successive ``cv2.pyrDown``."""
    pyr = [arr.astype(np.float32)]
    for _ in range(n_levels - 1):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def _center_surround(
    pyr: list[np.ndarray], c_idx: int, s_idx: int, target_hw: tuple[int, int],
) -> np.ndarray:
    """``|center - upsample(surround)|`` resampled to ``target_hw``."""
    cmap = pyr[c_idx]
    smap = pyr[s_idx]
    s_up = cv2.resize(smap, (cmap.shape[1], cmap.shape[0]), interpolation=cv2.INTER_LINEAR)
    diff = np.abs(cmap - s_up)
    th, tw = target_hw
    if diff.shape != target_hw:
        diff = cv2.resize(diff, (tw, th), interpolation=cv2.INTER_LINEAR)
    return diff


def _normalize_n(m: np.ndarray) -> np.ndarray:
    """Approximate Itti N(.): rescale to [0, 1] then weight by ``(1 - mean)^2``.

    The original uses ``(M - m_local)^2`` over local maxima for stronger
    suppression of cluttered maps; the global-mean form is faster and
    sufficient for a classical baseline.
    """
    m = m - m.min()
    m_max = m.max()
    if m_max <= 0:
        return m
    m = m / m_max
    return m * (1.0 - m.mean()) ** 2


def _itti_koch_one(
    img_uint8: np.ndarray,
    n_levels: int = 5,
    sigma_smooth: float = 8.0,
) -> np.ndarray:
    """Saliency for one ``(H, W, 3)`` uint8 RGB image."""
    img = img_uint8.astype(np.float32) / 255.0
    H, W = img.shape[:2]
    target = (H, W)

    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    intensity = (r + g + b) / 3.0

    # Itti's intensity mask: zero out pixels below 10% of max before computing
    # colour opponents to avoid spurious saliency in dark regions.
    max_i = intensity.max()
    valid = intensity > 0.1 * max_i
    r2 = np.where(valid, r, 0.0)
    g2 = np.where(valid, g, 0.0)
    b2 = np.where(valid, b, 0.0)

    R_ = np.clip(r2 - 0.5 * (g2 + b2), 0, None)
    G_ = np.clip(g2 - 0.5 * (r2 + b2), 0, None)
    B_ = np.clip(b2 - 0.5 * (r2 + g2), 0, None)
    Y_ = np.clip(0.5 * (r2 + g2) - 0.5 * np.abs(r2 - g2) - b2, 0, None)
    RG = R_ - G_
    BY = B_ - Y_

    sx = ndi.sobel(intensity, axis=1, mode="reflect")
    sy = ndi.sobel(intensity, axis=0, mode="reflect")
    orient = np.sqrt(sx ** 2 + sy ** 2)

    I_pyr = _build_pyramid(intensity, n_levels)
    RG_pyr = _build_pyramid(RG, n_levels)
    BY_pyr = _build_pyramid(BY, n_levels)
    O_pyr = _build_pyramid(orient, n_levels)

    cs_pairs = [(0, 2), (1, 3), (2, 4)]

    def conspicuity(pyr: list[np.ndarray]) -> np.ndarray:
        cs_maps = [_center_surround(pyr, c, s, target) for c, s in cs_pairs]
        return _normalize_n(np.sum(cs_maps, axis=0))

    I_c = conspicuity(I_pyr)
    RG_c = conspicuity(RG_pyr)
    BY_c = conspicuity(BY_pyr)
    O_c = conspicuity(O_pyr)
    C_c = (RG_c + BY_c) / 2.0

    saliency = (I_c + C_c + O_c) / 3.0
    saliency = ndi.gaussian_filter(saliency, sigma=sigma_smooth)
    return saliency.astype(np.float32)


class IttiKochSaliency(SaliencySource):
    """Compact Itti-Koch (1998) saliency."""

    name = "itti_koch"

    def __init__(self, n_levels: int = 5, sigma_smooth: float = 8.0):
        if n_levels < 5:
            raise ValueError(
                f"n_levels must be ≥ 5 for the (0, 2)/(1, 3)/(2, 4) "
                f"center-surround pairs, got {n_levels}"
            )
        if sigma_smooth <= 0:
            raise ValueError(f"sigma_smooth must be positive, got {sigma_smooth}")
        self.n_levels = int(n_levels)
        self.sigma_smooth = float(sigma_smooth)

    def _compute(self, images: torch.Tensor) -> torch.Tensor:
        b, _, h, w = images.shape
        if images.dtype == torch.uint8:
            arr = images.detach().cpu().numpy()
        else:
            arr = (images.detach().clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        out = np.empty((b, h, w), dtype=np.float32)
        for i in range(b):
            img_hwc = np.transpose(arr[i], (1, 2, 0))
            out[i] = _itti_koch_one(
                img_hwc, n_levels=self.n_levels, sigma_smooth=self.sigma_smooth,
            )
        return torch.from_numpy(out).to(images.device)


if __name__ == "__main__":
    src = IttiKochSaliency(n_levels=5, sigma_smooth=4.0)
    img = torch.full((1, 3, 64, 96), 0.4, dtype=torch.float32)
    img[:, :, 20:35, 30:50] = 1.0  # bright square
    sal = src(img)
    assert_saliency_contract(sal, batch_size=1)
    idx = int(sal[0].flatten().argmax().item())
    py, px = idx // sal.shape[-1], idx % sal.shape[-1]
    in_box = 18 <= py <= 37 and 28 <= px <= 52
    print(f"IttiKochSaliency OK — shape={tuple(sal.shape)}, "
          f"argmax at (x={px}, y={py}) {'inside' if in_box else 'NEAR'} bright square")
    assert in_box, f"Argmax ({px}, {py}) not inside bright-square ROI"
