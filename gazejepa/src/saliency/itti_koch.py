"""Classical saliency via OpenCV's spectral residual detector.

This uses cv2.saliency.StaticSaliencySpectralResidual, which implements
the algorithm from:
    Hou, X. & Zhang, L. (2007). Saliency detection: A spectral residual approach.
    CVPR 2007.

This is in the same intellectual tradition as Itti & Koch (1998) — both are
image-statistic-based, non-learned methods — but the actual algorithm differs.
The report is explicit about this distinction.

If OpenCV's saliency module is unavailable, IttiKochSaliency raises ImportError
with a clear message. See fallback_saliency() below for a minimal alternative.
"""

import numpy as np
import torch
from . import SaliencySource


def _check_opencv_saliency() -> bool:
    try:
        import cv2
        _ = cv2.saliency.StaticSaliencySpectralResidual_create()
        return True
    except (AttributeError, Exception):
        return False


class IttiKochSaliency(SaliencySource):
    """Spectral residual saliency (Hou & Zhang 2007) via OpenCV.

    Processes each image in the batch independently, normalizes output to sum=1.
    Input images are expected as (B, 3, H, W) ImageNet-normalized tensors;
    they are de-normalized internally before passing to OpenCV.
    """

    name = "itti_koch"

    # ImageNet mean/std for de-normalization
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self):
        import cv2
        if not _check_opencv_saliency():
            raise ImportError(
                "cv2.saliency module not available. Install opencv-contrib-python: "
                "pip install opencv-contrib-python"
            )
        self._detector = cv2.saliency.StaticSaliencySpectralResidual_create()

    def _tensor_to_bgr(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Convert (3, H, W) ImageNet-normalized tensor to uint8 BGR for OpenCV."""
        import cv2
        img = image_tensor.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        img = img * self._STD + self._MEAN
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        maps = []
        for i in range(images.shape[0]):
            bgr = self._tensor_to_bgr(images[i])
            _, sal = self._detector.computeSaliency(bgr)
            sal = sal.astype(np.float32)
            total = sal.sum()
            if total > 0:
                sal /= total
            maps.append(torch.from_numpy(sal))
        return torch.stack(maps, dim=0).to(images.device)


class LocalContrastSaliency(SaliencySource):
    """Minimal classical saliency based on local contrast (fallback).

    Computes the difference between a blurred and a more-blurred version
    of the luminance channel across multiple scales. Used when OpenCV's
    saliency module is unavailable.
    """

    name = "local_contrast"

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        B, _, H, W = images.shape
        # Convert to luminance
        r, g, b = images[:, 0], images[:, 1], images[:, 2]
        lum = (0.2126 * r + 0.7152 * g + 0.0722 * b).unsqueeze(1)  # (B, 1, H, W)

        maps = torch.zeros(B, H, W, device=images.device)
        for sigma in [2, 4, 8, 16]:
            k = max(int(4 * sigma) | 1, 3)
            blur = F.avg_pool2d(lum, kernel_size=k, stride=1, padding=k // 2)
            blur2 = F.avg_pool2d(lum, kernel_size=k * 2 + 1, stride=1, padding=k)
            maps += (blur - blur2).abs().squeeze(1)

        maps = maps - maps.flatten(1).min(1).values.view(B, 1, 1)
        sums = maps.flatten(1).sum(1).clamp(min=1e-8).view(B, 1, 1)
        return maps / sums


def get_classical_saliency() -> SaliencySource:
    """Return IttiKochSaliency if OpenCV is available, else LocalContrastSaliency."""
    if _check_opencv_saliency():
        return IttiKochSaliency()
    print("OpenCV saliency module not found, using LocalContrastSaliency fallback.")
    return LocalContrastSaliency()


if __name__ == "__main__":
    x = torch.zeros(2, 3, 224, 224)
    x[0, :, 50:80, 60:100] = 1.0  # bright rectangle

    src = get_classical_saliency()
    out = src(x)
    print(f"Shape: {out.shape}, sum: {out[0].sum():.6f}")
    print(f"Source used: {src.name}")
    print("Classical saliency OK")
