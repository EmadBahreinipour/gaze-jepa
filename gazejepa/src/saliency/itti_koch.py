"""Classical saliency via OpenCV's spectral residual detector.

Implements the algorithm from:
    Hou, X. & Zhang, L. (2007). Saliency detection: A spectral residual approach.
    CVPR 2007.

Requires: opencv-contrib-python (pip install opencv-contrib-python)
"""

import numpy as np
import torch
from . import SaliencySource


class IttiKochSaliency(SaliencySource):
    """Spectral residual saliency (Hou & Zhang 2007) via OpenCV.

    Processes each image in the batch independently, normalizes output to sum=1.
    Input images are (B, 3, H, W) ImageNet-normalized tensors;
    de-normalized internally before passing to OpenCV.
    """

    name = "itti_koch"

    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self):
        import cv2
        self._detector = cv2.saliency.StaticSaliencySpectralResidual_create()

    def _tensor_to_bgr(self, image_tensor: torch.Tensor) -> np.ndarray:
        import cv2
        img = image_tensor.permute(1, 2, 0).cpu().numpy()
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


def get_classical_saliency() -> SaliencySource:
    return IttiKochSaliency()


if __name__ == "__main__":
    x = torch.zeros(2, 3, 224, 224)
    x[0, :, 50:80, 60:100] = 1.0

    src = get_classical_saliency()
    out = src(x)
    print(f"Shape: {out.shape}, sum: {out[0].sum():.6f}")
    print("IttiKochSaliency OK")
