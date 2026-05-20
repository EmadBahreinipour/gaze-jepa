"""Spectral-residual saliency (Hou & Zhang, CVPR 2007) via OpenCV's contrib module.

"""

from __future__ import annotations

import numpy as np
import torch

from gazejepa.saliency.base import SaliencySource


class SpectralResidualSaliency(SaliencySource):
    """Hou-Zhang spectral-residual saliency via OpenCV's StaticSaliencySpectralResidual.

    Input: (B, 3, H, W) float tensor in [0, 1]. Runs per-image on a uint8
    BGR frame; the SaliencySource ABC resamples and renormalises the output.
    """

    name = "spectral_residual"

    def __init__(self) -> None:
        try:
            import cv2
            self._detector = cv2.saliency.StaticSaliencySpectralResidual_create()
        except (AttributeError, ImportError) as exc:
            raise ImportError(
                "cv2.saliency is not available. "
                "Install opencv-contrib-python or opencv-contrib-python-headless."
            ) from exc

    def _compute(self, images: torch.Tensor) -> torch.Tensor:
        import cv2

        b = images.shape[0]
        if images.dtype == torch.uint8:
            np_imgs = images.detach().cpu().numpy()
        else:
            np_imgs = (
                images.detach().clamp(0, 1).cpu().numpy() * 255.0
            ).astype(np.uint8)

        maps: list[torch.Tensor] = []
        for i in range(b):
            rgb = np.transpose(np_imgs[i], (1, 2, 0))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            ok, sal = self._detector.computeSaliency(bgr)
            if not ok:
                sal = np.ones(bgr.shape[:2], dtype=np.float32)
            maps.append(torch.from_numpy(sal.astype(np.float32)))
        return torch.stack(maps, dim=0).to(images.device)


if __name__ == "__main__":
    from gazejepa.saliency.base import assert_saliency_contract

    src = SpectralResidualSaliency()
    img = torch.full((2, 3, 64, 96), 0.4)
    img[0, :, 20:35, 30:50] = 1.0
    sal = src(img)
    assert_saliency_contract(sal, batch_size=2)
    print(f"SpectralResidualSaliency OK — shape={tuple(sal.shape)}")
