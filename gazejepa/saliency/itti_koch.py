"""Spectral-residual saliency (Hou & Zhang 2007) via OpenCV's contrib module.

This is the "classical" saliency source Arash used in his four-source
comparison; the class is named ``IttiKochSaliency`` for continuity with his
report and his ``saliency_comparison.csv`` row labels. The underlying
algorithm is the spectral residual approach (Hou & Zhang 2007), accessed
through ``cv2.saliency.StaticSaliencySpectralResidual``.

Requires the ``opencv-contrib-python`` (or ``opencv-contrib-python-headless``)
distribution — the regular ``opencv-python`` and ``opencv-python-headless``
packages do not ship the ``cv2.saliency`` namespace. If you have only the
plain OpenCV installed, use :class:`gazejepa.saliency.LocalContrastSaliency`
as a fallback (Arash's published "local_contrast" row).
"""

from __future__ import annotations

import numpy as np
import torch

from gazejepa.saliency.base import SaliencySource


class IttiKochSaliency(SaliencySource):
    """Hou-Zhang spectral-residual saliency via OpenCV's contrib saliency module.

    Input is expected as a ``(B, 3, H, W)`` float tensor in ``[0, 1]`` (or
    ``uint8``). The detector is run per-image on the de-tensored uint8 BGR
    frame, and the resulting raw map is returned for the
    :class:`SaliencySource` ABC to resample and renormalise.
    """

    name = "itti_koch"

    def __init__(self) -> None:
        try:
            import cv2
            self._detector = cv2.saliency.StaticSaliencySpectralResidual_create()
        except (AttributeError, ImportError) as exc:
            raise ImportError(
                "cv2.saliency.StaticSaliencySpectralResidual is not available. "
                "Install with: pip install opencv-contrib-python "
                "(or opencv-contrib-python-headless on a server). "
                "Note that opencv-contrib-python conflicts with the plain "
                "opencv-python / opencv-python-headless package — uninstall "
                "those first."
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
            rgb = np.transpose(np_imgs[i], (1, 2, 0))  # (H, W, 3)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            ok, sal = self._detector.computeSaliency(bgr)
            if not ok:
                # Detector occasionally fails on near-uniform frames; a
                # uniform map is the right "no preference" fallback.
                sal = np.ones(bgr.shape[:2], dtype=np.float32)
            maps.append(torch.from_numpy(sal.astype(np.float32)))
        return torch.stack(maps, dim=0).to(images.device)


if __name__ == "__main__":
    from gazejepa.saliency.base import assert_saliency_contract

    src = IttiKochSaliency()
    img = torch.full((2, 3, 64, 96), 0.4)
    img[0, :, 20:35, 30:50] = 1.0
    sal = src(img)
    assert_saliency_contract(sal, batch_size=2)
    print(
        f"IttiKochSaliency (spectral residual) OK — shape={tuple(sal.shape)}"
    )
