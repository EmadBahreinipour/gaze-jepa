"""Center bias saliency — parametric Gaussian and data-fit variants."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from gazejepa.saliency.base import SaliencySource, assert_saliency_contract


class CenterBiasSaliency(SaliencySource):
    """Fixed 2D Gaussian centered on the image, σ = ``image_size / 4``."""

    name = "center_bias_parametric"

    def __init__(self, image_size: int = 224):
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        self.image_size = int(image_size)
        self._map = self._build_map(self.image_size)

    @staticmethod
    def _build_map(size: int) -> torch.Tensor:
        cx = cy = size / 2.0
        sigma = size / 4.0
        ys, xs = torch.meshgrid(
            torch.arange(size, dtype=torch.float32),
            torch.arange(size, dtype=torch.float32),
            indexing="ij",
        )
        g = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma ** 2))
        return g / g.sum()

    def _compute(self, images: torch.Tensor) -> torch.Tensor:
        b = images.shape[0]
        m = self._map.to(images.device)
        return m.unsqueeze(0).expand(b, -1, -1).contiguous()


CenterBiasParametric = CenterBiasSaliency


class CenterBiasDataFit(SaliencySource):
    """Mean fixation heatmap fitted over the FIND training split.

    Loads from cache on first use; call build_and_cache() once to create it.
    """

    name = "center_bias_data_fit"

    def __init__(
        self,
        cache_path: str | os.PathLike[str] = "outputs/saliency/center_bias_train_avg.npy",
        image_size: int = 224,
    ):
        if image_size <= 0:
            raise ValueError(f"image_size must be positive, got {image_size}")
        self.cache_path = Path(cache_path)
        self.image_size = int(image_size)
        self._map: torch.Tensor | None = None

    def build_and_cache(
        self,
        data_root: str | os.PathLike[str],
        sigma: float = 20.0,
        frame_stride: int = 10,
        min_observers: int = 5,
        seed: int = 42,
    ) -> None:
        """Compute and cache the average training-split fixation heatmap."""
        from gazejepa.data import (
            get_frame_count,
            get_frame_fixations,
            get_split,
            load_fixations,
            make_heatmap,
            rescale_fixations,
        )

        train_ids = get_split(data_root, seed=seed)["train"]
        accumulator = np.zeros(
            (self.image_size, self.image_size), dtype=np.float64,
        )
        count = 0

        for vid in train_ids:
            fix_array = load_fixations(data_root, vid)
            n_frames = get_frame_count(data_root, vid)
            for f in range(0, n_frames, frame_stride):
                fixations = get_frame_fixations(fix_array, f)
                if len(fixations) < min_observers:
                    continue
                fixations_scaled = rescale_fixations(
                    fixations,
                    src_size=(1280, 720),
                    dst_size=(self.image_size, self.image_size),
                )
                heatmap = make_heatmap(
                    fixations_scaled, self.image_size, self.image_size, sigma=sigma,
                )
                accumulator += heatmap
                count += 1

        if count == 0:
            raise RuntimeError(
                "No frames passed the min-observer filter; cannot fit center bias."
            )

        avg = accumulator / count
        avg /= max(avg.sum(), 1e-12)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.cache_path, avg.astype(np.float32))
        self._map = torch.from_numpy(avg.astype(np.float32))

    def _ensure_loaded(self, device: torch.device) -> torch.Tensor:
        if self._map is None:
            if not self.cache_path.is_file():
                raise FileNotFoundError(
                    f"CenterBiasDataFit cache missing at {self.cache_path}; "
                    "call build_and_cache(data_root) once first."
                )
            arr = np.load(self.cache_path)
            self._map = torch.from_numpy(arr.astype(np.float32))
        return self._map.to(device)

    def _compute(self, images: torch.Tensor) -> torch.Tensor:
        b = images.shape[0]
        m = self._ensure_loaded(images.device)
        return m.unsqueeze(0).expand(b, -1, -1).contiguous()


if __name__ == "__main__":
    src = CenterBiasSaliency(image_size=64)
    img = torch.zeros(2, 3, 64, 96)
    sal = src(img)
    assert_saliency_contract(sal, batch_size=2)

    flat = sal[0].flatten()
    idx = int(flat.argmax().item())
    h, w = sal.shape[-2:]
    py, px = idx // w, idx % w
    assert abs(py - h // 2) <= 1 and abs(px - w // 2) <= 1, (
        f"Argmax {(px, py)} not at center {(w // 2, h // 2)}"
    )
    print(
        f"CenterBiasSaliency OK — shape={tuple(sal.shape)}, "
        f"argmax at (x={px}, y={py}) ≈ center ({w // 2}, {h // 2}); "
        f"alias CenterBiasParametric={CenterBiasParametric is CenterBiasSaliency}"
    )
