"""Center-bias saliency baselines.

Two variants:
  - CenterBiasParametric: fixed Gaussian at image center (no data required)
  - CenterBiasDataFit: average fixation heatmap over the training split (cached)

The data-fit version is the standard in the saliency literature and is used
as the canonical center-bias baseline in all reported results.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from . import SaliencySource


class CenterBiasParametric(SaliencySource):
    """Fixed 2D Gaussian centered on the image, sigma = image_size / 4."""

    name = "center_bias_parametric"

    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self._map = self._build_map(image_size)

    def _build_map(self, size: int) -> torch.Tensor:
        cx = cy = size / 2.0
        sigma = size / 4.0
        ys, xs = torch.meshgrid(torch.arange(size, dtype=torch.float32),
                                torch.arange(size, dtype=torch.float32),
                                indexing="ij")
        g = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        B = images.shape[0]
        m = self._map.to(images.device)
        return m.unsqueeze(0).expand(B, -1, -1)


class CenterBiasDataFit(SaliencySource):
    """Average fixation heatmap over the FIND training split.

    On first call, iterates through training frames, accumulates heatmaps,
    and caches the result to cache_path. Subsequent calls load from cache.
    """

    name = "center_bias"

    def __init__(self, cache_path: str = "outputs/center_bias_train_avg.npy",
                 image_size: int = 224):
        self.cache_path = cache_path
        self.image_size = image_size
        self._map: torch.Tensor | None = None

    def build_and_cache(self, data_root: str, sigma: float = 20.0,
                        frame_stride: int = 10,
                        min_observers: int = 5) -> None:
        """Compute the average fixation heatmap and save to disk.

        Only needs to run once. Takes ~10–20 minutes on a laptop.
        """
        from src.find_data import (get_video_ids, load_fixations,
                                    get_frame_count, get_frame_fixations,
                                    rescale_fixations, make_heatmap,
                                    get_video_size, MIN_OBSERVERS_PER_FRAME)

        video_ids = get_video_ids(data_root, "train")
        accumulator = np.zeros((self.image_size, self.image_size), dtype=np.float64)
        count = 0

        for vid in video_ids:
            print(f"  Processing {vid}...")
            src_size = get_video_size(data_root, vid)
            fix_array = load_fixations(data_root, vid)
            n_frames = get_frame_count(data_root, vid)

            for f in range(0, n_frames, frame_stride):
                fixations = get_frame_fixations(fix_array, f)
                if len(fixations) < min_observers:
                    continue
                fixations_scaled = rescale_fixations(fixations, src_size,
                                                     (self.image_size, self.image_size))
                heatmap = make_heatmap(fixations_scaled, self.image_size, sigma)
                accumulator += heatmap
                count += 1

        if count == 0:
            raise RuntimeError("No valid frames found for center-bias computation.")

        avg = accumulator / count
        avg /= avg.sum()
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        np.save(self.cache_path, avg.astype(np.float32))
        print(f"Cached center-bias map ({count} frames) to {self.cache_path}")
        self._map = torch.from_numpy(avg.astype(np.float32))

    def _load(self) -> torch.Tensor:
        if not os.path.exists(self.cache_path):
            raise FileNotFoundError(
                f"Center-bias cache not found at {self.cache_path}. "
                "Run build_and_cache(data_root) first."
            )
        arr = np.load(self.cache_path)
        return torch.from_numpy(arr)

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        if self._map is None:
            self._map = self._load()
        B = images.shape[0]
        m = self._map.to(images.device)
        return m.unsqueeze(0).expand(B, -1, -1)


if __name__ == "__main__":
    x = torch.rand(2, 3, 224, 224)

    src = CenterBiasParametric()
    out = src(x)
    print(f"Parametric shape: {out.shape}, sum: {out[0].sum():.6f}")
    assert out.shape == (2, 224, 224)
    print("CenterBiasParametric OK")
