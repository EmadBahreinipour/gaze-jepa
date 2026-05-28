"""FIND-backed (image, heatmap) Dataset for training saliency predictors.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from gazejepa.data import (
    FIND_NATIVE_H,
    FIND_NATIVE_W,
    MIN_OBSERVERS_PER_FRAME,
    get_frame_count,
    get_frame_fixations,
    get_split,
    load_fixations,
    load_frame,
    rescale_fixations,
)


def _make_heatmap(fixations: np.ndarray, image_size: int, sigma: float) -> np.ndarray:
    """Sum-to-1 Gaussian heatmap at ``image_size × image_size``."""
    heatmap = np.zeros((image_size, image_size), dtype=np.float64)
    radius = int(3 * sigma)
    for x, y in fixations:
        xi, yi = int(round(float(x))), int(round(float(y)))
        if not (0 <= xi < image_size and 0 <= yi < image_size):
            continue
        y_lo, y_hi = max(0, yi - radius), min(image_size, yi + radius + 1)
        x_lo, x_hi = max(0, xi - radius), min(image_size, xi + radius + 1)
        yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
        heatmap[y_lo:y_hi, x_lo:x_hi] += np.exp(
            -((xx - xi) ** 2 + (yy - yi) ** 2) / (2.0 * sigma ** 2)
        )
    total = heatmap.sum()
    if total > 0:
        heatmap /= total
    else:
        heatmap[:] = 1.0 / (image_size * image_size)
    return heatmap.astype(np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────

class FINDSaliencyDataset(Dataset):
    """Yields ``(image_tensor, heatmap_tensor)`` pairs from a FIND split.

    Args:
        data_root: FIND data root directory.
        split: ``"train"`` / ``"val"`` / ``"test"``.
        image_size: Resize each frame to ``(image_size, image_size)``.
        sigma: Gaussian sigma in image_size pixels for the GT heatmap.
        frame_stride: Sample every Nth frame.
        min_observers: Skip frames with fewer fixations than this.
        cache_dir: Where to cache the index pickle for fast re-instantiation.
        seed: Forwarded to :func:`gazejepa.data.get_split`.
    """

    def __init__(
        self,
        data_root: str | os.PathLike[str],
        split: str,
        image_size: int = 224,
        sigma: float = 20.0,
        frame_stride: int = 10,
        min_observers: int = MIN_OBSERVERS_PER_FRAME,
        cache_dir: str | os.PathLike[str] = "outputs/saliency",
        seed: int = 42,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {split!r}")

        self.data_root = str(data_root)
        self.split = split
        self.image_size = int(image_size)
        self.sigma = float(sigma)

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / (
            f"find_index_{split}_s{frame_stride}_m{min_observers}_seed{seed}.pkl"
        )

        if cache_path.is_file():
            with cache_path.open("rb") as fh:
                self.index, self.fix_arrays = pickle.load(fh)
        else:
            video_ids = get_split(self.data_root, seed=seed)[split]
            self.fix_arrays: dict[str, np.ndarray] = {}
            self.index: list[tuple[str, int]] = []

            for vid in video_ids:
                fix_array = load_fixations(self.data_root, vid)
                self.fix_arrays[vid] = fix_array
                n_frames = get_frame_count(self.data_root, vid)
                for f in range(0, n_frames, frame_stride):
                    if len(get_frame_fixations(fix_array, f)) >= min_observers:
                        self.index.append((vid, f))

            with cache_path.open("wb") as fh:
                pickle.dump((self.index, self.fix_arrays), fh)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        video_id, frame_idx = self.index[i]
        frame = load_frame(self.data_root, video_id, frame_idx, size=self.image_size)
        image = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

        fixations = get_frame_fixations(self.fix_arrays[video_id], frame_idx)
        fixations_scaled = rescale_fixations(
            fixations,
            src_size=(FIND_NATIVE_W, FIND_NATIVE_H),
            dst_size=(self.image_size, self.image_size),
        )
        heatmap = _make_heatmap(fixations_scaled, self.image_size, self.sigma)
        return image, torch.from_numpy(heatmap)


if __name__ == "__main__":
    import sys
    data_root = sys.argv[1] if len(sys.argv) > 1 else "data/find_dataset"
    ds = FINDSaliencyDataset(data_root, split="val", image_size=224, frame_stride=30)
    print(f"FINDSaliencyDataset(val) — {len(ds)} frames.")
    if len(ds):
        image, heat = ds[0]
        print(
            f"  sample: image {tuple(image.shape)} max={image.max():.3f}; "
            f"heatmap {tuple(heat.shape)} sum={heat.sum():.4f}"
        )
