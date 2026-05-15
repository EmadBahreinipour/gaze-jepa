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
    get_frame_count,
    get_frame_fixations,
    get_split,
    load_fixations,
    load_frame,
    make_heatmap,
    rescale_fixations,
)


class FINDSaliencyDataset(Dataset):
    """Yields ``(image_tensor, heatmap_tensor)`` pairs from a FIND split.

    Args:
        data_root: FIND data root directory.
        split: ``"train"`` / ``"val"`` / ``"test"``. Resolved through
            :func:`gazejepa.data.get_split` with the canonical ``seed=42``
            split shared across the project.
        image_size: Resize each frame to ``(image_size, image_size)``.
        sigma: Gaussian sigma (in image_size pixels) for the GT heatmap.
        frame_stride: Sample every Nth frame.
        min_observers: Skip frames with fewer valid fixations than this.
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
        min_observers: int = 5,
        cache_dir: str | os.PathLike[str] = "outputs/saliency",
        seed: int = 42,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {split!r}")

        self.data_root = str(data_root)
        self.split = split
        self.image_size = int(image_size)
        self.sigma = float(sigma)
        self.frame_stride = int(frame_stride)
        self.min_observers = int(min_observers)
        self.seed = int(seed)

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / (
            f"find_index_{split}_s{frame_stride}_m{min_observers}_seed{seed}.pkl"
        )

        if cache_path.is_file():
            with cache_path.open("rb") as fh:
                self.index, self.fix_arrays = pickle.load(fh)
        else:
            video_ids = get_split(self.data_root, seed=self.seed)[split]
            self.fix_arrays: dict[str, np.ndarray] = {}
            self.index: list[tuple[str, int]] = []

            for vid in video_ids:
                fix_array = load_fixations(self.data_root, vid)
                self.fix_arrays[vid] = fix_array
                n_frames = get_frame_count(self.data_root, vid)
                for f in range(0, n_frames, self.frame_stride):
                    fixations = get_frame_fixations(fix_array, f)
                    if len(fixations) >= self.min_observers:
                        self.index.append((vid, f))

            with cache_path.open("wb") as fh:
                pickle.dump((self.index, self.fix_arrays), fh)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        video_id, frame_idx = self.index[i]
        frame = load_frame(self.data_root, video_id, frame_idx, size=self.image_size)
        image = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0  # (3, H, W)

        fixations = get_frame_fixations(self.fix_arrays[video_id], frame_idx)
        fixations_scaled = rescale_fixations(
            fixations,
            src_size=(FIND_NATIVE_W, FIND_NATIVE_H),
            dst_size=(self.image_size, self.image_size),
        )
        heatmap = make_heatmap(
            fixations_scaled, self.image_size, self.image_size, sigma=self.sigma,
        )
        # make_heatmap returns max-normalised; convert to a probability for KL.
        heatmap = heatmap.astype(np.float32)
        total = heatmap.sum()
        if total > 0:
            heatmap = heatmap / total
        return image, torch.from_numpy(heatmap)


if __name__ == "__main__":
    from gazejepa.data import resolve_data_root

    data_root = resolve_data_root()
    ds = FINDSaliencyDataset(
        data_root, split="val", image_size=224, frame_stride=30,
    )
    print(f"FINDSaliencyDataset(val) — {len(ds)} frames cached.")
    if len(ds):
        image, heat = ds[0]
        print(
            f"  sample: image {tuple(image.shape)} dtype={image.dtype} "
            f"max={image.max():.3f}; heatmap {tuple(heat.shape)} sum={heat.sum():.4f}"
        )
