"""FIND-backed (image, heatmap) Dataset for training saliency predictors.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset

from gazejepa.data import (
    FIND_NATIVE_H,
    FIND_NATIVE_W,
    get_split,
    load_frame,
)


# ── fixation helpers (read from fix_data_NEW, the original FIND format) ──────

def _load_fixations(data_root: str, video_id: str) -> np.ndarray:
    """``(n_observers, n_frames, 2)`` float64 from fix_data_NEW/{video_id}.mat."""
    path = Path(data_root) / "Our_database" / "fix_data_NEW" / f"{video_id}.mat"
    mat = scipy.io.loadmat(str(path))
    arr = mat["curr_v_all_s"]          # (39, 1) object array
    n_obs = arr.shape[0]
    n_frames = arr[0][0].shape[0]
    out = np.zeros((n_obs, n_frames, 2), dtype=np.float64)
    for obs in range(n_obs):
        out[obs] = arr[obs][0].astype(np.float64)
    return out


def _get_frame_fixations(fix_array: np.ndarray, frame_idx: int) -> np.ndarray:
    """``(n_observers, 2)`` (x, y) for one frame."""
    return fix_array[:, frame_idx, :]


def _rescale_fixations(
    fixations: np.ndarray, src_w: int, src_h: int, dst: int,
) -> np.ndarray:
    out = fixations.copy().astype(np.float64)
    out[:, 0] = out[:, 0] * dst / src_w
    out[:, 1] = out[:, 1] * dst / src_h
    return out


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
                fix_array = _load_fixations(self.data_root, vid)
                self.fix_arrays[vid] = fix_array
                n_frames = fix_array.shape[1]
                for f in range(0, n_frames, frame_stride):
                    fixations = _get_frame_fixations(fix_array, f)
                    # count non-zero observers (zeros = missing in the mat file)
                    valid = np.any(fixations > 0, axis=1)
                    if valid.sum() >= min_observers:
                        self.index.append((vid, f))

            with cache_path.open("wb") as fh:
                pickle.dump((self.index, self.fix_arrays), fh)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        video_id, frame_idx = self.index[i]
        frame = load_frame(self.data_root, video_id, frame_idx, size=self.image_size)
        image = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

        fixations = _get_frame_fixations(self.fix_arrays[video_id], frame_idx)
        fixations_scaled = _rescale_fixations(
            fixations, FIND_NATIVE_W, FIND_NATIVE_H, self.image_size,
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
