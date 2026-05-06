"""Learned saliency — small CNN trained on FIND fixation heatmaps.

Architecture: 4 conv layers with GroupNorm + GELU, 1× stride-2 downsampling,
output upsampled back to input resolution and softmax-normalized.
~50K parameters. Intentionally small: the goal is data-driven saliency from
the FIND training split, not a general-purpose saliency model.

Also contains FINDSaliencyDataset, the PyTorch Dataset for training.
"""

import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from . import SaliencySource


# ImageNet normalization used for all model inputs
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class LearnedSaliency(nn.Module, SaliencySource):
    """Small 4-layer CNN saliency predictor.

    Input:  (B, 3, H, W) ImageNet-normalized tensor
    Output: (B, H, W) softmax-normalized saliency map (sums to 1 per sample)
    """

    name = "learned"

    def __init__(self, in_channels: int = 3, hidden: int = 32):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden * 2, 3, padding=1, stride=2),
            nn.GroupNorm(8, hidden * 2), nn.GELU(),
            nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1),
            nn.GroupNorm(8, hidden * 2), nn.GELU(),
            nn.Conv2d(hidden * 2, 1, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Returns logits at half resolution, shape (B, H/2, W/2)."""
        return self.net(images).squeeze(1)

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.forward(images)
        logits = F.interpolate(logits.unsqueeze(1), size=images.shape[-2:],
                               mode="bilinear", align_corners=False).squeeze(1)
        B = logits.shape[0]
        flat = logits.view(B, -1).softmax(dim=1)
        return flat.view(B, *logits.shape[-2:])

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "LearnedSaliency":
        model = cls()
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model


class FINDSaliencyDataset(Dataset):
    """PyTorch Dataset yielding (image_tensor, heatmap_tensor) pairs from FIND.

    Splits by video ID to prevent temporal data leakage. The index is cached
    to disk after first build so repeated instantiation is fast.

    Args:
        data_root: path to the find_dataset/ directory
        split: 'train', 'val', or 'test'
        image_size: resize frames to this square size (default 224)
        sigma: Gaussian sigma for fixation heatmap in pixels (default 20)
        frame_stride: use every Nth frame (default 10)
        min_observers: skip frames with fewer valid fixations (default 5)
        cache_dir: where to store the index pickle (default 'outputs/')
    """

    def __init__(self, data_root: str, split: str, image_size: int = 224,
                 sigma: float = 20.0, frame_stride: int = 10,
                 min_observers: int = 5, cache_dir: str = "outputs"):
        from src.find_data import (get_video_ids, load_fixations, get_frame_count,
                                    get_frame_fixations, get_video_size)

        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.sigma = sigma

        cache_path = os.path.join(cache_dir, f"find_index_{split}_s{frame_stride}_m{min_observers}.pkl")
        os.makedirs(cache_dir, exist_ok=True)

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.index, self.fix_arrays, self.video_sizes = pickle.load(f)
        else:
            print(f"Building index for split={split}...")
            video_ids = get_video_ids(data_root, split)
            self.fix_arrays = {}
            self.video_sizes = {}
            self.index = []

            for vid in video_ids:
                fix_array = load_fixations(data_root, vid)
                vid_size = get_video_size(data_root, vid)
                n_frames = get_frame_count(data_root, vid)
                self.fix_arrays[vid] = fix_array
                self.video_sizes[vid] = vid_size

                for f in range(0, n_frames, frame_stride):
                    fixations = get_frame_fixations(fix_array, f)
                    if len(fixations) >= min_observers:
                        self.index.append((vid, f))

            with open(cache_path, "wb") as f:
                pickle.dump((self.index, self.fix_arrays, self.video_sizes), f)
            print(f"  Index built: {len(self.index)} frames. Cached to {cache_path}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        from src.find_data import (load_frame, get_frame_fixations,
                                    rescale_fixations, make_heatmap)

        video_id, frame_idx = self.index[i]
        src_size = self.video_sizes[video_id]

        image = load_frame(self.data_root, video_id, frame_idx, size=self.image_size)
        fixations = get_frame_fixations(self.fix_arrays[video_id], frame_idx)
        fixations_scaled = rescale_fixations(fixations, src_size,
                                             (self.image_size, self.image_size))
        heatmap = make_heatmap(fixations_scaled, self.image_size, self.sigma)

        image_tensor = IMAGE_TRANSFORM(image)
        heatmap_tensor = torch.from_numpy(heatmap)
        return image_tensor, heatmap_tensor


if __name__ == "__main__":
    model = LearnedSaliency()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"LearnedSaliency params: {n_params:,}")

    x = torch.rand(2, 3, 224, 224)
    out = model.predict(x)
    print(f"Output shape: {out.shape}")
    print(f"Sum per sample: {out.sum(dim=(1,2))}")
    assert out.shape == (2, 224, 224)
    assert torch.allclose(out.sum(dim=(1, 2)), torch.ones(2), atol=1e-5)
    print("LearnedSaliency OK")
