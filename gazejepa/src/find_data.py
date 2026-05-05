"""FIND dataset loader — single source of truth for all data access.

The FIND dataset (Liu & Xu, Beihang 2016) contains fixation data from 39
observers watching 65 multi-face YouTube videos. Three videos (019, 022, 057)
are excluded due to quality issues, leaving 62 usable videos.

Fix data format: .mat files in fix_data/, each containing a numpy array
`fix_data` of shape (2, 70, n_frames):
  - dim 0: coordinate axis (0=x, 1=y), pixels at original resolution
  - dim 1: observer slot (up to 70; NaN-padded where no fixation exists)
  - dim 2: frame index (0-based in Python)

Original video resolution: 1280×720.
"""

import os
import cv2
import numpy as np
import scipy.io


# Frames with fewer than this many valid observer fixations are excluded
# from evaluation — heatmaps from sparse observations are too noisy.
MIN_OBSERVERS_PER_FRAME = 5

# The 62 usable video IDs (019, 022, 057 excluded)
ALL_VIDEO_IDS = [
    "001", "002", "003", "004", "005", "006", "007", "008", "009", "010",
    "011", "012", "013", "014", "015", "016", "018", "020", "021", "023",
    "024", "025", "026", "027", "028", "030", "031", "033", "034", "035",
    "036", "037", "038", "040", "042", "043", "044", "045", "047", "048",
    "049", "050", "053", "054", "056", "058", "060", "061", "062", "063",
    "064", "065", "066", "067", "068", "069", "070", "071", "072", "073",
    "074", "075",
]

# Fixed split — never change these without coordinating with collaborators
_TEST_VIDEOS  = ["070", "071", "072", "073", "074", "075"]
_VAL_VIDEOS   = ["064", "065", "066", "067", "068", "069"]
_TRAIN_VIDEOS = [v for v in ALL_VIDEO_IDS if v not in _TEST_VIDEOS and v not in _VAL_VIDEOS]


def get_video_ids(data_root: str, split: str) -> list[str]:
    """Return video IDs for the given split ('train', 'val', or 'test')."""
    if split == "train":
        return _TRAIN_VIDEOS
    elif split == "val":
        return _VAL_VIDEOS
    elif split == "test":
        return _TEST_VIDEOS
    else:
        raise ValueError(f"Unknown split '{split}'. Use 'train', 'val', or 'test'.")


def _video_path(data_root: str, video_id: str) -> str:
    base = os.path.join(data_root, "Our_database", "raw_videos", video_id)
    if os.path.exists(base + ".mp4"):
        return base + ".mp4"
    elif os.path.exists(base + ".avi"):
        return base + ".avi"
    raise FileNotFoundError(f"No video file found for video_id={video_id} in {data_root}")


def load_frame(data_root: str, video_id: str, frame_idx: int,
               size: int = None) -> np.ndarray:
    """Load a single frame from a FIND video.

    Returns an (H, W, 3) uint8 RGB array. If size is given, resizes to
    (size, size) with bilinear interpolation.
    """
    path = _video_path(data_root, video_id)
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx} from {path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if size is not None:
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
    return frame


def load_fixations(data_root: str, video_id: str) -> np.ndarray:
    """Load fixation array for one video.

    Returns shape (2, 70, n_frames): axis 0 is (x, y), axis 1 is observer
    slot (NaN-padded), axis 2 is frame index (0-based).
    """
    path = os.path.join(data_root, "Our_database", "fix_data", f"{video_id}.mat")
    mat = scipy.io.loadmat(path)
    return mat["fix_data"].astype(np.float64)


def get_frame_fixations(fix_array: np.ndarray, frame_idx: int) -> np.ndarray:
    """Extract valid (x, y) fixation coordinates for a single frame.

    Returns shape (N, 2) with N valid fixations. Returns empty array if none.
    Coordinates are in the original video pixel space.
    """
    xy = fix_array[:, :, frame_idx]  # (2, 70)
    x = xy[0]
    y = xy[1]
    valid = ~np.isnan(x) & ~np.isnan(y)
    if not valid.any():
        return np.empty((0, 2), dtype=np.float64)
    return np.stack([x[valid], y[valid]], axis=1)  # (N, 2)


def rescale_fixations(fixations: np.ndarray, source_size: tuple,
                      target_size: tuple) -> np.ndarray:
    """Rescale fixation coordinates from source to target pixel space.

    source_size and target_size are (width, height) tuples.
    fixations is (N, 2) array of (x, y) pixel coordinates.
    """
    if len(fixations) == 0:
        return fixations
    src_w, src_h = source_size
    tgt_w, tgt_h = target_size
    scaled = fixations.copy().astype(np.float64)
    scaled[:, 0] = scaled[:, 0] * tgt_w / src_w  # x
    scaled[:, 1] = scaled[:, 1] * tgt_h / src_h  # y
    return scaled


def make_heatmap(fixations: np.ndarray, image_size: int,
                 sigma_pixels: float) -> np.ndarray:
    """Build a fixation density heatmap by summing Gaussians at each fixation.

    fixations: (N, 2) array of (x, y) coordinates already in image_size space.
    Returns (image_size, image_size) float32 normalized to sum=1.
    If fixations is empty, returns a uniform map.
    """
    heatmap = np.zeros((image_size, image_size), dtype=np.float64)
    if len(fixations) == 0:
        return np.ones_like(heatmap, dtype=np.float32) / (image_size * image_size)

    ys, xs = np.mgrid[0:image_size, 0:image_size]
    for x, y in fixations:
        heatmap += np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma_pixels ** 2))

    total = heatmap.sum()
    if total > 0:
        heatmap /= total
    return heatmap.astype(np.float32)


def get_frame_count(data_root: str, video_id: str) -> int:
    """Return total number of frames in a video."""
    path = _video_path(data_root, video_id)
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def get_video_size(data_root: str, video_id: str) -> tuple[int, int]:
    """Return (width, height) of the video in pixels."""
    path = _video_path(data_root, video_id)
    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt

    data_root = sys.argv[1] if len(sys.argv) > 1 else "data/find_dataset"

    print("Loading frame 100 of video 001...")
    frame = load_frame(data_root, "001", 100, size=224)
    print(f"  Frame shape: {frame.shape}, dtype: {frame.dtype}")

    print("Loading fixations for video 001...")
    fix_array = load_fixations(data_root, "001")
    print(f"  Fix array shape: {fix_array.shape}")

    fixations = get_frame_fixations(fix_array, 100)
    print(f"  Valid fixations at frame 100: {len(fixations)}")

    src_size = get_video_size(data_root, "001")
    print(f"  Video size: {src_size}")

    fixations_scaled = rescale_fixations(fixations, src_size, (224, 224))
    heatmap = make_heatmap(fixations_scaled, 224, sigma_pixels=20)
    print(f"  Heatmap sum: {heatmap.sum():.4f} (should be ~1.0)")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(frame)
    axes[0].scatter(fixations_scaled[:, 0], fixations_scaled[:, 1],
                    c="red", s=30, alpha=0.8, label="fixations")
    axes[0].set_title("Frame + fixations")
    axes[0].legend()
    axes[1].imshow(heatmap, cmap="hot")
    axes[1].set_title("Fixation heatmap")
    plt.tight_layout()
    plt.savefig("outputs/find_data_test.png", dpi=100)
    print("  Plot saved to outputs/find_data_test.png")
