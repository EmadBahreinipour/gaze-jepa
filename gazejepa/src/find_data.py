"""FIND dataset loader -- single source of truth for all data access.

The FIND dataset (Liu & Xu, Beihang 2016) contains fixation data from 39
observers watching 65 multi-face YouTube videos.

Fix data format: .mat files in fix_data_NEW/, each containing an array
`curr_v_all_s` of shape (39, 1) where arr[observer][0] is (n_frames, 2)
with (x, y) pixel coordinates at original resolution (1280x720).
"""

import os
import cv2
import numpy as np
import scipy.io


# Frames with fewer than this many valid observer fixations are excluded
# from evaluation -- heatmaps from sparse observations are too noisy.
MIN_OBSERVERS_PER_FRAME = 5

# All 65 video IDs
ALL_VIDEO_IDS = [
    "001", "002", "003", "004", "005", "006", "007", "008", "009", "010",
    "011", "012", "013", "014", "015", "016", "018", "019", "020", "021",
    "022", "023", "024", "025", "026", "027", "028", "030", "031", "033",
    "034", "035", "036", "037", "038", "040", "042", "043", "044", "045",
    "047", "048", "049", "050", "053", "054", "056", "057", "058", "060",
    "061", "062", "063", "064", "065", "066", "067", "068", "069", "070",
    "071", "072", "073", "074", "075",
]

# Fixed split
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
    """Load fixation array for one video from fix_data_NEW.

    Returns shape (39, n_frames, 2): 39 observers, n_frames frames,
    (x, y) pixel coordinates at original resolution.
    """
    path = os.path.join(data_root, "Our_database", "fix_data_NEW", f"{video_id}.mat")
    mat = scipy.io.loadmat(path)
    arr = mat["curr_v_all_s"]  # (39, 1) object array
    # arr[obs][0] is (n_frames, 2) uint16
    n_observers = arr.shape[0]
    n_frames = arr[0][0].shape[0]
    out = np.zeros((n_observers, n_frames, 2), dtype=np.float64)
    for obs in range(n_observers):
        out[obs] = arr[obs][0].astype(np.float64)
    return out  # (39, n_frames, 2)


def get_frame_fixations(fix_array: np.ndarray, frame_idx: int) -> np.ndarray:
    """Extract (x, y) fixation coordinates for a single frame.

    fix_array: (39, n_frames, 2) from load_fixations.
    Returns (39, 2) array of (x, y) coordinates.
    """
    return fix_array[:, frame_idx, :]  # (39, 2)


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

    data_root = sys.argv[1] if len(sys.argv) > 1 else "f:/university/milan/NI/info/Gaze_Jepa/find_dataset"

    print("Loading frame 100 of video 001...")
    frame = load_frame(data_root, "001", 100, size=224)
    print(f"  Frame shape: {frame.shape}, dtype: {frame.dtype}")

    print("Loading fixations for video 001...")
    fix_array = load_fixations(data_root, "001")
    print(f"  Fix array shape: {fix_array.shape}")

    fixations = get_frame_fixations(fix_array, 100)
    print(f"  Fixations at frame 100: {len(fixations)} observers")

    src_size = get_video_size(data_root, "001")
    print(f"  Video size: {src_size}")

    fixations_scaled = rescale_fixations(fixations, src_size, (224, 224))
    heatmap = make_heatmap(fixations_scaled, 224, sigma_pixels=20)
    print(f"  Heatmap sum: {heatmap.sum():.4f} (should be ~1.0)")

    # Also verify one of the previously missing videos loads correctly
    print("Loading fixations for video 019 (previously missing)...")
    fix_019 = load_fixations(data_root, "019")
    print(f"  Fix array shape: {fix_019.shape}")
    print("All OK.")
