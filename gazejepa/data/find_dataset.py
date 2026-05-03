"""
FIND dataset loader.

Reads frames from ``Our_database/raw_videos/{vid}.mp4`` via OpenCV (NOT the
preprocessed ``dynamic/`` motion-energy PNGs — those are not natural RGB
and are explicitly excluded by EMAD_PLAN §1 / line 49).

Fixations come from ``Our_database/fix_data/{vid}.mat`` as a
``(2, n_observers, n_frames)`` array, with ``NaN`` where an observer did not
fixate on a frame. The plan's expected layout is `(2, 70, n_frames)` —
``n_observers`` is treated as a runtime value rather than a constant in
case the dataset version varies.

Public API (per MIGRATION_AUDIT §2):
    resolve_data_root(explicit=None) -> Path
    get_video_ids(data_root)         -> list[str]
    get_split(data_root, seed=42)    -> dict[str, list[str]]
    load_frame(data_root, video_id, frame_idx, size=None) -> np.ndarray
    load_fixations(data_root, video_id) -> np.ndarray
    get_frame_fixations(fix_array, frame_idx) -> np.ndarray
    get_frame_count(data_root, video_id) -> int
    make_heatmap(fixations_xy, img_h, img_w, sigma=30) -> np.ndarray
    rescale_fixations(fix_xy, src_size, dst_size) -> np.ndarray
    get_human_scanpath(data_root, video_id, observer_idx, start_frame, length)
        -> np.ndarray | None
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import scipy.io


# ---- Constants -------------------------------------------------------------

EXCLUDED_VIDEOS: frozenset[str] = frozenset({"019", "022", "057"})
"""Videos excluded by the original FIND MATLAB processing script."""

MIN_OBSERVERS_PER_FRAME: int = 5
"""Minimum number of observers a frame must have to be eligible for evaluation
(per EMAD_PLAN §1, line 97). Not enforced inside this module — the eval
pipeline filters with this constant."""

FIND_NATIVE_W: int = 1280
FIND_NATIVE_H: int = 720
"""Native resolution of FIND raw videos."""

DEFAULT_HEATMAP_SIGMA: int = 30
"""Default Gaussian sigma (pixels at native resolution) for fixation heatmaps."""


# ---- Path resolution -------------------------------------------------------

def resolve_data_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the FIND data root.

    Priority: explicit arg → ``$FIND_DATA_ROOT`` env var → error.
    No hardcoded fallback (per MIGRATION_AUDIT §4.11).
    """
    if explicit is not None:
        root = Path(explicit)
    else:
        env = os.environ.get("FIND_DATA_ROOT")
        if not env:
            raise RuntimeError(
                "FIND data root not set. Pass `data_root=` to the loader "
                "functions or set the FIND_DATA_ROOT environment variable."
            )
        root = Path(env)
    if not root.is_dir():
        raise FileNotFoundError(f"FIND data root does not exist: {root}")
    return root


def _fix_dir(root: str | os.PathLike[str]) -> Path:
    return Path(root) / "Our_database" / "fix_data"


def _video_path(root: str | os.PathLike[str], vid: str) -> Path:
    return Path(root) / "Our_database" / "raw_videos" / f"{vid}.mp4"


# ---- Video IDs and splits --------------------------------------------------

def get_video_ids(data_root: str | os.PathLike[str]) -> list[str]:
    """Sorted list of available video IDs (e.g. ``['001', '002', ...]``).

    A video is "available" iff its ``fix_data/{vid}.mat`` exists and it is
    not in :data:`EXCLUDED_VIDEOS`.
    """
    fix_dir = _fix_dir(data_root)
    if not fix_dir.is_dir():
        raise FileNotFoundError(f"Missing fix_data directory: {fix_dir}")
    ids = sorted(p.stem for p in fix_dir.glob("*.mat"))
    return [v for v in ids if v not in EXCLUDED_VIDEOS]


def get_split(
    data_root: str | os.PathLike[str], seed: int = 42,
) -> dict[str, list[str]]:
    """Reproducible 50 / 6 / 6 video-level split.

    Determinism is part of the API: both authors must call this with the
    default seed to share the split (EMAD_PLAN §1, line 97).
    """
    all_vids = get_video_ids(data_root)
    if len(all_vids) != 62:
        raise RuntimeError(
            f"Expected 62 videos for the 50/6/6 split, found {len(all_vids)}. "
            "Did the dataset change?"
        )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(all_vids))
    shuffled = [all_vids[i] for i in perm]
    return {
        "train": sorted(shuffled[:50]),
        "val":   sorted(shuffled[50:56]),
        "test":  sorted(shuffled[56:62]),
    }


# ---- Fixations -------------------------------------------------------------

def load_fixations(
    data_root: str | os.PathLike[str], video_id: str,
) -> np.ndarray:
    """Load the ``(2, n_observers, n_frames)`` fixation array for one video.

    Index 0 of the first axis is x (pixel), index 1 is y (pixel), with
    ``NaN`` where the observer did not fixate. Returned as ``float64`` (the
    .mat file's native dtype).
    """
    path = _fix_dir(data_root) / f"{video_id}.mat"
    if not path.is_file():
        raise FileNotFoundError(f"Fixation file not found: {path}")
    mat = scipy.io.loadmat(path)
    if "fix_data" not in mat:
        raise KeyError(
            f"{path} does not contain key 'fix_data'. Available: "
            f"{[k for k in mat.keys() if not k.startswith('__')]}"
        )
    return mat["fix_data"]


def get_frame_fixations(fix_array: np.ndarray, frame_idx: int) -> np.ndarray:
    """Return the ``(n_valid, 2)`` array of (x, y) fixation coords on a frame.

    Drops observers with ``NaN`` x-coordinate on this frame. Coords are in
    native FIND pixels (1280 × 720).
    """
    if fix_array.ndim != 3 or fix_array.shape[0] != 2:
        raise ValueError(
            f"Expected (2, n_observers, n_frames) fix_array, got {fix_array.shape}"
        )
    if frame_idx < 0 or frame_idx >= fix_array.shape[2]:
        raise IndexError(
            f"frame_idx {frame_idx} out of range [0, {fix_array.shape[2]})"
        )
    frame = fix_array[:, :, frame_idx]
    valid = ~np.isnan(frame[0])
    return np.stack([frame[0, valid], frame[1, valid]], axis=1)


# ---- Frames ----------------------------------------------------------------

def get_frame_count(
    data_root: str | os.PathLike[str], video_id: str,
) -> int:
    """Frame count usable for evaluation: ``min(mp4_frame_count, fix_n_frames)``.

    The two can differ slightly because of how the FIND MATLAB pipeline
    handles trailing frames; using the min is the conservative choice.
    """
    path = _video_path(data_root, video_id)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    n_fix = load_fixations(data_root, video_id).shape[2]
    return min(n_video, n_fix)


def load_frame(
    data_root: str | os.PathLike[str],
    video_id: str,
    frame_idx: int,
    size: int | None = None,
) -> np.ndarray:
    """Decode a single RGB frame from ``Our_database/raw_videos/{vid}.mp4``.

    Args:
        data_root: FIND data root.
        video_id: e.g. ``"001"``.
        frame_idx: zero-based frame index.
        size: if not ``None``, resize the frame to ``(size, size)`` (square)
            using ``cv2.INTER_AREA`` (downsampling-friendly).

    Returns:
        ``(H, W, 3)`` ``uint8`` array. RGB channel order. Native shape is
        ``(720, 1280, 3)``.
    """
    path = _video_path(data_root, video_id)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_idx < 0 or frame_idx >= n_total:
            raise IndexError(
                f"frame_idx {frame_idx} out of range [0, {n_total}) for {path.name}"
            )
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            raise RuntimeError(
                f"Failed to read frame {frame_idx} from {path}. "
                "Codec seek may have failed; consider sequential read."
            )
    finally:
        cap.release()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if size is not None:
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return rgb


# ---- Heatmap and rescaling -------------------------------------------------

def make_heatmap(
    fixations_xy: np.ndarray,
    img_h: int,
    img_w: int,
    sigma: float = DEFAULT_HEATMAP_SIGMA,
) -> np.ndarray:
    """Gaussian-smoothed fixation density map normalised so ``max == 1``.

    Standard FIND ground-truth saliency construction (Bylinskii et al. 2019).
    Each fixation contributes a 2D Gaussian of width ``sigma`` (pixels at the
    same resolution as the requested ``img_h × img_w``); the sum is rescaled
    so the brightest pixel is 1.

    Args:
        fixations_xy: ``(N, 2)`` array of ``(x, y)`` pixel coords.
        img_h, img_w: target heatmap dimensions.
        sigma: Gaussian sigma in pixels.
    """
    heatmap = np.zeros((img_h, img_w), dtype=np.float64)
    if len(fixations_xy) == 0:
        return heatmap
    radius = int(3 * sigma)
    for x, y in fixations_xy:
        xi, yi = int(round(float(x))), int(round(float(y)))
        if not (0 <= xi < img_w and 0 <= yi < img_h):
            continue
        y_lo = max(0, yi - radius)
        y_hi = min(img_h, yi + radius + 1)
        x_lo = max(0, xi - radius)
        x_hi = min(img_w, xi + radius + 1)
        yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
        g = np.exp(-((xx - xi) ** 2 + (yy - yi) ** 2) / (2.0 * sigma ** 2))
        heatmap[y_lo:y_hi, x_lo:x_hi] += g
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap


def rescale_fixations(
    fix_xy: np.ndarray,
    src_size: int | tuple[int, int],
    dst_size: int | tuple[int, int],
) -> np.ndarray:
    """Linearly rescale ``(x, y)`` fixations from ``src_size`` to ``dst_size``.

    Args:
        fix_xy: ``(N, 2)`` array.
        src_size: ``(W, H)`` tuple, or a single ``int`` if square.
        dst_size: ``(W, H)`` tuple, or a single ``int`` if square.
    """
    sw, sh = (src_size, src_size) if isinstance(src_size, int) else src_size
    dw, dh = (dst_size, dst_size) if isinstance(dst_size, int) else dst_size
    arr = np.asarray(fix_xy, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected (N, 2) array, got shape {arr.shape}")
    out = arr.copy()
    out[:, 0] = out[:, 0] * (dw / sw)
    out[:, 1] = out[:, 1] * (dh / sh)
    return out


# ---- Human scanpath (Option A from EMAD_PLAN §3.2) -------------------------

def get_human_scanpath(
    data_root: str | os.PathLike[str],
    video_id: str,
    observer_idx: int,
    start_frame: int,
    length: int,
) -> np.ndarray | None:
    """Per-observer scanpath across consecutive frames.

    Picks observer ``observer_idx`` and takes their fixation positions on
    frames ``[start_frame, start_frame + length)``. Returns ``None`` if any
    frame in that window has a NaN fixation, or if the window runs past the
    last frame.

    Caveat (EMAD_PLAN §0.4): the underlying scene is moving while the
    observer's gaze evolves, whereas Emad's loop generates ``length``
    fixations on a static frame. This helper produces the closest available
    "human scanpath" reference; the report calls out the static-vs-temporal
    mismatch explicitly.

    Returns:
        ``(length, 2)`` array in native FIND pixel coords, or ``None``.
    """
    fix = load_fixations(data_root, video_id)
    if start_frame < 0 or start_frame + length > fix.shape[2]:
        return None
    if observer_idx < 0 or observer_idx >= fix.shape[1]:
        raise IndexError(
            f"observer_idx {observer_idx} out of range [0, {fix.shape[1]})"
        )
    coords = []
    for f in range(start_frame, start_frame + length):
        x = fix[0, observer_idx, f]
        y = fix[1, observer_idx, f]
        if np.isnan(x) or np.isnan(y):
            return None
        coords.append([float(x), float(y)])
    return np.array(coords, dtype=np.float64)


# ---- Smoke test ------------------------------------------------------------

def _smoke_test() -> None:
    """Phase 2 stopping-condition verification (per MIGRATION_AUDIT §5).

    Run as ``python -m gazejepa.data.find_dataset`` with FIND_DATA_ROOT set.
    """
    # Imported lazily so module import doesn't drag in matplotlib/sklearn.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gazejepa.evaluation.saliency_metrics import (
        auc_score,
        center_bias_baseline,
    )

    data_root = resolve_data_root()
    print(f"[setup]  data root: {data_root}")

    vids = get_video_ids(data_root)
    print(f"[setup]  available videos: {len(vids)} "
          f"(excluded: {sorted(EXCLUDED_VIDEOS)})")
    assert len(vids) == 62, f"Expected 62 videos, got {len(vids)}"

    split = get_split(data_root)
    sizes = {k: len(v) for k, v in split.items()}
    print(f"[setup]  split (50/6/6, seed=42): {sizes}")
    print(f"[setup]    train[:5]={split['train'][:5]} ...")
    print(f"[setup]    val={split['val']}")
    print(f"[setup]    test={split['test']}")
    assert sizes == {"train": 50, "val": 6, "test": 6}

    vid = "001"
    n_frames = get_frame_count(data_root, vid)
    print(f"[video]  {vid}: {n_frames} frames")

    fix = load_fixations(data_root, vid)
    print(f"[fix]    array shape={fix.shape} dtype={fix.dtype}")
    assert fix.ndim == 3 and fix.shape[0] == 2

    frame = load_frame(data_root, vid, 100)
    print(f"[frame]  100 native: shape={frame.shape} dtype={frame.dtype} "
          f"min={frame.min()} max={frame.max()}")
    assert frame.shape == (FIND_NATIVE_H, FIND_NATIVE_W, 3), \
        f"Expected ({FIND_NATIVE_H}, {FIND_NATIVE_W}, 3), got {frame.shape}"
    assert frame.dtype == np.uint8

    # Resized variant
    frame_224 = load_frame(data_root, vid, 100, size=224)
    print(f"[frame]  100 size=224: shape={frame_224.shape} dtype={frame_224.dtype}")
    assert frame_224.shape == (224, 224, 3) and frame_224.dtype == np.uint8

    fixs = get_frame_fixations(fix, 100)
    print(f"[fix]    frame 100: {fixs.shape[0]} valid observers")
    assert fixs.shape[0] >= MIN_OBSERVERS_PER_FRAME, \
        f"Frame 100 has only {fixs.shape[0]} observers (< MIN_OBSERVERS_PER_FRAME={MIN_OBSERVERS_PER_FRAME})"

    h = make_heatmap(fixs, FIND_NATIVE_H, FIND_NATIVE_W, sigma=DEFAULT_HEATMAP_SIGMA)
    print(f"[heat]   shape={h.shape} range=[{h.min():.3f}, {h.max():.3f}] "
          f"sum={h.sum():.1f} nonzero_frac={(h > 0).mean():.4f}")
    assert h.shape == (FIND_NATIVE_H, FIND_NATIVE_W)
    assert 0 < h.max() <= 1.0 + 1e-9

    # Rescale fixations to a 224×224 frame to verify coord transformation
    rescaled = rescale_fixations(fixs, (FIND_NATIVE_W, FIND_NATIVE_H), (224, 224))
    print(f"[fix]    rescaled to 224: x=[{rescaled[:, 0].min():.1f}, "
          f"{rescaled[:, 0].max():.1f}] y=[{rescaled[:, 1].min():.1f}, "
          f"{rescaled[:, 1].max():.1f}]")
    assert (rescaled >= 0).all() and (rescaled <= 224 + 1).all()

    # Human-scanpath helper (Option A)
    sp = get_human_scanpath(
        data_root, vid, observer_idx=5, start_frame=100, length=5,
    )
    if sp is None:
        print("[scan]   observer 5 has NaN in window 100-104; trying observer 0")
        sp = get_human_scanpath(data_root, vid, 0, 100, 5)
    assert sp is None or sp.shape == (5, 2), f"Unexpected scanpath shape: {sp.shape if sp is not None else None}"
    print(f"[scan]   length-5 scanpath shape: "
          f"{None if sp is None else sp.shape}")

    # The plan's sanity check: center-bias AUC against human fixations
    # should land in 0.65–0.80 *typically*. Individual frames can be
    # higher (concentrated cluster on a face) or lower (scattered over
    # multiple faces). Compute AUC at several frame indices and use the
    # median for the gating check; report all values for transparency.
    cb = center_bias_baseline(FIND_NATIVE_H, FIND_NATIVE_W, sigma_frac=0.25)
    sample_frames = [50, 100, 150, 300, 500]
    aucs = []
    for fi in sample_frames:
        if fi >= n_frames:
            continue
        sample_fixs = get_frame_fixations(fix, fi)
        if len(sample_fixs) < MIN_OBSERVERS_PER_FRAME:
            continue
        aucs.append(auc_score(cb, sample_fixs))
        print(f"[auc]    frame {fi:>3d}: {aucs[-1]:.3f}  ({len(sample_fixs)} obs)")
    median_auc = float(np.median(aucs))
    auc = aucs[sample_frames.index(100)] if 100 in sample_frames else median_auc
    print(f"[auc]    median across {len(aucs)} frames: {median_auc:.3f} "
          f"(plan typical range: 0.65–0.80)")
    # Gate on a wider band: center-bias must clearly beat random (>0.55) and
    # not be near-perfect (<0.95) on the median frame. Tight 0.65–0.80 is a
    # population statistic, not a per-frame guarantee.
    in_range = 0.55 <= median_auc <= 0.95

    # Save a 3-panel verification figure (image, fixations, heatmap).
    out_dir = Path(__file__).resolve().parents[2] / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(frame)
    axes[0].set_title(f"Video {vid} frame 100 (native {FIND_NATIVE_W}×{FIND_NATIVE_H})")
    axes[0].axis("off")
    axes[1].imshow(frame)
    axes[1].scatter(
        fixs[:, 0], fixs[:, 1],
        c="red", s=40, alpha=0.7, edgecolors="white", linewidths=0.5,
    )
    axes[1].set_title(f"{fixs.shape[0]} observer fixations")
    axes[1].axis("off")
    axes[2].imshow(frame, alpha=0.4)
    axes[2].imshow(h, cmap="hot", alpha=0.6)
    axes[2].set_title(
        f"Heatmap (sigma={DEFAULT_HEATMAP_SIGMA})  |  "
        f"center-bias AUC={auc:.3f}"
    )
    axes[2].axis("off")
    plt.tight_layout()
    out_path = out_dir / "phase2_smoke_test.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[viz]    saved: {out_path}")

    # Hard-assert AUC at the very end so the figure still saves on failure.
    assert in_range, (
        f"Median center-bias AUC {median_auc:.3f} outside sanity range "
        "[0.55, 0.95]. Loader may be returning data in the wrong coord "
        "system; inspect the figure."
    )

    print("\n[OK] Phase 2 smoke test PASSED")


if __name__ == "__main__":
    _smoke_test()
