"""FIND loader. Frames from raw_videos/*.mp4 (not the dynamic/ motion-energy PNGs).

Fixations: ``(2, n_observers, n_frames)`` float64 from fix_data/*.mat, NaN-padded.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import scipy.io


EXCLUDED_VIDEOS: frozenset[str] = frozenset({"019", "022", "057"})
"""Videos excluded by the original FIND MATLAB processing script."""

MIN_OBSERVERS_PER_FRAME: int = 5
"""Eval gates frames below this — not enforced here."""

FIND_NATIVE_W: int = 1280
FIND_NATIVE_H: int = 720
"""Native resolution of FIND raw videos."""

DEFAULT_HEATMAP_SIGMA: int = 20
"""Default Gaussian sigma (pixels at native resolution) for fixation heatmaps."""


_REPO_DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "find_dataset"


def resolve_data_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the FIND data root."""
    if explicit is not None:
        root = Path(explicit)
    elif env := os.environ.get("FIND_DATA_ROOT"):
        root = Path(env)
    else:
        root = _REPO_DATA_ROOT
    if not root.is_dir():
        raise FileNotFoundError(f"FIND data root does not exist: {root}")
    return root


def _fix_dir(root: str | os.PathLike[str]) -> Path:
    return Path(root) / "Our_database" / "fix_data"


def _video_path(root: str | os.PathLike[str], vid: str) -> Path:
    return Path(root) / "Our_database" / "raw_videos" / f"{vid}.mp4"


def get_video_ids(data_root: str | os.PathLike[str]) -> list[str]:
    """Sorted video IDs — fix_data/*.mat present and not in EXCLUDED_VIDEOS."""
    fix_dir = _fix_dir(data_root)
    if not fix_dir.is_dir():
        raise FileNotFoundError(f"Missing fix_data directory: {fix_dir}")
    ids = sorted(p.stem for p in fix_dir.glob("*.mat"))
    return [v for v in ids if v not in EXCLUDED_VIDEOS]


def get_split(
    data_root: str | os.PathLike[str], seed: int = 42,
) -> dict[str, list[str]]:
    """Reproducible 50/6/6 split."""
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


def load_fixations(
    data_root: str | os.PathLike[str], video_id: str,
) -> np.ndarray:
    """``(2, n_observers, n_frames)`` float64: row 0 = x, row 1 = y, NaN if unfixated."""
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
    """``(n_valid, 2)`` (x, y) for one frame in native FIND pixels (NaN observers dropped)."""
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


def get_frame_count(
    data_root: str | os.PathLike[str], video_id: str,
) -> int:
    """``min(mp4_frame_count, fix_n_frames)`` — they sometimes differ at the tail."""
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
    """One RGB frame from raw_videos/{vid}.mp4 as ``(H, W, 3)`` uint8 (native 720×1280).

    If ``size`` is given, square-resampled with INTER_AREA.
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


def make_heatmap(
    fixations_xy: np.ndarray,
    img_h: int,
    img_w: int,
    sigma: float = DEFAULT_HEATMAP_SIGMA,
) -> np.ndarray:
    """Gaussian-smoothed fixation density, max-normalised to 1 (Bylinskii et al. 2019)."""
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
    """Linearly rescale ``(x, y)`` from ``src_size`` to ``dst_size`` — both ``(W, H)`` or int."""
    sw, sh = (src_size, src_size) if isinstance(src_size, int) else src_size
    dw, dh = (dst_size, dst_size) if isinstance(dst_size, int) else dst_size
    arr = np.asarray(fix_xy, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected (N, 2) array, got shape {arr.shape}")
    out = arr.copy()
    out[:, 0] = out[:, 0] * (dw / sw)
    out[:, 1] = out[:, 1] * (dh / sh)
    return out


def get_human_scanpath(
    data_root: str | os.PathLike[str],
    video_id: str,
    observer_idx: int,
    start_frame: int,
    length: int,
) -> np.ndarray | None:
    """One observer's fixations across ``[start_frame, start_frame + length)``, native pixels.

    Returns None on any NaN or past-end. Caveat: the scene moves while gaze
    evolves, so this isn't directly comparable to the loop's static-frame
    scanpath — the report flags this.
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


def _smoke_test() -> None:
    """Run as ``python -m gazejepa.data.find_dataset`` with FIND_DATA_ROOT set."""
    # lazy: keeps module import light
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

    rescaled = rescale_fixations(fixs, (FIND_NATIVE_W, FIND_NATIVE_H), (224, 224))
    print(f"[fix]    rescaled to 224: x=[{rescaled[:, 0].min():.1f}, "
          f"{rescaled[:, 0].max():.1f}] y=[{rescaled[:, 1].min():.1f}, "
          f"{rescaled[:, 1].max():.1f}]")
    assert (rescaled >= 0).all() and (rescaled <= 224 + 1).all()

    sp = get_human_scanpath(
        data_root, vid, observer_idx=5, start_frame=100, length=5,
    )
    if sp is None:
        print("[scan]   observer 5 has NaN in window 100-104; trying observer 0")
        sp = get_human_scanpath(data_root, vid, 0, 100, 5)
    assert sp is None or sp.shape == (5, 2), f"Unexpected scanpath shape: {sp.shape if sp is not None else None}"
    print(f"[scan]   length-5 scanpath shape: "
          f"{None if sp is None else sp.shape}")

    # Center-bias AUC ≈ 0.65–0.80; gate on median since per-frame swings widely.
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
          f"(typical: 0.65–0.80)")
    # Wider gate: must beat random clearly but not be near-perfect.
    in_range = 0.55 <= median_auc <= 0.95

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
