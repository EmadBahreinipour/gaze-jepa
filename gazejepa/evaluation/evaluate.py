"""Gaze loop vs human FIND scanpaths."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from gazejepa.data.find_dataset import (
    FIND_NATIVE_H,
    FIND_NATIVE_W,
    get_human_scanpath,
    load_fixations,
    load_frame,
    rescale_fixations,
)
from gazejepa.evaluation.scanpath_metrics import (
    frechet_distance,
    multimatch,
)
from gazejepa.gaze_loop.loop import GazeLoop


def evaluate_gaze_loop(
    *,
    data_root: str,
    saliency_source: Any,
    saliency_name: str,
    video_ids: Sequence[str],
    frame_indices: Sequence[int] = (50, 150, 300, 500),
    n_observers: int = 5,
    n_fixations: int = 5,
    image_size: int = 224,
    ior_sigma: float = 20.0,
    ior_decay: float = 0.7,
    sampling_mode: str = "stochastic",
    seed: int = 42,
) -> list[dict[str, Any]]:
    """One row per (video, frame, observer) with MultiMatch + Fréchet scores.

    Rows sharing (video_id, start_frame) reuse the same generated_scanpath.
    Seed is reset each call for reproducibility across script/notebook runs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    rows: list[dict[str, Any]] = []
    eval_image_size = (image_size, image_size)

    for video_id in video_ids:
        try:
            fix = load_fixations(data_root, video_id)
        except FileNotFoundError:
            continue
        n_frames_video = fix.shape[2]
        n_observers_total = fix.shape[1]

        for start_frame in frame_indices:
            start_frame = int(start_frame)
            if start_frame + n_fixations > n_frames_video:
                continue

            frame_np = load_frame(data_root, video_id, start_frame, size=image_size)
            image_t = torch.from_numpy(frame_np).permute(2, 0, 1).float() / 255.0

            loop = GazeLoop(
                saliency_source=saliency_source,
                n_fixations=n_fixations,
                image_size=eval_image_size,
                ior_sigma=ior_sigma,
                ior_decay=ior_decay,
                sampling_mode=sampling_mode,
            )
            with torch.no_grad():
                result = loop(image_t)
            gen_scanpath = result["scanpath"].cpu().numpy().astype(np.float64)

            valid: list[tuple[int, np.ndarray]] = []
            for obs_idx in range(n_observers_total):
                hp = get_human_scanpath(
                    data_root, video_id, obs_idx, start_frame, n_fixations,
                )
                if hp is None:
                    continue
                hp_resized = rescale_fixations(
                    hp, (FIND_NATIVE_W, FIND_NATIVE_H), eval_image_size,
                )
                valid.append((obs_idx, hp_resized.astype(np.float64)))
                if len(valid) >= n_observers:
                    break

            if not valid:
                continue

            for obs_idx, human_sp in valid:
                mm_scores = multimatch(gen_scanpath, human_sp, img_size=eval_image_size)
                fd = frechet_distance(gen_scanpath, human_sp, img_size=eval_image_size)
                rows.append({
                    "saliency_source":      saliency_name,
                    "video_id":             video_id,
                    "start_frame":          start_frame,
                    "observer_idx":         int(obs_idx),
                    "n_observers_used":     len(valid),
                    "multimatch_position":  float(mm_scores.get("position", 0.0)),
                    "multimatch_shape":     float(mm_scores.get("shape",    0.0)),
                    "multimatch_direction": float(mm_scores.get("direction", 0.0)),
                    "multimatch_length":    float(mm_scores.get("length",   0.0)),
                    "frechet":              float(fd["frechet"]),
                    "frechet_normalized":   float(fd["normalized"]),
                    "generated_scanpath":   gen_scanpath.copy(),
                    "human_scanpath":       human_sp.copy(),
                })

    return rows


def collect_generated_amplitudes(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    """Generated saccade amplitudes, deduped per (video, frame) for the KS test."""
    seen: set[tuple[str, int]] = set()
    out: list[np.ndarray] = []
    for r in rows:
        key = (r["video_id"], int(r["start_frame"]))
        if key in seen:
            continue
        seen.add(key)
        sp = np.asarray(r["generated_scanpath"])
        if len(sp) < 2:
            continue
        out.append(np.linalg.norm(np.diff(sp, axis=0), axis=1))
    return np.concatenate(out) if out else np.array([], dtype=np.float64)


def collect_paired_human_amplitudes(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    """Per-pair human amplitudes — dominated by drift; use :func:`collect_video_saccade_amplitudes` for the KS test."""
    out: list[np.ndarray] = []
    for r in rows:
        sp = np.asarray(r["human_scanpath"])
        if len(sp) < 2:
            continue
        out.append(np.linalg.norm(np.diff(sp, axis=0), axis=1))
    return np.concatenate(out) if out else np.array([], dtype=np.float64)


collect_human_amplitudes = collect_paired_human_amplitudes


def collect_video_saccade_amplitudes(
    data_root: str,
    video_ids: Sequence[str],
    image_size: int = 224,
    *,
    max_frame_gap: int = 5,
    min_amplitude_px: float = 5.0,
) -> np.ndarray:
    """Cross-observer human saccade amplitudes, rescaled to ``image_size``.

    max_frame_gap=5 ≈ 167 ms at 30 fps; larger risks crossing saccades.
    """
    scale_x = image_size / FIND_NATIVE_W
    scale_y = image_size / FIND_NATIVE_H

    amps: list[float] = []
    for video_id in video_ids:
        try:
            fix = load_fixations(data_root, video_id)
        except FileNotFoundError:
            continue
        n_obs = fix.shape[1]
        for obs in range(n_obs):
            xs = fix[0, obs, :]
            ys = fix[1, obs, :]
            valid = ~np.isnan(xs)
            valid_idx = np.where(valid)[0]
            for i in range(1, len(valid_idx)):
                gap = valid_idx[i] - valid_idx[i - 1]
                if gap > max_frame_gap:
                    continue
                dx = (xs[valid_idx[i]] - xs[valid_idx[i - 1]]) * scale_x
                dy = (ys[valid_idx[i]] - ys[valid_idx[i - 1]]) * scale_y
                amp = float(np.sqrt(dx * dx + dy * dy))
                if amp >= min_amplitude_px:
                    amps.append(amp)
    return np.asarray(amps, dtype=np.float64)
