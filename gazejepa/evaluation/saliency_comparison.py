"""Saliency comparison on a FIND split.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm

from gazejepa.data import (
    FIND_NATIVE_H,
    FIND_NATIVE_W,
    MIN_OBSERVERS_PER_FRAME,
    get_frame_count,
    get_frame_fixations,
    get_split,
    load_fixations,
    load_frame,
    make_heatmap,
    rescale_fixations,
)
from gazejepa.evaluation.saliency_metrics import auc_score, cc_score, nss_score
from gazejepa.saliency.base import SaliencySource


def evaluate_source(
    source: SaliencySource,
    data_root: str | os.PathLike[str],
    *,
    split: str = "test",
    image_size: int = 224,
    sigma: float = 20.0,
    frame_stride: int = 10,
    min_observers: int = MIN_OBSERVERS_PER_FRAME,
    seed: int = 42,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """AUC / NSS / CC for one saliency source on one FIND split."""
    device = torch.device(device)
    video_ids = get_split(data_root, seed=seed)[split]

    aucs: list[float] = []
    nsses: list[float] = []
    ccs: list[float] = []

    for vid in tqdm(video_ids, desc=f"Evaluating {source.name}"):
        fix_array = load_fixations(data_root, vid)
        n_frames = get_frame_count(data_root, vid)

        for f in range(0, n_frames, frame_stride):
            fixations = get_frame_fixations(fix_array, f)
            if len(fixations) < min_observers:
                continue

            frame = load_frame(data_root, vid, f, size=image_size)
            image = (
                torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            ).unsqueeze(0).to(device)
            fixations_scaled = rescale_fixations(
                fixations,
                src_size=(FIND_NATIVE_W, FIND_NATIVE_H),
                dst_size=(image_size, image_size),
            )
            gt_heatmap = make_heatmap(
                fixations_scaled, image_size, image_size, sigma=sigma,
            )

            with torch.no_grad():
                pred = source(image)[0].cpu().numpy()  # (H, W) sum-to-1

            aucs.append(auc_score(pred, fixations_scaled))
            nsses.append(nss_score(pred, fixations_scaled))
            ccs.append(cc_score(pred, gt_heatmap))

    return {
        "n_frames": len(aucs),
        "auc_mean": float(np.mean(aucs)) if aucs else 0.0,
        "auc_std":  float(np.std(aucs))  if aucs else 0.0,
        "nss_mean": float(np.mean(nsses)) if nsses else 0.0,
        "nss_std":  float(np.std(nsses))  if nsses else 0.0,
        "cc_mean":  float(np.mean(ccs))  if ccs  else 0.0,
        "cc_std":   float(np.std(ccs))   if ccs  else 0.0,
    }


def run_full_comparison(
    data_root: str | os.PathLike[str],
    *,
    sources: Iterable[SaliencySource] | None = None,
    resnet_checkpoint: str | os.PathLike[str] | None = "gazejepa/checkpoints/resnet_saliency_best.pt",
    ijepa_checkpoint: str | os.PathLike[str] | None = "gazejepa/checkpoints/ijepa_saliency_best.pt",
    output_csv: str | os.PathLike[str] = "report_arash/data/saliency_comparison.csv",
    split: str = "test",
    image_size: int = 224,
    seed: int = 42,
    device: torch.device | str = "cpu",
):  # -> pd.DataFrame (lazy-imported)
    """Evaluate all comparison sources, save a CSV, return a DataFrame.
    """
    if sources is None:
        from gazejepa.saliency.center_bias import CenterBiasSaliency
        from gazejepa.saliency.spectral_residual import SpectralResidualSaliency
        from gazejepa.saliency.random import RandomSaliency
        from gazejepa.saliency.resnet_saliency import ResNetSaliency

        built: list[SaliencySource] = [
            RandomSaliency(),
            CenterBiasSaliency(),
            SpectralResidualSaliency(),
        ]
        if resnet_checkpoint and Path(resnet_checkpoint).is_file():
            built.append(ResNetSaliency.load(str(resnet_checkpoint), device=device))
        else:
            print(
                f"[saliency_comparison] ResNet checkpoint not found at "
                f"{resnet_checkpoint}; skipping."
            )
        if ijepa_checkpoint and Path(ijepa_checkpoint).is_file():
            from gazejepa.saliency.ijepa_saliency import IJepaSaliency
            built.append(IJepaSaliency.load(str(ijepa_checkpoint), device=device))
        else:
            print(
                f"[saliency_comparison] I-JEPA checkpoint not found at "
                f"{ijepa_checkpoint}; skipping."
            )
        sources_iter: Iterable[SaliencySource] = built
    else:
        sources_iter = sources

    rows: dict[str, dict[str, float]] = {}
    for source in sources_iter:
        rows[source.name] = evaluate_source(
            source, data_root,
            split=split, image_size=image_size, seed=seed, device=device,
        )
        r = rows[source.name]
        print(
            f"  {source.name:>25}  "
            f"AUC={r['auc_mean']:.4f}±{r['auc_std']:.4f}  "
            f"NSS={r['nss_mean']:.4f}±{r['nss_std']:.4f}  "
            f"CC={r['cc_mean']:.4f}±{r['cc_std']:.4f}  (n={r['n_frames']})"
        )

    import pandas as pd  # heavy import; only needed for the CSV table.
    df = pd.DataFrame(rows).T
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv)
    print(f"\nSaved comparison to {output_csv}")
    return df
