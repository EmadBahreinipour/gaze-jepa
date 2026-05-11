"""Evaluation loop for all four saliency sources on the FIND test split.

This is the most important function in the project. It produces the
headline comparison table for the report.
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.find_data import (get_video_ids, load_fixations, get_frame_count,
                            get_frame_fixations, rescale_fixations, make_heatmap,
                            get_video_size, MIN_OBSERVERS_PER_FRAME)
from src.metrics.saliency_metrics import auc_judd, nss, cc
from src.saliency import SaliencySource
from src.saliency.learned import IMAGE_TRANSFORM


def evaluate_source(source: SaliencySource,
                    data_root: str,
                    split: str = "test",
                    image_size: int = 224,
                    sigma: float = 20.0,
                    frame_stride: int = 10,
                    min_observers: int = MIN_OBSERVERS_PER_FRAME) -> dict:
    """Evaluate a saliency source on a FIND split.

    All four sources must be evaluated with identical parameters to ensure
    fair comparison. Uses the same frame_stride, sigma, and min_observers.

    Returns dict with mean/std of AUC, NSS, CC and the frame count.
    """
    video_ids = get_video_ids(data_root, split)
    aucs, nsses, ccs = [], [], []

    for vid in tqdm(video_ids, desc=f"Evaluating {source.name}"):
        fix_array = load_fixations(data_root, vid)
        src_size  = get_video_size(data_root, vid)
        n_frames  = get_frame_count(data_root, vid)

        for f in range(0, n_frames, frame_stride):
            fixations = get_frame_fixations(fix_array, f)
            if len(fixations) < min_observers:
                continue

            from src.find_data import load_frame
            image = load_frame(data_root, vid, f, size=image_size)
            fixations_scaled = rescale_fixations(fixations, src_size,
                                                 (image_size, image_size))
            gt_heatmap = make_heatmap(fixations_scaled, image_size, sigma)

            image_tensor = IMAGE_TRANSFORM(image).unsqueeze(0)
            with torch.no_grad():
                pred = source(image_tensor)[0].numpy()  # (H, W)

            fix_int = fixations_scaled.astype(int)
            aucs.append(auc_judd(pred, fix_int))
            nsses.append(nss(pred, fix_int))
            ccs.append(cc(pred, gt_heatmap))

    return {
        "n_frames": len(aucs),
        "auc_mean": float(np.mean(aucs)),  "auc_std": float(np.std(aucs)),
        "nss_mean": float(np.mean(nsses)), "nss_std": float(np.std(nsses)),
        "cc_mean":  float(np.mean(ccs)),   "cc_std":  float(np.std(ccs)),
    }


def run_full_comparison(data_root: str,
                        checkpoint_path: str = "checkpoints/resnet_saliency_best.pt",
                        output_csv: str = "outputs/saliency_comparison.csv",
                        split: str = "test") -> pd.DataFrame:
    """Evaluate all four sources and return a comparison DataFrame.

    Sanity checks:
      - Random AUC should be ~0.50 (if not, AUC implementation is biased)
      - Center-bias AUC should be 0.70–0.85 on FIND
      - Itti-Koch AUC should be 0.55–0.70
      - Learned AUC should be >= center-bias if training succeeded
    """
    from src.saliency.random_saliency import RandomSaliency
    from src.saliency.center_bias import CenterBiasParametric
    from src.saliency.itti_koch import get_classical_saliency
    from src.saliency.resnet_saliency import ResNetSaliency

    sources = [
        RandomSaliency(),
        CenterBiasParametric(),
        get_classical_saliency(),
        ResNetSaliency.load(checkpoint_path),
    ]

    results = {}
    for source in sources:
        print(f"\n=== {source.name} ===")
        results[source.name] = evaluate_source(source, data_root, split)
        r = results[source.name]
        print(f"  AUC={r['auc_mean']:.4f}±{r['auc_std']:.4f}  "
              f"NSS={r['nss_mean']:.4f}±{r['nss_std']:.4f}  "
              f"CC={r['cc_mean']:.4f}±{r['cc_std']:.4f}  "
              f"(n={r['n_frames']})")

    # Sanity checks
    random_auc = results["random"]["auc_mean"]
    if random_auc > 0.55:
        print(f"\nWARNING: Random AUC={random_auc:.4f} > 0.55. AUC implementation may be biased.")

    df = pd.DataFrame(results).T
    import os
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv)
    print(f"\nResults saved to {output_csv}")
    print(df[["auc_mean", "nss_mean", "cc_mean"]].round(4))
    return df


if __name__ == "__main__":
    import sys
    data_root = sys.argv[1] if len(sys.argv) > 1 else "f:/university/milan/NI/info/Gaze_Jepa/find_dataset"
    run_full_comparison(data_root)
