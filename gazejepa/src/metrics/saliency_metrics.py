"""Saliency evaluation metrics following Bylinskii et al. (2019).

All metrics compare a predicted saliency map against human fixation data.
Higher values = better alignment with human gaze.

Reference:
    Bylinskii, Z., Judd, T., Oliva, A., Torralba, A., & Durand, F. (2019).
    What do different evaluation metrics tell us about saliency models?
    IEEE TPAMI.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def auc_judd(saliency_map: np.ndarray, fixation_points: np.ndarray) -> float:
    """Area Under the ROC Curve (Judd version).

    Treats each fixation as a positive sample and non-fixated locations as
    negatives. Sweeps a threshold over the saliency map to compute the ROC
    curve, then returns the AUC.

    Args:
        saliency_map: (H, W) float array, any non-negative values.
        fixation_points: (N, 2) int array of (x, y) pixel coordinates.

    Returns:
        AUC in [0, 1]. 0.5 = chance level.
    """
    if len(fixation_points) == 0:
        return 0.5

    H, W = saliency_map.shape
    flat = saliency_map.flatten()

    # Positive: saliency at fixation locations
    fx = np.clip(fixation_points[:, 0].astype(int), 0, W - 1)
    fy = np.clip(fixation_points[:, 1].astype(int), 0, H - 1)
    pos_indices = fy * W + fx
    pos_values = flat[pos_indices]

    # Negative: all non-fixated locations
    mask = np.ones(H * W, dtype=bool)
    mask[pos_indices] = False
    neg_values = flat[mask]

    scores = np.concatenate([pos_values, neg_values])
    labels = np.concatenate([np.ones(len(pos_values)), np.zeros(len(neg_values))])

    if scores.max() == scores.min():
        return 0.5

    return float(roc_auc_score(labels, scores))


def nss(saliency_map: np.ndarray, fixation_points: np.ndarray) -> float:
    """Normalized Scanpath Saliency.

    Normalizes the saliency map to zero mean and unit variance, then returns
    the mean value at fixation locations. Positive values indicate saliency
    is higher at fixations than the mean; negative means the model is worse
    than chance.

    Args:
        saliency_map: (H, W) float array.
        fixation_points: (N, 2) int array of (x, y) pixel coordinates.

    Returns:
        NSS score (float). Typical values: 1.0–2.5 for good saliency models.
    """
    if len(fixation_points) == 0:
        return 0.0

    H, W = saliency_map.shape
    std = saliency_map.std()
    if std < 1e-8:
        return 0.0

    normalized = (saliency_map - saliency_map.mean()) / std

    fx = np.clip(fixation_points[:, 0].astype(int), 0, W - 1)
    fy = np.clip(fixation_points[:, 1].astype(int), 0, H - 1)
    return float(normalized[fy, fx].mean())


def cc(saliency_map: np.ndarray, ground_truth_heatmap: np.ndarray) -> float:
    """Pearson Correlation Coefficient between saliency map and ground-truth heatmap.

    Args:
        saliency_map: (H, W) float array.
        ground_truth_heatmap: (H, W) float array, same spatial resolution.

    Returns:
        Pearson CC in [-1, 1]. 1.0 = perfect correlation, 0.0 = no correlation.
    """
    s = saliency_map.flatten().astype(np.float64)
    g = ground_truth_heatmap.flatten().astype(np.float64)

    if s.std() < 1e-8 or g.std() < 1e-8:
        return 0.0

    return float(np.corrcoef(s, g)[0, 1])


if __name__ == "__main__":
    import numpy as np

    H, W = 64, 64
    cx, cy = 32, 32
    ys, xs = np.mgrid[0:H, 0:W]
    perfect_map = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * 8 ** 2)).astype(np.float32)

    fixations = np.array([[cx, cy], [cx+5, cy-3], [cx-4, cy+6]], dtype=int)

    print("Saliency map peaked at center, fixations near center:")
    print(f"  AUC  = {auc_judd(perfect_map, fixations):.4f}  (expect > 0.5)")
    print(f"  NSS  = {nss(perfect_map, fixations):.4f}   (expect > 1.0)")

    flat_map = np.ones((H, W), dtype=np.float32) / (H * W)
    gt_heatmap = perfect_map / perfect_map.sum()
    print(f"  CC(perfect, gt)  = {cc(perfect_map, gt_heatmap):.4f}  (expect ~1.0)")
    print(f"  CC(uniform, gt)  = {cc(flat_map, gt_heatmap):.4f}   (expect ~0.0)")
