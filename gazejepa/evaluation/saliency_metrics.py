"""AUC / NSS / CC saliency metrics, following Bylinskii et al. (2019)."""

import numpy as np
from scipy.stats import rankdata


def _roc_auc(labels, scores):
    """ROC AUC via Mann-Whitney rank-sum (sklearn-free); ties get averaged ranks."""
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = rankdata(scores, method="average")
    rank_sum_pos = float(ranks[labels].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def auc_score(saliency_map, fixation_points, n_neg_samples=10000):
    """AUC treating saliency as a fixated/non-fixated classifier. 0.5 = chance, 1.0 = perfect."""
    h, w = saliency_map.shape

    if len(fixation_points) == 0:
        return 0.5

    pos_values = []
    for x, y in fixation_points:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            pos_values.append(saliency_map[yi, xi])

    if len(pos_values) == 0:
        return 0.5

    fix_set = set()
    for x, y in fixation_points:
        fix_set.add((int(round(x)), int(round(y))))

    neg_values = []
    rng = np.random.default_rng(42)
    attempts = 0
    while len(neg_values) < n_neg_samples and attempts < n_neg_samples * 3:
        rx = rng.integers(0, w)
        ry = rng.integers(0, h)
        if (rx, ry) not in fix_set:
            neg_values.append(saliency_map[ry, rx])
        attempts += 1

    if len(neg_values) == 0:
        return 0.5

    labels = np.concatenate([np.ones(len(pos_values)), np.zeros(len(neg_values))])
    scores = np.concatenate([pos_values, neg_values])

    return _roc_auc(labels, scores)


def nss_score(saliency_map, fixation_points):
    """Normalized Scanpath Saliency: mean z-scored saliency at fixations. 0 = chance."""
    h, w = saliency_map.shape

    if len(fixation_points) == 0:
        return 0.0

    mu = saliency_map.mean()
    sigma = saliency_map.std()
    if sigma < 1e-8:
        return 0.0
    normalized = (saliency_map - mu) / sigma

    values = []
    for x, y in fixation_points:
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            values.append(normalized[yi, xi])

    if len(values) == 0:
        return 0.0

    return float(np.mean(values))


def cc_score(saliency_map, gt_heatmap):
    """Pearson correlation between predicted and ground-truth maps. ``[-1, 1]``."""
    s = saliency_map.flatten().astype(np.float64)
    g = gt_heatmap.flatten().astype(np.float64)

    if s.std() < 1e-8 or g.std() < 1e-8:
        return 0.0

    return float(np.corrcoef(s, g)[0, 1])


def compute_all_metrics(saliency_map, fixation_points, gt_heatmap):
    """Compute AUC, NSS, and CC. Returns ``{'AUC': ..., 'NSS': ..., 'CC': ...}``."""
    return {
        'AUC': auc_score(saliency_map, fixation_points),
        'NSS': nss_score(saliency_map, fixation_points),
        'CC': cc_score(saliency_map, gt_heatmap),
    }


def center_bias_baseline(img_h, img_w, sigma_frac=0.25):
    """Image-centered Gaussian. On FIND, any saliency model should beat this."""
    cy, cx = img_h / 2, img_w / 2
    sigma_y = img_h * sigma_frac
    sigma_x = img_w * sigma_frac

    yy, xx = np.mgrid[0:img_h, 0:img_w]
    baseline = np.exp(-((xx - cx)**2 / (2 * sigma_x**2) + (yy - cy)**2 / (2 * sigma_y**2)))
    return baseline / baseline.max()


def uniform_random_baseline(img_h, img_w):
    """Flat saliency. Yields AUC ≈ 0.5, NSS ≈ 0, CC ≈ 0."""
    return np.ones((img_h, img_w))


if __name__ == "__main__":
    from gazejepa.data.find_dataset import (
        get_frame_fixations, get_video_ids, load_fixations,
        load_frame, make_heatmap, resolve_data_root,
    )

    data_root = resolve_data_root()
    vid = get_video_ids(data_root)[0]
    fd = load_fixations(data_root, vid)
    frame = load_frame(data_root, vid, 100)
    img_h, img_w = frame.shape[:2]

    fixations = get_frame_fixations(fd, 100)
    gt_heatmap = make_heatmap(fixations, img_h, img_w, sigma=30)

    print(f"Video {vid}, Frame 100")
    print(f"Image size: {img_w}x{img_h}")
    print(f"Number of fixation points: {len(fixations)}")
    print()

    baselines = {
        "Ground truth (upper bound)": gt_heatmap,
        "Center bias": center_bias_baseline(img_h, img_w),
        "Uniform random (lower bound)": uniform_random_baseline(img_h, img_w),
    }

    print(f"{'Baseline':<35} {'AUC':>6} {'NSS':>6} {'CC':>6}")
    print("-" * 58)
    for name, sal_map in baselines.items():
        metrics = compute_all_metrics(sal_map, fixations, gt_heatmap)
        print(f"{name:<35} {metrics['AUC']:6.3f} {metrics['NSS']:6.2f} {metrics['CC']:6.3f}")

    print()
    print("Expected ranges for a good learned saliency model:")
    print("  AUC:  0.85 – 0.95  (must beat center bias)")
    print("  NSS:  1.5 – 3.0")
    print("  CC:   0.4 – 0.7")