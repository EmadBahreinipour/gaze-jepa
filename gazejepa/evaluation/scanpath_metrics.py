"""Scanpath comparison metrics: MultiMatch (Dewhurst et al. 2012), AOI edit distance,
Fréchet, and saccade-amplitude statistics."""

import numpy as np
from scipy.spatial.distance import directed_hausdorff

# Prefer the canonical multimatch_gaze package when installed; fall back
# to the simplified implementation below otherwise.
try:
    import multimatch_gaze as _mm_pkg  # type: ignore
    _MULTIMATCH_BACKEND = "canonical"
except ImportError:  # pragma: no cover
    _mm_pkg = None  # type: ignore
    _MULTIMATCH_BACKEND = "simplified"


def multimatch_backend() -> str:
    """Return ``"canonical"`` if multimatch_gaze is in use, else ``"simplified"``."""
    return _MULTIMATCH_BACKEND


def multimatch(scanpath1, scanpath2, img_size=(1280, 720)):
    """MultiMatch comparison; returns ``{shape, direction, length, position}``, all in ``[0, 1]``.

    Duration is omitted because FIND has no per-fixation durations.
    """
    if _MULTIMATCH_BACKEND == "canonical":
        return _multimatch_canonical(scanpath1, scanpath2, img_size)
    return _multimatch_simplified(scanpath1, scanpath2, img_size)


def _multimatch_canonical(scanpath1, scanpath2, img_size):
    """MultiMatch via the multimatch_gaze package (Dewhurst et al. 2012)."""
    if len(scanpath1) < 2 or len(scanpath2) < 2:
        return {"shape": 0.0, "direction": 0.0, "length": 0.0, "position": 0.0}

    import pandas as pd  # local import; pandas is heavy

    def _to_df(sp):
        return pd.DataFrame({
            "start_x": np.asarray(sp[:, 0], dtype=float),
            "start_y": np.asarray(sp[:, 1], dtype=float),
            "duration": np.full(len(sp), 200.0),  # placeholder; FIND has no per-fix durations
        })

    result = _mm_pkg.docomparison(  # type: ignore[union-attr]
        _to_df(scanpath1), _to_df(scanpath2), screensize=img_size,
    )
    return {
        "shape":     float(np.clip(result[0], 0, 1)),
        "direction": float(np.clip(result[1], 0, 1)),
        "length":    float(np.clip(result[2], 0, 1)),
        "position":  float(np.clip(result[3], 0, 1)),
    }


def _multimatch_simplified(scanpath1, scanpath2, img_size=(1280, 720)):
    """Simplified MultiMatch (used only when multimatch_gaze is unavailable)."""
    if len(scanpath1) < 2 or len(scanpath2) < 2:
        return {'shape': 0.0, 'direction': 0.0, 'length': 0.0, 'position': 0.0}

    w, h = img_size
    diag = np.sqrt(w**2 + h**2)

    s1 = scanpath1.copy().astype(np.float64)
    s2 = scanpath2.copy().astype(np.float64)
    s1[:, 0] /= w
    s1[:, 1] /= h
    s2[:, 0] /= w
    s2[:, 1] /= h

    n = max(len(s1), len(s2))
    s1_interp = _interpolate_scanpath(s1, n)
    s2_interp = _interpolate_scanpath(s2, n)

    pos_dists = np.sqrt(np.sum((s1_interp - s2_interp)**2, axis=1))
    position_sim = 1.0 - np.mean(pos_dists) / np.sqrt(2)  # max dist in [0,1]² is sqrt(2)

    s1_centered = s1_interp - s1_interp.mean(axis=0, keepdims=True)
    s2_centered = s2_interp - s2_interp.mean(axis=0, keepdims=True)
    shape_dists = np.sqrt(np.sum((s1_centered - s2_centered)**2, axis=1))
    shape_sim = 1.0 - np.mean(shape_dists) / np.sqrt(2)

    sac1 = np.diff(s1_interp, axis=0)
    sac2 = np.diff(s2_interp, axis=0)

    angles1 = np.arctan2(sac1[:, 1], sac1[:, 0])
    angles2 = np.arctan2(sac2[:, 1], sac2[:, 0])
    angle_diffs = np.abs(angles1 - angles2)
    angle_diffs = np.minimum(angle_diffs, 2 * np.pi - angle_diffs)
    direction_sim = 1.0 - np.mean(angle_diffs) / np.pi

    lens1 = np.sqrt(np.sum(sac1**2, axis=1))
    lens2 = np.sqrt(np.sum(sac2**2, axis=1))
    max_lens = np.maximum(lens1, lens2) + 1e-8
    length_sim = np.mean(1.0 - np.abs(lens1 - lens2) / max_lens)

    return {
        'shape': float(np.clip(shape_sim, 0, 1)),
        'direction': float(np.clip(direction_sim, 0, 1)),
        'length': float(np.clip(length_sim, 0, 1)),
        'position': float(np.clip(position_sim, 0, 1)),
    }


def _interpolate_scanpath(scanpath, n_points):
    """Linearly interpolate a scanpath to have exactly n_points."""
    if len(scanpath) == n_points:
        return scanpath
    old_idx = np.linspace(0, 1, len(scanpath))
    new_idx = np.linspace(0, 1, n_points)
    x_new = np.interp(new_idx, old_idx, scanpath[:, 0])
    y_new = np.interp(new_idx, old_idx, scanpath[:, 1])
    return np.stack([x_new, y_new], axis=1)


def string_edit_distance(scanpath1, scanpath2, img_size=(1280, 720), grid_size=7):
    """Levenshtein distance on AOI sequences (image discretised to a ``grid_size`` grid).

    Returns ``{edit_distance, normalized, similarity}``; ``normalized`` and
    ``similarity`` are in ``[0, 1]``.
    """
    def to_string(scanpath):
        w, h = img_size
        chars = []
        for x, y in scanpath:
            gx = min(int(x / w * grid_size), grid_size - 1)
            gy = min(int(y / h * grid_size), grid_size - 1)
            cell_id = gy * grid_size + gx
            chars.append(chr(65 + cell_id))  # A, B, C, ...
        return ''.join(chars)

    s1 = to_string(scanpath1)
    s2 = to_string(scanpath2)

    dist = _levenshtein(s1, s2)
    max_len = max(len(s1), len(s2))
    norm = dist / max_len if max_len > 0 else 0.0

    return {
        'edit_distance': dist,
        'normalized': float(norm),
        'similarity': float(1.0 - norm),
    }


def _levenshtein(s1, s2):
    """Standard Levenshtein distance."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i-1] == s2[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def frechet_distance(scanpath1, scanpath2, img_size=(1280, 720)):
    """Discrete Fréchet distance — order-sensitive, unlike Hausdorff.

    Returns ``{frechet, normalized}``; ``normalized = frechet / image_diagonal``.
    """
    p = scanpath1.astype(np.float64)
    q = scanpath2.astype(np.float64)
    n, m = len(p), len(q)

    if n == 0 or m == 0:
        return {'frechet': float('inf'), 'normalized': 1.0}

    ca = np.full((n, m), -1.0)

    def _dist(i, j):
        return np.sqrt(np.sum((p[i] - q[j])**2))

    def _rec(i, j):
        if ca[i, j] > -0.5:
            return ca[i, j]
        d = _dist(i, j)
        if i == 0 and j == 0:
            ca[i, j] = d
        elif i == 0:
            ca[i, j] = max(_rec(0, j-1), d)
        elif j == 0:
            ca[i, j] = max(_rec(i-1, 0), d)
        else:
            ca[i, j] = max(min(_rec(i-1, j), _rec(i-1, j-1), _rec(i, j-1)), d)
        return ca[i, j]

    # Use iterative DP to avoid recursion depth issues
    for i in range(n):
        for j in range(m):
            d = _dist(i, j)
            if i == 0 and j == 0:
                ca[i, j] = d
            elif i == 0:
                ca[i, j] = max(ca[0, j-1], d)
            elif j == 0:
                ca[i, j] = max(ca[i-1, 0], d)
            else:
                ca[i, j] = max(min(ca[i-1, j], ca[i-1, j-1], ca[i, j-1]), d)

    w, h = img_size
    diag = np.sqrt(w**2 + h**2)
    frechet_val = ca[n-1, m-1]

    return {
        'frechet': float(frechet_val),
        'normalized': float(np.clip(frechet_val / diag, 0, 1)),
    }


def saccade_statistics(scanpath, img_size=(1280, 720)):
    """Saccade-level amplitude and direction statistics for a single scanpath."""
    if len(scanpath) < 2:
        return {
            'n_fixations': len(scanpath),
            'amplitudes': np.array([]),
            'mean_amplitude': 0.0,
            'std_amplitude': 0.0,
            'directions': np.array([]),
        }

    saccades = np.diff(scanpath, axis=0)
    amplitudes = np.sqrt(np.sum(saccades**2, axis=1))
    directions = np.arctan2(saccades[:, 1], saccades[:, 0])

    return {
        'n_fixations': len(scanpath),
        'amplitudes': amplitudes,
        'mean_amplitude': float(np.mean(amplitudes)),
        'std_amplitude': float(np.std(amplitudes)),
        'median_amplitude': float(np.median(amplitudes)),
        'directions': directions,
        'mean_direction': float(np.mean(directions)),
    }


def compare_saccade_distributions(scanpath1, scanpath2, img_size=(1280, 720)):
    """KS test on the two scanpaths' amplitude distributions. Returns ``{ks_statistic, p_value}``."""
    from scipy.stats import ks_2samp

    stats1 = saccade_statistics(scanpath1, img_size)
    stats2 = saccade_statistics(scanpath2, img_size)

    if len(stats1['amplitudes']) < 2 or len(stats2['amplitudes']) < 2:
        return {'ks_statistic': 1.0, 'p_value': 0.0}

    ks_stat, p_val = ks_2samp(stats1['amplitudes'], stats2['amplitudes'])

    return {
        'ks_statistic': float(ks_stat),
        'p_value': float(p_val),
    }


def compare_scanpaths(human_scanpath, model_scanpath, img_size=(1280, 720)):
    """Run every scanpath metric in this module and return them keyed in one dict."""
    results = {}
    results['multimatch'] = multimatch(human_scanpath, model_scanpath, img_size)
    results['string_edit'] = string_edit_distance(human_scanpath, model_scanpath, img_size)
    results['frechet'] = frechet_distance(human_scanpath, model_scanpath, img_size)
    results['saccade_dist'] = compare_saccade_distributions(human_scanpath, model_scanpath, img_size)
    results['human_stats'] = saccade_statistics(human_scanpath, img_size)
    results['model_stats'] = saccade_statistics(model_scanpath, img_size)
    return results


def summarize_comparison(results):
    """Print a human-readable summary of scanpath comparison results."""
    mm = results['multimatch']
    sed = results['string_edit']
    fd = results['frechet']
    sac = results['saccade_dist']
    hs = results['human_stats']
    ms = results['model_stats']

    print("=" * 60)
    print("SCANPATH COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n{'Metric':<30} {'Value':>10}")
    print("-" * 42)
    print(f"{'MultiMatch — Position':<30} {mm['position']:>10.3f}")
    print(f"{'MultiMatch — Shape':<30} {mm['shape']:>10.3f}")
    print(f"{'MultiMatch — Direction':<30} {mm['direction']:>10.3f}")
    print(f"{'MultiMatch — Length':<30} {mm['length']:>10.3f}")
    print(f"{'String Edit Similarity':<30} {sed['similarity']:>10.3f}")
    print(f"{'Fréchet Distance (norm)':<30} {fd['normalized']:>10.3f}")
    print(f"{'Saccade Amp KS-stat':<30} {sac['ks_statistic']:>10.3f}")
    print(f"\n{'Statistic':<30} {'Human':>10} {'Model':>10}")
    print("-" * 52)
    print(f"{'N fixations':<30} {hs['n_fixations']:>10d} {ms['n_fixations']:>10d}")
    print(f"{'Mean saccade amp (px)':<30} {hs['mean_amplitude']:>10.1f} {ms['mean_amplitude']:>10.1f}")
    print(f"{'Median saccade amp (px)':<30} {hs['median_amplitude']:>10.1f} {ms['median_amplitude']:>10.1f}")
    print(f"{'Std saccade amp (px)':<30} {hs['std_amplitude']:>10.1f} {ms['std_amplitude']:>10.1f}")


if __name__ == "__main__":
    np.random.seed(42)

    human = np.array([
        [640, 360],  # center
        [580, 280],  # left eye
        [700, 280],  # right eye
        [640, 350],  # nose
        [640, 420],  # mouth
        [500, 300],  # left face edge
    ], dtype=np.float64)

    model_good = np.array([
        [650, 370],
        [570, 290],
        [710, 270],
        [630, 340],
        [650, 430],
    ], dtype=np.float64)

    model_random = np.random.uniform(
        [0, 0], [1280, 720], size=(5, 2)
    )

    print("=== Good Model vs Human ===")
    results_good = compare_scanpaths(human, model_good)
    summarize_comparison(results_good)

    print("\n\n=== Random Model vs Human ===")
    results_random = compare_scanpaths(human, model_random)
    summarize_comparison(results_random)