#!/usr/bin/env python
"""
evaluate_jepa_reuse.py — scanpath comparison for the trained jepa_reuse model.

Loads a trained ``make_gaze_jepa`` checkpoint, then evaluates the
saliency × IOR-driven scanpaths it produces against the human FIND
scanpath pool with three metrics:

  * **KS test** on saccade-amplitude distributions (primary).
    Generated amplitudes vs. the cross-observer video-pool human
    distribution (see ``collect_video_saccade_amplitudes``). The pool is
    aggregated over the full split's videos rather than the curated
    grid because the per-pair Option A scanpaths are too short (5
    frames at 30 fps = 167 ms) to contain real saccades.
  * **MultiMatch** position / shape / direction / length similarity vs.
    Option A human scanpaths (per-pair, on the curated grid).
  * **Fréchet** distance vs. the same Option A pairs.

Scope
-----
The ``GazeCropper`` samples T fixations deterministically from
``saliency × IOR`` given a fixed seed; the trained predictor's weights
do not influence *where* it looks, only its representation-space output.
So this script evaluates the saliency × IOR scanpath mechanism, not the
predictor itself, and only reads cropper hyperparameters from
``checkpoint["args"]`` (``ior_sigma``, ``ior_decay``, ``n_fixations``,
``image_size``) — never the predictor weights. The scanpath it
reproduces is byte-identical to what the trained GazeCropper sampled,
given the same ``--saliency`` source (default ``resnet``).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import ks_2samp

# Allow `python scripts/evaluate_jepa_reuse.py` without `pip install -e .`.
HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

from gazejepa.data.find_dataset import get_split, resolve_data_root  # noqa: E402
from gazejepa.evaluation.evaluate import (  # noqa: E402
    collect_generated_amplitudes,
    collect_paired_human_amplitudes,
    collect_video_saccade_amplitudes,
    evaluate_gaze_loop,
)
from gazejepa.evaluation.scanpath_metrics import multimatch_backend  # noqa: E402
from gazejepa.saliency import (  # noqa: E402
    CenterBiasParametric,
    SpectralResidualSaliency,
    RandomSaliency,
    ResNetSaliency,
    SaliencySource,
)


# --------------------------------------------------------------------- #
# Saliency factory                                                       #
# --------------------------------------------------------------------- #


_SALIENCY_CHOICES = (
    "resnet", "spectral_residual", "center_bias", "random",
)


def build_saliency(
    name: str,
    image_size: int,
    resnet_checkpoint: str,
    device: str | torch.device = "cpu",
) -> SaliencySource:
    """Instantiate a saliency source by name.

    The default ``resnet`` source loads the trained ResNetSaliency checkpoint.
    """
    name = name.lower()
    if name == "resnet":
        ckpt = Path(resnet_checkpoint).expanduser().resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"ResNet saliency checkpoint not found: {ckpt}. Train it with "
                f"scripts/train_saliency.py or pass --resnet-checkpoint."
            )
        return ResNetSaliency.load(str(ckpt), device=device)
    if name == "spectral_residual":
        return SpectralResidualSaliency()
    if name == "center_bias":
        return CenterBiasParametric(image_size=image_size)
    if name == "random":
        return RandomSaliency()
    raise ValueError(
        f"Unknown saliency source {name!r}; choose from {_SALIENCY_CHOICES}."
    )


# --------------------------------------------------------------------- #
# CLI                                                                    #
# --------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument(
        "--checkpoint",
        default="outputs/jepa_reuse/checkpoints/final.pt",
        help="Path to a trained jepa_reuse checkpoint (default: %(default)s).",
    )
    p.add_argument(
        "--data-root", default=None,
        help="FIND dataset root (else $FIND_DATA_ROOT or repo default).",
    )
    p.add_argument(
        "--output-dir", default="outputs/jepa_reuse/eval",
        help="Where the CSV + PNG go (default: %(default)s).",
    )
    p.add_argument(
        "--splits", default="val,test",
        help="Comma-separated splits to evaluate (default: %(default)s).",
    )
    p.add_argument(
        "--n-videos", type=int, default=5,
        help="Videos per split (default: %(default)s; max 6).",
    )
    p.add_argument(
        "--n-observers", type=int, default=5,
        help="Max human observers per (video, frame) (default: %(default)s).",
    )
    p.add_argument(
        "--frames", default="50,150,300,500",
        help="Comma-separated start frame indices (default: %(default)s).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help=(
            "RNG seed for the cropper. Should match the training seed "
            "(default: %(default)s) so the evaluated scanpaths reproduce "
            "what the trained model's GazeCropper sampled."
        ),
    )
    p.add_argument(
        "--saliency",
        default="resnet",
        choices=_SALIENCY_CHOICES,
        help=(
            "Saliency source feeding the cropper (default: %(default)s, "
            "matching the training-time default in train_jepa_reuse.py)."
        ),
    )
    p.add_argument(
        "--resnet-checkpoint",
        default="checkpoints/resnet_saliency_best.pt",
        help=(
            "Checkpoint for --saliency resnet "
            "(default: %(default)s, matching train_jepa_reuse.py; "
            "ignored for other saliency sources)."
        ),
    )
    return p.parse_args()


# --------------------------------------------------------------------- #
# Checkpoint                                                             #
# --------------------------------------------------------------------- #


def load_cropper_hparams(ckpt_path: Path) -> dict[str, Any]:
    """Extract the cropper-relevant hyperparameters from a checkpoint.

    The training script saves ``vars(args)`` under the ``"args"`` key.
    We only need the subset that drives the GazeCropper's RNG path —
    everything else (lr, weight_decay, etc.) is irrelevant to the
    scanpath itself.

    The training script hardcodes ``sampling_mode="stochastic"`` (see
    ``train_jepa_reuse.py``) so we hardcode it here too rather than
    expecting it in ``args``.
    """
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Run scripts/train_jepa_reuse.py "
            f"first, or pass --checkpoint to point at an existing one."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "args" not in ckpt:
        raise KeyError(
            f"Checkpoint {ckpt_path} has no 'args' field. Cannot recover "
            f"the GazeCropper hyperparameters used during training."
        )
    a = ckpt["args"]
    needed = ["ior_sigma", "ior_decay", "n_fixations", "image_size", "seed"]
    missing = [k for k in needed if k not in a]
    if missing:
        raise KeyError(
            f"Checkpoint args missing keys: {missing}. The checkpoint may "
            f"have been produced by an older training script."
        )
    return {
        "ior_sigma":     float(a["ior_sigma"]),
        "ior_decay":     float(a["ior_decay"]),
        "n_fixations":   int(a["n_fixations"]),
        "image_size":    int(a["image_size"]),
        "seed":          int(a["seed"]),
        # SaccadeJepa/GazeCropper sampling_mode is hardcoded in the
        # training script; surface it here for the eval log.
        "sampling_mode": "stochastic",
    }


# --------------------------------------------------------------------- #
# CSV / plot                                                             #
# --------------------------------------------------------------------- #


CSV_FIELDNAMES = [
    "split", "saliency_source", "video_id", "start_frame", "observer_idx",
    "n_observers_used",
    "multimatch_position", "multimatch_shape",
    "multimatch_direction", "multimatch_length",
    "frechet", "frechet_normalized",
    "generated_scanpath", "human_scanpath",
]


def _scanpath_to_str(sp: np.ndarray) -> str:
    return ";".join(f"{float(x):.2f},{float(y):.2f}" for x, y in sp)


def write_csv(
    all_rows_per_split: dict[str, list[dict[str, Any]]],
    csv_path: Path,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for split_name, rows in all_rows_per_split.items():
            for r in rows:
                writer.writerow({
                    "split":                split_name,
                    "saliency_source":      r["saliency_source"],
                    "video_id":             r["video_id"],
                    "start_frame":          r["start_frame"],
                    "observer_idx":         r["observer_idx"],
                    "n_observers_used":     r["n_observers_used"],
                    "multimatch_position":  f"{r['multimatch_position']:.6f}",
                    "multimatch_shape":     f"{r['multimatch_shape']:.6f}",
                    "multimatch_direction": f"{r['multimatch_direction']:.6f}",
                    "multimatch_length":    f"{r['multimatch_length']:.6f}",
                    "frechet":              f"{r['frechet']:.4f}",
                    "frechet_normalized":   f"{r['frechet_normalized']:.6f}",
                    "generated_scanpath":   _scanpath_to_str(r["generated_scanpath"]),
                    "human_scanpath":       _scanpath_to_str(r["human_scanpath"]),
                })


def save_amplitude_histogram(
    rows_per_split: dict[str, list[dict[str, Any]]],
    human_pool_per_split: dict[str, np.ndarray],
    out_path: Path,
    saliency_label: str = "saliency × IOR",
) -> Path:
    """One subplot per split: generated vs. video-pool human amplitudes.

    Density-normalised so the small generated sample is visible
    alongside the much larger human pool.
    """
    splits = [s for s in rows_per_split.keys() if rows_per_split[s]]
    if not splits:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Empty figure with explanatory title — keeps the path stable.
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title("amplitude_histogram — no data")
        ax.axis("off")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out_path

    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 5),
                             sharey=False)
    if len(splits) == 1:
        axes = [axes]

    for ax, split_name in zip(axes, splits):
        rows = rows_per_split[split_name]
        gen_amps = collect_generated_amplitudes(rows)
        hum_amps = human_pool_per_split.get(split_name, np.array([]))

        if len(gen_amps) == 0 or len(hum_amps) == 0:
            ax.set_title(f"{split_name} — no data")
            ax.axis("off")
            continue

        upper = max(gen_amps.max(), float(np.percentile(hum_amps, 99)), 1.0)
        bins = np.linspace(0, upper * 1.05, 30)
        ax.hist(hum_amps, bins=bins, alpha=0.6, color="steelblue",
                density=True, label=f"human pool (n={len(hum_amps)})")
        ax.hist(gen_amps, bins=bins, alpha=0.55, color="coral",
                density=True, label=f"generated (n={len(gen_amps)})")
        ax.axvline(np.median(hum_amps), color="steelblue", linestyle="--",
                   label=f"hum med {np.median(hum_amps):.0f}px")
        ax.axvline(np.median(gen_amps), color="coral", linestyle="--",
                   label=f"gen med {np.median(gen_amps):.0f}px")

        ks_stat, ks_p = ks_2samp(gen_amps, hum_amps)
        ax.set_title(
            f"{split_name} — jepa_reuse ({saliency_label} + IOR)\n"
            f"KS={ks_stat:.3f}  p={ks_p:.3g}"
        )
        ax.set_xlabel("saccade amplitude (px @ image_size)")
        ax.set_ylabel("density")
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


# --------------------------------------------------------------------- #
# Aggregates / printing                                                  #
# --------------------------------------------------------------------- #


def per_split_summary(
    rows: Sequence[dict[str, Any]],
    human_pool: np.ndarray,
) -> dict[str, Any] | None:
    if not rows:
        return None
    gen_amps = collect_generated_amplitudes(rows)
    paired_hum_amps = collect_paired_human_amplitudes(rows)
    if len(gen_amps) < 2 or len(human_pool) < 2:
        return None
    ks_stat, ks_p = ks_2samp(gen_amps, human_pool)
    return {
        "n_pairs":              len(rows),
        "n_gen_scanpaths":      len({(r["video_id"], r["start_frame"]) for r in rows}),
        "n_gen_amplitudes":     len(gen_amps),
        "n_hum_amps_video":     len(human_pool),
        "n_hum_amps_paired":    len(paired_hum_amps),
        "gen_median_amp":       float(np.median(gen_amps)),
        "hum_median_amp_video": float(np.median(human_pool)),
        "ks_statistic":         float(ks_stat),
        "ks_pvalue":            float(ks_p),
        "mean_mm_position":     float(np.mean([r["multimatch_position"] for r in rows])),
        "mean_mm_shape":        float(np.mean([r["multimatch_shape"] for r in rows])),
        "mean_mm_direction":    float(np.mean([r["multimatch_direction"] for r in rows])),
        "mean_mm_length":       float(np.mean([r["multimatch_length"] for r in rows])),
        "mean_frechet":         float(np.mean([r["frechet"] for r in rows])),
        "mean_frechet_norm":    float(np.mean([r["frechet_normalized"] for r in rows])),
    }


def print_summary_table(
    all_rows_per_split: dict[str, list[dict[str, Any]]],
    human_pool_per_split: dict[str, np.ndarray],
) -> None:
    print()
    print("=" * 110)
    print("HEADLINE — jepa_reuse cropper scanpath comparison vs FIND human gaze")
    print("KS test: video-wide cross-observer human pool (≥5px, max 5-frame gap) vs generated.")
    print("MultiMatch + Frechet: Option A per-pair (5 consecutive frames per observer).")
    print("=" * 110)
    cols = (
        f"{'split':<6} {'pairs':>5} {'gen_med':>8} {'hum_med':>8} "
        f"{'KS_stat':>8} {'KS_p':>10} "
        f"{'MM_pos':>7} {'MM_sha':>7} {'MM_dir':>7} {'MM_len':>7} "
        f"{'Frech_px':>9} {'Frech/d':>8}"
    )
    print(cols)
    print("-" * 110)
    for split_name, rows in all_rows_per_split.items():
        pool = human_pool_per_split.get(split_name, np.array([]))
        s = per_split_summary(rows, pool)
        if s is None:
            print(f"{split_name:<6} (insufficient data)")
            continue
        print(
            f"{split_name:<6} "
            f"{s['n_pairs']:>5} "
            f"{s['gen_median_amp']:>8.1f} "
            f"{s['hum_median_amp_video']:>8.1f} "
            f"{s['ks_statistic']:>8.3f} "
            f"{s['ks_pvalue']:>10.4g} "
            f"{s['mean_mm_position']:>7.3f} "
            f"{s['mean_mm_shape']:>7.3f} "
            f"{s['mean_mm_direction']:>7.3f} "
            f"{s['mean_mm_length']:>7.3f} "
            f"{s['mean_frechet']:>9.2f} "
            f"{s['mean_frechet_norm']:>8.3f}"
        )
    print()


# --------------------------------------------------------------------- #
# Main                                                                   #
# --------------------------------------------------------------------- #


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data_root = resolve_data_root(args.data_root)

    hparams = load_cropper_hparams(ckpt_path)
    # Use the training seed if the user did not override.
    if args.seed == 42 and hparams["seed"] != args.seed:
        # Note but don't override — the user can pass --seed explicitly to
        # match. The default (42) matches the typical training default.
        pass

    splits = get_split(str(data_root))
    frame_indices = tuple(int(x) for x in args.frames.split(","))
    splits_to_run = [s.strip() for s in args.splits.split(",") if s.strip()]

    print(f"Checkpoint:      {ckpt_path}")
    print(f"Data root:       {data_root}")
    print(f"Output:          {out_dir}")
    print(f"MultiMatch:      {multimatch_backend()}")
    print(f"Frames:          {frame_indices}")
    print(f"Cropper hparams (from checkpoint['args']):")
    print(f"  image_size     = {hparams['image_size']}")
    print(f"  n_fixations    = {hparams['n_fixations']}")
    print(f"  ior_sigma      = {hparams['ior_sigma']}")
    print(f"  ior_decay      = {hparams['ior_decay']}")
    print(f"  sampling_mode  = {hparams['sampling_mode']}  (hardcoded in train script)")
    print(f"  training seed  = {hparams['seed']}  (eval seed = {args.seed})")
    print()
    print(
        "Note: this evaluates the saliency × IOR scanpath mechanism "
        "(deterministic given seed). The trained predictor's weights "
        "do not influence WHERE the cropper looks; they shape the "
        "predictor's representation-space output. See script docstring."
    )

    image_size = hparams["image_size"]
    saliency = build_saliency(
        args.saliency,
        image_size=image_size,
        resnet_checkpoint=args.resnet_checkpoint,
        device="cpu",
    )
    print(f"Saliency source: {args.saliency}")

    all_rows_per_split: dict[str, list[dict[str, Any]]] = {}
    human_pool_per_split: dict[str, np.ndarray] = {}

    for split_name in splits_to_run:
        if split_name not in splits:
            print(f"Unknown split {split_name!r}; skipping")
            continue
        video_ids = splits[split_name][: args.n_videos]

        print()
        print("=" * 64)
        print(f"Split: {split_name}  ({len(video_ids)} videos: {video_ids})")
        print("=" * 64)

        # Reference human saccade-amplitude pool for the KS test:
        # cross-observer, full videos, NaN-tolerant, ≥5px.
        hum_pool = collect_video_saccade_amplitudes(
            data_root=str(data_root),
            video_ids=video_ids,
            image_size=image_size,
            max_frame_gap=5,
            min_amplitude_px=5.0,
        )
        human_pool_per_split[split_name] = hum_pool
        if len(hum_pool):
            print(
                f"Human pool: {len(hum_pool)} saccades "
                f"(median {np.median(hum_pool):.1f}px, "
                f"mean {hum_pool.mean():.1f}px)"
            )
        else:
            print("Human pool: empty (no fixations)")

        # Run GazeLoop with the same cropper hparams as training. GazeLoop
        # is now scanpath-only (no fovea/periphery args) — same saliency ×
        # IOR × FixationSampler the trained GazeCropper uses internally,
        # so the scanpaths match byte-for-byte given the same seed.
        rows = evaluate_gaze_loop(
            data_root=str(data_root),
            saliency_source=saliency,
            saliency_name="jepa_reuse",
            video_ids=video_ids,
            frame_indices=frame_indices,
            n_observers=args.n_observers,
            n_fixations=hparams["n_fixations"],
            image_size=image_size,
            ior_sigma=hparams["ior_sigma"],
            ior_decay=hparams["ior_decay"],
            sampling_mode=hparams["sampling_mode"],
            seed=args.seed,
        )
        all_rows_per_split[split_name] = rows

        s = per_split_summary(rows, hum_pool)
        if s is None:
            print(f"  pairs={len(rows)} | insufficient data for summary")
        else:
            print(
                f"  pairs={s['n_pairs']:>3} "
                f"({s['n_gen_scanpaths']} gen scanpaths × ~"
                f"{s['n_pairs'] // max(s['n_gen_scanpaths'], 1)} obs) | "
                f"gen_med={s['gen_median_amp']:.1f}px "
                f"hum_med={s['hum_median_amp_video']:.1f}px | "
                f"KS={s['ks_statistic']:.3f} p={s['ks_pvalue']:.4g}"
            )
            print(
                f"        MM_pos={s['mean_mm_position']:.3f} "
                f"MM_sha={s['mean_mm_shape']:.3f} "
                f"MM_dir={s['mean_mm_direction']:.3f} "
                f"MM_len={s['mean_mm_length']:.3f} | "
                f"Frechet={s['mean_frechet']:.2f}px "
                f"(/diag={s['mean_frechet_norm']:.3f})"
            )

    csv_path = out_dir / "scanpath_comparison.csv"
    write_csv(all_rows_per_split, csv_path)
    print(f"\nCSV: {csv_path}")

    fig_path = out_dir / "amplitude_histogram.png"
    save_amplitude_histogram(
        all_rows_per_split, human_pool_per_split, fig_path,
        saliency_label=args.saliency,
    )
    print(f"Histogram: {fig_path}")

    print_summary_table(all_rows_per_split, human_pool_per_split)


if __name__ == "__main__":
    main()
