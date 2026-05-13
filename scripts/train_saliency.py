"""CLI entry point for saliency-predictor training on FIND.

Usage:
    python scripts/train_saliency.py --config configs/saliency_train.yml
    python scripts/train_saliency.py --config configs/saliency_train.yml \
        --data-root data/find_dataset --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from gazejepa.saliency.training import train_saliency


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config", default="configs/saliency_train.yml",
        help="YAML hyperparameter file.",
    )
    p.add_argument(
        "--data-root", default=None,
        help="Override data_root from the YAML config.",
    )
    p.add_argument(
        "--device", default=None,
        help="Override device (cpu / cuda / mps).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.device:
        cfg["device"] = args.device

    train_saliency(
        data_root=cfg["data_root"],
        model_type=cfg.get("model_type", "resnet"),
        checkpoint_dir=cfg.get("checkpoint_dir", "checkpoints/saliency"),
        log_dir=cfg.get("log_dir", "outputs/saliency"),
        image_size=cfg.get("image_size", 224),
        sigma=cfg.get("heatmap_sigma", 20.0),
        frame_stride=cfg.get("frame_subsample", 10),
        min_observers=cfg.get("min_observers", 5),
        epochs=cfg.get("epochs", 20),
        batch_size=cfg.get("batch_size", 16),
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 1e-4),
        num_workers=cfg.get("num_workers", 2),
        seed=cfg.get("seed", 42),
        device=cfg.get("device"),
    )


if __name__ == "__main__":
    main()
