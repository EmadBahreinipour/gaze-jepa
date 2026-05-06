"""CLI entry point for SaliencyNet training.

Usage:
    python scripts/train_saliency.py --config configs/saliency_train.yml
    python scripts/train_saliency.py --config configs/saliency_train.yml --data_root /path/to/find
"""

import argparse
import sys
import yaml

sys.path.insert(0, ".")
from src.saliency.train import train_saliency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/saliency_train.yml")
    parser.add_argument("--data_root", default=None,
                        help="Overrides data_root in config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.data_root:
        cfg["data_root"] = args.data_root

    train_saliency(
        data_root=cfg["data_root"],
        model_type=cfg.get("model_type", "resnet"),
        checkpoint_dir=cfg.get("checkpoint_dir", "checkpoints"),
        log_dir=cfg.get("log_dir", "outputs"),
        image_size=cfg.get("image_size", 224),
        sigma=cfg.get("heatmap_sigma", 20.0),
        frame_stride=cfg.get("frame_subsample", 10),
        min_observers=cfg.get("min_observers", 5),
        epochs=cfg.get("epochs", 20),
        batch_size=cfg.get("batch_size", 16),
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 1e-4),
        num_workers=cfg.get("num_workers", 2),
    )


if __name__ == "__main__":
    main()
