"""CLI entry point for the four-source saliency comparison on a FIND split.

Usage:
    python scripts/evaluate_saliency.py
    python scripts/evaluate_saliency.py --split val --device cuda \
        --resnet-checkpoint checkpoints/saliency/resnet_saliency_best.pt
"""

from __future__ import annotations

import argparse

from gazejepa.evaluation.saliency_comparison import run_full_comparison


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-root", default="data/find_dataset",
        help="FIND data root.",
    )
    p.add_argument(
        "--split", default="test", choices=["train", "val", "test"],
        help="Split to evaluate on (canonical seed=42 split).",
    )
    p.add_argument(
        "--image-size", type=int, default=224,
        help="Square frame size all sources are evaluated at.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Split seed shared with training.",
    )
    p.add_argument(
        "--resnet-checkpoint",
        default="checkpoints/saliency/resnet_saliency_best.pt",
        help="ResNetSaliency checkpoint; skipped if missing.",
    )
    p.add_argument(
        "--output-csv",
        default="report_arash/data/saliency_comparison.csv",
    )
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_full_comparison(
        args.data_root,
        resnet_checkpoint=args.resnet_checkpoint,
        output_csv=args.output_csv,
        split=args.split,
        image_size=args.image_size,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
