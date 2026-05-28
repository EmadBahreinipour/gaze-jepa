"""Train a saliency model (ResNet-18 or I-JEPA) on the FIND dataset.

Usage:
    python scripts/train_saliency.py --model resnet --data-root data/find_dataset
    python scripts/train_saliency.py --model ijepa  --data-root data/find_dataset --epochs 20
"""

from __future__ import annotations

import argparse

from gazejepa.saliency.training import train_saliency


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", default="resnet", choices=["resnet", "ijepa"],
        help="Which backbone to train.",
    )
    p.add_argument(
        "--data-root", default="data/find_dataset",
        help="FIND data root directory.",
    )
    p.add_argument(
        "--checkpoint-dir", default="gazejepa/checkpoints",
        help="Where to save the best checkpoint.",
    )
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--batch-size",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--frame-stride", type=int,   default=10)
    p.add_argument("--num-workers",  type=int,   default=2)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--device",       default=None,
                   help="cuda / mps / cpu (auto-detected if omitted).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    train_saliency(
        data_root=args.data_root,
        model_type=args.model,
        checkpoint_dir=args.checkpoint_dir,
        log_dir="outputs/saliency",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        frame_stride=args.frame_stride,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
