"""Train the patched SaccadeJEPA (GazeCropper swapped in) on FIND frames."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from gazejepa.data import (
    EXCLUDED_VIDEOS,
    get_frame_count,
    get_split,
    load_frame,
    resolve_data_root,
)
from gazejepa.jepa_reuse import make_gaze_jepa
from gazejepa.saliency import (
    CenterBiasSaliency,
    SpectralResidualSaliency,
    RandomSaliency,
    ResNetSaliency,
)


# Vendored from upstream utils.py / train.py to avoid pulling in the
# upstream repo's module-level YAML and ImageNet imports.


def ema_update(
    updated_model: torch.nn.Module,
    new_model: torch.nn.Module,
    ema_decay: float = 0.996,
) -> torch.nn.Module:
    for ema_param, new_param in zip(
        updated_model.parameters(), new_model.parameters()
    ):
        ema_param.data.copy_(
            ema_param.data * ema_decay + (1 - ema_decay) * new_param.data
        )
        ema_param.requires_grad_(False)
    return updated_model


class WarmUpScheduler:
    """Linear warmup followed by cosine annealing to ``min_lr``."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_iter: int,
        total_iter: int,
        min_lr: float = 1e-6,
    ):
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, total_iter - warmup_iter, eta_min=min_lr,
        )
        self.warmup_iter = max(1, warmup_iter)
        self.iter = 0

    def step(self) -> None:
        if self.iter < self.warmup_iter:
            target_lr = self.scheduler.get_last_lr()[0]
            lr = self.iter / self.warmup_iter * target_lr
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        else:
            self.scheduler.step()
        self.iter += 1


def saccade_loss(
    config: dict,
    model: torch.nn.Module,
    x: torch.Tensor,
    vicreg: bool = True,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Upstream ``saccade_loss`` for the ``predict_affines=False, use_cycle_consistency=True`` path."""
    target, context, target_pred, cycle_loss = model(x)

    loss = F.huber_loss(target_pred, target)
    loss = loss + cycle_loss * config["cycle_loss_weight"]

    if vicreg:
        context_mean = context.mean(dim=0, keepdim=True)
        variance_context = config["vicreg_gamma"] - (
            context.var(dim=0) + eps
        ).sqrt()
        variance = F.relu(variance_context).mean()
        cov_context = (context - context_mean).T @ (context - context_mean)
        cov_context = cov_context / context.shape[0]
        covariance = cov_context.triu().pow(2).sum() / cov_context.shape[0]
        loss = loss + (
            config["variance_weight"] * variance
            + config["covariance_weight"] * covariance
        )

    loss = loss / config["accumulation_steps"]
    return loss, target_pred


class FindFramesDataset(Dataset):
    """Evenly-spaced (video, frame) pairs from a FIND split as ``(3, image_size, image_size)`` float32 tensors in [0, 1].

    ``image_size`` must match the ``full_input_size`` passed to ``make_gaze_jepa``.
    """

    def __init__(
        self,
        data_root: str | Path,
        video_ids: list[str],
        image_size: int = 200,
        frames_per_video: int = 5,
        seed: int = 42,
    ):
        self.data_root = data_root
        self.image_size = int(image_size)
        rng = np.random.default_rng(seed)
        self.items: list[tuple[str, int]] = []
        for vid in video_ids:
            if vid in EXCLUDED_VIDEOS:
                continue
            try:
                n_frames = get_frame_count(data_root, vid)
            except Exception:
                continue
            if n_frames <= 0:
                continue
            n = min(int(frames_per_video), n_frames)
            base = np.linspace(0, n_frames - 1, n).astype(int)
            jitter = rng.integers(-2, 3, size=n)
            indices = np.clip(base + jitter, 0, n_frames - 1)
            for fi in indices:
                self.items.append((vid, int(fi)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> torch.Tensor:
        vid, fi = self.items[idx]
        frame = load_frame(self.data_root, vid, fi)  # (H, W, 3) uint8
        t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        t = F.interpolate(
            t.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return t


def build_saliency_source(args: argparse.Namespace):
    name = args.saliency
    if name == "resnet":
        ckpt = Path(args.resnet_checkpoint)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"--saliency resnet requires a trained checkpoint at "
                f"{ckpt} (override with --resnet-checkpoint)."
            )
        return ResNetSaliency.load(str(ckpt))
    if name == "spectral_residual":
        return SpectralResidualSaliency()
    if name == "center_bias":
        return CenterBiasSaliency(size=(args.image_size, args.image_size))
    if name == "random":
        return RandomSaliency()
    raise ValueError(f"unknown --saliency: {name}")


def pick_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(args: argparse.Namespace) -> dict:
    device = pick_device(args.device)
    print(f"Device: {device}")

    if args.data_root:
        data_root = resolve_data_root(args.data_root)
    else:
        data_root = resolve_data_root()
    print(f"FIND data root: {data_root}")

    split = get_split(str(data_root), seed=args.seed)
    train_videos = split["train"][: args.n_train_videos] if args.n_train_videos else split["train"]
    val_videos = split["val"][: args.n_val_videos] if args.n_val_videos else split["val"]
    print(f"Train videos: {len(train_videos)} | Val videos: {len(val_videos)}")

    train_ds = FindFramesDataset(
        data_root=str(data_root),
        video_ids=train_videos,
        image_size=args.image_size,
        frames_per_video=args.frames_per_video,
        seed=args.seed,
    )
    val_ds = FindFramesDataset(
        data_root=str(data_root),
        video_ids=val_videos,
        image_size=args.image_size,
        frames_per_video=args.frames_per_video,
        seed=args.seed + 1,
    )
    print(f"Train items: {len(train_ds)} | Val items: {len(val_ds)}")
    if len(train_ds) == 0:
        raise RuntimeError("Training dataset is empty — check --data-root.")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    saliency = build_saliency_source(args)
    print(f"Saliency source: {type(saliency).__name__}")
    model = make_gaze_jepa(
        saliency_source=saliency,
        full_input_size=(args.image_size, args.image_size),
        model_input_size=(args.crop_size, args.crop_size),
        n_fixations=args.n_fixations,
        ior_sigma=args.ior_sigma,
        ior_decay=args.ior_decay,
        sampling_mode="stochastic",
    ).to(device)
    print(
        f"Model: SaccadeJepa with GazeCropper, "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable params"
    )

    # Defaults from upstream config/training.yml.
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )
    total_iters = max(1, args.n_epochs * (len(train_loader) // args.accumulation_steps))
    warmup_iters = max(1, int(total_iters * args.warmup_fraction))
    scheduler = WarmUpScheduler(
        optimizer=optimizer, warmup_iter=warmup_iters,
        total_iter=total_iters, min_lr=args.min_lr,
    )

    config = {
        "cycle_loss_weight": args.cycle_loss_weight,
        "variance_weight": args.variance_weight,
        "covariance_weight": args.covariance_weight,
        "vicreg_gamma": args.vicreg_gamma,
        "accumulation_steps": args.accumulation_steps,
    }

    ckpt_dir = Path(args.output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(args.output_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    history: dict[str, list] = {
        "train_loss": [], "train_pred_mse": [],
        "val_pred_mse": [], "lr": [],
    }
    step = 0
    t0 = time.time()
    for epoch in range(args.n_epochs):
        model.train()
        epoch_losses: list[float] = []
        epoch_pred_mses: list[float] = []

        for batch_i, x in enumerate(train_loader):
            x = x.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss, target_pred = saccade_loss(
                config, model, x, vicreg=args.use_vicreg,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.clip_grad_norm,
            )
            optimizer.step()
            scheduler.step()

            model.target_encoder = ema_update(
                model.target_encoder, model.context_encoder,
                ema_decay=args.ema_decay,
            )

            with torch.no_grad():
                # Re-extract raw target for MSE logging (saccade_loss uses Huber).
                target, _, target_pred_log, _ = model(x)
                pred_mse = F.mse_loss(target_pred_log, target).item()
            epoch_losses.append(loss.item() * args.accumulation_steps)
            epoch_pred_mses.append(pred_mse)
            history["train_loss"].append(loss.item() * args.accumulation_steps)
            history["train_pred_mse"].append(pred_mse)
            history["lr"].append(optimizer.param_groups[0]["lr"])
            step += 1

            if batch_i % max(1, len(train_loader) // 10) == 0:
                elapsed = time.time() - t0
                print(
                    f"  epoch {epoch + 1}/{args.n_epochs} "
                    f"batch {batch_i + 1}/{len(train_loader)} "
                    f"loss={loss.item() * args.accumulation_steps:.4f} "
                    f"pred_mse={pred_mse:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e} "
                    f"elapsed={elapsed:.0f}s"
                )

        model.eval()
        val_mses: list[float] = []
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                target, _, target_pred, _ = model(x)
                val_mses.append(F.mse_loss(target_pred, target).item())
        val_mse = float(np.mean(val_mses)) if val_mses else float("nan")
        history["val_pred_mse"].append(val_mse)

        train_loss_mean = float(np.mean(epoch_losses))
        train_mse_mean = float(np.mean(epoch_pred_mses))
        print(
            f"== epoch {epoch + 1}/{args.n_epochs} done — "
            f"train_loss={train_loss_mean:.4f} "
            f"train_pred_mse={train_mse_mean:.4f} "
            f"val_pred_mse={val_mse:.4f} =="
        )

        ckpt_path = ckpt_dir / f"epoch_{epoch + 1:03d}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "args": vars(args),
                "history": history,
            },
            ckpt_path,
        )
        print(f"  saved {ckpt_path}")

    final_path = ckpt_dir / "final.pt"
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, final_path)
    print(f"Final checkpoint: {final_path}")

    history_path = metrics_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2))
    print(f"Loss history: {history_path}")

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history["train_loss"], label="train_loss")
        axes[0].plot(history["train_pred_mse"], label="train_pred_mse")
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("loss")
        axes[0].set_title("Training loss")
        axes[0].legend()
        axes[1].plot(history["val_pred_mse"], "o-", label="val_pred_mse")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("MSE")
        axes[1].set_title("Validation prediction MSE")
        axes[1].legend()
        fig.tight_layout()
        loss_path = metrics_dir / "loss_curve.png"
        fig.savefig(loss_path, dpi=120, bbox_inches="tight")
        print(f"Loss curve: {loss_path}")
    except Exception as e:  # pragma: no cover
        print(f"(loss-curve plot skipped: {e})")

    return history


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Train the patched SaccadeJepa (GazeCropper-substituted) on "
            "FIND. Defaults are small for a quick demo; raise --n-epochs, "
            "--frames-per-video, --batch-size for a real run with GPU."
        ),
    )
    p.add_argument("--data-root", default=None,
                   help="FIND data root (else $FIND_DATA_ROOT or repo default).")
    p.add_argument("--n-train-videos", type=int, default=0,
                   help="Cap train videos (0 = all 50).")
    p.add_argument("--n-val-videos", type=int, default=0,
                   help="Cap val videos (0 = all 6).")
    p.add_argument("--frames-per-video", type=int, default=5,
                   help="Frames sampled per video per epoch.")
    p.add_argument("--image-size", type=int, default=200,
                   help="Full image size fed to the model.")
    p.add_argument("--crop-size", type=int, default=128,
                   help="Per-fixation crop size (encoder input).")

    p.add_argument("--n-fixations", type=int, default=5)
    p.add_argument("--ior-sigma", type=float, default=20.0)
    p.add_argument("--ior-decay", type=float, default=0.7)

    p.add_argument(
        "--saliency",
        default="resnet",
        choices=["resnet", "spectral_residual", "center_bias", "random"],
        help=(
            "Saliency source for GazeCropper. Default 'resnet' loads the "
            "trained ResNetSaliency checkpoint (best AUC on FIND); the "
            "other options are for ablations or fallbacks."
        ),
    )
    p.add_argument(
        "--resnet-checkpoint",
        default="checkpoints/resnet_saliency_best.pt",
        help="Path to ResNetSaliency checkpoint when --saliency resnet.",
    )

    p.add_argument("--n-epochs", type=int, default=3,
                   help="Default 3 keeps the demo short; raise for a real run.")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accumulation-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--warmup-fraction", type=float, default=0.1,
                   help="Fraction of total iters used for LR warmup.")
    p.add_argument("--clip-grad-norm", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.996)

    # Loss weights (defaults from upstream config/training.yml).
    p.add_argument("--cycle-loss-weight", type=float, default=0.25)
    p.add_argument("--variance-weight", type=float, default=25.0)
    p.add_argument("--covariance-weight", type=float, default=1.0)
    p.add_argument("--vicreg-gamma", type=float, default=4.0)
    p.add_argument("--no-vicreg", dest="use_vicreg", action="store_false")
    p.set_defaults(use_vicreg=True)

    p.add_argument("--device", default="auto",
                   help="auto | cpu | cuda | mps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="outputs/jepa_reuse",
                   help="Where checkpoints/ and metrics/ go.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
