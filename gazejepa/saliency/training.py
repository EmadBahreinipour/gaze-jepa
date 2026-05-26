"""Training driver for saliency models on FIND.

Supports ResNetSaliency (14×14 output grid), IJepaSaliency (16×16), and
LearnedSaliency. KL(GT ∥ pred) is computed at the model's native low
resolution — keeping the loss in the few-hundred-bin regime where the
gradient is well-conditioned.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from gazejepa.evaluation.saliency_metrics import auc_score, cc_score, nss_score
from gazejepa.saliency.dataset import FINDSaliencyDataset

if TYPE_CHECKING:
    from gazejepa.saliency.ijepa_saliency import IJepaSaliency
    from gazejepa.saliency.learned import LearnedSaliency
    from gazejepa.saliency.resnet_saliency import ResNetSaliency


def _kl_loss(logits: torch.Tensor, gt_heatmaps: torch.Tensor) -> torch.Tensor:
    """KL(gt || softmax(logits)) at logits' resolution."""
    b, h, w = logits.shape
    pred = logits.view(b, -1).softmax(dim=1)

    gt_small = F.interpolate(
        gt_heatmaps.unsqueeze(1), size=(h, w),
        mode="bilinear", align_corners=False,
    ).squeeze(1)
    gt_small = gt_small / gt_small.view(b, -1).sum(dim=1).clamp(min=1e-8).view(b, 1, 1)

    return -(gt_small.view(b, -1) * (pred + 1e-8).log()).sum(dim=1).mean()


def _evaluate(
    model: "ResNetSaliency | IJepaSaliency | LearnedSaliency",
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """AUC / NSS / CC averaged over a dataloader at full input resolution."""
    was_training = model.training
    model.eval()
    aucs: list[float] = []
    nsses: list[float] = []
    ccs: list[float] = []
    with torch.no_grad():
        for images, gt_heatmaps in loader:
            images = images.to(device)
            preds = model(images).cpu().numpy()  # (B, H, W) sum-to-1
            gts = gt_heatmaps.numpy()
            for i in range(preds.shape[0]):
                gt = gts[i]
                threshold = gt.mean() + gt.std()
                ys, xs = np.where(gt > threshold)
                if len(ys) == 0:
                    continue
                fixations = np.stack([xs, ys], axis=1).astype(np.float64)
                aucs.append(auc_score(preds[i], fixations))
                nsses.append(nss_score(preds[i], fixations))
                ccs.append(cc_score(preds[i], gt))
    if was_training:
        model.train()
    return {
        "n_frames": len(aucs),
        "auc_mean": float(np.mean(aucs)) if aucs else 0.0,
        "auc_std":  float(np.std(aucs))  if aucs else 0.0,
        "nss_mean": float(np.mean(nsses)) if nsses else 0.0,
        "nss_std":  float(np.std(nsses))  if nsses else 0.0,
        "cc_mean":  float(np.mean(ccs))  if ccs  else 0.0,
        "cc_std":   float(np.std(ccs))   if ccs  else 0.0,
    }


def train_saliency(
    data_root: str | os.PathLike[str],
    *,
    model_type: str = "resnet",
    checkpoint_dir: str | os.PathLike[str] = "checkpoints/saliency",
    log_dir: str | os.PathLike[str] = "outputs/saliency",
    image_size: int = 224,
    sigma: float = 20.0,
    frame_stride: int = 10,
    min_observers: int = 5,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 2,
    seed: int = 42,
    device: torch.device | str | None = None,
) -> "ResNetSaliency | IJepaSaliency | LearnedSaliency":
    """Train a saliency predictor on FIND. Returns the in-memory model."""
    if model_type not in {"resnet", "ijepa", "learned"}:
        raise ValueError(f"model_type must be 'resnet', 'ijepa', or 'learned', got {model_type!r}")

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(device)

    train_ds = FINDSaliencyDataset(
        data_root, "train", image_size, sigma, frame_stride, min_observers, seed=seed,
    )
    val_ds = FINDSaliencyDataset(
        data_root, "val", image_size, sigma, frame_stride, min_observers, seed=seed,
    )
    train_loader = DataLoader(
        train_ds, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
    )

    if model_type == "resnet":
        from gazejepa.saliency.resnet_saliency import ResNetSaliency
        model: "ResNetSaliency | IJepaSaliency | LearnedSaliency" = ResNetSaliency(
            pretrained=True, freeze_backbone=True,
        ).to(device)
    elif model_type == "ijepa":
        from gazejepa.saliency.ijepa_saliency import IJepaSaliency
        model = IJepaSaliency(pretrained=True).to(device)
    else:
        from gazejepa.saliency.learned import LearnedSaliency
        model = LearnedSaliency().to(device)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    checkpoint_dir = Path(checkpoint_dir)
    log_dir = Path(log_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / f"{model_type}_saliency_best.pt"
    log_path = log_dir / f"{model_type}_training_log.json"

    best_val_auc = 0.0
    log: list[dict] = []

    for epoch in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        for images, gt_heatmaps in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images = images.to(device)
            gt_heatmaps = gt_heatmaps.to(device)
            # Dataset yields raw [0,1]. ResNet/I-JEPA forward_logits expects
            # ImageNet-normalised input; LearnedSaliency normalises internally.
            if model_type in {"resnet", "ijepa"}:
                images_in = (images - model._imagenet_mean) / model._imagenet_std
            else:
                images_in = images
            logits = model.forward_logits(images_in)
            loss = _kl_loss(logits, gt_heatmaps)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        scheduler.step()
        val_metrics = _evaluate(model, val_loader, device)
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")

        entry = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "lr": float(scheduler.get_last_lr()[0]),
            **val_metrics,
        }
        log.append(entry)
        print(
            f"  loss={train_loss:.4f}  val_auc={val_metrics['auc_mean']:.4f}  "
            f"val_nss={val_metrics['nss_mean']:.4f}  val_cc={val_metrics['cc_mean']:.4f}"
        )

        if val_metrics["auc_mean"] > best_val_auc:
            best_val_auc = val_metrics["auc_mean"]
            model.save(str(save_path))

        with log_path.open("w") as fh:
            json.dump(log, fh, indent=2)

    print(f"\nTraining complete. Best val AUC: {best_val_auc:.4f}")
    print(f"Checkpoint: {save_path}")
    return model
