"""Training script for saliency models on the FIND dataset.

Loss: KL divergence KL(gt || pred) computed at the model's native low
resolution (14x14 for ResNet, 112x112 for the simple CNN). This gives
the model a clean learning signal — KL over 196 bins is tractable,
KL over 50176 bins (full 224x224) is not.

Usage (from repo root):
    python scripts/train_saliency.py --config configs/saliency_train.yml
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from .learned import FINDSaliencyDataset
from src.metrics.saliency_metrics import auc_judd, nss, cc


def evaluate_full(model, loader: DataLoader, device: torch.device) -> dict:
    """Compute AUC, NSS, CC over all frames in a dataloader."""
    model.eval()
    aucs, nsses, ccs = [], [], []

    with torch.no_grad():
        for images, gt_heatmaps in loader:
            images = images.to(device)
            preds = model.predict(images).cpu().numpy()  # (B, 224, 224)
            gts = gt_heatmaps.numpy()

            for i in range(preds.shape[0]):
                gt = gts[i]
                threshold = gt.mean() + gt.std()
                ys, xs = np.where(gt > threshold)
                if len(ys) == 0:
                    continue
                fixations = np.stack([xs, ys], axis=1)
                aucs.append(auc_judd(preds[i], fixations))
                nsses.append(nss(preds[i], fixations))
                ccs.append(cc(preds[i], gt))

    model.train()
    return {
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "nss_mean": float(np.mean(nsses)), "nss_std": float(np.std(nsses)),
        "cc_mean":  float(np.mean(ccs)),  "cc_std":  float(np.std(ccs)),
        "n_frames": len(aucs),
    }


def _kl_loss(logits: torch.Tensor, gt_heatmaps: torch.Tensor) -> torch.Tensor:
    """KL(gt || pred) at the model's native output resolution.

    logits: (B, H, W) raw model output at low resolution
    gt_heatmaps: (B, 224, 224) ground truth, will be downsampled to match
    """
    B, H, W = logits.shape
    pred_flat = logits.view(B, -1).softmax(dim=1)

    gt_small = F.interpolate(gt_heatmaps.unsqueeze(1), size=(H, W),
                             mode="bilinear", align_corners=False).squeeze(1)
    gt_small = gt_small / gt_small.sum(dim=(1, 2), keepdim=True).clamp(min=1e-8)

    return -(gt_small.view(B, -1) * (pred_flat + 1e-8).log()).sum(dim=1).mean()


def train_saliency(data_root: str,
                   model_type: str = "resnet",
                   checkpoint_dir: str = "checkpoints",
                   log_dir: str = "outputs",
                   image_size: int = 224,
                   sigma: float = 20.0,
                   frame_stride: int = 10,
                   min_observers: int = 5,
                   epochs: int = 20,
                   batch_size: int = 16,
                   lr: float = 1e-3,
                   weight_decay: float = 1e-4,
                   num_workers: int = 2):

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Build datasets (uses cached index if available)
    train_ds = FINDSaliencyDataset(data_root, "train", image_size, sigma, frame_stride, min_observers)
    val_ds   = FINDSaliencyDataset(data_root, "val",   image_size, sigma, frame_stride, min_observers)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    print(f"Train frames: {len(train_ds)}, Val frames: {len(val_ds)}")

    # Model selection
    if model_type == "resnet":
        from .resnet_saliency import ResNetSaliency
        model = ResNetSaliency(pretrained=True, freeze_backbone=True).to(device)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: ResNetSaliency (pretrained backbone frozen, {n_trainable:,} trainable params)")
    else:
        from .learned import LearnedSaliency
        model = LearnedSaliency().to(device)
        print(f"Model: LearnedSaliency ({sum(p.numel() for p in model.parameters()):,} params)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, f"{model_type}_saliency_best.pt")
    log_path  = os.path.join(log_dir, f"{model_type}_training_log.json")

    best_val_auc = 0.0
    log = []

    for epoch in range(epochs):
        model.train()
        epoch_losses = []

        for images, gt_heatmaps in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images      = images.to(device)
            gt_heatmaps = gt_heatmaps.to(device)

            # Get raw logits at model's native low resolution
            if model_type == "resnet":
                logits = model.forward_logits(images)   # (B, 14, 14)
            else:
                logits = model.forward(images)           # (B, 112, 112)

            loss = _kl_loss(logits, gt_heatmaps)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()
        train_loss = float(np.mean(epoch_losses))
        val_metrics = evaluate_full(model, val_loader, device)
        val_auc = val_metrics["auc_mean"]

        entry = {"epoch": epoch + 1, "train_loss": train_loss, **val_metrics,
                 "lr": scheduler.get_last_lr()[0]}
        log.append(entry)
        print(f"  loss={train_loss:.4f}  val_auc={val_auc:.4f}  "
              f"val_nss={val_metrics['nss_mean']:.4f}  val_cc={val_metrics['cc_mean']:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            model.save(save_path)
            print(f"  *** New best val AUC={best_val_auc:.4f}, checkpoint saved to {save_path} ***")

        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

    print(f"\nTraining complete. Best val AUC: {best_val_auc:.4f}")
    print(f"Checkpoint: {save_path}")
    return model
