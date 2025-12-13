"""
Training script for U-Net semantic segmentation.

Uses prepared masks from data/unet_original or data/unet_augmented.

Usage (run from models/U-Net/ directory):
    python scripts/train_unet.py --data-dir ../../data/unet_original --epochs 100
    python scripts/train_unet.py --data-dir ../../data/unet_augmented --epochs 100
"""

# Fix OpenMP duplicate library error on Windows
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import UNet, UNetSmall
from data import EcoliSegmentationDataset, get_train_transforms, collate_fn
from utils import (
    set_seed, load_config, get_device, count_parameters,
    save_checkpoint, SegmentationMetrics
)


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device, metrics_tracker):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    metrics_tracker.reset()

    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        # Update metrics
        preds = outputs.argmax(dim=1).cpu().numpy()
        targets = masks.cpu().numpy()
        metrics_tracker.update(preds, targets)

    avg_loss = total_loss / len(loader.dataset)
    metrics = metrics_tracker.compute()

    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, metrics_tracker):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0
    metrics_tracker.reset()

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        total_loss += loss.item() * images.size(0)

        # Update metrics
        preds = outputs.argmax(dim=1).cpu().numpy()
        targets = masks.cpu().numpy()
        metrics_tracker.update(preds, targets)

    avg_loss = total_loss / len(loader.dataset)
    metrics = metrics_tracker.compute()

    return avg_loss, metrics


# =============================================================================
# DATASET WRAPPER WITH TRANSFORMS
# =============================================================================

class TransformedDataset(torch.utils.data.Dataset):
    """Wrapper to apply transforms to segmentation dataset."""

    def __init__(self, dataset, transform=None, augment=True):
        self.dataset = dataset
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        mask = item["mask"]

        # Convert tensors back to numpy for transforms
        if isinstance(image, torch.Tensor):
            image = image.squeeze(0).numpy()
            image = ((image * 0.5 + 0.5) * 255).astype(np.uint8)
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()

        # Apply augmentation
        if self.transform is not None and self.augment:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Convert to tensors and normalize
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.copy()).float().unsqueeze(0) / 255.0
            image = (image - 0.5) / 0.5

        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask.copy()).long()

        return {
            "image": image,
            "mask": mask,
            "filename": item.get("filename", f"image_{idx}.png")
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train U-Net for E. coli segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data-dir", required=True, help="Data directory with train/val/test subdirs")
    parser.add_argument("--output-dir", default=None, help="Output directory for results")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--model", choices=["unet", "unet_small"], default="unet_small", help="Model architecture")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs")
    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)

    # Determine output directory (relative to models/U-Net/)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Auto-determine based on data directory name
        if "augmented" in str(data_dir).lower():
            output_dir = Path("results/augmented")
        else:
            output_dir = Path("results/original")

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = Path("weights")
    weights_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = EcoliSegmentationDataset(data_dir / "train")
    val_dataset = EcoliSegmentationDataset(data_dir / "val")

    # Wrap with transforms
    train_transform = get_train_transforms()
    train_dataset_aug = TransformedDataset(train_dataset, train_transform, augment=True)
    val_dataset_aug = TransformedDataset(val_dataset, None, augment=False)

    print(f"Train samples: {len(train_dataset_aug)}")
    print(f"Val samples:   {len(val_dataset_aug)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset_aug,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset_aug,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn
    )

    # Model
    if args.model == "unet":
        model = UNet(n_channels=1, n_classes=4).to(device)
    else:
        model = UNetSmall(n_channels=1, n_classes=4).to(device)

    print(f"\nModel: {args.model}")
    print(f"Parameters: {count_parameters(model):,}")

    # Compute class weights from training masks
    print("\nComputing class weights...")
    class_counts = np.zeros(4)
    for batch in tqdm(train_loader, desc="Counting classes", leave=False):
        masks = batch["mask"].numpy()
        for cls in range(4):
            class_counts[cls] += (masks == cls).sum()

    # Inverse frequency weighting
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * 4  # Normalize
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )

    # Metrics tracker
    train_metrics = SegmentationMetrics(num_classes=4)
    val_metrics = SegmentationMetrics(num_classes=4)

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_miou": [],
        "val_miou": [],
        "train_miou_no_bg": [],
        "val_miou_no_bg": [],
        "val_iou_per_class": [],
        "lr": []
    }

    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    best_miou = 0
    model_name = "augmented" if "augmented" in str(data_dir).lower() else "original"

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss, train_m = train_one_epoch(model, train_loader, criterion, optimizer, device, train_metrics)
        print(f"  Train - Loss: {train_loss:.4f}, mIoU: {train_m['mean_iou']:.4f}, mIoU (no bg): {train_m['mean_iou_no_bg']:.4f}")

        # Validate
        val_loss, val_m = evaluate(model, val_loader, criterion, device, val_metrics)
        print(f"  Val   - Loss: {val_loss:.4f}, mIoU: {val_m['mean_iou']:.4f}, mIoU (no bg): {val_m['mean_iou_no_bg']:.4f}")

        # Per-class IoU
        print(f"  IoU per class: ", end="")
        for name, iou in val_m['iou_per_class'].items():
            if not np.isnan(iou):
                print(f"{name[:3]}={iou:.3f} ", end="")
        print()

        # Update scheduler
        scheduler.step(val_m['mean_iou_no_bg'])

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_miou"].append(train_m['mean_iou'])
        history["val_miou"].append(val_m['mean_iou'])
        history["train_miou_no_bg"].append(train_m['mean_iou_no_bg'])
        history["val_miou_no_bg"].append(val_m['mean_iou_no_bg'])
        history["val_iou_per_class"].append(val_m['iou_per_class'])
        history["lr"].append(optimizer.param_groups[0]['lr'])

        # Save best model
        if val_m['mean_iou_no_bg'] > best_miou:
            best_miou = val_m['mean_iou_no_bg']
            save_checkpoint(
                model, optimizer, epoch,
                {"mean_iou": val_m['mean_iou'], "mean_iou_no_bg": val_m['mean_iou_no_bg']},
                weights_dir / f"best_unet_{model_name}.pth"
            )
            print(f"  --> Saved best model (mIoU no bg: {best_miou:.4f})")

        # Periodic save
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch,
                {"mean_iou": val_m['mean_iou']},
                weights_dir / f"unet_{model_name}_epoch{epoch+1}.pth"
            )

    # Save final model
    save_checkpoint(
        model, optimizer, args.epochs - 1,
        {"mean_iou": val_m['mean_iou']},
        weights_dir / f"final_unet_{model_name}.pth"
    )

    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        # Convert numpy types to Python types
        serializable_history = {}
        for key, values in history.items():
            if key == "val_iou_per_class":
                serializable_history[key] = [
                    {k: float(v) if not np.isnan(v) else None for k, v in d.items()}
                    for d in values
                ]
            else:
                serializable_history[key] = [float(v) for v in values]
        json.dump(serializable_history, f, indent=2)

    # Final summary
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Best mIoU (no background): {best_miou:.4f}")
    print(f"Model saved to: {weights_dir / f'best_unet_{model_name}.pth'}")
    print(f"History saved to: {history_path}")

    # Print final per-class performance
    print("\nFinal Per-Class IoU:")
    for name, iou in val_m['iou_per_class'].items():
        if not np.isnan(iou):
            print(f"  {name:12}: {iou:.4f}")


if __name__ == "__main__":
    main()
