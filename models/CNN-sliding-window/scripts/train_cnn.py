"""
Training script for the patch CNN classifier.

Uses the prepared CNN patches from data/cnn_patches/ (created by prepare_cnn_data.py).
Expects patches organized in ImageFolder structure:
    data/cnn_patches/{train,val,test}/{rod,dividing,microcolony,background}/
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models import PatchCNN
from utils import set_seed, load_config, get_device, compute_metrics


# =============================================================================
# TRANSFORMS
# =============================================================================

def get_train_transforms():
    """Training transforms with augmentation."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(45),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1, 1]
    ])


def get_val_transforms():
    """Validation transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    metrics["loss"] = total_loss / len(all_labels)

    return metrics


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train CNN patch classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", default="configs/config.yaml", help="Config file")
    parser.add_argument("--data-dir", default="../../data/cnn_patches", help="Patch data directory (relative to models/CNN-sliding-window/)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    args = parser.parse_args()

    # Setup
    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = get_device()
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)

    # Create datasets using ImageFolder
    print(f"\nLoading patches from: {data_dir}")
    train_dataset = datasets.ImageFolder(
        root=data_dir / "train",
        transform=get_train_transforms()
    )
    val_dataset = datasets.ImageFolder(
        root=data_dir / "val",
        transform=get_val_transforms()
    )

    # Print class mapping and counts
    print(f"\nClass mapping: {train_dataset.class_to_idx}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Count samples per class
    train_targets = torch.tensor(train_dataset.targets)
    val_targets = torch.tensor(val_dataset.targets)
    for class_name, class_idx in train_dataset.class_to_idx.items():
        train_count = (train_targets == class_idx).sum().item()
        val_count = (val_targets == class_idx).sum().item()
        print(f"  {class_name}: train={train_count}, val={val_count}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Windows compatibility
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Model
    num_classes = len(train_dataset.classes)
    model = PatchCNN(num_classes=num_classes).to(device)
    print(f"\nModel: PatchCNN with {num_classes} classes")

    # Loss with class weighting (to handle imbalance)
    class_counts = torch.bincount(train_targets)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * num_classes  # Normalize
    class_weights = class_weights.to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    best_f1 = 0
    weights_dir = Path("weights")
    weights_dir.mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"  Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        val_metrics = evaluate(model, val_loader, criterion, device)
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Precision: {val_metrics['precision']:.4f}, "
              f"Recall: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}")

        scheduler.step()

        # Save best model
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_f1": best_f1,
                "class_to_idx": train_dataset.class_to_idx,
            }, weights_dir / "best_cnn.pth")
            print(f"  --> Saved best model (F1: {best_f1:.4f})")

    print("\n" + "=" * 60)
    print(f"Training complete. Best F1: {best_f1:.4f}")
    print(f"Model saved to: {weights_dir / 'best_cnn.pth'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
