"""
Training script for the patch CNN classifier.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import EcoliPatchDataset, get_patch_transforms
from src.models import PatchCNN
from src.utils import set_seed, load_config, get_device, compute_metrics


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(loader, desc="Training"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
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
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(loader, desc="Evaluating"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item() * images.size(0)
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    import numpy as np
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    metrics["loss"] = total_loss / len(all_labels)
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    
    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = get_device()
    print(f"Using device: {device}")
    
    # Datasets
    train_dataset = EcoliPatchDataset(
        images_dir="data/processed/images/train",
        labels_dir="data/processed/labels/train",
        transform=get_patch_transforms(train=True),
    )
    val_dataset = EcoliPatchDataset(
        images_dir="data/processed/images/val",
        labels_dir="data/processed/labels/val",
        transform=get_patch_transforms(train=False),
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Model
    model = PatchCNN(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_f1 = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        
        val_metrics = evaluate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        scheduler.step()
        
        # Save best model
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), "weights/best_cnn.pth")
            print(f"Saved best model (F1: {best_f1:.4f})")
    
    print(f"\nTraining complete. Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    Path("weights").mkdir(exist_ok=True)
    main()

