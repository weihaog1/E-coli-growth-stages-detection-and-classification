"""
Training script for YOLOv8.

Uses the Ultralytics library for training.
Expects data in YOLO format.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n.pt", help="Base model")
    parser.add_argument("--data", default="data/dataset.yaml", help="Dataset config")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--name", default="ecoli_yolo")
    args = parser.parse_args()
    
    # Load model (pretrained on COCO)
    model = YOLO(args.model)
    
    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project="runs/train",
        
        # Transfer learning settings
        freeze=10,  # Freeze first 10 layers
        
        # Optimization
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        
        # Augmentation (built-in)
        hsv_h=0.0,  # No hue (grayscale)
        hsv_s=0.0,  # No saturation
        hsv_v=0.2,  # Brightness
        degrees=45.0,
        flipud=0.5,
        fliplr=0.5,
        
        # Misc
        patience=20,
        save=True,
        plots=True,
    )
    
    print("Training complete")
    print(f"Best model saved to: runs/train/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()

