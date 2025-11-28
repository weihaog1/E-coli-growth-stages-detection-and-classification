"""
PyTorch Dataset classes.

Two dataset types:
1. EcoliDataset: For object detection (YOLO-style)
2. EcoliPatchDataset: For patch classification (sliding window CNN)
"""

from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .convert_annotations import compute_iou


class EcoliDataset(Dataset):
    """
    Object detection dataset.
    
    Directory structure expected:
        images_dir/
            image1.png
            image2.png
        labels_dir/
            image1.txt  (YOLO format)
            image2.txt
    """
    
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        transform: Optional[Callable] = None,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        
        self.image_files = sorted(
            list(self.images_dir.glob("*.png")) +
            list(self.images_dir.glob("*.jpg"))
        )
        
        print(f"Loaded {len(self.image_files)} images from {images_dir}")
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_files[idx]
        
        # Load grayscale, convert to 3-channel for YOLO compatibility
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        image = np.stack([image, image, image], axis=-1)
        
        # Load YOLO labels
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        bboxes = []
        class_labels = []
        
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_labels.append(int(parts[0]))
                        bboxes.append([float(x) for x in parts[1:5]])
        
        # Apply transforms
        if self.transform:
            result = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            image = result["image"]
            bboxes = result["bboxes"]
            class_labels = result["class_labels"]
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        # Convert lists to tensors
        if bboxes:
            bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
            labels_tensor = torch.tensor(class_labels, dtype=torch.long)
        else:
            bboxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros(0, dtype=torch.long)
        
        return {
            "image": image,
            "bboxes": bboxes_tensor,
            "labels": labels_tensor,
            "path": str(img_path),
        }


class EcoliPatchDataset(Dataset):
    """
    Patch classification dataset for sliding window CNN.
    
    Extracts fixed-size patches from images.
    Each patch is labeled based on IoU overlap with ground truth boxes.
    Class 3 = background (no significant overlap).
    """
    
    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        transform: Optional[Callable] = None,
        patch_size: int = 64,
        stride: int = 32,
        iou_threshold: float = 0.3,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
        self.iou_threshold = iou_threshold
        
        self.patches = self._build_patch_list()
        print(f"Created {len(self.patches)} patches")
    
    def _build_patch_list(self) -> list:
        """
        Pre-compute all patch locations and their labels.
        """
        patches = []
        
        image_files = sorted(
            list(self.images_dir.glob("*.png")) +
            list(self.images_dir.glob("*.jpg"))
        )
        
        for img_path in image_files:
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            h, w = image.shape
            
            # Load and convert YOLO labels to pixel coords
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            objects = self._load_labels(label_path, w, h)
            
            # Generate patches with sliding window
            for y in range(0, h - self.patch_size + 1, self.stride):
                for x in range(0, w - self.patch_size + 1, self.stride):
                    patch_box = (x, y, x + self.patch_size, y + self.patch_size)
                    class_id = self._get_patch_class(patch_box, objects)
                    
                    patches.append({
                        "path": str(img_path),
                        "x": x,
                        "y": y,
                        "class_id": class_id,
                    })
        
        return patches
    
    def _load_labels(self, label_path: Path, img_w: int, img_h: int) -> list:
        """
        Load YOLO labels and convert to pixel coordinates.
        """
        objects = []
        
        if not label_path.exists():
            return objects
        
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                cls = int(parts[0])
                cx, cy, bw, bh = [float(x) for x in parts[1:5]]
                
                # YOLO to pixel coords
                x1 = (cx - bw / 2) * img_w
                y1 = (cy - bh / 2) * img_h
                x2 = (cx + bw / 2) * img_w
                y2 = (cy + bh / 2) * img_h
                
                objects.append({"class_id": cls, "bbox": (x1, y1, x2, y2)})
        
        return objects
    
    def _get_patch_class(self, patch_box: tuple, objects: list) -> int:
        """
        Determine class for a patch based on IoU with objects.
        Returns 3 (background) if no significant overlap.
        """
        best_iou = 0.0
        best_class = 3
        
        for obj in objects:
            iou = compute_iou(patch_box, obj["bbox"])
            if iou > best_iou and iou > self.iou_threshold:
                best_iou = iou
                best_class = obj["class_id"]
        
        return best_class
    
    def __len__(self) -> int:
        return len(self.patches)
    
    def __getitem__(self, idx: int) -> dict:
        info = self.patches[idx]
        
        # Extract patch from image
        image = cv2.imread(info["path"], cv2.IMREAD_GRAYSCALE)
        x, y = info["x"], info["y"]
        patch = image[y : y + self.patch_size, x : x + self.patch_size]
        
        # Apply transforms
        if self.transform:
            result = self.transform(image=patch)
            patch = result["image"]
        else:
            patch = torch.from_numpy(patch).unsqueeze(0).float() / 255.0
        
        return {
            "image": patch,
            "label": torch.tensor(info["class_id"], dtype=torch.long),
        }
