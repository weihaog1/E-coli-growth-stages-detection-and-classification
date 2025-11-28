"""
Data augmentation using Albumentations.

Albumentations is used because:
- It automatically transforms bounding boxes with images
- Faster than torchvision (OpenCV backend)
- Rich augmentation options for object detection
"""

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path


def get_train_transforms(img_size: int = 256):
    """
    Augmentation pipeline for training.
    Includes geometric and pixel-level transforms.
    """
    return A.Compose(
        [
            # Geometric
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(rotate=(-45, 45), scale=(0.8, 1.2), p=0.5),
            
            # Blur (one of)
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 5)),
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
            ], p=0.3),
            
            # Noise (one of)
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50)),
                A.ISONoise(),
            ], p=0.3),
            
            # Brightness/contrast
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            
            # Normalize and convert
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


def get_val_transforms(img_size: int = 256):
    """
    Transform pipeline for validation/testing.
    No augmentation, only normalization.
    """
    return A.Compose(
        [
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
        ),
    )


def get_patch_transforms(train: bool = True):
    """
    Transforms for patch classification (no bounding boxes).
    """
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.GaussNoise(var_limit=(10, 30), p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])


def augment_image(
    image_path: str,
    label_path: str,
    output_dir: str,
    num_copies: int = 5,
) -> None:
    """
    Create augmented copies of a single image and its labels.
    
    Saves to output_dir as {stem}_aug{i}.png and {stem}_aug{i}.txt
    """
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Load YOLO labels
    bboxes = []
    class_labels = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_labels.append(int(parts[0]))
                bboxes.append([float(x) for x in parts[1:5]])
    
    # Augmentation pipeline (no normalization for saving)
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(rotate=(-45, 45), p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    
    for i in range(num_copies):
        result = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        
        # Save image
        cv2.imwrite(str(output_path / f"{stem}_aug{i}.png"), result["image"])
        
        # Save labels
        with open(output_path / f"{stem}_aug{i}.txt", "w") as f:
            for cls, box in zip(result["class_labels"], result["bboxes"]):
                coords = " ".join(f"{v:.6f}" for v in box)
                f.write(f"{cls} {coords}\n")
