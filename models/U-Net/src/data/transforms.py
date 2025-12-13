"""
Data augmentation transforms for U-Net segmentation.

Uses Albumentations for synchronized image+mask transforms.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def get_train_transforms():
    """
    Training transforms with augmentation.

    Applies geometric transforms that are synchronized for image and mask.
    Does NOT include normalization (handled in dataset).
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.9, 1.1),
            rotate=(-45, 45),
            p=0.5
        ),
        # Pixel-level augmentation (image only, not mask)
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MedianBlur(blur_limit=5),
        ], p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
    ])


def get_val_transforms():
    """
    Validation/test transforms.

    No augmentation, just ensures consistent format.
    """
    return None  # No transforms needed


def get_normalize_transform():
    """
    Normalization transform for images.

    Converts to tensor and normalizes to [-1, 1].
    """
    return A.Compose([
        A.Normalize(mean=[0.5], std=[0.5], max_pixel_value=255.0),
        ToTensorV2(),
    ])


class SegmentationTransform:
    """
    Combined transform for segmentation that handles both image and mask.

    Applies augmentation, then normalization and tensor conversion.
    """

    def __init__(self, augment: bool = True):
        self.augment = augment

        if augment:
            self.aug_transform = get_train_transforms()
        else:
            self.aug_transform = None

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        """
        Apply transforms to image and mask.

        Args:
            image: Grayscale image (H, W) or (H, W, 1), values 0-255
            mask: Segmentation mask (H, W), values 0-3

        Returns:
            dict with 'image' (tensor) and 'mask' (tensor)
        """
        # Ensure image is 2D for albumentations
        if image.ndim == 3:
            image = image[:, :, 0]

        # Apply augmentation
        if self.aug_transform is not None:
            transformed = self.aug_transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Convert image to tensor and normalize
        import torch

        # Normalize to [-1, 1]
        image = torch.from_numpy(image.copy()).float().unsqueeze(0) / 255.0
        image = (image - 0.5) / 0.5

        # Convert mask to tensor (no normalization)
        mask = torch.from_numpy(mask.copy()).long()

        return {"image": image, "mask": mask}


def collate_fn(batch):
    """
    Custom collate function for segmentation batches.

    Stacks images and masks into batched tensors.
    """
    import torch

    images = torch.stack([item["image"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])
    filenames = [item["filename"] for item in batch]

    return {
        "image": images,
        "mask": masks,
        "filename": filenames
    }
