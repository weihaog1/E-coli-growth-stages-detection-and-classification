"""
Dataset classes for U-Net segmentation training.

Loads paired image and mask files for semantic segmentation.
"""

from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class EcoliSegmentationDataset(Dataset):
    """
    Dataset for E. coli semantic segmentation.

    Expects directory structure:
        root/
        ├── images/
        │   ├── image1.png
        │   └── image2.png
        └── masks/
            ├── image1.png
            └── image2.png

    Masks are single-channel images where pixel values represent class IDs:
        0 = background
        1 = rod
        2 = dividing
        3 = microcolony

    Args:
        root: Root directory containing 'images' and 'masks' subdirectories
        transform: Optional transform to apply to both image and mask
        image_transform: Optional transform for image only (e.g., normalization)
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        image_transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "masks"
        self.transform = transform
        self.image_transform = image_transform

        # Get list of image files
        self.image_files = sorted(list(self.images_dir.glob("*.png")))

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.images_dir}")

        print(f"Loaded {len(self.image_files)} images from {root}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict:
        # Load image
        image_path = self.image_files[idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Load corresponding mask
        mask_path = self.masks_dir / image_path.name
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")

        # Apply joint transforms (augmentation)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Apply image-only transforms (normalization)
        if self.image_transform is not None:
            transformed = self.image_transform(image=image)
            image = transformed["image"]

        # Convert to tensors if not already
        if not isinstance(image, torch.Tensor):
            # Normalize to [0, 1] then to [-1, 1]
            image = torch.from_numpy(image).float().unsqueeze(0) / 255.0
            image = (image - 0.5) / 0.5

        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()

        return {
            "image": image,
            "mask": mask,
            "filename": image_path.name
        }


class EcoliSegmentationDatasetFromXML(Dataset):
    """
    Dataset that creates masks on-the-fly from XML annotations.

    Useful for quick experimentation without pre-generating masks.
    Slower than pre-generated masks but more flexible.

    Args:
        images_dir: Directory containing images
        annotations_dir: Directory containing XML annotations
        transform: Optional transform to apply
        class_priority: Order of class priority for overlapping boxes
    """

    CLASS_TO_ID = {
        "rod": 1,
        "dividing": 2,
        "microcolony": 3,
    }

    def __init__(
        self,
        images_dir: str,
        annotations_dir: str,
        transform: Optional[Callable] = None,
        class_priority: Tuple[int, ...] = (1, 2, 3),  # rod, dividing, microcolony
    ):
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.transform = transform
        self.class_priority = class_priority

        self.image_files = sorted(list(self.images_dir.glob("*.png")))

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def _parse_xml(self, xml_path: Path) -> list:
        """Parse XML annotation file."""
        import xml.etree.ElementTree as ET

        if not xml_path.exists():
            return []

        tree = ET.parse(xml_path)
        root = tree.getroot()

        objects = []
        for obj in root.findall("object"):
            name = obj.find("name").text.lower()
            box = obj.find("bndbox")

            bbox = (
                int(float(box.find("xmin").text)),
                int(float(box.find("ymin").text)),
                int(float(box.find("xmax").text)),
                int(float(box.find("ymax").text)),
            )

            class_id = self.CLASS_TO_ID.get(name, 0)
            if class_id > 0:
                objects.append({"class_id": class_id, "bbox": bbox})

        return objects

    def _create_mask(self, objects: list, height: int, width: int) -> np.ndarray:
        """Create segmentation mask from bounding boxes."""
        mask = np.zeros((height, width), dtype=np.uint8)

        # Sort by class priority (draw lower priority first)
        sorted_objects = sorted(objects, key=lambda x: self.class_priority.index(x["class_id"])
                                if x["class_id"] in self.class_priority else -1)

        for obj in sorted_objects:
            x1, y1, x2, y2 = obj["bbox"]
            # Clip to image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            mask[y1:y2, x1:x2] = obj["class_id"]

        return mask

    def __getitem__(self, idx: int) -> dict:
        # Load image
        image_path = self.image_files[idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        h, w = image.shape

        # Load and parse annotation
        xml_path = self.annotations_dir / f"{image_path.stem}.xml"
        objects = self._parse_xml(xml_path)

        # Create mask
        mask = self._create_mask(objects, h, w)

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Convert to tensors
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float().unsqueeze(0) / 255.0
            image = (image - 0.5) / 0.5

        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()

        return {
            "image": image,
            "mask": mask,
            "filename": image_path.name
        }
