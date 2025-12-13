"""
Mask Preparation Script for U-Net Training

Converts Pascal VOC XML annotations to segmentation masks.
Creates output structure suitable for U-Net training.

Usage:
    python scripts/prepare_masks.py --source data/raw/image_patches --output data/unet_original
    python scripts/prepare_masks.py --source data/Augmented/augmented_images --output data/unet_augmented
"""

import argparse
import shutil
import random
from pathlib import Path
from collections import defaultdict
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from tqdm import tqdm


# =============================================================================
# CONFIGURATION
# =============================================================================

CLASS_TO_ID = {
    "rod": 1,
    "dividing": 2,
    "microcolony": 3,
}

CLASS_NAMES = {
    0: "background",
    1: "rod",
    2: "dividing",
    3: "microcolony"
}

# Class priority for overlapping boxes (higher ID = higher priority)
# This ensures rare classes (microcolony) are not overwritten
CLASS_PRIORITY = [1, 2, 3]  # rod, dividing, microcolony

DEFAULT_SEED = 42
VAL_SPLIT = 0.2  # 20% of training data for validation


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_xml(xml_path: str) -> dict:
    """
    Parse a Pascal VOC XML file.

    Returns:
        {
            "width": int,
            "height": int,
            "objects": [{"class_name": str, "class_id": int, "bbox": (x1,y1,x2,y2)}]
        }
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

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

        class_id = CLASS_TO_ID.get(name, 0)
        if class_id > 0:
            objects.append({
                "class_name": name,
                "class_id": class_id,
                "bbox": bbox,
            })

    return {"width": width, "height": height, "objects": objects}


def create_mask(objects: list, height: int, width: int) -> np.ndarray:
    """
    Create segmentation mask from bounding boxes.

    Fills bounding box regions with class IDs.
    Higher priority classes overwrite lower priority ones.

    Args:
        objects: List of objects with 'class_id' and 'bbox' keys
        height, width: Image dimensions

    Returns:
        Mask array of shape (height, width) with values 0-3
    """
    mask = np.zeros((height, width), dtype=np.uint8)

    # Sort objects by class priority (draw lower priority first)
    sorted_objects = sorted(
        objects,
        key=lambda x: CLASS_PRIORITY.index(x["class_id"]) if x["class_id"] in CLASS_PRIORITY else -1
    )

    for obj in sorted_objects:
        x1, y1, x2, y2 = obj["bbox"]

        # Clip to image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        # Fill bounding box with class ID
        mask[y1:y2, x1:x2] = obj["class_id"]

    return mask


def process_split(
    images_dir: Path,
    annotations_dir: Path,
    output_dir: Path,
    split_name: str,
    image_files: list
) -> dict:
    """
    Process a dataset split (train/val/test).

    Args:
        images_dir: Directory containing images
        annotations_dir: Directory containing XML annotations
        output_dir: Base output directory
        split_name: Name of split ('train', 'val', 'test')
        image_files: List of image file paths

    Returns:
        Dictionary with class statistics
    """
    # Create output directories
    split_dir = output_dir / split_name
    images_out = split_dir / "images"
    masks_out = split_dir / "masks"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)

    stats = defaultdict(int)

    for img_path in tqdm(image_files, desc=f"Processing {split_name}"):
        # Load image (to verify it exists)
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Could not load {img_path}")
            continue

        h, w = image.shape

        # Find annotation file
        xml_path = annotations_dir / f"{img_path.stem}.xml"
        if not xml_path.exists():
            print(f"Warning: No annotation for {img_path.name}")
            continue

        # Parse annotation and create mask
        annotation = parse_xml(str(xml_path))
        mask = create_mask(annotation["objects"], h, w)

        # Copy image
        shutil.copy(img_path, images_out / img_path.name)

        # Save mask
        cv2.imwrite(str(masks_out / img_path.name), mask)

        # Collect statistics
        for cls_id in range(4):
            stats[CLASS_NAMES[cls_id]] += (mask == cls_id).sum()

        stats["num_images"] += 1
        stats["num_objects"] += len(annotation["objects"])

    return dict(stats)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare segmentation masks from XML annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (run from models/U-Net/ directory):
    # Original dataset
    python scripts/prepare_masks.py --source ../../data/raw/image_patches --output ../../data/unet_original

    # Augmented dataset
    python scripts/prepare_masks.py --source ../../data/Augmented/augmented_images --output ../../data/unet_augmented
        """
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source directory containing train_images, train_annotations, test_images, test_annotations"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for processed data"
    )

    parser.add_argument(
        "--val-split",
        type=float,
        default=VAL_SPLIT,
        help=f"Fraction of training data for validation (default: {VAL_SPLIT})"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})"
    )

    args = parser.parse_args()

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    source_dir = Path(args.source)
    output_dir = Path(args.output)

    # Verify source structure
    train_images = source_dir / "train_images"
    train_annotations = source_dir / "train_annotations"
    test_images = source_dir / "test_images"
    test_annotations = source_dir / "test_annotations"

    if not train_images.exists():
        raise ValueError(f"train_images not found: {train_images}")
    if not train_annotations.exists():
        raise ValueError(f"train_annotations not found: {train_annotations}")

    # Clear output directory
    if output_dir.exists():
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    print("=" * 60)
    print("U-Net Mask Preparation")
    print("=" * 60)
    print(f"Source:     {source_dir}")
    print(f"Output:     {output_dir}")
    print(f"Val split:  {args.val_split}")
    print(f"Seed:       {args.seed}")

    # Get training images and split into train/val
    train_image_files = sorted(list(train_images.glob("*.png")))
    random.shuffle(train_image_files)

    n_val = int(len(train_image_files) * args.val_split)
    val_files = train_image_files[:n_val]
    train_files = train_image_files[n_val:]

    print(f"\nTrain images: {len(train_files)}")
    print(f"Val images:   {len(val_files)}")

    # Process training split
    print("\n" + "-" * 40)
    train_stats = process_split(
        train_images, train_annotations, output_dir, "train", train_files
    )

    # Process validation split
    print("\n" + "-" * 40)
    val_stats = process_split(
        train_images, train_annotations, output_dir, "val", val_files
    )

    # Process test split (if exists)
    test_stats = {}
    if test_images.exists() and test_annotations.exists():
        test_files = sorted(list(test_images.glob("*.png")))
        print(f"\nTest images: {len(test_files)}")
        print("\n" + "-" * 40)
        test_stats = process_split(
            test_images, test_annotations, output_dir, "test", test_files
        )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for split_name, stats in [("Train", train_stats), ("Val", val_stats), ("Test", test_stats)]:
        if not stats:
            continue
        print(f"\n{split_name}:")
        print(f"  Images:  {stats.get('num_images', 0)}")
        print(f"  Objects: {stats.get('num_objects', 0)}")
        print(f"  Pixels per class:")
        for cls_name in ["background", "rod", "dividing", "microcolony"]:
            pixels = stats.get(cls_name, 0)
            total = sum(stats.get(c, 0) for c in ["background", "rod", "dividing", "microcolony"])
            pct = 100 * pixels / total if total > 0 else 0
            print(f"    {cls_name:12}: {pixels:10,} ({pct:5.1f}%)")

    print("\n" + "=" * 60)
    print(f"Done! Data saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
