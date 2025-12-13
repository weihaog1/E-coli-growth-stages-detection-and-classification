"""
CNN Data Preparation Script
============================

This script prepares the DeepBacs dataset specifically for our sliding-window CNN.

The Problem with the Original Approach:
--------------------------------------
The original code used IoU (Intersection over Union) to label patches:
  - If IoU > 0.3, label patch as that bacteria class
  - Otherwise, label as background

But bacteria are TINY compared to patches:
  - Average bacteria: ~28 x 19 pixels = 530 sq pixels
  - Patch size: 64 x 64 = 4096 sq pixels
  - Maximum possible IoU = 530/4096 = 0.13
  - Threshold was 0.3 → IMPOSSIBLE to label bacteria!

The Solution:
------------
Instead of IoU, we use CENTER-BASED labeling:
  - If a bacteria's CENTER POINT falls inside the patch → label as that class
  - This works regardless of bacteria size

Additionally, we:
  - Extract patches centered on each bacteria (guaranteed positive samples)
  - Extract random background patches
  - Balance the classes for better training

Usage:
------
    python scripts/prepare_cnn_data.py

Output:
-------
    data/cnn_patches/
    ├── train/
    │   ├── rod/           # Patches containing rod bacteria
    │   ├── dividing/      # Patches containing dividing bacteria
    │   ├── microcolony/   # Patches containing microcolonies
    │   └── background/    # Patches with no bacteria
    ├── val/
    │   └── (same structure)
    └── test/
        └── (same structure)
"""

import argparse
import random
import shutil
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.convert_annotations import parse_xml


# =============================================================================
# CONFIGURATION
# =============================================================================

CLASS_NAMES = {
    0: "rod",
    1: "dividing",
    2: "microcolony",
    3: "background"
}

DEFAULT_PATCH_SIZE = 64
DEFAULT_SEED = 42


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_bacteria_center(bbox: tuple) -> tuple:
    """
    Calculate the center point of a bounding box.

    Args:
        bbox: (x1, y1, x2, y2) in pixel coordinates

    Returns:
        (center_x, center_y)
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y


def is_point_in_patch(point: tuple, patch_box: tuple) -> bool:
    """
    Check if a point falls inside a patch.

    Args:
        point: (x, y) coordinates
        patch_box: (x1, y1, x2, y2) patch boundaries

    Returns:
        True if point is inside patch
    """
    px, py = point
    x1, y1, x2, y2 = patch_box
    return x1 <= px < x2 and y1 <= py < y2


def extract_patch(image: np.ndarray, center_x: int, center_y: int,
                  patch_size: int) -> np.ndarray:
    """
    Extract a patch centered at (center_x, center_y).

    Handles edge cases by clamping to image boundaries.

    Args:
        image: Grayscale image array
        center_x, center_y: Center point of patch
        patch_size: Size of patch (square)

    Returns:
        Extracted patch as numpy array, or None if invalid
    """
    h, w = image.shape
    half = patch_size // 2

    # Calculate patch boundaries
    x1 = center_x - half
    y1 = center_y - half
    x2 = x1 + patch_size
    y2 = y1 + patch_size

    # Clamp to image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    # Check if patch is valid size
    if x2 - x1 != patch_size or y2 - y1 != patch_size:
        return None

    return image[y1:y2, x1:x2].copy()


def find_random_background_position(image_shape: tuple, objects: list,
                                     patch_size: int, max_attempts: int = 50) -> tuple:
    """
    Find a random position that doesn't contain any bacteria.

    Args:
        image_shape: (height, width) of image
        objects: List of bacteria objects with 'bbox' key
        patch_size: Size of patch
        max_attempts: Maximum random attempts before giving up

    Returns:
        (x, y) position, or None if couldn't find valid position
    """
    h, w = image_shape
    half = patch_size // 2

    for _ in range(max_attempts):
        # Random center point (ensuring patch fits in image)
        cx = random.randint(half, w - half - 1)
        cy = random.randint(half, h - half - 1)

        # Check if any bacteria center falls in this patch
        patch_box = (cx - half, cy - half, cx + half, cy + half)

        has_bacteria = False
        for obj in objects:
            bacteria_center = get_bacteria_center(obj["bbox"])
            if is_point_in_patch(bacteria_center, patch_box):
                has_bacteria = True
                break

        if not has_bacteria:
            return cx, cy

    return None


# =============================================================================
# MAIN EXTRACTION LOGIC
# =============================================================================

def extract_patches_from_image(image_path: Path, annotation_path: Path,
                                patch_size: int,
                                background_per_image: int = 5) -> dict:
    """
    Extract all patches from a single image.

    Strategy:
    1. For each bacteria, extract a patch centered on it
    2. Extract additional random background patches

    Args:
        image_path: Path to image file
        annotation_path: Path to XML annotation file
        patch_size: Size of patches to extract
        background_per_image: Number of background patches per image

    Returns:
        Dictionary with class names as keys, list of patches as values
    """
    # Load image
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"  Warning: Could not load {image_path}")
        return {}

    # Load annotations
    if not annotation_path.exists():
        print(f"  Warning: No annotation for {image_path.name}")
        return {}

    annotation = parse_xml(str(annotation_path))
    objects = annotation["objects"]

    # Storage for extracted patches
    patches = defaultdict(list)

    # ----- Extract patches centered on each bacteria -----
    for obj in objects:
        class_id = obj["class_id"]
        if class_id == -1:
            continue  # Unknown class

        class_name = CLASS_NAMES[class_id]

        # Get bacteria center
        cx, cy = get_bacteria_center(obj["bbox"])
        cx, cy = int(cx), int(cy)

        # Extract patch
        patch = extract_patch(image, cx, cy, patch_size)
        if patch is not None:
            patches[class_name].append(patch)

    # ----- Extract random background patches -----
    for _ in range(background_per_image):
        pos = find_random_background_position(image.shape, objects, patch_size)
        if pos is not None:
            cx, cy = pos
            patch = extract_patch(image, cx, cy, patch_size)
            if patch is not None:
                patches["background"].append(patch)

    return patches


def process_dataset(source_dir: Path, output_dir: Path, split: str,
                    patch_size: int, background_per_image: int) -> dict:
    """
    Process all images in a dataset split.

    Args:
        source_dir: Directory containing image_patches folder
        output_dir: Output directory for patches
        split: 'train', 'val', or 'test'
        patch_size: Size of patches
        background_per_image: Background patches per image

    Returns:
        Dictionary with class counts
    """
    # Determine source directories based on split
    if split in ["train", "val"]:
        images_dir = source_dir / "train_images"
        annotations_dir = source_dir / "train_annotations"
    else:
        images_dir = source_dir / "test_images"
        annotations_dir = source_dir / "test_annotations"

    # Get image list
    image_files = sorted(images_dir.glob("*.png"))

    # For train/val, we need to split the training images
    if split in ["train", "val"]:
        random.seed(DEFAULT_SEED)  # Consistent split
        image_files = list(image_files)
        random.shuffle(image_files)

        n_val = int(len(image_files) * 0.2)
        if split == "val":
            image_files = image_files[:n_val]
        else:
            image_files = image_files[n_val:]

    print(f"\nProcessing {split} split: {len(image_files)} images")

    # Create output directories
    for class_name in CLASS_NAMES.values():
        (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    # Process each image
    class_counts = defaultdict(int)
    patch_id = 0

    for img_path in image_files:
        xml_path = annotations_dir / f"{img_path.stem}.xml"

        patches = extract_patches_from_image(
            img_path, xml_path, patch_size, background_per_image
        )

        # Save patches
        for class_name, patch_list in patches.items():
            for patch in patch_list:
                output_path = output_dir / split / class_name / f"{patch_id:05d}.png"
                cv2.imwrite(str(output_path), patch)
                patch_id += 1
                class_counts[class_name] += 1

    return dict(class_counts)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare patch dataset for CNN training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/prepare_cnn_data.py
    python scripts/prepare_cnn_data.py --patch-size 32
    python scripts/prepare_cnn_data.py --background-per-image 10
        """
    )

    parser.add_argument(
        "--source",
        type=str,
        default="../../data/raw/image_patches",
        help="Path to source dataset (relative to models/CNN-sliding-window/)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="../../data/cnn_patches",
        help="Path to output directory (relative to models/CNN-sliding-window/)"
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=DEFAULT_PATCH_SIZE,
        help=f"Size of patches (default: {DEFAULT_PATCH_SIZE})"
    )

    parser.add_argument(
        "--background-per-image",
        type=int,
        default=5,
        help="Number of background patches to extract per image (default: 5)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})"
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    source_dir = Path(args.source)
    output_dir = Path(args.output)

    # Clear output directory if it exists
    if output_dir.exists():
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    print("=" * 60)
    print("CNN Patch Extraction")
    print("=" * 60)
    print(f"Source:     {source_dir}")
    print(f"Output:     {output_dir}")
    print(f"Patch size: {args.patch_size}x{args.patch_size}")
    print(f"Background patches per image: {args.background_per_image}")

    # Process each split
    all_counts = {}

    for split in ["train", "val", "test"]:
        counts = process_dataset(
            source_dir, output_dir, split,
            args.patch_size, args.background_per_image
        )
        all_counts[split] = counts

        # Print summary for this split
        total = sum(counts.values())
        print(f"\n  {split} patches extracted: {total}")
        for class_name in CLASS_NAMES.values():
            count = counts.get(class_name, 0)
            pct = 100 * count / total if total > 0 else 0
            print(f"    {class_name:12}: {count:5} ({pct:5.1f}%)")

    # Print final summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for split in ["train", "val", "test"]:
        counts = all_counts[split]
        total = sum(counts.values())
        print(f"\n{split.upper()}: {total} patches")
        for class_name in CLASS_NAMES.values():
            count = counts.get(class_name, 0)
            print(f"  {class_name:12}: {count}")

    print("\n" + "=" * 60)
    print("Done! Patches saved to:", output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
